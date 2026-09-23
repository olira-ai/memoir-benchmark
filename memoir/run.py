"""Run MEMOIR against a substrate.

    python -m memoir.run --substrate examples.local_events:LocalEventsSubstrate

Questions are grouped by patient, so one agent, with one set of store connections,
serves all of that patient's questions. Patients run concurrently; within a single
question the tool loop is sequential.

Results stream to JSONL as they complete, one row per question, so a long run can
be resumed with ``--resume`` and inspected while it is still going.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from memoir.dataset import Question, load_questions
from memoir.harness.agent import BenchmarkAgent
from memoir.report import format_report
from memoir.scoring.score import Score, aggregate, score_answer
from memoir.substrate import Substrate
from memoir.tasks import TASKS


def load_substrate(spec: str) -> Substrate:
    """Instantiate a substrate from a ``module.path:Attribute`` spec.

    The attribute may be a :class:`~memoir.substrate.Substrate` subclass, an
    instance, or a zero-argument factory returning one.
    """
    if ":" not in spec:
        raise SystemExit(
            f"--substrate must look like 'my_package.my_module:MySubstrate', got {spec!r}"
        )
    module_name, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"could not import {module_name!r}: {exc}") from exc
    try:
        obj = getattr(module, attr)
    except AttributeError as exc:
        raise SystemExit(f"{module_name!r} has no attribute {attr!r}") from exc

    if isinstance(obj, Substrate):
        return obj
    if isinstance(obj, type) and issubclass(obj, Substrate):
        return obj()
    if callable(obj):
        built = obj()
        if isinstance(built, Substrate):
            return built
    raise SystemExit(f"{spec} is not a Substrate, a Substrate subclass, or a factory")


def _completed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["question_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


async def _run_patient(
    substrate: Substrate,
    patient_id: str,
    questions: list[Question],
    *,
    llm: BaseChatModel | None,
    model: str | None,
    judge_llm: BaseChatModel | None,
    judge_model: str | None,
    with_grounding: bool,
    include_trace: bool,
    write: Any,
    lock: asyncio.Lock,
    progress: dict[str, int],
) -> list[Score]:
    """Answer and score every question for one patient over a warm agent."""
    agent = BenchmarkAgent(substrate, patient_id, llm=llm, model=model)
    scores: list[Score] = []
    for q in questions:
        error: str | None = None
        try:
            result = await agent.ask(
                q.question, question_id=q.question_id, include_trace=include_trace
            )
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                raise
            # A failed row is recorded and scored 0 rather than aborting the run.
            error = f"{type(exc).__name__}: {exc}"[:500]
            from memoir.harness.result import AnswerResult

            result = AnswerResult(
                question_id=q.question_id,
                patient_id=patient_id,
                substrate=substrate.name,
                model=agent.model,
                question=q.question,
                answer="",
                error=error,
            )

        score = await score_answer(
            q,
            result.answer,
            error=error,
            judge_llm=judge_llm,
            judge_model=judge_model,
            with_grounding=with_grounding,
        )
        scores.append(score)

        row = {**result.to_dict(), **score.to_dict()}
        async with lock:
            write(json.dumps(row, ensure_ascii=False) + "\n")
            progress["done"] += 1
            if progress["done"] % 25 == 0 or progress["done"] == progress["total"]:
                print(
                    f"  {progress['done']}/{progress['total']} questions",
                    file=sys.stderr,
                    flush=True,
                )
    return scores


async def run(
    substrate: Substrate,
    questions: Iterable[Question],
    *,
    out_path: Path,
    model: str | None = None,
    llm: BaseChatModel | None = None,
    judge_llm: BaseChatModel | None = None,
    judge_model: str | None = None,
    concurrency: int = 4,
    with_grounding: bool = True,
    include_trace: bool = False,
    resume: bool = False,
) -> list[Score]:
    """Answer and score a question set, streaming results to ``out_path``."""
    by_patient: dict[str, list[Question]] = defaultdict(list)
    skip = _completed_ids(out_path) if resume else set()
    for q in questions:
        if q.question_id not in skip:
            by_patient[q.patient_id].append(q)

    total = sum(len(v) for v in by_patient.values())
    if skip:
        print(f"resuming: {len(skip)} already scored, {total} to go", file=sys.stderr)
    if not total:
        print("nothing to run", file=sys.stderr)
        return []

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()
    progress = {"done": 0, "total": total}
    started = time.perf_counter()

    with out_path.open("a" if resume else "w", encoding="utf-8") as fh:

        async def _guarded(pid: str, qs: list[Question]) -> list[Score]:
            async with gate:
                return await _run_patient(
                    substrate,
                    pid,
                    qs,
                    llm=llm,
                    model=model,
                    judge_llm=judge_llm,
                    judge_model=judge_model,
                    with_grounding=with_grounding,
                    include_trace=include_trace,
                    write=fh.write,
                    lock=lock,
                    progress=progress,
                )

        try:
            batches = await asyncio.gather(
                *(_guarded(pid, qs) for pid, qs in sorted(by_patient.items()))
            )
        finally:
            await substrate.aclose()

    elapsed = time.perf_counter() - started
    print(f"finished {total} questions in {elapsed / 60:.1f} min", file=sys.stderr)
    return [s for batch in batches for s in batch]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m memoir.run", description=__doc__.split("\n\n")[0]
    )
    p.add_argument(
        "--substrate",
        required=True,
        help="Substrate to evaluate, as 'module.path:Attribute'.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSONL results path (default: runs/<substrate>.jsonl).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Answering model id (default: MEMOIR_AGENT_MODEL).",
    )
    p.add_argument(
        "--judge-model",
        default=None,
        help="Grading model id (default: MEMOIR_JUDGE_MODEL).",
    )
    p.add_argument("--task", action="append", choices=TASKS, help="Repeatable filter.")
    p.add_argument("--patient", action="append", help="Repeatable patient id filter.")
    p.add_argument("--template", action="append", help="Repeatable template id filter.")
    p.add_argument(
        "--limit-per-patient",
        type=int,
        default=None,
        help="Keep only the first N questions per patient, useful for a smoke run.",
    )
    p.add_argument("--concurrency", type=int, default=4, help="Patients answered in parallel.")
    p.add_argument(
        "--no-grounding", action="store_true", help="Skip the two grounding checks."
    )
    p.add_argument(
        "--trace", action="store_true", help="Record the full message trace per row."
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing results file, skipping questions already scored.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    substrate = load_substrate(args.substrate)
    out_path = args.out or Path("runs") / f"{substrate.name}.jsonl"

    questions = load_questions(
        tasks=tuple(args.task) if args.task else None,
        patient_ids=tuple(args.patient) if args.patient else None,
        templates=tuple(args.template) if args.template else None,
        limit_per_patient=args.limit_per_patient,
    )
    print(
        f"{substrate.name}: {len(questions)} questions, "
        f"{len({q.patient_id for q in questions})} patients -> {out_path}",
        file=sys.stderr,
    )

    scores = asyncio.run(
        run(
            substrate,
            questions,
            out_path=out_path,
            model=args.model,
            judge_model=args.judge_model,
            concurrency=args.concurrency,
            with_grounding=not args.no_grounding,
            include_trace=args.trace,
            resume=args.resume,
        )
    )
    if scores:
        print(format_report(aggregate(scores, substrate=substrate.name)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
