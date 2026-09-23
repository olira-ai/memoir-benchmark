# MEMOIR: Memory Evaluation over Multi-year Oncology Inference & Retrieval

**M**emory **E**valuation over **M**ulti-year **O**ncology **I**nference & **R**etrieval

A benchmark for the memory substrate underneath a clinical agent: 117 synthetic
oncology patients with complete multi-year records, 3,617 questions across four
tasks, the scorers that grade them, and the agent harness that holds everything
above the memory layer constant.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22906151.svg)](https://doi.org/10.5281/zenodo.22906151)

**Study.** [How Context Representation Shapes Longitudinal Clinical Reasoning](https://www.olira.ai/blog/the-wrong-shape-of-memory)  
G. Efstathiadis and P. Emedom-Nnamdi · Olira · September 2026 · [10.5281/zenodo.22906151](https://doi.org/10.5281/zenodo.22906151)

---

## Why this exists

The current wave of agent-memory tools was built to solve a specific problem:
letting a chatbot remember its own conversations. The unit of memory is the
dialogue, and the data was produced by the agent and its user in the first place.

A patient record is a different shape. It arrives from many systems, in many
forms, at cadences that differ by orders of magnitude: vitals sampled by the
minute, a lab panel every few weeks, an imaging report once a year, a clinician's
note whenever someone happens to write one. It spans years, and much of it is too
dense to hand to a model directly. The question stops being *what did we say* and
becomes *what do we know about this patient*.

MEMOIR asks a narrow, falsifiable version of that: **which substrate lets an agent
answer clinical questions faithfully, without inventing dates, drugs, or diagnoses
that aren't in the record, and at what cost in tokens, latency, and round-trips?**

It is designed so the memory layer is the only thing that varies. Same data, same
agent scaffold, same prompt skeleton, same decoding settings, same scorers. What
changes between runs is the substrate underneath and the tools it exposes.

## Install

```bash
git clone https://github.com/olira-ai/memoir-benchmark
cd memoir-benchmark
pip install -e .
```

The corpus ships in the repository, so there is nothing further to download.

## Bring your own model

MEMOIR does not bundle a provider or pick a model for you. Point it at any
endpoint that speaks the OpenAI chat-completions wire protocol, which covers the
commercial APIs and most self-hosted servers (vLLM, Ollama, LiteLLM,
text-generation-inference), and name two models: one to answer questions, one to
grade the open-ended tasks.

```bash
export MEMOIR_LLM_BASE_URL=https://your-endpoint.example.com/v1
export MEMOIR_LLM_API_KEY=...
export MEMOIR_AGENT_MODEL=...    # answers questions
export MEMOIR_JUDGE_MODEL=...    # grades open-ended answers, usually smaller
```

| Variable | Purpose |
|---|---|
| `MEMOIR_LLM_BASE_URL` | Endpoint both roles use |
| `MEMOIR_LLM_API_KEY` | Key both roles use |
| `MEMOIR_AGENT_MODEL` | Model that answers questions |
| `MEMOIR_JUDGE_MODEL` | Model that grades the open-ended tasks |
| `MEMOIR_AGENT_MODEL_KWARGS` | JSON of extra parameters for the agent |
| `MEMOIR_JUDGE_MODEL_KWARGS` | JSON of extra parameters for the judge |
| `MEMOIR_AGENT_BASE_URL` / `MEMOIR_AGENT_API_KEY` | Endpoint override for the agent alone |
| `MEMOIR_JUDGE_BASE_URL` / `MEMOIR_JUDGE_API_KEY` | Endpoint override for the judge alone |
| `MEMOIR_DATA_DIR` | Read the corpus from somewhere other than `data/` |

Decoding should be held as fixed as your model family allows, and what that means
differs by family: `{"temperature": 0}` for most, `{"reasoning_effort": "low"}` for
a reasoning model that rejects temperature. Set it explicitly rather than relying
on a default, and report what you used alongside any number you publish.

```bash
export MEMOIR_AGENT_MODEL_KWARGS='{"temperature": 0}'
```

If none of this suits, skip the environment entirely and hand in a model you built
yourself. Anything implementing LangChain's `BaseChatModel` is accepted:

```python
BenchmarkAgent(substrate, patient_id, llm=my_chat_model)
await score_answer(question, answer, judge_llm=my_small_chat_model)
```

## Quickstart

Run the bundled reference substrate over two questions per patient:

```bash
python -m memoir.run \
  --substrate examples.local_events:LocalEventsSubstrate \
  --limit-per-patient 2
```

Results stream to `runs/local-events.jsonl`, one row per question, carrying the
answer, its score, and the cost of getting it. Report on them:

```bash
python -m memoir.report runs/local-events.jsonl
```

`examples/local_events.py` is a worked example of the interface, not a system from
the study. It reads the corpus straight off disk. Use it to confirm your endpoint
and models are wired up, and as the thing you copy when writing your own.

**Do not treat it as a baseline.** Lossless access to the typed record is a strong
position on this benchmark, not a weak one, and this substrate scores around 0.89
overall on a 59-question sample, level with the study's raw-access control. Its
`total_matched` gives exact counts for free and its `contains` greps full note text
with nothing in between, so it sidesteps the series-truncation failure that costs
the real control most of its trajectory score. Your substrate scoring below it is
not by itself evidence that your substrate is weak.

## Evaluating your own substrate

A substrate answers two questions: *what tools does the agent get for this
patient*, and *what does the agent need to know about them*. The harness supplies
everything else.

```python
from memoir import SimpleSubstrate


class MyStore(SimpleSubstrate):
    name = "my-store"

    def build_tools(self, patient_id):
        return [make_search_tool(patient_id)]

    def prompt_section(self, patient_id):
        return "`search(query)` runs semantic search over this patient's record."
```

```bash
python -m memoir.run --substrate my_package.my_module:MyStore
```

Subclass `Substrate` directly when a patient needs setup and teardown, such as a
connection pool, a client, or a session, and implement `session()` as an async
context manager.

**Ingestion is yours.** Load each patient's event stream into your store however
it ingests best. In the study each system was tuned rather than forced into one
shape, so that none was handicapped by a format it parses badly:

```python
from memoir import load_cohort

for patient in load_cohort():
    for event in patient.events():
        my_store.ingest(patient.patient_id, event.event_type, event.timestamp, event.payload)
```

### The trend calculator

Some questions come down to the direction of a single marker over the window, so a
substrate that rebuilds the series at query time could lose on regression
arithmetic rather than on what it stored. Every substrate that reconstructs at
runtime is therefore handed the same `compute_trend` tool, a plain OLS fit plus a
two-sided Student-t significance test, automatically.

A substrate that answers trend questions from something it computed ahead of time
declares `precomputes_trends = True` and does not receive it. Reading a
pre-computed trend is the capability under test there, not arithmetic.

## The data

The corpus is built from [HL7/CodeX mCODE Test Data][mcode], synthetic oncology
records generated with [Synthea][synthea], conformant to the mCODE standard, and
released without privacy or usage restrictions. Every record is a *complete*
patient history rather than just the cancer-relevant slice: non-cancer encounters,
unrelated conditions, incidental medications and all.

From the full corpus we selected 117 patients by picking, for each, the densest
window of clinical activity, a couple of hundred events at minimum and around 500
at the median, while still allowing multi-year quiet stretches.

[mcode]: https://confluence.hl7.org/spaces/COD/pages/80119851/mCODE+Test+Data
[synthea]: https://github.com/synthetichealth/synthea

| | |
|---|---|
| Patients | 117 |
| Clinical events | 65,377 |
| Questions | 3,617 |
| Question templates | 72 defined, 64 instantiated |
| Event types | 15 |
| Record spans | multi-decade, per patient |

```
data/
├── events/<patient_id>.ndjson   one event per line, ascending by timestamp
├── questions.ndjson             one question per line
├── cohort.json                  per-patient manifest
└── templates.json               the templates the questions were built from
```

See [`data/README.md`](data/README.md) for the full field-by-field schema, the
event-type vocabulary, and the known quirks of the source encoding.

```python
from memoir import load_cohort, load_events, load_questions

cohort = load_cohort()
events = load_events(cohort[0].patient_id)
trends = load_questions(tasks=("trajectory",))
```

## The tasks

Four task types, escalating roughly from point lookups to whole-timeline
reasoning. Each stands in for a question a clinician or patient might actually
ask, and each is phrased purely in clinical terms. A question never assumes a
particular internal representation exists.

| Task | Questions | Templates | Example |
|---|---|---|---|
| **Factual**, point-in-time lookup | 541 | 15 | *"How many medications is this patient currently on?"* |
| **Needle**, event detection | 584 | 17 | *"Was this patient ever prescribed an opioid?"* |
| **Compilation**, clinical episode synthesis | 1,328 | 15 | *"Walk me through what was happening around her last chemo session. What labs were drawn, and what do they suggest about how she's tolerating treatment?"* |
| **Time series & trajectory**, temporal reasoning | 1,164 | 25 | *"Has her creatinine been heading in the right direction, and overall, is she doing better or worse than she was a year ago?"* |

**Factual** answers are retrievable from the latest state of the record, with no
temporal reasoning required. The record treated as a static snapshot.

**Needle** questions force a scan of the entire record for a specific event or
pattern. This is the task that most directly tests the heavy tail, because a needle
is by definition a low-frequency event. Each is tagged by where the evidence
lives: `structured` for an atomic event in a typed payload, `free_text` for one
spun into the prose of a clinical note, `multi_event` for one that requires
joining several. The distinction matters, because the atomic rare event survives a
distill-to-facts memory and the free-text one is precisely what gets stripped first.

**Compilation** questions anchor on a pivotal event and ask for a narrative
drawing together everything that co-occurred around it, typically across two or
more event types. This is where you learn whether a system can assemble a coherent
clinical picture or only return isolated facts.

**Time series & trajectory** comes in two kinds, carried on each row as
`question_kind`. `marker_trend` asks for the first and last value of one biomarker
across the window and its direction. `timeline_trajectory` asks about the
directional arc of the whole record: is treatment escalating or tapering, is the
medication burden growing, are labs and vitals trending better or worse.

## How answers are scored

Different question types need different graders, so each task has one primary
scorer, named on every dataset row.

| Task | Scorer | How |
|---|---|---|
| Factual | `semantic_equivalence` | Model judge, because an exact string match fails on paraphrase and on questions where either of two answers is correct |
| Needle | `token_recall` | Recall of the key facts, so a thorough answer is not punished for context |
| Compilation | `rubric_judge` | Model judge against the rubric written for that question, 0 to 3 rescaled to 0 and 1 |
| Trajectory (`marker_trend`) | `direction_match` | Deterministic on the direction label |
| Trajectory (`timeline_trajectory`) | `rubric_judge` | As compilation, with its own calibration anchors |

The deterministic direction scorer is what makes the trend result load-bearing: a
wrong label means the *series itself was built wrong*, not that a judge was harsh.

Judge prompts are kept disciplined: a rubric, a few calibration examples, and
little else. The calibration matters more than it looks. Without anchors the judge
drifts, and its most common error is deducting for *extra* correct detail the
reference answer simply did not enumerate. A rubric judge is still a rubric judge,
with the usual caveats.

### Grounding

Alongside those sit two checks that ask not whether an answer is *right* but
whether it is *invented*. For every answer, the dates and clinical entities it
cites are pulled out and verified against the patient's own record: a date matching
no real event, a medication or diagnosis appearing nowhere in the history.

In a clinical setting this is the failure mode that matters most. A fluent,
confident, wrong answer is far more dangerous than "I couldn't find that", and a
grader reading only for plausibility will happily reward the first. Both checks are
deliberately permissive, so a low score is a strong fabrication signal rather than
a noisy artifact.

### Overall

`overall` is the **mean of the four task scores, equally weighted**, not a mean
across all questions. The tasks differ in size by more than a factor of two, so a
question-weighted mean would let the largest task quietly decide the headline
number.

Efficiency is reported as a first-class result, not a footnote: tool calls,
end-to-end latency, and token usage per question, plus each per unit of score
actually earned. The claim a benchmark like this can make is not just "more
accurate" but "more accurate *and* cheaper", and that claim is only available if
the cost was measured from the start.

## The harness

Everything above the memory substrate is held constant: the same agent scaffold,
the same prompt structure, the same decoding settings, the same orchestration.

The evaluation is **single-turn**, one question in and one answer out, with no
multi-session conversation. But single-turn is not single-tool-call. Within a turn
the agent runs a full loop, calling tools, reading results, calling again, until it
has what it needs. What is measured is whether a substrate lets an agent retrieve
the right information in one sitting, and with how much effort.

The loop is bounded by a graph recursion limit, roughly two dozen tool
round-trips, rather than an arbitrary iteration cap. One agent is built per
patient and reused across that patient's questions, so its model client and store
connections stay warm. A failed row is recorded and scored 0 rather than aborting
the run.

The prompt skeleton in [`memoir/harness/prompts/base.yaml`](memoir/harness/prompts/base.yaml)
is byte-identical across every substrate. Three slots are filled per run: the
event-type vocabulary, a calendar of when this patient's events actually occurred,
and the substrate's own tool documentation. A substrate customises only the last.

## Reference results

From the study accompanying this release. Every system was driven by the same
harness, the same questions, and the same scorers; the substrate was the only
variable. `Olira State` is a materialized patient state; `Olira State lite` is the
same platform with materialization switched off, which makes the pair a controlled
ablation on *when* the work of structuring the record happens.

| Substrate | Paradigm | Overall | Factual | Needle | Compilation | Time series & trajectory |
|---|---|---|---|---|---|---|
| **Olira State** | Materialized patient state | **0.923** | **0.971** | **0.802** | **0.958** | **0.962** |
| **Olira State lite** | Raw typed event stream (control) | 0.889 | 0.970 | 0.769 | 0.954 | 0.863 |
| **mem0** | Passive vector extraction | 0.832 | 0.867 | 0.760 | 0.852 | 0.850 |
| **Graphiti** | Temporal knowledge graph | 0.823 | 0.852 | 0.722 | 0.861 | 0.857 |
| **Letta** | Agentic context paging | 0.612 | 0.734 | 0.642 | 0.399 | 0.672 |

| Substrate | Tokens | Latency | Tool calls | Tokens / correct | Calls / correct |
|---|---|---|---|---|---|
| Olira State | 56.7K | **15.0s** | **2.15** | 60.5K | **2.30** |
| Olira State lite | **38.4K** | 16.0s | 2.97 | **42.2K** | 3.26 |
| mem0 | 43.6K | 24.7s | 3.05 | 51.2K | 3.57 |
| Graphiti | 39.2K | 23.0s | 2.87 | 46.0K | 3.37 |
| Letta | 42.2K | 21.6s | 5.65 | 69.8K | 9.36 |

Three of these systems were built for conversational recall, not for reasoning over
a dense, multi-year clinical record. Pointed at a problem they were never designed
for, it is unsurprising that they give ground, and that is not the interesting part
of the result. What *is* interesting is that every system fails in a way that maps
precisely onto how it represents the record. The full reading is in the [blog
post][blog].

A note on reproducing these exactly: scores depend on the answering model, the
judge model, and the ingestion each substrate was given. Report those alongside any
number you publish. Rubric judges also carry run-to-run variance, so take
replicates before reading a small gap as real.

## Fairness and limitations

A benchmark whose authors also built one of the entrants owes the reader its
counter-evidence up front.

**The questions were written by a model.** The corpus was built by a reproducible
authoring engine rather than curated by hand: it explores each patient's raw
events, identifies signal-rich anchor points, and instantiates parameterized
templates against the specific facts it finds. That makes the set reproducible and
auditable end to end, and it also means the gold answers inherit an authoring
model's phrasing and its blind spots. What keeps it honest is that every gold
answer is anchored to specific, verifiable events in the raw record.
`test_every_anchor_resolves_to_its_event` checks all 17,232 of them against the
event streams on every run, so there are no answers that exist only in our heads.

**Scope.** mCODE is EHR data: notes, plans, labs, visits, medications, conditions.
It contains no wearables, no behavioral signals, no continuous sensor time series.
The strongest case for building state offline, that a single day of a
high-frequency stream blows past a context window on its own, is exactly the case
this dataset cannot exercise. MEMOIR tests the EHR slice.

## Repository layout

```
memoir/
├── dataset.py              corpus loaders
├── substrate.py            the interface you implement
├── tasks.py                the task taxonomy
├── models.py               chat models for the agent and the judge
├── run.py                  batch runner
├── report.py               accuracy grid and efficiency table
├── harness/
│   ├── agent.py            the shared agent scaffold
│   ├── prompt.py           prompt assembly
│   ├── prompts/base.yaml   the skeleton, identical across substrates
│   └── result.py           one answer and its measurements
├── scoring/
│   ├── score.py            routing and aggregation
│   ├── text.py             deterministic scorers
│   ├── judge.py            the rubric judge and its calibration
│   └── grounding.py        date and entity grounding
└── tools/trend.py          the shared compute_trend calculator
```

## Development

```bash
pip install -e '.[dev]'
pytest            # corpus integrity, scorers, harness wiring; no model calls
ruff check .
```

## Citation

Please cite the study ([DOI](https://doi.org/10.5281/zenodo.22906151)):

```bibtex
@misc{memoir2026,
  title        = {MEMOIR: Memory Evaluation over Multi-year Oncology Inference \& Retrieval},
  author       = {Olira},
  year         = {2026},
  howpublished = {\url{https://github.com/olira-ai/memoir-benchmark}},
  doi          = {10.5281/zenodo.22906151}
}
```

## License

Code is released under [Apache 2.0](LICENSE). The corpus is derived from HL7/CodeX
mCODE Test Data, which is synthetic and released without privacy or usage
restrictions; the derived questions and answer keys are released under
[CC BY 4.0](data/LICENSE). No real patient data is included.
