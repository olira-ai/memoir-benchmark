"""Tests for the harness wiring that does not need a model call."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from langchain_core.tools import StructuredTool

from memoir.dataset import load_cohort
from memoir.harness.agent import BenchmarkAgent
from memoir.harness.prompt import build_system_prompt, patient_event_calendar
from memoir.run import load_substrate
from memoir.substrate import SimpleSubstrate, Substrate


def _noop_tool(name: str = "search") -> StructuredTool:
    async def _run(query: str) -> str:
        return "{}"

    return StructuredTool.from_function(name=name, description="test tool", coroutine=_run)


class DemoSubstrate(SimpleSubstrate):
    name = "demo"

    def build_tools(self, patient_id):
        return [_noop_tool()]

    def prompt_section(self, patient_id):
        return "## Your tools\n\n`search(query)` searches this patient's record."


class PrecomputedSubstrate(DemoSubstrate):
    name = "demo-precomputed"
    precomputes_trends = True


@pytest.fixture(scope="module")
def patient_id():
    return load_cohort()[0].patient_id


class TestPromptSkeleton:
    def test_every_slot_is_filled(self, patient_id):
        prompt = build_system_prompt(
            patient_id=patient_id,
            substrate_section="## Your tools\n\n`search(query)` does a thing.",
            event_types=["lab_results_received", "medication_action"],
        )
        assert "{{" not in prompt
        assert "`search(query)` does a thing." in prompt
        assert "`lab_results_received`" in prompt

    def test_the_calendar_locates_the_record_in_time(self, patient_id):
        calendar = patient_event_calendar(patient_id)
        assert "span" in calendar
        assert any(str(year) in calendar for year in range(1900, 2031))

    def test_the_skeleton_is_identical_across_substrates(self, patient_id):
        common = dict(patient_id=patient_id, event_types=["lab_results_received"])
        a = build_system_prompt(substrate_section="SECTION-A", **common)
        b = build_system_prompt(substrate_section="SECTION-B", **common)
        assert a.replace("SECTION-A", "X") == b.replace("SECTION-B", "X")


class TestTrendToolInjection:
    """The shared calculator goes to whoever rebuilds a series at query time."""

    async def _tools_for(self, substrate: Substrate, patient_id: str) -> list[str]:
        async with substrate.session(patient_id) as tools:
            names = [t.name for t in tools]
        if not substrate.precomputes_trends:
            names.append("compute_trend")
        return names

    @pytest.mark.asyncio
    async def test_runtime_reconstruction_gets_the_calculator(self, patient_id):
        assert "compute_trend" in await self._tools_for(DemoSubstrate(), patient_id)

    @pytest.mark.asyncio
    async def test_precomputed_trends_do_not(self, patient_id):
        assert "compute_trend" not in await self._tools_for(PrecomputedSubstrate(), patient_id)


class TestSubstrateLoading:
    def test_loads_a_class_by_spec(self):
        s = load_substrate("examples.local_events:LocalEventsSubstrate")
        assert s.name == "local-events"

    def test_rejects_a_spec_without_an_attribute(self):
        with pytest.raises(SystemExit):
            load_substrate("examples.local_events")

    def test_rejects_a_non_substrate(self):
        with pytest.raises(SystemExit):
            load_substrate("memoir.tasks:TASKS")


class TestLocalEventsExample:
    @pytest.mark.asyncio
    async def test_get_events_filters_and_pages(self, patient_id):
        from examples.local_events import LocalEventsSubstrate

        substrate = LocalEventsSubstrate()
        async with substrate.session(patient_id) as tools:
            get_events = next(t for t in tools if t.name == "get_events")
            import json

            everything = json.loads(await get_events.coroutine(limit=200))
            labs = json.loads(
                await get_events.coroutine(event_types=["lab_results_received"], limit=200)
            )
            assert labs["total_matched"] <= everything["total_matched"]
            assert all(e["event_type"] == "lab_results_received" for e in labs["events"])

            page = json.loads(await get_events.coroutine(limit=5))
            assert page["returned"] == 5
            assert page["truncated"] is True


class TestAgentConstruction:
    def test_builds_its_prompt_without_calling_a_model(self, patient_id):
        agent = BenchmarkAgent(DemoSubstrate(), patient_id, event_types=["vitals_measurement"])
        assert "`search(query)` searches this patient's record." in agent.system_prompt
        assert "`vitals_measurement`" in agent.system_prompt

    @pytest.mark.asyncio
    async def test_a_session_may_hold_per_patient_resources(self, patient_id):
        opened, closed = [], []

        class Stateful(Substrate):
            name = "stateful"

            @asynccontextmanager
            async def session(self, pid):
                opened.append(pid)
                try:
                    yield [_noop_tool()]
                finally:
                    closed.append(pid)

        substrate = Stateful()
        async with substrate.session(patient_id) as tools:
            assert [t.name for t in tools] == ["search"]
        assert opened == [patient_id] and closed == [patient_id]
