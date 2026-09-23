"""The memory substrate interface.

MEMOIR holds everything above the memory layer constant (the same agent scaffold,
the same prompt skeleton, the same decoding settings, the same scorers) and varies
only what sits underneath. A substrate is that variable: whatever stores the
patient's record and exposes tools the agent can call to read it back.

Implementing one means answering two questions:

1. *What tools does the agent get for this patient?* :meth:`Substrate.session`
2. *What does the agent need to know about them?* :meth:`Substrate.prompt_section`

The harness supplies everything else, including the shared trend calculator for
any substrate that has to reconstruct a series at query time.

    class MyStore(SimpleSubstrate):
        name = "my-store"

        def build_tools(self, patient_id):
            return [make_search_tool(patient_id)]

        def prompt_section(self, patient_id):
            return "`search(query)` runs semantic search over this patient's record."
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.tools import BaseTool


class Substrate(abc.ABC):
    """A memory layer under test.

    One instance serves one substrate across the whole run; :meth:`session` is
    entered once per patient and may hold connections open for that patient's
    questions.
    """

    #: Short identifier recorded on every result row.
    name: ClassVar[str] = "substrate"

    #: True when the substrate answers trend questions from something it computed
    #: ahead of time. When False, meaning the substrate reconstructs a series at
    #: query time, the harness hands the agent the shared ``compute_trend`` calculator,
    #: so a trend result reflects the substrate's data rather than one agent's
    #: ability to fit a regression.
    precomputes_trends: ClassVar[bool] = False

    @abc.abstractmethod
    def session(self, patient_id: str) -> AbstractAsyncContextManager[Sequence[BaseTool]]:
        """Async context manager yielding the tools for one patient.

        Re-entered on each retry of a failed answer, so open connections inside
        the context rather than reusing one across attempts.
        """

    def prompt_section(self, patient_id: str) -> str:
        """Markdown describing this substrate's tools, spliced into the prompt.

        Document each tool's parameters and how the agent should reach for it.
        Returning ``""`` leaves the agent with only the tool schemas.
        """
        return ""

    async def aclose(self) -> None:
        """Release run-scoped resources. Called once when the run finishes."""
        return None


class SimpleSubstrate(Substrate):
    """A substrate whose tools need no per-patient setup or teardown.

    Implement :meth:`build_tools` and the session plumbing is handled for you.
    """

    @abc.abstractmethod
    def build_tools(self, patient_id: str) -> Sequence[BaseTool]:
        """Return the tools the agent gets for this patient."""

    @asynccontextmanager
    async def session(self, patient_id: str) -> AsyncIterator[Sequence[BaseTool]]:
        yield self.build_tools(patient_id)
