"""Core types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    """One unit of work in an execution trace."""
    # Is this span type matching with the other frameworks? I don't have much
    # idea how this datastructure should look like depending on the LLM
    # providers and the tracing library maybe.
    name: str
    kind: str  # agent, tool, llm, http, retrieval, custom, etc.
    status: str = "ok"  # ok, error
    input: Any = None
    output: Any = None
    parent_id: str | None = None
    duration_ms: float = 0
    children: list[Span] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)


@dataclass
class Trace:
    """An agent's execution trace."""

    root: Span
    input: Any = None
    output: Any = None


@dataclass
class Sample:
    """One item in an offline eval dataset."""

    id: str
    input: Any
    expected: Any


@dataclass
class Score:
    """One dimension of evaluation."""

    name: str
    value: float | bool
    reason: str | None = None


@dataclass
class SampleResult:
    """Eval result for one sample."""

    id: str
    input: Any
    output: Any
    expected: Any | None
    scores: list[Score]
    error: str | None = None


@dataclass
class EvalResults:
    """Eval results for a batch."""

    results: list[SampleResult]


# ── Helpers ───────────────────────────────────────────────────────────

# Have to check if this can be optimized or this is the simplest and the
# efficient way. What does the other frameworks do here?
def find_spans(root: Span, kind: str | None = None, status: str | None = None) -> list[Span]:
    """Walk the span tree and return matching spans."""
    matches: list[Span] = []
    if (kind is None or root.kind == kind) and (status is None or root.status == status):
        matches.append(root)
    for child in root.children:
        matches.extend(find_spans(child, kind, status))
    return matches
