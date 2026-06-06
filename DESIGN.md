# Evals MVP — Design

## Public API

```python
from evaris_evals import evaluate

# Offline: dataset is required, solvers produce traces
results = await evaluate(
    dataset=dataset,         # list of {input, expected}
    solver=my_agent,         # fn(input) -> trace
    scorers=[exact_match, tool_count],
    mode="offline",
)

# Online: trace is required, no dataset
results = await evaluate(
    trace=trace,             # the agent's execution trace
    scorers=[error_check, cost_budget],
    mode="online",
)
```

## Core Types

```python
@dataclass
class Span:
    """Trace-agnostic span. Works with etrace, Langfuse, Langsmith, or hand-rolled."""
    name: str
    kind: str               # agent, tool, llm, http, etc.
    status: str             # ok, error
    input: Any | None = None
    output: Any | None = None
    parent_id: str | None = None
    duration_ms: float = 0
    children: list[Span] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)

@dataclass
class Trace:
    """The agent's execution trace."""
    root: Span              # root span with children
    input: Any | None = None
    output: Any | None = None

@dataclass
class Sample:
    """One item in a dataset."""
    id: str
    input: Any
    expected: Any

@dataclass
class Score:
    name: str               # "exact_match", "tool_count"
    value: float | bool
    reason: str | None = None

@dataclass
class SampleResult:
    id: str
    input: Any
    output: Any
    expected: Any | None
    scores: list[Score]
    error: str | None = None

@dataclass
class EvalResults:
    results: list[SampleResult]
```

## Evaluator

A function that takes a trace and expected, returns scores.

```python
def exact_match(trace: Trace, expected: Any) -> list[Score]:
    return [Score(name="exact_match", value=trace.output == expected)]

def tool_count(trace: Trace, expected: Any) -> list[Score]:
    tools = find_spans(trace.root, kind="tool")
    return [Score(name="tool_count", value=len(tools))]
```

## Adapter

Converts any tracing library's format to our `Trace` / `Span`.

```python
def from_etrace(span: etrace.Span) -> Trace: ...
def from_langfuse(trace: dict) -> Trace: ...
def from_langsmith(run_tree: dict) -> Trace: ...
```

## File Structure

```
mvp-evals/
├── DESIGN.md
├── pyproject.toml
└── src/evaris/
    ├── __init__.py      # evals()
    ├── types.py         # Span, Trace, Sample, Score, SampleResult, EvalResults
    ├── runner.py         # evals() execution
    ├── adapters.py       # from_etrace(), from_langfuse(), from_langsmith()
    └── scorers.py        # built-in: exact_match
```

## Out of Scope

- Solver (mode=offline without solver = user passes traces directly)
- ClickHouse extraction
- LLM-as-judge built-in
- Platform UI
