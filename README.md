# Evaris Evals

Trace-based evaluation and self-improvement experiments for AI agents.

This repo is the MVP eval layer behind the Chameleon demo: **self-healing agents**. The core idea is simple:

```text
agent -> traces -> evals -> distillation -> optimizers -> heal
```

Tracing and evals tell you what happened and what failed. Chameleon explores the next step: turning those failures into repair records, routing them to the right optimizer, and accepting only candidates that improve on held-out tasks.

## What is here

- `src/evaris/`: a small Python library for scoring agent traces.
- `examples/workflow.py`: GAIA eval and self-improvement workflow.
- `examples/langchain-gaia-agent/`: a LangChain GAIA agent wired for tracing.
- `src/evaris/distill.py`: failure distillation into root cause / fix records.
- `examples/optimizers/`: experimental optimizer glue, including agent-side fix proposals.
- `chameleon-demo.html`: single-file slide demo for the self-healing agent story.
- `chameleon-demo-script.md`: demo narration / storyboard.

## Why trace-based evals?

Final answers are not enough for agent systems. A wrong answer can come from a bad search query, skipped tool, tool error, weak verification policy, formatting issue, or runtime config problem.

Evaris scores the execution trace so evals can inspect:

- final output correctness
- tool calls and tool errors
- LLM/tool span status
- cost and latency metadata when present
- full trace behavior through an LLM-as-judge scorer

That trace evidence is what makes distillation and repair possible.

## Install

This repo uses `uv`.

```bash
uv sync
```

For local development:

```bash
uv run python tests/smoke_test.py
```

If you prefer plain Python for the smoke test:

```bash
python3 tests/smoke_test.py
```

## Basic Usage

### Offline eval

Offline eval runs a solver over a dataset. The solver returns an `evaris.Trace`, and scorers evaluate that trace against the expected answer.

```python
import asyncio

from evaris import Sample, Span, Trace, evaluate, exact_match, no_errors, tool_call_count


def solve(question: str) -> Trace:
    tool = Span(
        name="search",
        kind="tool",
        status="ok",
        input="capital of France",
        output="Paris",
        duration_ms=120,
    )
    root = Span(
        name="agent",
        kind="agent",
        status="ok",
        input=question,
        output="Paris",
        children=[tool],
    )
    return Trace(root=root, input=question, output="Paris")


async def main():
    results = await evaluate(
        mode="offline",
        dataset=[Sample(id="1", input="Capital of France?", expected="Paris")],
        solver=solve,
        scorers=[exact_match, no_errors, tool_call_count],
    )
    print(results)


asyncio.run(main())
```

### Online eval

Online eval scores an existing trace, usually from production traffic or a live run.

```python
results = await evaluate(
    mode="online",
    trace=trace,
    scorers=[no_errors, tool_call_count],
)
```

## Built-in Scorers

Current scorers include:

- `exact_match`
- `includes`
- `no_errors`
- `tool_call_count`
- `tool_efficiency`
- `llm_judge`
- `trace_llm_judge`

LLM judge scorers require model/API environment configuration. See `src/evaris/scorers.py` for the current implementation details.

## Distillation

Distillation turns eval results into compact failure records for optimization.

Each failure record can include:

- task id
- input, expected output, actual output
- exact match result
- judge score and reason
- failure type
- root cause
- suggested fix
- fix type: `prompt`, `config`, `code`, or `tool`
- fix target

Example:

```bash
uv run python -c "from pathlib import Path; from evaris import distill; distill(Path('path/to/eval-results.json'))"
```

## Chameleon Self-Healing Loop

Chameleon is the demo narrative built on top of this eval layer:

```text
1. Agent runs on a task.
2. Traces capture LLM calls, tool calls, errors, cost, and latency.
3. Evals turn traces into quality and safety signals.
4. Distillation converts failures into root causes and fix targets.
5. Optimizers propose repairs:
   - GEPA handles prompts and policies.
   - Agent optimization handles code, tools, and config.
6. A holdout gate accepts only candidates that improve without unsafe regressions.
```

Open the demo directly in a browser:

```bash
open chameleon-demo.html
```

Navigation:

- `Space` or `ArrowRight`: next scene
- `ArrowLeft`: previous scene
- click the stage: next scene

## GAIA Example Workflow

The GAIA experiment plan is documented here:

- [GAIA self-improve plan](docs/GAIA_SELF_IMPROVE_PLAN.md)

Set up local GAIA manifests:

```bash
uv run python examples/setup_gaia_dataset.py
```

Run a smoke eval:

```bash
uv run python examples/workflow.py \
  --mode eval \
  --manifest examples/data/local.gaia_level1_smoke.jsonl \
  --config examples/smoke.config.json \
  --run-tag smoke \
  --limit 3 \
  --concurrency 2
```

Run the self-improvement workflow after baseline and optimize evals exist:

```bash
uv run python examples/workflow.py \
  --mode improve \
  --optimize-eval "examples/.evaris/eval-runs/baseline-optimize-*.json" \
  --baseline-eval "examples/.evaris/eval-runs/baseline-*.json" \
  --manifest-holdout examples/data/local.gaia_level1_holdout_15.jsonl \
  --config examples/mvp.config.json
```

## Trace Model

The core trace type is intentionally small and adapter-friendly:

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    name: str
    kind: str
    status: str
    input: Any | None = None
    output: Any | None = None
    parent_id: str | None = None
    duration_ms: float = 0
    children: list["Span"] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)


@dataclass
class Trace:
    root: Span
    input: Any | None = None
    output: Any | None = None
```

Adapters can convert traces from etrace, Langfuse, LangSmith, or hand-rolled logs into this shape.

## Repo Layout

```text
mvp-evals/
  src/evaris/
    runner.py       # evaluate()
    types.py        # Trace, Span, Sample, Score, EvalResults
    scorers.py      # built-in scorers
    distill.py      # failure distillation
    adapters.py     # tracing adapters
  examples/
    workflow.py
    langchain-gaia-agent/
    optimizers/
    data/
  tests/
    smoke_test.py
    cross_framework_eval.py
  docs/
    GAIA_SELF_IMPROVE_PLAN.md
  chameleon-demo.html
  chameleon-demo-script.md
```

## Status

This is an MVP / hackathon research repo. The trace eval core is usable for local experiments, while the self-healing loop is still experimental and intentionally scoped around GAIA-style agent tasks.
