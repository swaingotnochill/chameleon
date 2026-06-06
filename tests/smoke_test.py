import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaris import (
    evaluate,
    Span,
    Trace,
    Sample,
    find_spans,
    exact_match,
    no_errors,
    tool_call_count,
    tool_efficiency,
)


def fake_agent(input: str) -> Trace:
    """Simulate an agent running. Returns a trace."""

    if "error" in input:
        tool_span = Span(
            name="search",
            kind="tool",
            status="error",
            input="query",
            output="timeout",
            duration_ms=500,
        )
    else:
        tool_span = Span(
            name="search",
            kind="tool",
            status="ok",
            input="query",
            output="found 3 results",
            duration_ms=200,
        )

    llm_span = Span(
        name="gpt-4o",
        kind="llm",
        status="ok",
        input="summarize results",
        output="The answer is Paris",
        duration_ms=1500,
        attributes={"model": "gpt-4o", "tokens": 150},
        children=[tool_span],
    )

    root = Span(
        name="agent",
        kind="agent",
        status="ok",
        input=input,
        output="The answer is Paris",
        duration_ms=2000,
        children=[llm_span],
    )

    return Trace(root=root, input=input, output="The answer is Paris")


dataset = [
    Sample(id="1", input="What is the capital of France?", expected="Paris"),
    Sample(id="2", input="What is 2+2?", expected="4"),
    Sample(id="3", input="error case: bad query", expected="error output"),
]


async def main():
    print("=== Offline Eval ===\n")
    results = await evaluate(
        mode="offline",
        dataset=dataset,
        solver=fake_agent,
        scorers=[exact_match, no_errors, tool_call_count, tool_efficiency],
        concurrency=5,
    )

    for r in results.results:
        print(f"Sample {r.id}:")
        print(f"  input:    {r.input}")
        print(f"  output:   {r.output}")
        print(f"  expected: {r.expected}")
        print(f"  scores:")
        for s in r.scores:
            print(f"    {s.name}: {s.value}  ({s.reason or '-'})")
        if r.error:
            print(f"  ERROR: {r.error}")
        print()

    # Aggregation
    print("--- Aggregation ---")
    for scorer_name in ["exact_match", "no_errors"]:
        scores = [
            s for r in results.results for s in r.scores if s.name == scorer_name]
        passed = sum(1 for s in scores if s.value is True)
        total = len(scores)
        print(f"  {scorer_name}: {passed}/{total} ({passed/total*100:.0f}%)")

    print("\n=== Online Eval ===\n")
    trace = fake_agent("What is the capital of France?")
    online_results = await evaluate(
        mode="online",
        trace=trace,
        scorers=[no_errors, tool_call_count],
    )
    for r in online_results.results:
        print(f"Online trace eval:")
        print(f"  output: {r.output}")
        print(f"  scores:")
        for s in r.scores:
            print(f"    {s.name}: {s.value}  ({s.reason or '-'})")
        print()

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
