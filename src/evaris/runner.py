from __future__ import annotations

import asyncio
from typing import Any, Callable

from .types import EvalResults, Sample, SampleResult, Score, Trace


async def evaluate(
    *,
    mode: str,  # this should be a pydantic type.
    dataset: list[Sample] | None = None,
    solver: Callable[[Any], Any] | None = None,  # this or traces should be
    # present at once. Basically a
    # Union Type?
    # There shouldn't be need for a separate trace.
    trace: Trace | None = None,
    # we should just have a trace list.
    traces: list[Trace] | None = None,
    scorers: list[Callable],
    concurrency: int = 10,  # concurrency should not be enforced here. Maybe pass
    # a config reference which can be changed at runtime.
) -> EvalResults:
    """Score traces with evaluators.

    Offline:  evaluate(mode="offline", dataset=..., solver=..., scorers=...)
    Online:   evaluate(mode="online",  trace=..., scorers=...)
    """

    # this should be wrapped like this: if mode == "online" ->
    # run_online_evaluate() else if mode == "offline" run_offline_evaluate() and
    # then we have that thing in there. Kind of abstraction.
    if mode == "online":
        if traces is not None:
            results = []
            for t in traces:
                # for
                r = await _score_trace(t, expected=None, scorers=scorers)
                # an agent action, is this a single trace containing all the
                # spans for that action? It should be that only right?
                results.append(r)
            return EvalResults(results=results)
        if trace is None:
            raise ValueError("trace or traces is required for online mode")
        return EvalResults(results=[await _score_trace(trace, expected=None, scorers=scorers)])

    if mode == "offline":
        if dataset is None:
            raise ValueError("dataset is required for offline mode")

        # this is batched evaluation right?
        sem = asyncio.Semaphore(concurrency)
        tasks = [_run_offline_sample(sample, solver, scorers, sem)
                 for sample in dataset]
        results = await asyncio.gather(*tasks)
        return EvalResults(results=list(results))

    raise ValueError(f"unknown mode: {mode!r}")


async def _run_offline_sample(
    sample: Sample,
    solver: Callable[[Any], Any] | None,
    scorers: list[Callable],
    sem: asyncio.Semaphore,
) -> SampleResult:
    async with sem:
        try:
            # Produce trace: from solver or from pre-computed traces
            if solver is not None:
                result = solver(sample.input)
                if asyncio.iscoroutine(result):
                    result = await result
                t: Trace = result
            else:
                raise ValueError(
                    "solver is required. Pass a callable that takes input and returns a Trace."
                )

            return await _score_trace(
                t, expected=sample.expected, scorers=scorers, sample_id=sample.id
            )
        except Exception as e:
            return SampleResult(
                id=sample.id,
                input=sample.input,
                output=None,
                expected=sample.expected,
                scores=[],
                error=str(e),
            )


async def _score_trace(
    trace: Trace,
    expected: Any,
    scorers: list[Callable],
    sample_id: str = "",
) -> SampleResult:
    scores: list[Score] = []
    for scorer in scorers:
        result = scorer(trace, expected)
        if asyncio.iscoroutine(result):
            result = await result
        if isinstance(result, list):
            scores.extend(result)
        else:
            scores.append(result)

    return SampleResult(
        id=sample_id,
        input=trace.input,
        output=trace.output,
        expected=expected,
        scores=scores,
    )
