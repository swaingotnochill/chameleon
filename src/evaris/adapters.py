from __future__ import annotations

from typing import Any

from .types import Span, Trace


def from_etrace(span: Any) -> Trace:
    """Convert an etrace.Span to an evaris.Trace.

    Builds the nested tree from flat parent_span_id references.
    Requires all spans in the trace to be accessible via the processor
    or passed as a list.
    """
    # Collect all spans in the trace (walk etrace's flat structure)
    all_spans = _collect_etrace_spans(span)

    # Build our nested tree
    by_id: dict[str, Span] = {}
    roots: list[Span] = []

    for es in all_spans:
        s = _convert_single_etrace_span(es)
        by_id[s.parent_id or "__root__"] = by_id.get(s.parent_id or "__root__")
        by_id[es.span_id] = s

    for es in all_spans:
        s = by_id[es.span_id]
        parent = by_id.get(es.parent_span_id) if es.parent_span_id else None
        if parent and parent is not s:
            parent.children.append(s)
        elif s not in roots:
            roots.append(s)

    root = roots[0] if roots else _convert_single_etrace_span(span)
    return Trace(root=root, input=span.input, output=span.output)


def _collect_etrace_spans(span: Any) -> list[Any]:
    """Collect a span and all its children from etrace's flat structure.

    etrace uses parent_span_id (flat), not nested children.
    This requires access to the processor's span buffer or the RunTracker's _spans.
    For now, we work with what we're given — a single root span.
    """
    # If the etrace processor or tracker exposes collected spans, use that.
    # Otherwise, we only have the root span.
    try:
        from etrace._processor import _span_buffer
        trace_id = span.trace_id
        return [s for s in _span_buffer.values() if s.trace_id == trace_id]
    except (ImportError, AttributeError):
        return [span]


def _convert_single_etrace_span(es: Any) -> Span:
    kind = str(es.kind.value) if hasattr(es.kind, "value") else str(es.kind)
    status = str(es.status.value) if hasattr(
        es.status, "value") else str(es.status)
    duration_ms = (es.duration_ns or 0) / 1_000_000
    return Span(
        name=es.name,
        kind=kind,
        status=status,
        input=es.input,
        output=es.output,
        parent_id=es.parent_span_id,
        duration_ms=duration_ms,
        attributes=dict(es.attributes) if es.attributes else {},
    )


def from_langfuse(trace_data: dict) -> Trace:
    """Convert a Langfuse trace dict to an evaris.Trace."""
    root_span = trace_data.get("data", trace_data)
    observations = root_span.get("observations", [])

    by_id: dict[str, Span] = {}

    for obs in observations:
        obs_id = obs.get("id", "")
        s = Span(
            name=obs.get("name", ""),
            kind=_langfuse_kind(obs),
            status="ok" if obs.get("level", "default") != "error" else "error",
            input=obs.get("input"),
            output=obs.get("output"),
            parent_id=obs.get("parent_observation_id"),
            duration_ms=obs.get("duration_ms", 0),
            attributes={
                k: v for k, v in obs.items()
                if k not in ("id", "name", "type", "input", "output",
                             "parent_observation_id", "duration_ms", "level")
            },
        )
        by_id[obs_id] = s

    # Build tree
    roots: list[Span] = []
    for obs_id, s in by_id.items():
        parent = by_id.get(s.parent_id) if s.parent_id else None
        if parent and parent is not s:
            parent.children.append(s)
        else:
            roots.append(s)

    root = roots[0] if roots else Span(name="trace", kind="agent")
    return Trace(root=root, input=root_span.get("input"), output=root_span.get("output"))


def _langfuse_kind(obs: dict) -> str:
    ot = obs.get("type", "DEFAULT")
    mapping = {
        "SPAN": "tool",
        "GENERATION": "llm",
        "DEFAULT": "custom",
        "EVENT": "custom",
    }
    return mapping.get(ot, "custom")


def from_langsmith(run_tree: dict) -> Trace:
    """Convert a Langsmith run tree dict to an evaris.Trace."""
    root = _convert_langsmith_run(run_tree)
    return Trace(root=root, input=run_tree.get("inputs"), output=run_tree.get("outputs"))


def _convert_langsmith_run(run: dict) -> Span:
    children = [_convert_langsmith_run(child)
                for child in run.get("child_runs", [])]
    return Span(
        name=run.get("name", ""),
        kind=run.get("run_type", "custom"),
        status="ok" if run.get("error") is None else "error",
        input=run.get("inputs"),
        output=run.get("outputs"),
        parent_id=run.get("parent_run_id"),
        duration_ms=run.get("execution_time", 0) *
        1000 if run.get("execution_time") else 0,
        children=children,
        attributes={
            k: v for k, v in run.items()
            if k not in ("name", "run_type", "error", "inputs", "outputs",
                         "parent_run_id", "execution_time", "child_runs", "id")
        },
    )
