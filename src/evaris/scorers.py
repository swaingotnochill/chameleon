"""Built-in scorers."""

from __future__ import annotations

from typing import Any

from .types import Score, Trace, find_spans


def exact_match(trace: Trace, expected: Any) -> list[Score]:
    """Check if trace output exactly matches expected."""
    if expected is None:
        return []
    output = trace.output
    # Compare as floats if both are numeric
    try:
        return [Score(name="exact_match", value=abs(float(output) - float(expected)) < 1e-9)]
    except (ValueError, TypeError):
        return [Score(name="exact_match", value=str(output) == str(expected))]


def includes(trace: Trace, expected: Any) -> list[Score]:
    """Check if expected string is contained in trace output."""
    if expected is None:
        return []
    return [Score(name="includes", value=str(expected) in str(trace.output))]


def no_errors(trace: Trace, expected: Any) -> list[Score]:
    """Check that no spans in the trace have error status."""
    errors = find_spans(trace.root, status="error")
    return [
        Score(
            name="no_errors",
            value=len(errors) == 0,
            reason=f"{len(errors)} error span(s)" if errors else "clean",
        )
    ]


def tool_call_count(trace: Trace, expected: Any) -> list[Score]:
    """Count tool spans in the trace."""
    tools = find_spans(trace.root, kind="tool")
    return [
        Score(
            name="tool_call_count",
            value=float(len(tools)),
            reason=f"{len(tools)} tool call(s)",
        )
    ]


def tool_efficiency(trace: Trace, expected: Any) -> list[Score]:
    """Score based on how few tool calls were made. Fewer = better."""
    tools = find_spans(trace.root, kind="tool")
    count = len(tools)
    if count == 0:
        return [Score(name="tool_efficiency", value=1.0, reason="no tool calls")]
    # 1 tool call = 1.0, each additional call reduces score by 0.2, min 0
    value = max(0.0, 1.0 - (count - 1) * 0.2)
    return [Score(name="tool_efficiency", value=value, reason=f"{count} tool call(s)")]


def llm_judge(
    trace: Trace,
    expected: Any,
    *,
    model: str = "openai/gpt-4o",
    criteria: str | None = None,
    template: str | None = None,
    grade_pattern: str | None = None,
    scale: str = "binary",  # binary | 1-5
) -> list[Score]:
    """LLM-as-judge scorer.

    Asks an LLM to grade the trace output against the expected answer.
    Returns a score based on the LLM's verdict.

    Args:
        model: Model to use for grading (e.g. "openai/gpt-4o", "anthropic/claude-sonnet-4-20250514")
        criteria: Custom grading criteria. If None, uses default.
        template: Custom prompt template with {question}, {answer}, {criterion} variables.
        grade_pattern: Regex to extract grade from LLM response.
        scale: "binary" (C/I) or "1-5" (numeric).

    Returns:
        Async function that can be awaited by the runner.
    """
    return _create_llm_judge(
        model=model,
        criteria=criteria,
        template=template,
        grade_pattern=grade_pattern,
        scale=scale,
    )


def _create_llm_judge(
    *,
    model: str,
    criteria: str | None,
    template: str | None,
    grade_pattern: str | None,
    scale: str,
):
    """Create an async LLM judge evaluator function."""

    tmpl = template or _default_template()
    instructions = criteria or _default_instructions(scale)
    pattern = grade_pattern or _default_pattern(scale)

    async def _judge(trace: Trace, expected: Any) -> list[Score]:
        if expected is None:
            return []

        prompt = tmpl.format(
            question=str(trace.input or ""),
            answer=str(trace.output or ""),
            criterion=str(expected),
            instructions=instructions,
        )

        completion = await _call_model(model, prompt)

        # Extract grade
        import re

        match = re.search(pattern, completion)
        if not match:
            return [
                Score(
                    name="llm_judge",
                    value=0.0,
                    reason=f"grade not found in response: {completion[:200]}",
                )
            ]

        raw = match.group(1)

        if scale == "1-5":
            try:
                value = float(raw)
            except ValueError:
                value = 0.0
            passed = value >= 4.0
        else:
            # binary: C=correct, P=partial, I=incorrect
            grade = raw.strip().upper()
            if grade == "C":
                value = 1.0
            elif grade == "P":
                value = 0.5
            else:
                value = 0.0
            passed = grade == "C"

        return [
            Score(
                name="llm_judge",
                value=value,
                reason=completion.strip(),
            )
        ]

    return _judge


# ── Prompt templates (based on OpenAI closedqa / inspect-ai model_graded_qa) ───


QA_TEMPLATE = """\
You are assessing a submitted answer on a given task based on a criterion.\
 Here is the data:

[BEGIN DATA]
***
[Task]: {question}
***
[Submission]: {answer}
***
[Criterion]: {criterion}
***
[END DATA]

Does the submission meet the criterion?

{instructions}
"""


def _default_template() -> str:
    return QA_TEMPLATE


def _default_instructions(scale: str) -> str:
    if scale == "1-5":
        return """\
Rate the submission on a scale of 1 to 5.
1 = completely incorrect
2 = mostly incorrect
3 = partially correct
4 = mostly correct
5 = fully correct

First, reason step by step about the criterion. Then end with your score as: SCORE: N\
 where N is the integer from 1 to 5."""

    return """\
After assessing the submitted answer, reply with 'GRADE: $LETTER' (without quotes)\
 where LETTER is one of CPI. Choose ONE option: "C" for correct, "P" for partially \
correct, or "I" for incorrect.

For example, after reviewing a correct answer you might write 'GRADE: C'.\
 After reviewing an incorrect answer you might write 'GRADE: I'.

First, reason step by step about the criterion to ensure your conclusion is \
correct. Avoid simply stating the answer at the outset. Then, end with your \
answer formatted as 'GRADE: $LETTER' where LETTER is one of CPI."""


def _default_pattern(scale: str) -> str:
    if scale == "1-5":
        return r"(?i)SCORE\s*:\s*([1-5])"
    return r"(?i)GRADE\s*:\s*([CPI])"


# ── Model calling ─────────────────────────────────────────────────────


async def _call_model(model: str, prompt: str) -> str:
    """Call an LLM and return the text completion.

    Uses the openai SDK by default. The model string can be a provider/model pair
    like "openai/gpt-4o" or "anthropic/claude-sonnet-4-20250514".

    Users can monkey-patch this function to use any provider:
        import evaris.scorers as scorers
        scorers._call_model = my_custom_call
    """
    try:
        from openai import AsyncOpenAI

        provider, model_name = _parse_model_string(model)
        client = AsyncOpenAI(base_url=_provider_base_url(
            provider), api_key=_provider_key(provider))
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""
    except ImportError:
        raise ImportError(
            "openai package is required for LLM-as-judge. "
            "Install it with: pip install openai"
        )


def _parse_model_string(model: str) -> tuple[str, str]:
    """Parse 'openai/gpt-4o' into ('openai', 'gpt-4o').

    If no provider prefix, defaults to 'openai'.
    """
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return provider, model_name
    return "openai", model


def _provider_base_url(provider: str) -> str | None:
    """Map provider name to base URL."""
    urls = {
        "anthropic": "https://api.anthropic.com/v1",
        "z-ai": "https://api.z.ai/api/paas/v4",
    }
    return urls.get(provider)


def _provider_key(provider: str) -> str | None:
    """Map provider name to env var for API key."""
    import os

    keys = {
        "openai": os.environ.get("OPENAI_API_KEY"),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
        "z-ai": os.environ.get("ZAI_API_KEY"),
    }
    return keys.get(provider)
