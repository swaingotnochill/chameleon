"""Distillation: extract failure records with reflection model root cause analysis."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

_llm_semaphore: asyncio.Semaphore | None = None


def set_llm_concurrency(limit: int) -> None:
    global _llm_semaphore
    _llm_semaphore = asyncio.Semaphore(limit)


def distill(
    eval_path: Path,
    *,
    output_path: Path | None = None,
    min_score: float = 1.0,
    reflection_model: str | None = None,
) -> Path:
    return asyncio.run(_distill_async(eval_path, output_path=output_path, min_score=min_score, reflection_model=reflection_model))


async def _distill_async(
    eval_path: Path,
    *,
    output_path: Path | None = None,
    min_score: float = 1.0,
    reflection_model: str | None = None,
) -> Path:
    data = json.loads(eval_path.read_text())
    results = data.get("results", [])

    failures: list[dict[str, Any]] = []
    for r in results:
        scores = {s["name"]: s for s in r.get("scores", [])}
        exact_pass = scores.get("normalized_exact_match", {}).get("value", False)
        judge_score = scores.get("trace_llm_judge", {}).get("value", 0.0)
        judge_reason = scores.get("trace_llm_judge", {}).get("reason", "")

        if not exact_pass or judge_score < min_score:
            failure_type = _classify_failure(r, scores)
            failures.append({
                "task_id": r["id"],
                "input": r["input"].get("input", r["input"]) if isinstance(r["input"], dict) else str(r.get("input", "")),
                "expected": r["expected"],
                "actual": r["output"],
                "exact_match": exact_pass,
                "judge_score": judge_score,
                "judge_reason": (judge_reason or "")[:500],
                "failure_type": failure_type,
                "tool_calls": scores.get("tool_call_count", {}).get("value", 0),
                "no_errors": scores.get("no_errors", {}).get("value", True),
            })

    if output_path is None:
        output_path = eval_path.parent / "distill" / "failures.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if reflection_model and failures:
        failures = await _enrich_with_reflection(failures, reflection_model)
    else:
        for failure in failures:
            failure.setdefault("root_cause", "")
            failure.setdefault("suggested_fix", "")
            failure.setdefault("fix_type", _guess_fix_type(failure))
            failure.setdefault("fix_target", "")

    with output_path.open("w") as f:
        for failure in failures:
            f.write(json.dumps(failure) + "\n")

    print(f"Distilled {len(failures)} failures from {len(results)} tasks → {output_path}")
    return output_path


async def _enrich_with_reflection(
    failures: list[dict[str, Any]],
    model: str,
) -> list[dict[str, Any]]:
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.z.ai/api/paas/v4")
    api_key = os.environ.get("OPENAI_API_KEY")
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def analyze(failure: dict[str, Any]) -> dict[str, Any]:
        prompt = REFLECTION_PROMPT.format(
            question=failure["input"][:500],
            expected=str(failure["expected"])[:200],
            actual=str(failure["actual"])[:200],
            judge_reason=failure.get("judge_reason", "")[:300],
            failure_type=failure["failure_type"],
            tool_calls=failure["tool_calls"],
        )
        sem = _llm_semaphore
        if sem:
            async with sem:
                resp = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    temperature=0.3, max_tokens=1024,
                )
        else:
            resp = await client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=1024,
            )
        raw = resp.choices[0].message.content or ""

        parsed = _parse_reflection(raw)
        failure["root_cause"] = parsed.get("root_cause", raw[:300])
        failure["suggested_fix"] = parsed.get("suggested_fix", "")
        failure["fix_type"] = parsed.get("fix_type", "prompt")
        failure["fix_target"] = parsed.get("fix_target", "")
        return failure

    tasks = [analyze(f) for f in failures]
    results = await asyncio.gather(*tasks)
    return list(results)


def _guess_fix_type(failure: dict[str, Any]) -> str:
    ftype = failure.get("failure_type", "")
    if ftype in ("agent_error", "empty_response"):
        return "config"
    if ftype == "no_tool_call":
        return "tool"
    return "prompt"


def _parse_reflection(raw: str) -> dict[str, str]:
    match = re.search(r"root_cause:\s*(.+?)(?=\n\s*suggested_fix:|$)", raw, re.DOTALL | re.IGNORECASE)
    fix = re.search(r"suggested_fix:\s*(.+?)(?=\n\s*fix_type:|$)", raw, re.DOTALL | re.IGNORECASE)
    ftype = re.search(r"fix_type:\s*(\w+)", raw, re.IGNORECASE)
    ftarget = re.search(r"fix_target:\s*(.+?)$", raw, re.MULTILINE | re.IGNORECASE)

    return {
        "root_cause": match.group(1).strip()[:300] if match else raw[:300],
        "suggested_fix": fix.group(1).strip()[:300] if fix else "",
        "fix_type": ftype.group(1).strip().lower() if ftype else "prompt",
        "fix_target": ftarget.group(1).strip() if ftarget else "",
    }


def _classify_failure(result: dict[str, Any], scores: dict[str, Any]) -> str:
    output = str(result.get("output", "")).strip()
    expected = str(result.get("expected", "")).strip()

    if not scores.get("no_errors", {}).get("value", True):
        return "agent_error"
    if not output or output.lower() in ("", "none", "null", "nan", "error"):
        return "empty_response"

    tool_calls = scores.get("tool_call_count", {}).get("value", 0)
    inp = str(result.get("input", {})).lower()
    needs_file = "attachment" in inp and tool_calls == 0
    needs_search = any(k in inp for k in ["website", "wikipedia", "published", "who won", "in the year"])
    if (needs_file or needs_search) and tool_calls == 0:
        return "no_tool_call"
    if _partial_overlap(output, expected):
        return "partial_match"
    if any(p in output.lower() for p in ["i think", "i believe", "probably", "uncertain", "not sure"]):
        return "uncertain_answer"
    return "wrong_answer"


def _partial_overlap(actual: str, expected: str) -> bool:
    def tokens(s: str) -> set[str]:
        return set(re.findall(r"\w+", s.lower()))

    actual_t = tokens(actual)
    expected_t = tokens(expected)
    if not actual_t or not expected_t:
        return False
    return len(actual_t & expected_t) >= max(2, len(expected_t) * 0.4)


REFLECTION_PROMPT = """Analyze this GAIA benchmark agent failure. Be concise.

Question: {question}
Expected: {expected}
Actual: {actual}
Judge: {judge_reason}
Failure type: {failure_type}
Tool calls made: {tool_calls}

Respond with exactly these 4 fields, one per line:
root_cause: <what went wrong in one sentence>
suggested_fix: <how to fix it in one sentence>
fix_type: prompt | config | code | tool
fix_target: <which component to change, e.g. tool_policy, normalize_answer, web_search, max_output_tokens>"""
