"""Agent optimizer: pi agent reads codebase + failures, proposes fixes."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .pi_agent import PiAgent

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def propose_agent_fixes(failures_path: Path, *, output_path: Path | None = None, cwd: str | None = None) -> Path:
    failures = [json.loads(l) for l in failures_path.read_text().strip().split('\n') if l.strip()]
    agent_failures = [f for f in failures if f.get("fix_type") in ("config", "code", "tool")]

    if not agent_failures:
        print("No agent-side failures. Nothing to propose.")
        out = output_path or failures_path.parent / "agent-fixes.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("[]\n")
        return out

    prompt = f"""Analyze these GAIA benchmark failures and propose fixes.

Read these files first to understand the code:
- examples/langchain-gaia-agent/agent.py
- examples/workflow.py

DO NOT change prompt text (system_prompt, tool_policy, etc.) — GEPA handles that.
Focus on: config values, Python code, tool implementations.

Failures:
{json.dumps(agent_failures, indent=2, default=str)}

Output a JSON array of fixes. Each fix: type ("config"|"code"), file, description, and for config: key+value, for code: search+replace. Only JSON, nothing else."""

    print(f"Asking pi agent to analyze {len(agent_failures)} failures...")
    pi = PiAgent(cwd=cwd or str(PROJECT_ROOT))
    try:
        response = pi.prompt(prompt)
    finally:
        pi.dispose()

    fixes = _parse_fixes(response)

    out = output_path or failures_path.parent / "agent-fixes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixes, indent=2))
    print(f"Proposed {len(fixes)} fixes → {out}")
    return out


def _parse_fixes(raw: str) -> list[dict[str, Any]]:
    m = re.search(r"\[[\s\S]*\]", raw)
    if m:
        try:
            fixes = json.loads(m.group())
            if isinstance(fixes, list):
                return fixes
        except json.JSONDecodeError:
            pass
    print(f"Warning: could not parse fixes from pi response. Raw: {raw[:300]}")
    return []
