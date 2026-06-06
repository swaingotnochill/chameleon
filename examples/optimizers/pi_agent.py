"""Pi agent RPC client — spawn pi subprocess, send prompts, collect responses."""

from __future__ import annotations

import json
import subprocess
from typing import Any


class PiAgent:
    def __init__(self, *, cwd: str | None = None):
        self.cwd = cwd
        self.proc: subprocess.Popen | None = None

    def prompt(self, message: str) -> str:
        cmd = ["pi", "--mode", "rpc", "--no-session"]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.cwd,
        )
        self.proc.stdin.write(json.dumps({"type": "prompt", "message": message}) + "\n")
        self.proc.stdin.flush()
        return self._collect()

    def _collect(self) -> str:
        chunks: list[str] = []
        for line in self.proc.stdout:
            event = json.loads(line.strip())
            if event.get("type") == "agent_end":
                break
            if event.get("type") == "message_update":
                delta = event.get("assistantMessageEvent", {})
                if delta.get("type") == "text_delta":
                    chunks.append(delta["delta"])
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        return "".join(chunks)

    def dispose(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"type": "abort"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
