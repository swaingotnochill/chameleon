# Chameleon — Self-Improving Agent Harness

Reproduce the GAIA benchmark experiment: agent → traces → evals → distillation → optimization (GEPA + pi agent) → apply fixes → compare.

## Prerequisites

- Python 3.11+
- Node.js 22+ (for pi agent optimizer)
- ZAI API key ([z.ai](https://z.ai))

## Setup

```bash
# Install dependencies
uv sync

# Set API key
cp .env.example .env
# Edit .env and add your ZAI_API_KEY

# Verify
chameleon --help
```

## Project Structure

```
examples/
├── workflow.py              # Main entry point (--mode eval|improve)
├── run_gepa.py              # Standalone GEPA optimization
├── langchain-gaia-agent/
│   └── agent.py              # LangChain GAIA agent (tools + tracing)
├── optimizers/
│   ├── pi_agent.py           # Pi RPC client (subprocess JSONL)
│   └── agent_optimizer.py    # Sends failures to pi, parses JSON fixes
├── artifacts/prompts/
│   └── baseline.json         # Prompt artifact (4 fields GEPA mutates)
├── mvp.config.json           # Config: models, GEPA params, budgets
└── data/
    ├── local.gaia_level1_optimize_15.jsonl   # 15 tasks for distillation
    └── local.gaia_level1_holdout_15.jsonl    # 15 tasks for evaluation only

src/evaris/
├── runner.py                 # Eval runner with checkpoints
├── distill.py                # Failure extraction + reflection
├── scorers.py                # exact_match, judge, tool_count
└── types.py                  # Trace, Span, Score dataclasses
```

## Step 1: Run Baseline

Run on both datasets separately. Use `--concurrency 1` to avoid rate limits.

```bash
# Terminal 1 — Holdout baseline
chameleon eval \
  --manifest examples/data/local.gaia_level1_holdout_15.jsonl \
  --config examples/mvp.config.json \
  --run-tag baseline-holdout \
  --concurrency 1 --task-timeout 180

# Terminal 2 — Optimize baseline
chameleon eval \
  --manifest examples/data/local.gaia_level1_optimize_15.jsonl \
  --config examples/mvp.config.json \
  --run-tag baseline-optimize \
  --concurrency 1 --task-timeout 180
```

Results are saved to `examples/.evaris/eval-runs/{run_tag}-*.json` with incremental checkpoints.

**Expected output:**
```
  ━━━━━━━━━━━━━━━━━━━━ chameleon ━━━━━━━━━━━━━━━━━━━

  $ chameleon eval
    exact_match: 4/12
    errors:      3
    avg_tools:   0.0
    run_tag:     baseline-holdout
    model:       glm-5
```

## Step 2: Run Improve Pipeline

The full self-improvement loop in one command:

```bash
chameleon improve \
  --optimize-eval examples/.evaris/eval-runs/baseline-optimize-*.json \
  --baseline-eval examples/.evaris/eval-runs/baseline-holdout-*.json \
  --manifest-holdout examples/data/local.gaia_level1_holdout_15.jsonl \
  --config examples/mvp.config.json \
  --concurrency 1 --task-timeout 180 \
  --output-dir examples/.evaris/experiment-runs/improve-run-1
```

This runs 5 steps:

| Step | Description | Output |
|------|-------------|--------|
| **1. Distill** | Extract failures, classify fix_type | `distill/failures.jsonl` |
| **2a. Agent Optimizer** | Pi agent proposes config/code fixes | `optimizers/agent-fixes.json` |
| **2b. GEPA** | Optimize 4 prompt fields via reflection | `artifacts/best-candidate.json` |
| **3. Eval** | Run optimized agent on holdout | `evals/gepa-optimized-*.json` |
| **4. Accept/Reject** | Compare vs baseline, gate on improvement | `report.md` |
| **5. Fixes Report** | Show pi agent fixes for manual review | console output |

## Step 3: Run Individual Steps

You can also run steps separately for debugging.

### Distill only
```bash
python -c "
import asyncio
from evaris.distill import _distill_async
asyncio.run(_distill_async(
    'examples/.evaris/eval-runs/baseline-optimize-*.json',
    output_path='examples/.evaris/distill/failures.jsonl',
    reflection_model='glm-5',
))
"
```

### GEPA only
```bash
chameleon eval \
  --manifest examples/data/local.gaia_level1_holdout_15.jsonl \
  --config examples/mvp.config.json \
  --artifact examples/.evaris/experiment-runs/improve-run-1/artifacts/best-candidate.json \
  --run-tag optimized \
  --concurrency 1 --task-timeout 180
```

### Eval with custom artifact
```bash
chameleon eval \
  --manifest examples/data/local.gaia_level1_holdout_15.jsonl \
  --config examples/mvp.config.json \
  --artifact examples/.evaris/experiment-runs/improve-run-1/artifacts/best-candidate.json \
  --run-tag optimized \
  --concurrency 1 --task-timeout 180
```

### Apply pi agent fixes manually
```bash
# Review the proposed fixes
cat examples/.evaris/experiment-runs/improve-run-1/optimizers/agent-fixes.json

# Apply manually (example: increase token limit)
# Edit examples/langchain-gaia-agent/agent.py as needed
```

## Configuration

Edit `examples/mvp.config.json`:

```json
{
  "task_model": "glm-5",          // Agent LLM
  "judge_model": "glm-5",         // LLM-as-judge
  "reflection_model": "glm-5",    // Root cause analysis in distill
  "gepa": {
    "max_candidates": 4,           // Max GEPA prompt variants
    "minibatch_size": 3,           // Reflection minibatch size
    "max_task_runs": 12            // Total GEPA rollouts (lower = faster)
  }
}
```

## CLI Reference

```
chameleon eval \
  --manifest <path>              # GAIA task manifest (JSONL)
  --config <path>                # Config file
  --run-tag <string>             # Name for this run
  --artifact <path>              # Custom prompt artifact (JSON)
  --concurrency <int>            # Parallel tasks (default: 2)
  --task-timeout <float>         # Seconds per task (default: 180)
  --limit <int>                  # Max tasks to run
  --no-studio                    # Disable etrace studio export
  --no-judge                     # Skip LLM judge scorer
  --debug                        # Verbose scorer output
  --llm-concurrency <int>        # Async LLM call cap (default: 5)
```

```
chameleon improve \
  --optimize-eval <path>         # Optimize eval JSON (glob)
  --baseline-eval <path>         # Holdout eval JSON (glob)
  --manifest-holdout <path>      # Holdout manifest (JSONL)
  --config <path>
  --output-dir <path>            # Experiment output directory
  --concurrency <int>
  --task-timeout <float>
  --debug
  --llm-concurrency <int>
```

## Checkpoints & Recovery

Each task saves an incremental checkpoint. If the process crashes or is killed:

```bash
# Promote checkpoint to final result
cp examples/.evaris/eval-runs/baseline-*.checkpoint.json \
   examples/.evaris/eval-runs/baseline-final.json

# Re-run only missing tasks
chameleon eval \
  --manifest /tmp/remaining.jsonl \
  --config examples/mvp.config.json \
  --run-tag baseline-fill \
  --concurrency 1 --task-timeout 180
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `429 Rate limit` | Reduce `--concurrency 1`, wait, retry |
| `asyncio.run() cannot be called` | Ensure `distill.py` uses `_distill_async` from async context |
| `tool_call_count = 0` always | Check `InMemoryExporter` is in `init_tracing()` exporters list |
| Agent refuses to use tools | Add "MUST" directives to `system_prompt` and `tool_policy` |
| GEPA scores all 0.0 | Check `evaluate()` uses `ThreadPoolExecutor` for async compatibility |
| Process hangs after GEPA | Kill and run eval separately with `--artifact` |
