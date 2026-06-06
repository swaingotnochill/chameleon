# LangChain GAIA Agent

Atomic LangChain GAIA agent.

This folder is only the product-under-test: a runnable GAIA agent with tools and
etrace wiring. Eval workflows should import `solve` from `agent.py` and pass it
as the mvp-evals solver.

## Run

Start etrace Studio from your separate `etrace` repo:

```bash
cd path/to/etrace/etrace-studio
pnpm dev
```

This agent sends traces to `http://localhost:3001/v1/traces` by default.

Prompt artifacts are optional and live outside the agent folder. The agent has
embedded prompt defaults and does not need an artifact unless you pass
`--artifact`.

In another terminal, run the demo task against Z.ai:

```bash
cp .env.example .env
```

Fill `ZAI_API_KEY` in the root `.env` file, then run:

```bash
uv run python examples/langchain-gaia-agent/agent.py
```

Run one local GAIA manifest task:

```bash
uv run python examples/langchain-gaia-agent/agent.py \
  --manifest examples/data/local.gaia_manifest.jsonl
```

Run with an explicit prompt artifact:

```bash
uv run python examples/langchain-gaia-agent/agent.py \
  --artifact examples/artifacts/prompts/baseline.json
```

The real manifest should use this JSONL shape and stay local:

```json
{"id":"gaia-dev-level1-local-001","level":1,"split":"level1_train","input":"...","expected":"...","file_path":null}
```

Run a manual question:

```bash
uv run python examples/langchain-gaia-agent/agent.py \
  --question "A rectangle has side lengths 17 and 23. What is its area?" \
  --expected "391"
```

Use in-memory tracing if Studio is not running:

```bash
uv run python examples/langchain-gaia-agent/agent.py --no-studio
```

## Eval Workflow

Set up local GAIA Level 1 manifests:

```bash
source ~/.zshrc
uv run python examples/setup_gaia_dataset.py
```

This uses GAIA Level 1 `validation` because the official `test` split has hidden
answers (`Final answer` is `?`) and cannot be scored locally.

Generated local manifests:

```text
examples/data/local.gaia_level1_smoke.jsonl      # 3 cheap plumbing tasks
examples/data/local.gaia_level1_optimize.jsonl   # GEPA/search only
examples/data/local.gaia_level1_dev.jsonl        # candidate selection/debugging
examples/data/local.gaia_level1_holdout.jsonl    # final baseline vs improved check
```

Run the first eval stage from the repo root:

```bash
uv run python examples/workflow.py \
  --manifest examples/data/local.gaia_level1_smoke.jsonl \
  --limit 3 \
  --concurrency 2 \
  --task-timeout 180
```

This calls `agent.solve` as the `mvp-evals` solver, sends traces to Studio, and
prints per-task start/done output as each task finishes.
