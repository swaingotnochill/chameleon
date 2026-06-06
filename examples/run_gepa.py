"""Run GEPA optimization standalone (STEP 2b of improve pipeline).

Usage:
    python examples/run_gepa.py \
        --baseline-artifact examples/artifacts/prompts/baseline.json \
        --optimize-manifest examples/data/local.gaia_level1_optimize_15.jsonl \
        --holdout-manifest examples/data/local.gaia_level1_holdout_15.jsonl \
        --config examples/mvp.config.json \
        --output-dir examples/.evaris/experiment-runs/improve-run-2
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Allow imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from workflow import GAIAAdapter, load_agent_module, normalize_answer
from evaris.types import EvalResults


def main():
    parser = argparse.ArgumentParser(description="Run GEPA optimization standalone")
    parser.add_argument("--baseline-artifact", required=True, help="Path to baseline prompts JSON")
    parser.add_argument("--optimize-manifest", required=True, help="Path to optimize set manifest")
    parser.add_argument("--holdout-manifest", required=True, help="Path to holdout set manifest")
    parser.add_argument("--config", default="examples/mvp.config.json", help="Config file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--task-timeout", type=float, default=180.0)
    parser.add_argument("--llm-concurrency", type=int, default=5)
    parser.add_argument("--debug", action="store_true", help="Verbose output")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    # Load artifacts and tasks
    baseline_artifact = json.loads(Path(args.baseline_artifact).read_text())
    optimize_tasks = [json.loads(l) for l in Path(args.optimize_manifest).read_text().strip().split("\n")]
    holdout_tasks = [json.loads(l) for l in Path(args.holdout_manifest).read_text().strip().split("\n")]

    if args.debug:
        print(f"Baseline artifact: {list(baseline_artifact.keys())}")
        print(f"Task model: {task_model}")
        print(f"Reflection model: {reflection_model}")
        print(f"GEPA config: {json.dumps(config.get('gepa', {}), indent=2)}")
        print(f"API base: {base_url}")
    print(f"\nOptimize set: {len(optimize_tasks)} tasks")
    print(f"Holdout set:   {len(holdout_tasks)} tasks")

    # Load agent
    agent_module = load_agent_module()
    agent_module.configure_environment()
    agent_module.init_tracing()

    task_model = config.get("task_model", "glm-5")
    reflection_model = config.get("reflection_model", "glm-5")

    adapter = GAIAAdapter(
        agent_module=agent_module,
        task_model=task_model,
        judge_model=config.get("judge_model"),
        llm_concurrency=args.llm_concurrency,
        task_timeout=args.task_timeout,
    )

    # Reflection LLM callable for GEPA
    from openai import AsyncOpenAI
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.z.ai/api/paas/v4")
    api_key = os.environ.get("OPENAI_API_KEY")
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    def reflection_lm(prompt):
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(
                asyncio.run,
                client.chat.completions.create(
                    model=reflection_model,
                    messages=[{"role": "user", "content": prompt if isinstance(prompt, str) else prompt[-1]["content"]}],
                    temperature=0.7,
                    max_tokens=2048,
                ),
            ).result().choices[0].message.content or ""

    # Seed candidate
    seed_candidate = {
        "system_prompt": baseline_artifact.get("system_prompt", ""),
        "tool_policy": baseline_artifact.get("tool_policy", ""),
        "answer_format_policy": baseline_artifact.get("answer_format_policy", ""),
        "verification_policy": baseline_artifact.get("verification_policy", ""),
    }

    print("\n" + "=" * 60)
    print("GEPA OPTIMIZATION")
    print("=" * 60)

    import gepa
    gepa_result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=optimize_tasks,
        valset=holdout_tasks,
        adapter=adapter,
        task_lm=None,
        reflection_lm=reflection_lm,
        candidate_selection_strategy="pareto",
        reflection_minibatch_size=config.get("gepa", {}).get("minibatch_size", 3),
        max_metric_calls=config.get("gepa", {}).get("max_task_runs", 24),
        run_dir=str(output_dir),
        display_progress_bar=True,
    )

    best_candidate = gepa_result.best_candidate
    print(f"\nGEPA completed. Explored {gepa_result.num_candidates} candidates.")
    print(f"Best candidate index: {gepa_result.best_idx}")

    # Save best candidate
    best_path = output_dir / "artifacts" / "best-candidate.json"
    best_path.write_text(json.dumps(best_candidate, indent=2))
    print(f"Saved best candidate: {best_path}")

    # Print diff
    for key in seed_candidate:
        if seed_candidate[key] != best_candidate.get(key, ""):
            print(f"\n--- {key} CHANGED ---")
            print(f"BEFORE: {seed_candidate[key]}")
            print(f"AFTER:  {best_candidate[key]}")
        else:
            print(f"\n--- {key} (unchanged) ---")


if __name__ == "__main__":
    main()
