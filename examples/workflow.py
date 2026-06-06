"""GAIA eval + self-improvement workflow.

Modes:
    eval     — Run agent on a manifest, score with rule-based + LLM judge.
    improve  — Distill failures from optimize eval → GEPA optimization → rerun holdout.

Usage:
    # Baseline eval (both datasets)
    python examples/workflow.py --mode eval \
        --manifest examples/data/local.gaia_level1_holdout_15.jsonl \
        --config examples/mvp.config.json --run-tag baseline

    python examples/workflow.py --mode eval \
        --manifest examples/data/local.gaia_level1_optimize_15.jsonl \
        --config examples/mvp.config.json --run-tag baseline-optimize

    # Self-improvement (distill → GEPA → accept/reject)
    python examples/workflow.py --mode improve \
        --optimize-eval "examples/.evaris/eval-runs/baseline-optimize-*.json" \
        --baseline-eval "examples/.evaris/eval-runs/baseline-*.json" \
        --manifest-holdout examples/data/local.gaia_level1_holdout_15.jsonl \
        --config examples/mvp.config.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import string
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaris import (
    EvalResults,
    Sample,
    SampleResult,
    Score,
    Trace,
    distill,
    no_errors,
    set_llm_concurrency,
    tool_call_count,
    trace_llm_judge,
)

import etrace

AGENT_PATH = ROOT / "examples" / "langchain-gaia-agent" / "agent.py"
DEFAULT_STUDIO_ENDPOINT = "http://localhost:3001/v1/traces"


# ── Shared helpers ─────────────────────────────────────────────────────


def load_agent_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("langchain_gaia_agent", AGENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent module from {AGENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_environment(no_studio: bool = False) -> None:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("ZAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["ZAI_API_KEY"]
    os.environ.setdefault("OPENAI_BASE_URL", "https://api.z.ai/api/paas/v4")
    os.environ.setdefault("OTEL_SERVICE_NAME", "gaia-eval-workflow")
    os.environ.setdefault("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", DEFAULT_STUDIO_ENDPOINT)
    if no_studio:
        os.environ["ETRACE_EXPORT"] = "memory"


def load_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_artifact(artifact_path: str | None) -> dict[str, Any]:
    if not artifact_path:
        return {}
    path = Path(artifact_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def normalize_answer(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(string.whitespace + "\"'")
    text = text.translate(str.maketrans("", "", string.punctuation.replace(".", "")))
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalized_exact_match(trace: Trace, expected: Any) -> list[Score]:
    actual_norm = normalize_answer(trace.output)
    expected_norm = normalize_answer(expected)
    return [
        Score(
            name="normalized_exact_match",
            value=actual_norm == expected_norm,
            reason=f"actual={actual_norm!r}, expected={expected_norm!r}",
        )
    ]


def load_manifest(path: Path, limit: int | None = None) -> list[Sample]:
    samples: list[Sample] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task = {
                "id": str(row.get("id") or row.get("task_id")),
                "level": int(row.get("level") or row.get("Level") or 1),
                "split": str(row.get("split") or "level1_train"),
                "input": str(row.get("input") or row.get("Question")),
                "expected": row.get("expected") or row.get("Final answer"),
                "file_path": row.get("file_path") or row.get("file_name"),
            }
            samples.append(Sample(id=task["id"], input=task, expected=task["expected"]))
            if limit is not None and len(samples) >= limit:
                break
    return samples


def resolve_eval_glob(glob_str: str, require: bool = True) -> Path:
    """Resolve a glob pattern to the most recent matching file."""
    p = Path(glob_str)
    if "*" not in glob_str and "?" not in glob_str:
        return p
    parent = p.parent if p.is_absolute() else ROOT
    matches = sorted(parent.glob(p.name))
    if not matches:
        if require:
            raise RuntimeError(f"No eval files matched: {glob_str}")
        return p
    print(f"Resolved glob → {matches[-1]}")
    return matches[-1]


def extract_baseline_score(eval_data: dict[str, Any]) -> dict[str, Any]:
    agg = eval_data.get("aggregate", {})
    em = agg.get("exact_match", {})
    judge = agg.get("trace_llm_judge", {})
    return {
        "exact_match_pass": em.get("passed", 0),
        "exact_match_total": em.get("total", 0),
        "judge_correct": judge.get("correct", 0),
        "judge_total": judge.get("total", 0),
        "errors": agg.get("errors", 0),
    }


# ── Eval output ───────────────────────────────────────────────────────


def export_results(
    results: EvalResults,
    output_path: Path,
    *,
    run_tag: str = "",
    config: dict[str, Any] | None = None,
    manifest: str = "",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = results.results
    exact_scores = [s for r in records for s in r.scores if s.name == "normalized_exact_match"]
    judge_scores = [s for r in records for s in r.scores if s.name == "trace_llm_judge"]
    error_count = sum(1 for r in records if r.error)

    output_path.write_text(
        json.dumps(
            {
                "run_tag": run_tag,
                "manifest": str(manifest),
                "model": (config or {}).get("task_model", "unknown"),
                "judge_model": (config or {}).get("judge_model"),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_samples": len(records),
                "aggregate": {
                    "exact_match": {
                        "passed": sum(1 for s in exact_scores if s.value is True),
                        "total": len(exact_scores),
                    },
                    "trace_llm_judge": {
                        "correct": sum(1 for s in judge_scores if s.value == 1.0),
                        "partial": sum(1 for s in judge_scores if s.value == 0.5),
                        "incorrect": sum(1 for s in judge_scores if s.value == 0.0),
                        "total": len(judge_scores),
                    },
                    "errors": error_count,
                },
                "results": [
                    {
                        **asdict(result),
                        "scores": [asdict(score) for score in result.scores],
                    }
                    for result in results.results
                ],
            },
            indent=2,
            default=str,
        )
    )


def print_pipeline_summary(
    *,
    eval_data: dict,
    failures: list[dict],
    agent_fixes: list[dict],
    seed_candidate: dict,
    best_candidate: dict | None = None,
    gepa_result=None,
    opt_eval_data: dict | None = None,
    accepted: bool | None = None,
    reject_reason: str = "",
) -> None:
    """Print a chameleon-style compact pipeline summary."""

    def dim(s):
        return f"\033[2m{s}\033[0m"

    def bold(s):
        return f"\033[1m{s}\033[0m"

    def green(s):
        return f"\033[32m{s}\033[0m"

    def red(s):
        return f"\033[31m{s}\033[0m"

    def yellow(s):
        return f"\033[33m{s}\033[0m"

    def cyan(s):
        return f"\033[36m{s}\033[0m"

    agg = eval_data.get("aggregate", {})
    em_pass = agg.get("exact_match", {}).get("passed", 0)
    em_total = agg.get("exact_match", {}).get("total", 0)
    n_err = agg.get("errors", 0)
    n_tasks = len(eval_data.get("results", []))

    print()
    print(f"  {bold('━━━━━━━━━━ chameleon ━━━━━━━━━━')}")
    print()
    print(f"  {bold('$ chameleon run baseline')}")
    print(f"    eval:     {em_pass}/{em_total} exact_match, {n_err} errors, {n_tasks} tasks")

    # Distill summary
    fix_types = {}
    root_causes = []
    for f in failures:
        ft = f.get("fix_type", "unknown")
        fix_types[ft] = fix_types.get(ft, 0) + 1
        rc = f.get("root_cause", "")
        if rc:
            root_causes.append(rc)

    print(f"  {bold('$ chameleon distill')}")
    print(f"    failures: {len(failures)}/{n_tasks} tasks failed")
    if fix_types:
        parts = [f"{v}→{k}" for k, v in sorted(fix_types.items())]
        print(f"    routes:   {', '.join(parts)}")
    # Show top root causes
    if root_causes:
        for rc in root_causes[:2]:
            print(f"    trace:    {dim(rc[:90])}")

    # Agent optimizer summary
    print(f"  {bold('$ chameleon optimize')}")
    if agent_fixes:
        for fix in agent_fixes:
            ftype = fix.get("type", "?")
            ffile = fix.get("file", "?")
            fdesc = fix.get("description", "")[:70]
            fkey = fix.get("key", "")
            print(f"    route:    {ftype} → {yellow('pi_agent')}")
            if ftype == "config":
                print(f"    patch:    {ffile} → {fkey}: {fix.get('old_value', '?')} → {green(str(fix.get('value')))}")
            else:
                print(f"    patch:    {cyan(ffile)}")
                print(f"              {dim(fdesc)}")
    else:
        print(f"    agent_optimizer: no code patch required")

    # GEPA summary
    if best_candidate:
        changed_fields = []
        for key in seed_candidate:
            if seed_candidate.get(key) != best_candidate.get(key, ""):
                changed_fields.append(key)

        if changed_fields:
            print(f"    route:    {', '.join(changed_fields)} → {yellow('GEPA')}")
            if gepa_result:
                print(f"    gepa:     {gepa_result.num_candidates} candidates explored")
            for field in changed_fields:
                old = seed_candidate[field][:50]
                new = best_candidate[field][:50]
                print(f"    patch:    {cyan(field)}")
                print(f"              {dim(old)}")
                print(f"              {green(new)}")
        else:
            print(f"    gepa:     no prompt mutations accepted")
    else:
        print(f"    gepa:     not run")

    # Eval + accept/reject
    if opt_eval_data:
        oagg = opt_eval_data.get("aggregate", {})
        oem_pass = oagg.get("exact_match", {}).get("passed", 0)
        oem_total = oagg.get("exact_match", {}).get("total", 0)
        o_err = oagg.get("errors", 0)
        delta = oem_pass - em_pass

        print(f"  {bold('$ chameleon heal --gate holdout')}")
        print(f"    eval:     {oem_pass}/{oem_total} exact_match, {o_err} errors")
        if delta > 0:
            print(f"    delta:    {green(f'+{delta} exact_match')} ({em_pass}→{oem_pass})")
        elif delta < 0:
            print(f"    delta:    {red(f'{delta} exact_match')} ({em_pass}→{oem_pass})")
        else:
            print(f"    delta:    {yellow('no change')} ({em_pass}→{oem_pass})")

        if accepted:
            print(f"    verdict:  {green('✅ ACCEPTED')}")
        elif accepted is False:
            print(f"    verdict:  {red('❌ REJECTED')} — {reject_reason}")
        else:
            print(f"    verdict:  {yellow('PENDING')}")

    print()
    print(f"  {bold('━━━━━━━━━━━━━━━━━━━━━━━━━━━━')}")
    print()


def print_summary(results: EvalResults) -> None:
    records = results.results
    errors = [result for result in records if result.error]
    exact_scores = [
        score
        for result in records
        for score in result.scores
        if score.name == "normalized_exact_match"
    ]
    passed = sum(1 for score in exact_scores if score.value is True)
    total = len(exact_scores)

    judge_scores = [
        score
        for result in records
        for score in result.scores
        if score.name == "trace_llm_judge"
    ]
    judge_passed = sum(1 for score in judge_scores if score.value == 1.0)
    judge_partial = sum(1 for score in judge_scores if score.value == 0.5)
    judge_total = len(judge_scores)

    print(
        f"\nEval summary: {passed}/{total} normalized exact matches, "
        f"{len(errors)}/{len(records)} agent/eval errors"
    )
    if judge_total:
        print(f"Trace judge:  {judge_passed}/{judge_total} correct, {judge_partial}/{judge_total} partial")
    for result in records:
        print(f"\n[{result.id}]")
        inp = result.input.get("input") if isinstance(result.input, dict) else result.input
        print(f"input:    {str(inp)[:120]}")
        print(f"output:   {result.output}")
        print(f"expected: {result.expected}")
        if result.error:
            print(f"error:    {result.error}")
        for score in result.scores:
            print(f"score:    {score.name}={score.value} ({score.reason or '-'})")


# ── Eval mode ─────────────────────────────────────────────────────────


def build_scorers(judge_model: str | None = None, debug: bool = False) -> list:
    if debug:
        scorers_list = [
            _debug_scorer(normalized_exact_match),
            _debug_scorer(no_errors),
            _debug_scorer(tool_call_count),
        ]
        if judge_model:
            scorers_list.append(trace_llm_judge(model=judge_model))
    else:
        scorers_list = [normalized_exact_match, no_errors, tool_call_count]
        if judge_model:
            scorers_list.append(trace_llm_judge(model=judge_model))
    return scorers_list


def _debug_scorer(scorer):
    name = getattr(scorer, "__name__", str(scorer))

    def wrapper(trace, expected):
        t0 = time.time()
        result = scorer(trace, expected)
        dt = time.time() - t0
        print(f"  [DEBUG] scorer={name} took {dt:.3f}s", flush=True)
        return result

    return wrapper


def print_result(result: SampleResult, elapsed: float) -> None:
    print(f"[DONE] {result.id} in {elapsed:.1f}s", flush=True)
    print(f"  output:   {result.output}", flush=True)
    print(f"  expected: {result.expected}", flush=True)
    if result.error:
        print(f"  error:    {result.error}", flush=True)
    for score in result.scores:
        print(f"  score:    {score.name}={score.value} ({score.reason or '-'})", flush=True)


async def run_sample_live(
    sample: Sample,
    *,
    agent: Any,
    timeout: float,
    semaphore: asyncio.Semaphore,
    scorers: list,
    debug: bool = False,
    artifact_override: dict[str, Any] | None = None,
    config_override: dict[str, Any] | None = None,
) -> SampleResult:
    async with semaphore:
        print(f"[START] {sample.id}: {sample.input['input'][:180]}", flush=True)
        started = time.time()

        try:
            agent_start = time.time()
            kwargs = {}
            if artifact_override is not None:
                kwargs["artifact"] = artifact_override
            if config_override is not None:
                kwargs["config"] = config_override
            trace = await asyncio.wait_for(agent.solve(sample.input, **kwargs), timeout=timeout)
            agent_elapsed = time.time() - agent_start
            if debug:
                print(f"  [DEBUG] agent.solve() done in {agent_elapsed:.1f}s", flush=True)

            scores: list[Score] = []
            for scorer in scorers:
                scorer_name = getattr(scorer, "__name__", "trace_llm_judge")
                if debug:
                    print(f"  [DEBUG] running scorer: {scorer_name}...", flush=True)
                t0 = time.time()
                result = scorer(trace, sample.expected)
                if asyncio.iscoroutine(result):
                    result = await result
                dt = time.time() - t0
                if isinstance(result, list):
                    scores.extend(result)
                else:
                    scores.append(result)
                if debug:
                    print(f"  [DEBUG] scorer={scorer_name} done in {dt:.1f}s", flush=True)

            sample_result = SampleResult(
                id=sample.id,
                input=trace.input,
                output=trace.output,
                expected=sample.expected,
                scores=scores,
                error=None,
            )
        except (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError, Exception) as exc:
            elapsed = time.time() - started
            print(f"[ERROR] {sample.id} after {elapsed:.1f}s: {type(exc).__name__}: {exc}", flush=True)
            sample_result = SampleResult(
                id=sample.id,
                input=sample.input,
                output="",
                expected=sample.expected,
                scores=[Score(name="no_errors", value=False, reason=str(exc)[:200])],
                error=f"{type(exc).__name__}: {exc}",
            )
        print_result(sample_result, time.time() - started)
        return sample_result


async def run_eval(args: argparse.Namespace) -> None:
    """Mode: eval — run agent on manifest and score."""
    configure_environment(args.no_studio)

    config = load_config(args.config)
    judge_model = None if args.no_judge else config.get("judge_model")
    if judge_model:
        print(f"LLM-as-judge enabled (model={judge_model})", flush=True)
        set_llm_concurrency(args.llm_concurrency)
        print(f"LLM concurrency cap: {args.llm_concurrency}", flush=True)
    else:
        print("LLM-as-judge disabled", flush=True)
    scorers = build_scorers(judge_model=judge_model, debug=args.debug)

    agent = load_agent_module()
    agent.configure_environment()
    agent.init_tracing()

    etrace.set_context(etrace._types.ContextOptions(tags=[args.run_tag]))
    print(f"Run tag: {args.run_tag}", flush=True)

    samples = load_manifest(Path(args.manifest), limit=args.limit)
    if not samples:
        raise RuntimeError(f"No samples found in manifest: {args.manifest}")

    print(
        f"Running {len(samples)} sample(s) with concurrency={args.concurrency}, "
        f"task_timeout={args.task_timeout}s",
        flush=True,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    output_path = (
        Path(args.output)
        if args.output
        else ROOT / "examples" / ".evaris" / "eval-runs" / f"{args.run_tag}-{int(time.time())}.json"
    )

    # Incremental save: checkpoint after each task
    completed: list[SampleResult] = []
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    async def run_and_save(sample: Sample) -> SampleResult:
        result = await run_sample_live(
            sample, agent=agent, timeout=args.task_timeout,
            semaphore=semaphore, scorers=scorers, debug=args.debug,
        )
        completed.append(result)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        export_results(
            EvalResults(results=list(completed)), checkpoint_path,
            run_tag=args.run_tag, config=config, manifest=args.manifest,
        )
        return result

    tasks = [run_and_save(sample) for sample in samples]
    results = EvalResults(results=list(await asyncio.gather(*tasks)))
    print_summary(results)

    export_results(results, output_path, run_tag=args.run_tag, config=config, manifest=args.manifest)
    checkpoint_path.unlink(missing_ok=True)
    print(f"\nSaved eval output: {output_path}")

    # Compact chameleon-style summary
    print()
    print(f"  {chr(0x1b)}[1m{'━' * 30} chameleon {chr(0x1b)}[0m")
    records = results.results
    em_scores = [s for r in records for s in r.scores if s.name == 'normalized_exact_match']
    em_pass = sum(1 for s in em_scores if s.value is True)
    em_total = len(em_scores)
    errs = sum(1 for r in records if r.error)
    tc_scores = [s for r in records for s in r.scores if s.name == 'tool_call_count']
    tc_avg = sum(s.value for s in tc_scores) / len(tc_scores) if tc_scores else 0
    recovered = sum(1 for r in records if any(s.name == 'normalized_exact_match' and s.value for s in r.scores))
    print(f"  {chr(0x1b)}[1m$ chameleon eval {chr(0x1b)}[0m")
    print(f"    exact_match: {em_pass}/{em_total}")
    print(f"    errors:      {errs}")
    print(f"    avg_tools:   {tc_avg:.1f}")
    if args.run_tag:
        print(f"    run_tag:     {args.run_tag}")
    print(f"    model:       {config.get('task_model', 'unknown')}")
    print()
    print(f"  {chr(0x1b)}[1m{'━' * 36}{chr(0x1b)}[0m")
    print()


# ── GEPA Adapter ───────────────────────────────────────────────────────


class GAIAAdapter:
    """GEPAAdapter for the GAIA agent.

    Bridges our LangChain GAIA agent into GEPA's optimization engine.
    A candidate is a dict[str, str] with keys: system_prompt, tool_policy,
    answer_format_policy, verification_policy.

    DataInst = GAIA task dict
    Trajectory = eval result info (answer, score, judge reason, failure type)
    RolloutOutput = raw agent answer string
    """

    def __init__(
        self,
        agent_module: Any,
        task_model: str,
        judge_model: str | None = None,
        llm_concurrency: int = 5,
        task_timeout: float = 180.0,
    ):
        self.agent = agent_module
        self.task_model = task_model
        self.judge_model = judge_model
        self.llm_concurrency = llm_concurrency
        self.task_timeout = task_timeout
        self.propose_new_texts = None

    def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ):
        """Evaluate a candidate prompt artifact on a batch of GAIA tasks.

        Runs the agent synchronously per task (GEPA calls this in threads).
        Returns scores: 1.0 for correct (exact match), 0.0 otherwise.
        Higher is better for GEPA.
        """
        from gepa.core.adapter import EvaluationBatch

        outputs: list[str] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None

        config = {
            "task_model": self.task_model,
            "candidate_id": candidate.get("candidate_id", "unknown"),
        }

        for task in batch:
            try:
                # GEPA may call evaluate() from the main thread or its own threads.
                # Always use a thread with a fresh loop to avoid nesting with the
                # outer asyncio.run(main()) event loop.
                from concurrent.futures import ThreadPoolExecutor
                loop = ThreadPoolExecutor(max_workers=1)
                trace = loop.submit(
                    asyncio.run,
                    asyncio.wait_for(
                        self.agent.solve(task, config=config, artifact=candidate),
                        timeout=self.task_timeout,
                    )
                ).result()
                loop.shutdown(wait=False)
                answer = trace.output
                correct = normalize_answer(answer) == normalize_answer(task.get("expected", ""))

                if trajectories is not None:
                    trajectories.append({
                        "task_id": task.get("id"),
                        "question": task.get("input", ""),
                        "expected": task.get("expected", ""),
                        "answer": answer,
                        "correct": correct,
                        "score": 1.0 if correct else 0.0,
                    })

                scores.append(1.0 if correct else 0.0)
                outputs.append(answer)

            except Exception as exc:
                scores.append(0.0)
                outputs.append(f"ERROR: {exc}")
                if trajectories is not None:
                    trajectories.append({
                        "task_id": task.get("id"),
                        "question": task.get("input", ""),
                        "expected": task.get("expected", ""),
                        "answer": "",
                        "correct": False,
                        "error": str(exc),
                        "score": 0.0,
                    })

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Format failures into reflective records for the teacher LLM."""
        ret: dict[str, list[dict[str, Any]]] = {}
        trajectories = eval_batch.trajectories
        if not trajectories:
            return ret

        for component_name in components_to_update:
            items: list[dict[str, Any]] = []
            for traj in trajectories:
                if traj.get("correct"):
                    continue  # Only reflect on failures
                items.append({
                    "Inputs": traj["question"],
                    "Generated Outputs": traj["answer"],
                    "Feedback": (
                        f"Incorrect. Expected: {traj['expected']}. "
                        f"The answer provided was wrong. Think about what instruction "
                        f"change would help the agent get this right."
                    ),
                })
            if items:
                ret[component_name] = items

        return ret


# ── Improve mode ──────────────────────────────────────────────────────


def should_accept(baseline: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, str]:
    """Acceptance gate: +1 exact match, no error regression."""
    b_em = baseline.get("exact_match_pass", 0)
    c_em = candidate.get("exact_match_pass", 0)
    diff = c_em - b_em

    if diff >= 1:
        reasons = [f"exact match +{diff} ({b_em}→{c_em})"]
    else:
        return False, f"No improvement: exact match {b_em}→{c_em} (need +1)"

    b_err = baseline.get("errors", 0)
    c_err = candidate.get("errors", 0)
    if c_err > b_err:
        return False, f"Error regression: {b_err}→{c_err}"

    b_judge = baseline.get("judge_correct", 0)
    c_judge = candidate.get("judge_correct", 0)
    if c_judge > b_judge:
        reasons.append(f"judge correct +{c_judge - b_judge} ({b_judge}→{c_judge})")

    reasons.append("All gates passed")
    return True, "; ".join(reasons)


async def run_improve(args: argparse.Namespace) -> None:
    """Mode: improve — distill failures → GEPA → accept/reject."""
    configure_environment(args.no_studio)

    config = load_config(args.config)
    from evaris.distill import _distill_async as distill, set_llm_concurrency as distill_set_llm
    distill_set_llm(args.llm_concurrency)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Distill failures from optimize eval ───────────────────
    print("=" * 60)
    print("STEP 1: DISTILL — extracting failures from optimize eval")
    print("=" * 60)

    optimize_eval_path = resolve_eval_glob(args.optimize_eval)
    failures_path = await distill(
        optimize_eval_path,
        output_path=output_dir / "distill" / "failures.jsonl",
        reflection_model=config.get("reflection_model"),
    )

    # ── Step 2a: Agent optimizer — pi proposes config/code fixes ────
    print("\n" + "=" * 60)
    print("STEP 2a: AGENT OPTIMIZER — pi agent proposes config/code fixes")
    print("=" * 60)

    sys.path.insert(0, str(ROOT / "examples"))
    from optimizers.agent_optimizer import propose_agent_fixes

    agent_fixes_path = propose_agent_fixes(
        failures_path,
        output_path=output_dir / "optimizers" / "agent-fixes.json",
        cwd=str(ROOT),
    )

    # ── Step 2b: GEPA — optimize prompt artifact ───────────────────
    print("\n" + "=" * 60)
    print("STEP 2b: GEPA — optimizing prompt artifact via reflection")
    print("=" * 60)

    baseline_artifact = load_artifact(args.artifact)
    if not baseline_artifact:
        agent_mod = load_agent_module()
        baseline_artifact = agent_mod.DEFAULT_ARTIFACT

    # Save baseline artifact
    (output_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (output_dir / "artifacts" / "baseline.json").write_text(
        json.dumps(baseline_artifact, indent=2)
    )

    # Load optimize set for GEPA training
    optimize_tasks = []
    with optimize_eval_path.open() as f:
        data = json.load(f)
    for r in data.get("results", []):
        task_input = r.get("input", {})
        if isinstance(task_input, dict):
            optimize_tasks.append(task_input)
        else:
            optimize_tasks.append({"input": str(task_input)})

    if not optimize_tasks:
        raise RuntimeError("No tasks found in optimize eval results")

    # Load holdout for validation
    holdout_path = Path(args.manifest_holdout)
    holdout_tasks = []
    with holdout_path.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                holdout_tasks.append({
                    "id": str(row.get("id") or row.get("task_id")),
                    "level": int(row.get("level") or row.get("Level") or 1),
                    "split": str(row.get("split") or "level1_train"),
                    "input": str(row.get("input") or row.get("Question")),
                    "expected": row.get("expected") or row.get("Final answer"),
                    "file_path": row.get("file_path") or row.get("file_name"),
                })

    # Build adapter
    agent_module = load_agent_module()
    agent_module.configure_environment()
    agent_module.init_tracing()

    task_model = config.get("task_model", "glm-5")
    reflection_model = config.get("reflection_model", "glm-5-turbo")

    adapter = GAIAAdapter(
        agent_module=agent_module,
        task_model=task_model,
        judge_model=config.get("judge_model"),
        llm_concurrency=args.llm_concurrency,
        task_timeout=args.task_timeout,
    )

    # Make reflection_lm callable for GEPA
    from openai import AsyncOpenAI

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.z.ai/api/paas/v4")
    api_key = os.environ.get("OPENAI_API_KEY")
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    def reflection_lm(prompt: str | list[dict[str, Any]]) -> str:
        loop = asyncio.get_event_loop()
        if loop.is_running():
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
        return loop.run_until_complete(
            client.chat.completions.create(
                model=reflection_model,
                messages=[{"role": "user", "content": prompt if isinstance(prompt, str) else prompt[-1]["content"]}],
                temperature=0.7,
                max_tokens=2048,
            )
        ).choices[0].message.content or ""

    # Seed candidate: the 4 prompt fields GEPA will mutate
    seed_candidate = {
        "system_prompt": baseline_artifact.get("system_prompt", ""),
        "tool_policy": baseline_artifact.get("tool_policy", ""),
        "answer_format_policy": baseline_artifact.get("answer_format_policy", ""),
        "verification_policy": baseline_artifact.get("verification_policy", ""),
    }

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
    (output_dir / "artifacts" / "best-candidate.json").write_text(
        json.dumps(best_candidate, indent=2)
    )

    # ── Step 3: Evaluate best candidate on holdout ────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: EVALUATE — running best candidate on holdout")
    print("=" * 60)

    baseline_eval_path = resolve_eval_glob(args.baseline_eval)
    baseline_data = json.loads(baseline_eval_path.read_text())
    baseline_score = extract_baseline_score(baseline_data)
    print(f"Baseline: {baseline_score['exact_match_pass']}/{baseline_score['exact_match_total']} exact match")

    holdout_samples = load_manifest(holdout_path)
    scorers = build_scorers(judge_model=config.get("judge_model"), debug=args.debug)

    etrace.set_context(etrace._types.ContextOptions(tags=["optimized", "best-candidate"]))

    semaphore = asyncio.Semaphore(args.concurrency)
    completed: list[SampleResult] = []
    opt_checkpoint = output_dir / "evals" / f"gepa-optimized-checkpoint-{int(time.time())}.json"

    async def run_and_checkpoint(sample: Sample) -> SampleResult:
        result = await run_sample_live(
            sample, agent=agent_module, timeout=args.task_timeout,
            semaphore=semaphore, scorers=scorers, debug=args.debug,
            artifact_override=best_candidate,
            config_override=config | {"candidate_id": "best-candidate"},
        )
        completed.append(result)
        export_results(
            EvalResults(results=list(completed)), opt_checkpoint,
            run_tag="optimized", config=config, manifest=str(holdout_path),
        )
        return result

    tasks = [run_and_checkpoint(sample) for sample in holdout_samples]
    results = EvalResults(results=list(await asyncio.gather(*tasks)))
    print_summary(results)

    opt_eval_path = output_dir / "evals" / f"gepa-optimized-{int(time.time())}.json"
    export_results(
        results, opt_eval_path,
        run_tag="optimized", config=config, manifest=str(holdout_path),
    )

    # ── Step 4: Accept/reject ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: ACCEPT/REJECT")
    print("=" * 60)

    opt_data = json.loads(opt_eval_path.read_text())
    opt_score = extract_baseline_score(opt_data)

    accepted, reason = should_accept(baseline_score, opt_score)
    print(f"\nBaseline:   {baseline_score['exact_match_pass']}/{baseline_score['exact_match_total']} exact match")
    print(f"GEPA:       {opt_score['exact_match_pass']}/{opt_score['exact_match_total']} exact match")
    print(f"Verdict:    {'✅ ACCEPT' if accepted else '❌ REJECT'}")
    if reason:
        print(f"Reason:     {reason}")

    if accepted:
        report_path = output_dir / "report.md"
        report_path.write_text(
            f"# Self-Improvement Report\n\n"
            f"## GEPA Accepted: {reason}\n\n"
            f"| Metric | Baseline | Optimized |\n"
            f"|--------|----------|-----------|\n"
            f"| Exact match | {baseline_score['exact_match_pass']}/{baseline_score['exact_match_total']} "
            f"| {opt_score['exact_match_pass']}/{opt_score['exact_match_total']} |\n"
            f"| Judge correct | {baseline_score['judge_correct']} | {opt_score['judge_correct']} |\n"
            f"| Errors | {baseline_score['errors']} | {opt_score['errors']} |\n"
            f"\n## GEPA Prompt Artifact\n\n```json\n{json.dumps(best_candidate, indent=2)}\n```\n"
        )
        print(f"\nSaved report: {report_path}")
        print(f"\nTo use the optimized artifact:")
        print(f"  python examples/workflow.py --mode eval \\")
        print(f"    --manifest examples/data/local.gaia_level1_holdout_15.jsonl \\")
        print(f"    --config examples/mvp.config.json \\")
        print(f"    --artifact {output_dir}/artifacts/best-candidate.json \\")
        print(f"    --run-tag optimized")
    else:
        print("\nNo improvement found. Baseline remains best.")

    # ── Step 5: Agent fixes report ────────────────────────────────────
    agent_fixes = []
    if agent_fixes_path.exists():
        agent_fixes = json.loads(agent_fixes_path.read_text())
    if agent_fixes:
        print(f"\n{'=' * 60}")
        print(f"STEP 5: AGENT FIXES — {len(agent_fixes)} proposed (manual review)")
        print(f"{'=' * 60}")
        for i, fix in enumerate(agent_fixes, 1):
            print(f"  Fix {i}: [{fix.get('type', '?')}] {fix.get('file', '?')}")
            print(f"    {fix.get('description', 'No description')[:120]}")
        fixes_path = output_dir / "optimizers" / "agent-fixes.json"
        print(f"\nReview fixes at: {fixes_path}")
        print(f"To apply, run manually or use: --apply-optimizer agent")

    print(f"\nExperiment output: {output_dir}")

    # ── Pipeline summary (chameleon UX) ──────────────────────────────
    failures_data = []
    if failures_path.exists():
        failures_data = [json.loads(l) for l in failures_path.read_text().strip().split("\n") if l.strip()]

    optimize_eval_data = json.loads(optimize_eval_path.read_text())

    print_pipeline_summary(
        eval_data=optimize_eval_data,
        failures=failures_data,
        agent_fixes=agent_fixes,
        seed_candidate=seed_candidate,
        best_candidate=best_candidate,
        gepa_result=gepa_result,
        opt_eval_data=opt_data if accepted else None,
        accepted=accepted,
        reject_reason=reason if not accepted else "",
    )


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAIA eval + self-improvement workflow")
    parser.add_argument("--mode", choices=["eval", "improve"], default="eval")
    parser.add_argument("--manifest")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--task-timeout", type=float, default=180.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-studio", action="store_true")
    parser.add_argument("--config", default=str(ROOT / "examples" / "mvp.config.json"))
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--llm-concurrency", type=int, default=5)
    parser.add_argument("--run-tag", default="baseline")
    parser.add_argument("--artifact", default=None)

    # Improve-mode args
    parser.add_argument("--optimize-eval", default=None, help="Path to optimize eval JSON (glob OK)")
    parser.add_argument("--baseline-eval", default=None, help="Path to baseline holdout eval JSON (glob OK)")
    parser.add_argument("--manifest-holdout", default=None, help="Path to holdout manifest")
    parser.add_argument("--output-dir", default=None, help="Experiment output directory")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    dotenv_path = ROOT / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("ZAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["ZAI_API_KEY"]
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set ZAI_API_KEY or OPENAI_API_KEY before running.")

    if args.mode == "eval":
        await run_eval(args)
    elif args.mode == "improve":
        await run_improve(args)


if __name__ == "__main__":
    asyncio.run(main())
