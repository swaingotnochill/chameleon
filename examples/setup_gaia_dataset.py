"""Create local GAIA manifests for baseline and optimization experiments.

Requires a Hugging Face token with access to gaia-benchmark/GAIA:

    source ~/.zshrc
    uv run python examples/setup_gaia_dataset.py

The generated files are intentionally gitignored.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import urllib.request
from pathlib import Path
from typing import Any

DATASET = "gaia-benchmark/GAIA"
DATASET_ENCODED = "gaia-benchmark%2FGAIA"
DEFAULT_CONFIG = "2023_level1"
DEFAULT_SOURCE_SPLIT = "validation"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "examples" / "data"
ATTACHMENT_DIR = DATA_DIR / "gaia_attachments"


def require_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN first, e.g. `source ~/.zshrc`.")
    return token


def authed_json(url: str, token: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def authed_download(url: str, token: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as response:
        output_path.write_bytes(response.read())


def fetch_rows(config: str, source_split: str, token: str) -> list[dict[str, Any]]:
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={DATASET_ENCODED}&config={config}&split={source_split}&offset=0&length=100"
    )
    payload = authed_json(url, token)
    total = int(payload["num_rows_total"])
    if total > len(payload["rows"]):
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={DATASET_ENCODED}&config={config}&split={source_split}&offset=0&length={total}"
        )
        payload = authed_json(url, token)
    return [item["row"] for item in payload["rows"]]


def download_attachment(remote_path: str, token: str) -> str:
    local_path = ATTACHMENT_DIR / remote_path
    if local_path.exists():
        return str(local_path)
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{remote_path}"
    authed_download(url, token, local_path)
    return str(local_path)


def to_task(row: dict[str, Any], token: str) -> dict[str, Any]:
    remote_file_path = row.get("file_path") or ""
    file_path = download_attachment(remote_file_path, token) if remote_file_path else None
    return {
        "id": row["task_id"],
        "level": int(row["Level"]),
        "source_dataset": DATASET,
        "source_split": DEFAULT_SOURCE_SPLIT,
        "input": row["Question"],
        "expected": row["Final answer"],
        "file_name": row.get("file_name") or None,
        "file_path": file_path,
        "remote_file_path": remote_file_path or None,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]], split_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps({**row, "split": split_name}, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create local GAIA Level 1 manifests.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source-split", default=DEFAULT_SOURCE_SPLIT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimize-size", type=int, default=20)
    parser.add_argument("--dev-size", type=int, default=10)
    parser.add_argument("--smoke-size", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = require_token()
    rows = fetch_rows(args.config, args.source_split, token)
    rows = [row for row in rows if row.get("Final answer") and row.get("Final answer") != "?"]

    tasks = [to_task(row, token) for row in rows]
    rng = random.Random(args.seed)
    rng.shuffle(tasks)

    optimize = tasks[: args.optimize_size]
    dev = tasks[args.optimize_size : args.optimize_size + args.dev_size]
    holdout = tasks[args.optimize_size + args.dev_size :]
    smoke_candidates = [task for task in optimize if not task.get("file_path")]
    smoke = smoke_candidates[: args.smoke_size]

    write_jsonl(DATA_DIR / "local.gaia_level1_all.jsonl", tasks, "level1_all")
    write_jsonl(DATA_DIR / "local.gaia_level1_optimize.jsonl", optimize, "level1_optimize")
    write_jsonl(DATA_DIR / "local.gaia_level1_dev.jsonl", dev, "level1_dev")
    write_jsonl(DATA_DIR / "local.gaia_level1_holdout.jsonl", holdout, "level1_holdout")
    write_jsonl(DATA_DIR / "local.gaia_level1_smoke.jsonl", smoke, "level1_smoke")

    summary = {
        "source": {
            "dataset": DATASET,
            "config": args.config,
            "split": args.source_split,
            "seed": args.seed,
        },
        "counts": {
            "all": len(tasks),
            "optimize": len(optimize),
            "dev": len(dev),
            "holdout": len(holdout),
            "smoke": len(smoke),
            "attachments": sum(1 for task in tasks if task.get("file_path")),
        },
        "policy": {
            "optimize": "Only use for GEPA/search.",
            "dev": "Use for candidate selection and debugging.",
            "holdout": "Use for baseline-vs-improved final comparison; do not feed to GEPA.",
            "smoke": "Use for cheap plumbing tests.",
        },
    }
    summary_path = DATA_DIR / "local.gaia_level1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote manifests under {DATA_DIR}")


if __name__ == "__main__":
    main()
