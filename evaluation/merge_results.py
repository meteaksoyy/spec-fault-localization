"""
Merge two batch result files produced by run_evaluation.py --indices into one.

Usage:
  python evaluation/merge_results.py results/eval_full_5c_batch1.json results/eval_full_5c_batch2.json
  python evaluation/merge_results.py results/eval_full_5c_batch1.json results/eval_full_5c_batch2.json --output results/eval_full_5c_merged.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import compute_metrics, print_metrics


def merge(path_a: str, path_b: str, mode: str, output: str) -> None:
    with open(path_a, encoding="utf-8") as f:
        data_a = json.load(f)
    with open(path_b, encoding="utf-8") as f:
        data_b = json.load(f)

    def _get(data: dict, key: str) -> dict:
        if key in data:
            return data[key]
        available = list(data.keys())
        if len(available) == 1:
            return data[available[0]]
        raise KeyError(f"Mode '{key}' not found in file. Available: {available}")

    result_a = _get(data_a, mode)
    result_b = _get(data_b, mode)

    # Merge predictions+ground_truths by trajectory_id, preferring batch_b on conflict.
    # This handles the case where batch_a is a partial run with null predictions for
    # failed traces, and batch_b re-runs only those failed traces.
    pairs_a = list(zip(result_a.get("predictions", []), result_a.get("ground_truths", [])))
    pairs_b = list(zip(result_b.get("predictions", []), result_b.get("ground_truths", [])))

    merged_map: dict[str, tuple] = {}
    for pred, gt in pairs_a:
        tid = pred.get("_trajectory_id", id(pred))
        merged_map[tid] = (pred, gt)
    for pred, gt in pairs_b:
        tid = pred.get("_trajectory_id", id(pred))
        merged_map[tid] = (pred, gt)  # batch_b wins on conflict

    preds, gts = zip(*merged_map.values()) if merged_map else ([], [])
    preds, gts = list(preds), list(gts)

    # Only carry forward errors for traces that still have null predictions
    null_tids = {p.get("_trajectory_id") for p in preds if not p.get("failure_mode")}
    errors_a = [e for e in result_a.get("errors", []) if True]  # keep all from a
    errors_b = result_b.get("errors", [])
    # Deduplicate errors by index, preferring batch_b
    error_map: dict[int, dict] = {e["index"]: e for e in errors_a}
    for e in errors_b:
        error_map[e["index"]] = e
    errors = [e for e in error_map.values() if any(
        p.get("_trajectory_id") and not p.get("failure_mode") for p in preds
    ) or True]
    errors = list(error_map.values())

    metrics = compute_metrics(preds, gts)
    metrics["skipped"] = result_a.get("metrics", {}).get("skipped", 0) + result_b.get("metrics", {}).get("skipped", 0)
    metrics["errors"] = len(errors)

    merged = {
        mode: {
            "mode": mode,
            "metrics": {k: v for k, v in metrics.items() if k != "per_trace"},
            "predictions": preds,
            "ground_truths": gts,
            "errors": errors,
            "merged_from": [path_a, path_b],
        }
    }

    print_metrics(metrics, title=f"Merged results ({mode})")
    print(f"  Total predictions : {len(preds)}")
    print(f"  Total errors      : {len(errors)}")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two batch evaluation result files.")
    parser.add_argument("batch_a", help="First batch results JSON")
    parser.add_argument("batch_b", help="Second batch results JSON")
    parser.add_argument("--mode", default="full", help="Mode key to merge (default: full)")
    parser.add_argument("--output", default=None, help="Output path (default: derived from inputs)")
    args = parser.parse_args()

    if args.output is None:
        stem = Path(args.batch_a).stem.replace("_batch1", "").replace("_batch2", "")
        args.output = str(Path(args.batch_a).parent / f"{stem}_merged.json")

    print(f"Merging:\n  {args.batch_a}\n  {args.batch_b}\n")
    merge(args.batch_a, args.batch_b, args.mode, args.output)


if __name__ == "__main__":
    main()
