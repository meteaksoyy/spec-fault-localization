"""
Inspect a saved evaluation results JSON file.

Usage:
  python evaluation/inspect_results.py
  python evaluation/inspect_results.py results/eval_full_20260508_151918.json
  python evaluation/inspect_results.py results/eval_full_20260508_151918.json --mode baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR


def _latest_results_file() -> str:
    files = sorted(Path(RESULTS_DIR).glob("eval_*.json"), key=os.path.getmtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No eval_*.json files found in {RESULTS_DIR}")
    return str(files[0])


def inspect(path: str, mode: str) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if mode not in data:
        available = list(data.keys())
        print(f"Mode '{mode}' not in file. Available: {available}")
        mode = available[0]
        print(f"Falling back to '{mode}'.\n")

    result = data[mode]
    preds  = result.get("predictions", [])
    gts    = result.get("ground_truths", [])
    errors = result.get("errors", [])
    m      = result.get("metrics", {})

    print(f"File   : {path}")
    print(f"Mode   : {mode}")
    print(f"Traces : {m.get('n_traces', len(preds))}  |  Errors: {len(errors)}  |  Skipped: {m.get('skipped', '?')}")
    print(f"Mode acc (strict/lenient): {m.get('failure_mode_accuracy', float('nan')):.1%} / {m.get('failure_mode_accuracy_lenient', float('nan')):.1%}")
    print(f"Fam  acc (strict/lenient): {m.get('failure_family_accuracy', float('nan')):.1%} / {m.get('failure_family_accuracy_lenient', float('nan')):.1%}")
    print()

    # Per-trace breakdown
    print("=" * 90)
    print(f"{'Trajectory':<20} {'GT mode':<28} {'Pred mode':<28} {'Fam GT>Pred':<12} {'Step'}")
    print("=" * 90)

    for p, g in zip(preds, gts):
        tid      = p.get("_trajectory_id", "?")
        gt_mode  = g.get("failure_mode", "")
        gt_fam   = g.get("failure_family", "")
        pr_mode  = p.get("failure_mode", "")
        pr_fam   = p.get("failure_family", "")
        step     = p.get("decisive_step", "?")
        match    = "✓" if pr_mode.strip().lower() == gt_mode.strip().lower() else " "
        fam_str  = f"{gt_fam}>{pr_fam}"
        print(f"{match} {tid:<18} {gt_mode:<28} {pr_mode:<28} {fam_str:<12} {step}")

    print()

    # Per-trace reasoning detail
    print("=" * 90)
    print("REASONING DETAIL")
    print("=" * 90)

    for p, g in zip(preds, gts):
        tid      = p.get("_trajectory_id", "?")
        gt_mode  = g.get("failure_mode", "")
        gt_fam   = g.get("failure_family", "")
        pr_mode  = p.get("failure_mode", "")
        pr_fam   = p.get("failure_family", "")
        all_gt   = [e["mode"] for e in g.get("all_failure_modes", [])]

        print(f"\n--- {tid} ---")
        print(f"  GT  : {gt_mode} ({gt_fam})  |  all labels: {all_gt}")
        print(f"  PRED: {pr_mode} ({pr_fam})  |  agent: {p.get('culprit_agent')}  step: {p.get('decisive_step')}")
        reason_step = p.get("reason_for_step", "").replace("\n", " ")
        reason_cat  = p.get("reason_for_category", "").replace("\n", " ")
        viol_steps  = p.get("supporting_violation_steps", [])
        print(f"  Step reason   : {reason_step[:300]}")
        print(f"  Category reason: {reason_cat[:200]}")
        print(f"  Violation steps: {viol_steps}")

    # Errors
    if errors:
        print()
        print("=" * 90)
        print("ERRORS")
        print("=" * 90)
        for e in errors:
            print(f"\n  Index {e['index']} | stage: {e['stage']}")
            print(f"  Error: {e['error']}")
            tb = e.get("traceback", "")
            if tb:
                # Print last 3 lines of traceback
                lines = [l for l in tb.strip().splitlines() if l.strip()]
                for line in lines[-3:]:
                    print(f"    {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a saved evaluation results JSON.")
    parser.add_argument("path", nargs="?", default=None, help="Path to results JSON (default: latest in results/)")
    parser.add_argument("--mode", default="full", help="Mode key inside the JSON (default: full)")
    args = parser.parse_args()

    path = args.path or _latest_results_file()
    print(f"Loading: {path}\n")
    inspect(path, args.mode)


if __name__ == "__main__":
    main()
