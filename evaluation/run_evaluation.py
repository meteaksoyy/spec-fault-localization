"""
Evaluation runner for specification-based fault localization.

Loads the MAST MAD dataset, runs the full pipeline on each trace,
and computes metrics against human-annotated ground truth.

Supported modes:
  full        — global specs + dynamic constraints + validation log + judge
  baseline    — raw LLM-as-a-judge (no spec extraction, no violation log)
  global_only — only global constraints (no dynamic generation)
  checklist   — judge with taxonomy checklist but empty violation log
  all         — runs full + baseline side-by-side and prints comparison

Usage:
  python evaluation/run_evaluation.py --mode full --max_traces 50
  python evaluation/run_evaluation.py --mode all --max_traces 20 --output results/run_01.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    MAD_HUMAN_FILE,
    MAD_FULL_FILE,
    RAW_DATA_DIR,
    RESULTS_DIR,
    CONSTRAINT_GENERATION_MODE,
)
from src.data.normalizer import normalize_mad_dataset, normalize_trace
from src.spec_extractor import extract_global_specs
from src.validation_log import build_validation_log, build_validation_log_one_shot
from src.judge import run_judge
from evaluation.metrics import compute_metrics, print_metrics, compare_methods


# ── Dataset Loading ────────────────────────────────────────────────────────────

def _load_mad(prefer_human: bool = True) -> list[dict]:
    """Load MAD records from local files (download_mast.py must have run first)."""
    candidates = []
    if prefer_human:
        candidates = [
            os.path.join(RAW_DATA_DIR, MAD_HUMAN_FILE),
            os.path.join(RAW_DATA_DIR, MAD_FULL_FILE),
        ]
    else:
        candidates = [
            os.path.join(RAW_DATA_DIR, MAD_FULL_FILE),
            os.path.join(RAW_DATA_DIR, MAD_HUMAN_FILE),
        ]

    for path in candidates:
        if os.path.exists(path):
            print(f"Loading dataset from: {path}")
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            # Some dumps wrap the list in a key
            for key in ("data", "records", "traces"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return list(data.values()) if isinstance(data, dict) else []

    raise FileNotFoundError(
        f"MAD dataset not found in {RAW_DATA_DIR}. "
        "Run: python data/download_mast.py --source huggingface"
    )


def _has_ground_truth(ir: dict) -> bool:
    """Return True if the IR has the minimum ground truth fields needed for evaluation."""
    gt = ir.get("ground_truth") or {}
    # MAD dataset only has failure_mode annotated (no culprit_agent or decisive_step)
    return bool(gt.get("failure_mode"))


# ── Per-trace Pipeline ─────────────────────────────────────────────────────────

def run_full_pipeline(ir: dict, dynamic: bool = True) -> tuple[dict, list[dict], list[dict]]:
    """
    Run the complete spec-based FL pipeline on a single trace IR.

    Args:
        ir:      Normalized trace IR.
        dynamic: If True, generate dynamic constraints per step. If False, global only.

    Returns:
        (prediction, validation_log, global_constraints)
        validation_log and global_constraints are saved alongside predictions for RQ4 analysis.
    """
    global_constraints = extract_global_specs(ir)

    if dynamic:
        if CONSTRAINT_GENERATION_MODE == "step_by_step":
            validation_log = build_validation_log(ir, global_constraints, verbose=False)
        else:
            validation_log = build_validation_log_one_shot(ir, global_constraints, verbose=False)
    else:
        # Global-only: evaluate global constraints step-by-step, no dynamic generation
        validation_log = build_validation_log_one_shot(ir, global_constraints, verbose=False)

    pred = run_judge(ir, validation_log, mode="auto")
    return pred, validation_log, global_constraints


def run_baseline_pipeline(ir: dict) -> dict:
    """Run baseline judge: no spec extraction, raw trajectory only."""
    return run_judge(ir, [], mode="baseline")


def run_checklist_pipeline(ir: dict) -> dict:
    """Run judge with MAST checklist but an empty violation log (ablation)."""
    return run_judge(ir, [], mode="auto")


# ── Constraint Log Builder ─────────────────────────────────────────────────────

def _build_constraint_log_entry(ir: dict, validation_log: list[dict], global_constraints: list[dict]) -> dict:
    """
    Build a per-trace constraint analysis entry for RQ4.

    Captures:
      - global_constraints: summary of all extracted global constraints (type, source, taxonomy_targets)
      - violations: all violations from the validation log with type/target/check_type
      - dynamic_constraints: subset of violations from dynamic sources for targeted analysis
    """
    def _constraint_summary(c: dict) -> dict:
        return {
            "assertion_name": c.get("assertion_name", ""),
            "constraint_type": c.get("constraint_type", "ANY"),
            "check_type": c.get("check_type", "nl_check"),
            "source": c.get("source", ""),
            "taxonomy_targets": c.get("taxonomy_targets", []),
            "agent_scope": c.get("agent_scope", "*"),
        }

    def _violation_summary(v: dict) -> dict:
        c = v.get("constraint", {})
        return {
            "step_index": v.get("step_index"),
            "agent_id": v.get("agent_id", ""),
            "assertion_name": v.get("assertion_name", ""),
            "constraint_type": v.get("constraint_type") or c.get("constraint_type", "ANY"),
            "check_type": c.get("check_type", "nl_check"),
            "source": c.get("source", ""),
            "taxonomy_targets": v.get("taxonomy_targets") or c.get("taxonomy_targets", []),
            "evidence": v.get("evidence", ""),
        }

    global_summaries = [_constraint_summary(c) for c in global_constraints]
    violation_summaries = [_violation_summary(v) for v in validation_log]
    dynamic_violations = [v for v in violation_summaries if v["source"] == "dynamic"]

    return {
        "framework": ir.get("framework", ""),
        "n_steps": len(ir.get("steps", [])),
        "global_constraints": global_summaries,
        "global_constraint_type_counts": _count_by(global_summaries, "constraint_type"),
        "violations": violation_summaries,
        "violation_type_counts": _count_by(violation_summaries, "constraint_type"),
        "violation_source_counts": _count_by(violation_summaries, "source"),
        "dynamic_violations": dynamic_violations,
        "dynamic_violation_type_counts": _count_by(dynamic_violations, "constraint_type"),
        "violated_taxonomy_targets": _flatten_targets(violation_summaries),
    }


def _count_by(items: list[dict], key: str) -> dict:
    counts: dict[str, int] = {}
    for item in items:
        val = item.get(key, "unknown")
        counts[val] = counts.get(val, 0) + 1
    return counts


def _flatten_targets(violations: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for v in violations:
        for t in v.get("taxonomy_targets", []):
            counts[t] = counts.get(t, 0) + 1
    return counts


# ── Evaluation Loop ────────────────────────────────────────────────────────────

def _write_checkpoint(output_path: str, mode: str, predictions, ground_truths, errors, skipped, complete: bool, constraint_logs=None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    checkpoint = {
        mode: {
            "mode": mode,
            "complete": complete,
            "predictions": predictions,
            "ground_truths": ground_truths,
            "errors": errors,
            "metrics": {"skipped": skipped, "errors": len(errors)},
        }
    }
    if constraint_logs is not None:
        checkpoint[mode]["constraint_logs"] = constraint_logs
    if complete:
        metrics = compute_metrics(predictions, ground_truths)
        metrics["skipped"] = skipped
        metrics["errors"] = len(errors)
        checkpoint[mode]["metrics"] = metrics
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False, default=str)


def evaluate(
    raw_records: list[dict],
    mode: str,
    max_traces: int | None,
    judge_mode: str = "auto",
    indices: list[int] | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """
    Run evaluation over the dataset.

    Args:
        raw_records:  Raw MAD records (pre-normalization).
        mode:         "full" | "baseline" | "global_only" | "checklist"
        max_traces:   Cap on number of traces to evaluate.
        judge_mode:   Passed to run_judge ("auto" | "all_at_once" | "step_then_category").
        indices:      If set, only evaluate traces at these dataset indices (0-based).
        output_path:  If set, writes a checkpoint after each trace so progress survives crashes.

    Returns:
        Dict with "metrics" and "predictions" keys.
    """
    predictions: list[dict] = []
    ground_truths: list[dict] = []
    errors: list[dict] = []
    constraint_logs: list[dict] = []  # per-trace constraint/violation data for RQ4
    skipped = 0
    processed = 0

    records = raw_records[:max_traces] if max_traces else raw_records
    if indices is not None:
        index_set = set(indices)
        records = [(i, r) for i, r in enumerate(records) if i in index_set]
    else:
        records = list(enumerate(records))

    for i, raw in tqdm(records, desc=f"[{mode}]", total=len(records)):
        # Normalize
        try:
            ir = normalize_trace(raw)
        except Exception as exc:
            errors.append({"index": i, "stage": "normalize", "error": str(exc)})
            skipped += 1
            continue

        if not _has_ground_truth(ir):
            skipped += 1
            continue

        gt = ir["ground_truth"]
        ground_truths.append(gt)

        # Run pipeline
        validation_log_entry = None
        try:
            if mode == "baseline":
                pred = run_baseline_pipeline(ir)
            elif mode == "global_only":
                pred, vlog, gconstraints = run_full_pipeline(ir, dynamic=False)
                validation_log_entry = _build_constraint_log_entry(ir, vlog, gconstraints)
            elif mode == "checklist":
                pred = run_checklist_pipeline(ir)
            else:  # "full"
                pred, vlog, gconstraints = run_full_pipeline(ir, dynamic=True)
                validation_log_entry = _build_constraint_log_entry(ir, vlog, gconstraints)
        except Exception as exc:
            tb = traceback.format_exc()
            errors.append({"index": i, "stage": "pipeline", "error": str(exc), "traceback": tb})
            # Emit a null prediction to keep ground_truths aligned
            pred = {
                "culprit_agent": None,
                "decisive_step": None,
                "failure_mode": "",
                "failure_family": "",
            }

        tid = ir.get("trajectory_id", f"trace_{i}")
        pred["_trajectory_id"] = tid
        predictions.append(pred)
        if validation_log_entry is not None:
            validation_log_entry["_trajectory_id"] = tid
            constraint_logs.append(validation_log_entry)
        processed += 1

        if output_path:
            _write_checkpoint(output_path, mode, predictions, ground_truths, errors, skipped, complete=False, constraint_logs=constraint_logs)

    metrics = compute_metrics(predictions, ground_truths)
    metrics["skipped"] = skipped
    metrics["errors"] = len(errors)

    return {
        "mode": mode,
        "metrics": metrics,
        "predictions": predictions,
        "ground_truths": ground_truths,
        "errors": errors,
        "constraint_logs": constraint_logs,
    }


# ── Multi-mode Runner ──────────────────────────────────────────────────────────

def run_all_modes(
    raw_records: list[dict],
    max_traces: int | None,
) -> dict[str, dict]:
    """Run full + baseline modes and return results keyed by mode name."""
    results = {}
    for mode in ("full", "baseline"):
        print(f"\n{'─'*50}")
        print(f"  Running mode: {mode}")
        print(f"{'─'*50}")
        result = evaluate(raw_records, mode=mode, max_traces=max_traces)
        results[mode] = result
        print_metrics(result["metrics"], title=f"Mode: {mode}")
    return results


# ── Result Persistence ─────────────────────────────────────────────────────────

def save_results(results: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    # Strip per_trace from metrics for cleaner top-level save; keep in predictions
    saveable = {}
    for key, val in results.items():
        if isinstance(val, dict) and "metrics" in val:
            m = {k: v for k, v in val["metrics"].items() if k != "per_trace"}
            saveable[key] = {**val, "metrics": m}
        else:
            saveable[key] = val
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(saveable, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {output_path}")
    if any(isinstance(v, dict) and v.get("constraint_logs") for v in results.values()):
        print("  └─ constraint_logs included (use for RQ4 analysis)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate specification-based fault localization on MAST MAD."
    )
    p.add_argument(
        "--mode",
        choices=["full", "baseline", "global_only", "checklist", "all"],
        default="full",
        help="Evaluation mode (default: full)",
    )
    p.add_argument(
        "--max_traces",
        type=int,
        default=None,
        help="Maximum number of traces to evaluate (default: all)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON (default: results/eval_<mode>_<timestamp>.json)",
    )
    p.add_argument(
        "--prefer_human",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer human-labelled MAD file over LLM-annotated (default: True). Use --no-prefer_human for full dataset.",
    )
    p.add_argument(
        "--judge_mode",
        choices=["auto", "all_at_once", "step_then_category"],
        default="auto",
        help="Judge invocation mode (default: auto)",
    )
    p.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Only evaluate specific dataset indices, e.g. --indices 10 11 12 13 15 17",
    )
    p.add_argument(
        "--framework",
        type=str,
        default=None,
        help="Filter to a specific MAS framework, e.g. --framework hyperagent",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    raw_records = _load_mad(prefer_human=args.prefer_human)

    if args.framework:
        fw = args.framework.lower()
        raw_records = [r for r in raw_records if r.get("mas_name", "").lower() == fw]
        print(f"Filtered to framework '{fw}': {len(raw_records)} records.")
    print(f"Loaded {len(raw_records)} records from MAD dataset.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = args.output or os.path.join(
        RESULTS_DIR, f"eval_{args.mode}_{timestamp}.json"
    )

    if args.mode == "all":
        results = run_all_modes(raw_records, max_traces=args.max_traces)
        # Print comparison table
        compare_methods({m: r["metrics"] for m, r in results.items()})
        save_results(results, output_path)
    else:
        result = evaluate(
            raw_records,
            mode=args.mode,
            max_traces=args.max_traces,
            judge_mode=args.judge_mode,
            indices=args.indices,
            output_path=output_path,
        )
        print_metrics(result["metrics"], title=f"Mode: {args.mode}")
        save_results({args.mode: result}, output_path)


if __name__ == "__main__":
    main()
