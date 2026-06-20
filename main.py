"""
Specification-Based Fault Localization — Main Entry Point.

Orchestrates the full pipeline end-to-end:
  1. Load and normalize a MAST trace (from file or MAD dataset)
  2. Extract global specifications
  3. Build validation log (step-by-step constraint evaluation)
  4. Run LLM judge for fault attribution
  5. Print results

For batch evaluation with metrics, use: evaluation/run_evaluation.py

Usage examples:
  # Run on a single trace file
  python main.py --trace data/raw/ag2/trace_001.json

  # Run on the first N traces of the MAD dataset
  python main.py --dataset --max_traces 5

  # Run baseline (no spec extraction) for comparison
  python main.py --trace data/raw/ag2/trace_001.json --mode baseline

  # Compare full vs baseline on a single trace
  python main.py --trace data/raw/ag2/trace_001.json --mode compare
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    MAD_HUMAN_FILE,
    MAD_FULL_FILE,
    RAW_DATA_DIR,
    RESULTS_DIR,
    CONSTRAINT_GENERATION_MODE,
)
from src.data.normalizer import normalize_trace, normalize_file, normalize_mad_dataset
from src.spec_extractor import extract_global_specs
from src.validation_log import build_validation_log, build_validation_log_one_shot, format_log_for_judge
from src.judge import run_judge


# ── Display helpers ────────────────────────────────────────────────────────────

def _print_header(title: str) -> None:
    bar = "═" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def _print_attribution(result: dict, label: str = "Attribution") -> None:
    _print_header(label)
    print(f"  Culprit agent    : {result.get('culprit_agent', 'unknown')}")
    print(f"  Decisive step    : {result.get('decisive_step', 'N/A')}")
    print(f"  Failure family   : {result.get('failure_family', '')}")
    print(f"  Failure mode     : {result.get('failure_mode', '')}")
    step_reason = result.get("reason_for_step", "")
    if step_reason:
        print(f"\n  Step reasoning:")
        for line in step_reason.split(". "):
            if line.strip():
                print(f"    {line.strip()}.")
    cat_reason = result.get("reason_for_category", "")
    if cat_reason:
        print(f"\n  Category reasoning:")
        for line in cat_reason.split(". "):
            if line.strip():
                print(f"    {line.strip()}.")
    sup = result.get("supporting_violation_steps", [])
    if sup:
        print(f"\n  Supporting violation steps: {sup}")
    print()


def _print_ground_truth(gt: dict) -> None:
    if not gt:
        return
    _print_header("Ground Truth")
    print(f"  Culprit agent    : {gt.get('culprit_agent', 'N/A')}")
    print(f"  Decisive step    : {gt.get('decisive_step', 'N/A')}")
    print(f"  Failure family   : {gt.get('failure_family', 'N/A')}")
    print(f"  Failure mode     : {gt.get('failure_mode', 'N/A')}")
    print()


def _print_violations(validation_log: list[dict], max_show: int = 10) -> None:
    if not validation_log:
        print("\n  [No violations detected]\n")
        return
    _print_header(f"Validation Log ({len(validation_log)} violations)")
    for entry in validation_log[:max_show]:
        step = entry.get("step_index", "?")
        name = entry.get("assertion_name", "?")
        agent = entry.get("agent_id", "?")
        targets = ", ".join(entry.get("taxonomy_targets", []))
        evidence = str(entry.get("evidence", ""))[:120]
        print(f"  step {step:>3}  [{agent}]  {name}")
        print(f"           targets: {targets}")
        print(f"           evidence: {evidence}")
        print()
    if len(validation_log) > max_show:
        print(f"  ... and {len(validation_log) - max_show} more violations.\n")


# ── Single trace pipeline ──────────────────────────────────────────────────────

def run_single_trace(ir: dict, mode: str, verbose: bool) -> dict:
    """Run the pipeline on a single normalized IR and return attribution."""
    task = ir.get("task_instruction", "")[:120]
    n_steps = len(ir.get("steps", []))
    framework = ir.get("framework", "unknown")

    _print_header(f"Trace: {ir.get('trajectory_id', 'unknown')}")
    print(f"  Framework : {framework}")
    print(f"  Task      : {task}{'...' if len(ir.get('task_instruction','')) > 120 else ''}")
    print(f"  Steps     : {n_steps}")
    print()

    if mode == "baseline":
        result = run_judge(ir, [], mode="baseline")
        _print_attribution(result, label="Baseline Attribution (no spec)")
        return result

    # Full pipeline
    print("  [1/3] Extracting global specifications...")
    global_constraints = extract_global_specs(ir)
    print(f"        → {len(global_constraints)} global constraints extracted.")

    print("  [2/3] Building validation log...")
    if CONSTRAINT_GENERATION_MODE == "step_by_step":
        validation_log = build_validation_log(ir, global_constraints, verbose=False)
    else:
        validation_log = build_validation_log_one_shot(ir, global_constraints, verbose=False)
    print(f"        → {len(validation_log)} violations found.")

    if verbose:
        _print_violations(validation_log)

    print("  [3/3] Running LLM judge...")
    result = run_judge(ir, validation_log, mode="auto")

    _print_attribution(result, label="Full Pipeline Attribution")

    if mode == "compare":
        print("  Running baseline for comparison...")
        baseline = run_judge(ir, [], mode="baseline")
        _print_attribution(baseline, label="Baseline Attribution (no spec)")

    _print_ground_truth(ir.get("ground_truth", {}))

    return result


# ── Dataset mode ───────────────────────────────────────────────────────────────

def _load_mad_records(prefer_human: bool = True) -> list[dict]:
    candidates = (
        [MAD_HUMAN_FILE, MAD_FULL_FILE] if prefer_human else [MAD_FULL_FILE, MAD_HUMAN_FILE]
    )
    for fname in candidates:
        path = os.path.join(RAW_DATA_DIR, fname)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            for key in ("data", "records", "traces"):
                if key in data and isinstance(data[key], list):
                    return data[key]
    raise FileNotFoundError(
        f"MAD dataset not found in {RAW_DATA_DIR}. "
        "Run: python data/download_mast.py --source huggingface"
    )


def run_dataset_mode(args: argparse.Namespace) -> None:
    """Run on a sample of the MAD dataset (for quick inspection, not full eval)."""
    records = _load_mad_records()
    print(f"Dataset loaded: {len(records)} records. Showing first {args.max_traces}.")

    for i, raw in enumerate(records[: args.max_traces]):
        try:
            ir = normalize_trace(raw)
        except Exception as exc:
            print(f"[trace {i}] Normalization failed: {exc}")
            continue

        try:
            run_single_trace(ir, mode=args.mode, verbose=args.verbose)
        except Exception as exc:
            print(f"[trace {i}] Pipeline failed: {exc}")
            continue

        if i < args.max_traces - 1:
            input("\n  Press ENTER to continue to next trace...")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Specification-Based Fault Localization for LLM Multi-Agent Systems"
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--trace",
        metavar="PATH",
        help="Path to a single raw trace JSON file.",
    )
    source.add_argument(
        "--dataset",
        action="store_true",
        help="Run on traces from the MAD dataset (use with --max_traces).",
    )

    p.add_argument(
        "--mode",
        choices=["full", "baseline", "compare"],
        default="full",
        help="Pipeline mode (default: full)",
    )
    p.add_argument(
        "--max_traces",
        type=int,
        default=3,
        help="Max traces to process in dataset mode (default: 3)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print full violation log.",
    )
    p.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Save attribution result to JSON file.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset:
        run_dataset_mode(args)
        return

    # Single trace mode
    trace_path = args.trace
    if not os.path.exists(trace_path):
        print(f"ERROR: Trace file not found: {trace_path}", file=sys.stderr)
        sys.exit(1)

    traces = normalize_file(trace_path)
    if not traces:
        print(f"ERROR: No valid traces found in {trace_path}", file=sys.stderr)
        sys.exit(1)

    ir = traces[0]
    result = run_single_trace(ir, mode=args.mode, verbose=args.verbose)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Result saved to: {args.output}")


if __name__ == "__main__":
    main()
