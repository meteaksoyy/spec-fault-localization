"""
Evaluation metrics for specification-based fault localization.

Following AgentRx §4.1, AgenTracer §5.1, ErrorProbe §5.1 metric definitions:

  - Agent-level accuracy  : â = a*  (exact culprit agent match)
  - Step-level accuracy   : ŝ = t*  (exact decisive step match)
  - Step accuracy @±r     : |ŝ - t*| ≤ r  for r ∈ {1, 3, 5}
  - Average step distance : E[|ŝ - t*|]  (lower is better)
  - Failure mode accuracy : ŷ = y*  (14-class MAST)
  - Failure family acc.   : ŷ_family = y*_family  (FC1/FC2/FC3)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any


def _agent_match(pred: str | None, gold: str | None) -> bool:
    """Case-insensitive partial match on agent names (handles role vs ID differences)."""
    if pred is None or gold is None:
        return False
    return gold.lower() in pred.lower() or pred.lower() in gold.lower()


def compute_metrics(
    predictions: list[dict],
    ground_truths: list[dict],
    tolerances: tuple[int, ...] = (1, 3, 5),
) -> dict:
    """
    Compute all fault localization metrics.

    Args:
        predictions:   List of judge output dicts (one per trace).
        ground_truths: List of ground truth dicts (one per trace), aligned by index.
                       Each must have: culprit_agent, decisive_step, failure_mode,
                       failure_family.
        tolerances:    Step-accuracy tolerance values r (for @±r metrics).

    Returns:
        Dict of metric name → value (floats in [0, 1] or raw numbers).
    """
    assert len(predictions) == len(ground_truths), (
        f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths"
    )

    n = len(predictions)
    if n == 0:
        return {}

    agent_correct = 0
    step_correct  = 0
    step_within   = defaultdict(int)
    step_dists    = []
    mode_correct        = 0
    mode_correct_lenient = 0
    family_correct        = 0
    family_correct_lenient = 0

    per_trace = []

    for pred, gt in zip(predictions, ground_truths):
        p_agent  = pred.get("culprit_agent")
        p_step   = pred.get("decisive_step")
        p_mode   = pred.get("failure_mode", "")
        p_family = pred.get("failure_family", "")

        g_agent  = gt.get("culprit_agent")
        g_step   = gt.get("decisive_step")
        g_mode   = gt.get("failure_mode", "")
        g_family = gt.get("failure_family", "")

        # Build sets of all annotated modes/families for lenient matching
        all_modes   = {e["mode"]   for e in gt.get("all_failure_modes", []) if e.get("mode")}
        all_families = {e["family"] for e in gt.get("all_failure_modes", []) if e.get("family")}
        if g_mode:
            all_modes.add(g_mode)
        if g_family:
            all_families.add(g_family)

        # Agent accuracy (skip when ground truth has no culprit_agent annotation)
        a_ok = _agent_match(p_agent, g_agent) if g_agent is not None else None
        if a_ok is not None:
            agent_correct += int(a_ok)

        # Step accuracy
        if p_step is not None and g_step is not None:
            dist = abs(int(p_step) - int(g_step))
            step_dists.append(dist)
            s_ok = dist == 0
            step_correct += int(s_ok)
            for r in tolerances:
                step_within[r] += int(dist <= r)
        else:
            step_dists.append(None)
            s_ok = False

        # Mode accuracy — strict (primary label only) and lenient (any annotated label)
        p_mode_norm = p_mode.strip().lower()
        m_ok         = bool(p_mode_norm and g_mode and p_mode_norm == g_mode.strip().lower())
        m_ok_lenient = bool(p_mode_norm and any(p_mode_norm == m.strip().lower() for m in all_modes))
        mode_correct         += int(m_ok)
        mode_correct_lenient += int(m_ok_lenient)

        # Family accuracy — strict and lenient
        p_fam_norm = p_family.strip().upper()
        f_ok         = bool(p_fam_norm and g_family and p_fam_norm == g_family.strip().upper())
        f_ok_lenient = bool(p_fam_norm and any(p_fam_norm == fam.strip().upper() for fam in all_families))
        family_correct         += int(f_ok)
        family_correct_lenient += int(f_ok_lenient)

        per_trace.append({
            "trajectory_id": pred.get("_trajectory_id"),
            "agent_correct": a_ok,  # None means ground truth unavailable
            "step_correct": s_ok,
            "step_dist": dist if p_step is not None and g_step is not None else None,
            "mode_correct": m_ok,
            "mode_correct_lenient": m_ok_lenient,
            "family_correct": f_ok,
            "family_correct_lenient": f_ok_lenient,
            "pred_agent": p_agent,
            "gold_agent": g_agent,
            "pred_step": p_step,
            "gold_step": g_step,
            "pred_mode": p_mode,
            "gold_mode": g_mode,
            "all_gold_modes": sorted(all_modes),
        })

    # Aggregate
    valid_dists = [d for d in step_dists if d is not None]
    avg_step_dist = sum(valid_dists) / len(valid_dists) if valid_dists else float("nan")

    # Count traces where agent/step ground truth was actually available
    n_with_agent_gt = sum(1 for gt in ground_truths if gt.get("culprit_agent") is not None)
    n_with_step_gt  = len(valid_dists)

    metrics: dict[str, Any] = {
        "n_traces": n,
        "agent_accuracy": agent_correct / n_with_agent_gt if n_with_agent_gt else float("nan"),
        "step_accuracy":  step_correct / n_with_step_gt  if n_with_step_gt  else float("nan"),
        "avg_step_distance": avg_step_dist,
        "failure_mode_accuracy":         mode_correct / n,
        "failure_mode_accuracy_lenient": mode_correct_lenient / n,
        "failure_family_accuracy":         family_correct / n,
        "failure_family_accuracy_lenient": family_correct_lenient / n,
    }
    for r in tolerances:
        denom = len(valid_dists) if valid_dists else n
        metrics[f"step_accuracy_at_{r}"] = step_within[r] / denom if denom else 0.0

    metrics["per_trace"] = per_trace
    return metrics


def print_metrics(metrics: dict, title: str = "Results") -> None:
    """Pretty-print the metrics dict."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    print(f"  Traces evaluated   : {metrics.get('n_traces', 0)}")
    print(f"  Agent accuracy     : {metrics.get('agent_accuracy', 0):.1%}")
    print(f"  Step accuracy      : {metrics.get('step_accuracy', 0):.1%}")
    for key, val in sorted(metrics.items()):
        if key.startswith("step_accuracy_at_"):
            r = key.split("_")[-1]
            print(f"  Step acc @±{r}       : {val:.1%}")
    dist = metrics.get("avg_step_distance", float("nan"))
    print(f"  Avg step distance  : {dist:.2f}" if not math.isnan(dist) else "  Avg step distance  : N/A")
    print(f"  Failure mode acc.  : {metrics.get('failure_mode_accuracy', 0):.1%}  (lenient: {metrics.get('failure_mode_accuracy_lenient', 0):.1%})")
    print(f"  Failure family acc.: {metrics.get('failure_family_accuracy', 0):.1%}  (lenient: {metrics.get('failure_family_accuracy_lenient', 0):.1%})")
    print(f"{'=' * 50}\n")


def compare_methods(results: dict[str, dict]) -> None:
    """Print a comparison table for multiple method results."""
    methods = list(results.keys())
    metrics_to_show = [
        "agent_accuracy", "step_accuracy",
        "step_accuracy_at_1", "step_accuracy_at_3",
        "avg_step_distance",
        "failure_mode_accuracy", "failure_mode_accuracy_lenient",
        "failure_family_accuracy", "failure_family_accuracy_lenient",
    ]
    labels = {
        "agent_accuracy": "Agent Acc.",
        "step_accuracy": "Step Acc.",
        "step_accuracy_at_1": "Step @±1",
        "step_accuracy_at_3": "Step @±3",
        "avg_step_distance": "Avg Dist↓",
        "failure_mode_accuracy": "Mode Acc. (strict)",
        "failure_mode_accuracy_lenient": "Mode Acc. (lenient)",
        "failure_family_accuracy": "Family Acc. (strict)",
        "failure_family_accuracy_lenient": "Family Acc. (lenient)",
    }

    col_w = 12
    header = f"{'Metric':<22}" + "".join(f"{m:>{col_w}}" for m in methods)
    print("\n" + "=" * (22 + col_w * len(methods)))
    print(header)
    print("-" * (22 + col_w * len(methods)))

    for key in metrics_to_show:
        label = labels.get(key, key)
        row = f"{label:<22}"
        for m in methods:
            val = results[m].get(key, float("nan"))
            if key == "avg_step_distance":
                row += f"{val:>{col_w}.2f}" if not math.isnan(val) else f"{'N/A':>{col_w}}"
            else:
                row += f"{val:>{col_w}.1%}" if not math.isnan(val) else f"{'N/A':>{col_w}}"
        print(row)

    print("=" * (22 + col_w * len(methods)) + "\n")
