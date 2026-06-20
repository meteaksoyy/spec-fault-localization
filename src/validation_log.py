"""
Stage 4: Validation Log Construction.

Runs the full step-by-step constraint evaluation loop over a trajectory and
builds the step-indexed validation log V:

  V := {(k, C, e) | C ∈ C_k, G_C(T≤k, s_k) = 1, (VIOL, e) = EVAL(C, T≤k, s_k)}

Following AgentRx §3.3:
  - At each step k, evaluates C_k = C^G ∪ C^D_k
  - Records every violated constraint with its evidence
  - The resulting log is the primary input to the LLM judge
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CONSTRAINT_GENERATION_MODE
from src.constraint_evaluator import evaluate_all_at_step, VERDICT_VIOL
from src.constraint_generator import generate_dynamic_constraints


def build_validation_log(
    ir: dict,
    global_constraints: list[dict],
    verbose: bool = False,
) -> list[dict]:
    """
    Run step-by-step constraint evaluation and return the validation log.

    Args:
        ir:                 Normalized trace IR.
        global_constraints: C^G extracted by spec_extractor.
        verbose:            Print progress to stdout.

    Returns:
        List of violation entries. Each entry:
        {
            "step_index": int,
            "assertion_name": str,
            "verdict": "VIOL",
            "evidence": str,
            "constraint_type": str,
            "taxonomy_targets": [...],
            "agent_id": str,
            "constraint": {...}   # full constraint dict
        }
    """
    steps = ir.get("steps", [])
    trajectory = steps  # alias for clarity

    validation_log: list[dict] = []
    # Accumulate dynamic constraints over steps
    dynamic_constraints: list[dict] = []
    dynamic_names: set[str] = set()

    for step in steps:
        k = step["step_index"]

        # Generate new dynamic constraints for this step prefix (step-by-step mode)
        if CONSTRAINT_GENERATION_MODE == "step_by_step":
            try:
                new_dynamic = generate_dynamic_constraints(ir, k, global_constraints)
                for c in new_dynamic:
                    name = c.get("assertion_name", "")
                    if name and name not in dynamic_names:
                        dynamic_constraints.append(c)
                        dynamic_names.add(name)
            except Exception as exc:
                if verbose:
                    print(f"  [warn] Dynamic constraint gen failed at step {k}: {exc}")

        # C_k = C^G ∪ C^D (accumulated so far)
        c_k = global_constraints + dynamic_constraints

        if verbose:
            print(f"  Step {k:3d} | constraints: {len(c_k):3d}", end="")

        # Evaluate all constraints at step k
        results = evaluate_all_at_step(c_k, trajectory, k)

        violations = [r for r in results if r["verdict"] == VERDICT_VIOL]

        if verbose:
            print(f" | violations: {len(violations)}")

        for v in violations:
            c = v["constraint"]
            agent = step.get("agent_id", "unknown")
            entry = {
                "step_index": k,
                "assertion_name": v["assertion_name"],
                "verdict": VERDICT_VIOL,
                "evidence": v["evidence"],
                "constraint_type": c.get("constraint_type", "ANY"),
                "taxonomy_targets": c.get("taxonomy_targets", []),
                "agent_id": agent,
                "constraint": c,
            }
            validation_log.append(entry)

    # Deduplicate: keep only the first occurrence of each constraint name.
    # Dynamic constraints tend to re-fire on every subsequent step once the
    # underlying condition persists, creating hundreds of redundant entries.
    # The first firing is the most informative (earliest signal).
    seen: set[str] = set()
    deduped: list[dict] = []
    for entry in validation_log:
        name = entry["assertion_name"]
        if name not in seen:
            seen.add(name)
            deduped.append(entry)
    return deduped


def build_validation_log_one_shot(
    ir: dict,
    global_constraints: list[dict],
    verbose: bool = False,
) -> list[dict]:
    """
    One-shot variant: generate all dynamic constraints from the full trajectory
    at once, then evaluate all constraints on every step.

    More cost-efficient but less reliable for long traces (AgentRx §4.5).
    """
    steps = ir.get("steps", [])
    trajectory = steps

    # Generate dynamic constraints from the full trajectory in one call
    dynamic_constraints: list[dict] = []
    if steps:
        last_step = steps[-1]["step_index"]
        try:
            dynamic_constraints = generate_dynamic_constraints(ir, last_step, global_constraints)
        except Exception as exc:
            if verbose:
                print(f"  [warn] One-shot dynamic constraint gen failed: {exc}")

    c_k = global_constraints + dynamic_constraints
    validation_log: list[dict] = []

    seen: set[str] = set()
    for step in steps:
        k = step["step_index"]
        results = evaluate_all_at_step(c_k, trajectory, k)
        for v in results:
            if v["verdict"] == VERDICT_VIOL:
                name = v["assertion_name"]
                if name in seen:
                    continue
                seen.add(name)
                c = v["constraint"]
                validation_log.append({
                    "step_index": k,
                    "assertion_name": name,
                    "verdict": VERDICT_VIOL,
                    "evidence": v["evidence"],
                    "constraint_type": c.get("constraint_type", "ANY"),
                    "taxonomy_targets": c.get("taxonomy_targets", []),
                    "agent_id": step.get("agent_id", "unknown"),
                    "constraint": c,
                })

    return validation_log


def format_log_for_judge(validation_log: list[dict]) -> str:
    """
    Render the validation log into a human-readable text block for the judge prompt.
    Numbered, step-indexed, with evidence. Mirrors AgentRx's 'auditable violation log'.
    """
    if not validation_log:
        return "No constraint violations detected."

    lines = []
    for i, entry in enumerate(validation_log, 1):
        lines.append(
            f"[Violation #{i}]"
            f"\n  Step:       {entry['step_index']}"
            f"\n  Agent:      {entry['agent_id']}"
            f"\n  Constraint: {entry['assertion_name']}"
            f"\n  Type:       {entry['constraint_type']}"
            f"\n  Targets:    {', '.join(entry['taxonomy_targets'])}"
            f"\n  Evidence:   {entry['evidence']}"
        )
    return "\n\n".join(lines)
