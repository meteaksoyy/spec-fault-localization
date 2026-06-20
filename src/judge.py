"""
Stage 5: LLM Judge for Fault Attribution.

Given the normalized trajectory and the validation log V, the judge outputs:
  - culprit_agent  : the agent responsible for the critical failure
  - decisive_step  : the step index of the first unrecoverable failure
  - failure_family : FC1 | FC2 | FC3
  - failure_mode   : one of the 14 MAST failure modes

Implements two modes (AgentRx §3.4 / §4.6):
  - "all_at_once"       (default): trajectory + full V log in one call
  - "step_then_category": first find step, then classify (can be brittle on long traces)

Also provides a BASELINE judge (no spec extraction, no violation log) for comparison.

Following AgentRx's asymmetric architecture: strong JUDGE_MODEL for final attribution.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    JUDGE_MODEL,
    LLM_TEMPERATURE,
    LONG_TRACE_THRESHOLD,
    MAST_FAILURE_MODES,
    PROMPTS_DIR,
    OPENAI_API_KEY,
)
from src.validation_log import format_log_for_judge

_client: OpenAI | None = None

# Max characters for trajectory serialization in judge prompt
MAX_TRAJ_CHARS = 24_000
# Max violations to include in judge prompt (avoid context overflow)
MAX_VIOLATIONS_IN_PROMPT = 30


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _load_prompt(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, "judge", filename)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _parse_output(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1:
            try:
                return json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                pass
    return {}


def _serialize_trajectory(steps: list[dict], max_chars: int = MAX_TRAJ_CHARS) -> str:
    """Serialize steps to a compact JSON string, truncating if too long."""
    compact = []
    for s in steps:
        compact.append({
            "step": s["step_index"],
            "agent": s["agent_id"],
            "role": s["role"],
            "content": s["content"][:800],
            "tool_calls": s["tool_calls"],
            "tool_results": [
                {**tr, "output": str(tr.get("output", ""))[:400]}
                for tr in s.get("tool_results", [])
            ],
        })
    serialized = json.dumps(compact, ensure_ascii=False, indent=2)
    if len(serialized) > max_chars:
        # Truncate from the middle to preserve start and end
        half = max_chars // 2
        serialized = serialized[:half] + "\n... [truncated for length] ...\n" + serialized[-half:]
    return serialized


def _normalize_output(raw: dict, steps: list[dict]) -> dict:
    """Validate and normalize the judge's output dict."""
    step_indices = {s["step_index"] for s in steps}

    decisive = raw.get("decisive_step")
    if decisive is not None:
        try:
            decisive = int(decisive)
        except (TypeError, ValueError):
            decisive = None

    mode = raw.get("failure_mode", "")
    # Always derive family from mode to prevent inconsistent mode/family pairs
    family = MAST_FAILURE_MODES.get(mode, raw.get("failure_family", ""))

    return {
        "culprit_agent": raw.get("culprit_agent", "unknown"),
        "decisive_step": decisive,
        "failure_family": family,
        "failure_mode": mode,
        "reason_for_step": raw.get("reason_for_step", ""),
        "reason_for_category": raw.get("reason_for_category", ""),
        "supporting_violation_steps": raw.get("supporting_violation_steps", []),
    }


# ── Main judge call ────────────────────────────────────────────────────────────

def judge_all_at_once(
    ir: dict,
    validation_log: list[dict],
) -> dict:
    """
    All-at-once judging: provide the full trajectory + violation log in a single
    LLM call. Best for traces ≤ LONG_TRACE_THRESHOLD steps.

    Returns normalized attribution dict.
    """
    steps = ir.get("steps", [])
    task = ir.get("task_instruction", "")

    traj_json = _serialize_trajectory(steps)

    # Include only the top MAX_VIOLATIONS violations in the prompt
    top_violations = validation_log[:MAX_VIOLATIONS_IN_PROMPT]
    violation_text = format_log_for_judge(top_violations)

    template = _load_prompt("fault_attribution_judge.txt")
    prompt = (
        template
        .replace("{task_instruction}", task)
        .replace("{trajectory_json}", traj_json)
        .replace("{violation_log_text}", violation_text)
    )

    client = _get_client()
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=LLM_TEMPERATURE,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise failure attribution expert for multi-agent systems. "
                    "Return only valid JSON with the exact fields requested."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    raw_text = resp.choices[0].message.content or ""
    raw = _parse_output(raw_text)
    return _normalize_output(raw, steps)


def judge_step_then_category(
    ir: dict,
    validation_log: list[dict],
) -> dict:
    """
    Two-call judging: first find the decisive step and culprit, then classify.
    More reliable when violation signals are noisy (AgentRx §4.6).
    """
    steps = ir.get("steps", [])
    task = ir.get("task_instruction", "")
    traj_json = _serialize_trajectory(steps)
    top_violations = validation_log[:MAX_VIOLATIONS_IN_PROMPT]
    violation_text = format_log_for_judge(top_violations)

    # ── Step 1: Find decisive step and culprit ─────────────────────────────
    step_prompt = f"""You are a failure attribution expert.

Given the trajectory and violation log below, identify:
1. The DECISIVE STEP: the earliest step index from which the task became unrecoverable
2. The CULPRIT AGENT: the agent active at that step

Use violations as primary evidence. The decisive step should be the earliest violation
that is sufficient to explain the terminal failure.

TASK: {task}

TRAJECTORY:
{traj_json}

VIOLATION LOG:
{violation_text}

Return JSON: {{"decisive_step": <int>, "culprit_agent": "<string>", "reason": "<1-2 sentences>"}}
"""

    client = _get_client()
    resp1 = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": step_prompt},
        ],
    )
    step_raw = _parse_output(resp1.choices[0].message.content or "")
    decisive_step = step_raw.get("decisive_step")
    culprit_agent = step_raw.get("culprit_agent", "unknown")
    step_reason = step_raw.get("reason", "")

    # ── Step 2: Classify failure mode ──────────────────────────────────────
    # Provide the context around the decisive step
    window = [s for s in steps if abs(s["step_index"] - (decisive_step or 0)) <= 5]

    cat_prompt = f"""You are a failure categorization expert for multi-agent systems.

The decisive failure occurred at step {decisive_step}, agent: {culprit_agent}.
Reason: {step_reason}

Context around the failure step:
{json.dumps([{"step": s["step_index"], "agent": s["agent_id"], "content": s["content"][:600]} for s in window], indent=2)}

Violation evidence at/near this step:
{format_log_for_judge([v for v in top_violations if abs(v['step_index'] - (decisive_step or 0)) <= 3])}

Classify this failure using the MAST taxonomy:
FC1: DisobeyTaskSpec, DisobeyRoleSpec
FC2: IgnoredOtherAgentInput, ReasoningActionMismatch, PrematureTermination, UnawareOfStoppingCond, IncompleteVerification, IncorrectVerification
FC3: NoVerification, Hallucination, FaultyReasoning, ContextLoss, FailTaskSpec, InsufficientInfo

Return JSON: {{"failure_family": "FC1|FC2|FC3", "failure_mode": "<mode name>", "reason": "<1-2 sentences>"}}
"""

    resp2 = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": cat_prompt},
        ],
    )
    cat_raw = _parse_output(resp2.choices[0].message.content or "")

    return _normalize_output({
        "culprit_agent": culprit_agent,
        "decisive_step": decisive_step,
        "failure_family": cat_raw.get("failure_family", ""),
        "failure_mode": cat_raw.get("failure_mode", ""),
        "reason_for_step": step_reason,
        "reason_for_category": cat_raw.get("reason", ""),
        "supporting_violation_steps": [v["step_index"] for v in top_violations],
    }, steps)


def judge_baseline(ir: dict) -> dict:
    """
    Baseline LLM-as-a-judge: no spec extraction, no violation log.
    Used as comparison baseline (AgentRx / ErrorProbe baseline).
    """
    steps = ir.get("steps", [])
    task = ir.get("task_instruction", "")
    traj_json = _serialize_trajectory(steps)

    template = _load_prompt("baseline_judge.txt")
    prompt = (
        template
        .replace("{task_instruction}", task)
        .replace("{trajectory_json}", traj_json)
    )

    client = _get_client()
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    raw = _parse_output(resp.choices[0].message.content or "")
    return _normalize_output(raw, steps)


# ── Dispatcher ─────────────────────────────────────────────────────────────────

def run_judge(
    ir: dict,
    validation_log: list[dict],
    mode: str = "auto",
) -> dict:
    """
    Run fault attribution.

    Args:
        ir:             Normalized trace IR.
        validation_log: Output of build_validation_log().
        mode:           "auto" | "all_at_once" | "step_then_category" | "baseline"

    Returns:
        Attribution dict with culprit_agent, decisive_step, failure_mode, etc.
    """
    n_steps = len(ir.get("steps", []))

    if mode == "baseline":
        return judge_baseline(ir)
    elif mode == "step_then_category":
        return judge_step_then_category(ir, validation_log)
    elif mode == "all_at_once":
        return judge_all_at_once(ir, validation_log)
    else:
        # Auto: use all-at-once for short traces, step-then-category for long ones
        if n_steps <= LONG_TRACE_THRESHOLD:
            return judge_all_at_once(ir, validation_log)
        else:
            return judge_step_then_category(ir, validation_log)
