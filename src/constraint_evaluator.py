"""
Stage 3: Constraint Evaluation.

For each step k and each constraint C in C_k = C^G ∪ C^D_k, evaluates whether
the constraint is violated. Returns a verdict + evidence pair.

Two check modes (following AgentRx §3.2):
  - python_check : runs dynamically generated Python code (deterministic)
  - nl_check     : calls GPT-4o with a structured rubric (semantic)

Follows the AgentRx guard/assertion pattern:
  EVAL_C(k) = (SKIP, ∅)                  if G_C(T≤k, s_k) = 0
             (Φ_C(T≤k, s_k), evidence)   otherwise
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any

from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EVAL_MODEL, LLM_TEMPERATURE, PROMPTS_DIR, OPENAI_API_KEY

_client: OpenAI | None = None

VERDICT_VIOL = "VIOL"
VERDICT_SAT  = "SAT"
VERDICT_SKIP = "SKIP"

# Context window: how many steps before/after current step to include in nl_check
CONTEXT_RADIUS = 3


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _load_nl_prompt() -> str:
    path = os.path.join(PROMPTS_DIR, "nl_check_judge.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def _parse_verdict_json(text: str) -> dict:
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
    return {"verdict": "UNCLEAR", "reasoning": "Failed to parse judge output", "evidence_quote": ""}


def _guard_applies(constraint: dict, step: dict, trajectory: list[dict]) -> bool:
    """
    Check whether the constraint's guard fires at this step.
    A guard fires when the event_trigger matches this step.
    If no event_trigger is specified, the constraint applies to every step.
    """
    trigger = constraint.get("event_trigger")
    if not trigger:
        return True

    # role_name match
    role_pattern = trigger.get("role_name", "*")
    if role_pattern != "*":
        agent_role = step.get("role", "")
        agent_id   = step.get("agent_id", "")
        if role_pattern.lower() not in agent_role.lower() and role_pattern.lower() not in agent_id.lower():
            return False

    # tool_name match: fires only when this step contains the specified tool call
    tool_pattern = trigger.get("tool_name", "*")
    if tool_pattern != "*":
        called_tools = [tc.get("name", "") for tc in step.get("tool_calls", [])]
        if not any(tool_pattern.lower() in t.lower() for t in called_tools):
            return False

    # agent_scope match
    scope = constraint.get("agent_scope", "*")
    if scope and scope != "*":
        if scope.lower() not in step.get("agent_id", "").lower() and scope.lower() not in step.get("role", "").lower():
            return False

    # step_index range match
    idx_pattern = trigger.get("step_index", "*")
    if idx_pattern != "*":
        if isinstance(idx_pattern, int):
            if step["step_index"] != idx_pattern:
                return False
        elif isinstance(idx_pattern, list) and len(idx_pattern) == 2:
            if not (idx_pattern[0] <= step["step_index"] <= idx_pattern[1]):
                return False

    return True


# ── Python check ──────────────────────────────────────────────────────────────

def _run_python_check(constraint: dict, trajectory: list[dict], step_index: int) -> tuple[str, str]:
    """
    Execute the python_check code block from the constraint.
    Returns (verdict, evidence_text).
    """
    pc = constraint.get("python_check", {})
    code_lines = pc.get("code_lines", [])
    if not code_lines:
        return VERDICT_SKIP, "No python_check code provided"

    code = "\n".join(code_lines)
    fn_name = pc.get("function_name", "check")

    local_ns: dict = {}
    try:
        exec(compile(code, "<constraint>", "exec"), local_ns)  # noqa: S102
    except Exception as exc:
        return VERDICT_SKIP, f"Compile error: {exc}"

    fn = local_ns.get(fn_name)
    if fn is None:
        return VERDICT_SKIP, f"Function '{fn_name}' not found after exec"

    try:
        result = fn(trajectory, step_index)
        if result is True:
            return VERDICT_SAT, "python_check returned True"
        elif result is False:
            return VERDICT_VIOL, "python_check returned False"
        else:
            return VERDICT_SKIP, f"python_check returned non-bool: {result}"
    except Exception as exc:
        return VERDICT_SKIP, f"Runtime error: {traceback.format_exc(limit=3)}"


# ── NL check ──────────────────────────────────────────────────────────────────

def _run_nl_check(constraint: dict, trajectory: list[dict], step_index: int) -> tuple[str, str]:
    """
    Ask the LLM to evaluate whether the constraint holds at the given step.
    Returns (verdict, evidence_quote).
    """
    step = next((s for s in trajectory if s["step_index"] == step_index), None)
    if step is None:
        return VERDICT_SKIP, "Step not found in trajectory"

    # Build context window
    context = [
        s for s in trajectory
        if abs(s["step_index"] - step_index) <= CONTEXT_RADIUS
    ]

    def _compact(s: dict) -> dict:
        return {
            "step_index": s["step_index"],
            "agent": s["agent_id"],
            "role": s["role"],
            "content": s["content"][:600],
            "tool_calls": s["tool_calls"],
            "tool_results": [
                {**tr, "output": str(tr.get("output", ""))[:300]}
                for tr in s.get("tool_results", [])
            ],
        }

    template = _load_nl_prompt()
    prompt = (
        template
        .replace("{assertion_name}", constraint.get("assertion_name", ""))
        .replace("{description}", constraint.get("description", ""))
        .replace("{check_hint}", constraint.get("check_hint", ""))
        .replace("{taxonomy_targets}", json.dumps(constraint.get("taxonomy_targets", [])))
        .replace("{current_step_index}", str(step_index))
        .replace("{current_step_json}", json.dumps(_compact(step), ensure_ascii=False, indent=2))
        .replace("{context_window_json}", json.dumps([_compact(s) for s in context], ensure_ascii=False, indent=2))
    )

    client = _get_client()
    resp = client.chat.completions.create(
        model=EVAL_MODEL,
        temperature=LLM_TEMPERATURE,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict compliance judge. "
                    "Evaluate constraint violations precisely. Return only valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    raw = resp.choices[0].message.content or ""
    parsed = _parse_verdict_json(raw)

    raw_verdict = parsed.get("verdict", "UNCLEAR")
    evidence = parsed.get("evidence_quote", "") or parsed.get("reasoning", "")

    if raw_verdict == "CLEAR_FAIL":
        return VERDICT_VIOL, evidence
    else:
        return VERDICT_SAT, evidence


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_constraint(
    constraint: dict,
    trajectory: list[dict],
    step_index: int,
) -> tuple[str, str]:
    """
    Evaluate a single constraint at the given step.

    Returns:
        (verdict, evidence) where verdict ∈ {VIOL, SAT, SKIP}
    """
    step = next((s for s in trajectory if s["step_index"] == step_index), None)
    if step is None:
        return VERDICT_SKIP, "Step not found"

    # Guard check
    if not _guard_applies(constraint, step, trajectory):
        return VERDICT_SKIP, "Guard did not fire"

    check_type = constraint.get("check_type", "nl_check")
    if check_type == "python_check":
        verdict, evidence = _run_python_check(constraint, trajectory, step_index)
        # If python check skips (code error), fall back to nl_check
        if verdict == VERDICT_SKIP:
            verdict, evidence = _run_nl_check(constraint, trajectory, step_index)
    else:
        verdict, evidence = _run_nl_check(constraint, trajectory, step_index)

    return verdict, evidence


def evaluate_all_at_step(
    constraints: list[dict],
    trajectory: list[dict],
    step_index: int,
) -> list[dict]:
    """
    Evaluate all constraints at the given step.

    Returns a list of result dicts for each constraint that FIRES (guard=1),
    regardless of verdict.
    """
    results = []
    for c in constraints:
        step = next((s for s in trajectory if s["step_index"] == step_index), None)
        if step is None:
            continue
        if not _guard_applies(c, step, trajectory):
            continue
        verdict, evidence = evaluate_constraint(c, trajectory, step_index)
        results.append({
            "assertion_name": c.get("assertion_name"),
            "step_index": step_index,
            "verdict": verdict,
            "evidence": evidence,
            "constraint": c,
        })
    return results
