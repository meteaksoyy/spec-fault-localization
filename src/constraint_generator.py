"""
Stage 2b: Dynamic Constraint Generation (C^D_k).

At each step k, generates new constraints conditioned on the trajectory prefix
T≤k. These capture invariants established by what the agent has done/stated so
far — e.g., "the count stated at step 3 must match future tool outputs".

Follows AgentRx §3.1: C_k = C^G ∪ C^D_k
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    EXTRACTION_MODEL,
    LLM_TEMPERATURE,
    MAX_DYNAMIC_CONSTRAINTS_PER_STEP,
    PROMPTS_DIR,
    OPENAI_API_KEY,
)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _load_prompt() -> str:
    import os
    path = os.path.join(PROMPTS_DIR, "dynamic_spec", "constraint_generator.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def _parse_json_array(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        s, e = text.find("["), text.rfind("]")
        if s != -1 and e != -1:
            try:
                return json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                pass
    return []


def _serialize_step(step: dict) -> dict:
    """Return a compact view of a step for prompt inclusion."""
    return {
        "step_index": step["step_index"],
        "agent": step["agent_id"],
        "role": step["role"],
        "content": step["content"][:800],   # truncate long content
        "tool_calls": step["tool_calls"],
        "tool_results": [
            {**tr, "output": str(tr.get("output", ""))[:400]}
            for tr in step["tool_results"]
        ],
    }


def generate_dynamic_constraints(
    ir: dict,
    step_index: int,
    global_constraints: list[dict],
) -> list[dict]:
    """
    Generate dynamic constraints C^D_{step_index} for the given step.

    Args:
        ir:                 Normalized trace IR dict.
        step_index:         Index of the current step (0-based).
        global_constraints: Already-established global constraints C^G.

    Returns:
        List of new constraint dicts for this step (may be empty).
    """
    steps = ir.get("steps", [])
    prefix = [_serialize_step(s) for s in steps if s["step_index"] <= step_index]

    if not prefix:
        return []

    template = _load_prompt()
    global_names = [c.get("assertion_name", "") for c in global_constraints]

    prompt = (
        template
        .replace("{current_step_index}", str(step_index))
        .replace("{max_constraints}", str(MAX_DYNAMIC_CONSTRAINTS_PER_STEP))
        .replace("{task_instruction}", ir.get("task_instruction", ""))
        .replace("{trajectory_prefix_json}", json.dumps(prefix, ensure_ascii=False, indent=2))
        .replace("{global_constraint_names_json}", json.dumps(global_names, ensure_ascii=False))
    )

    client = _get_client()
    resp = client.chat.completions.create(
        model=EXTRACTION_MODEL,
        temperature=LLM_TEMPERATURE,
        messages=[
            {
                "role": "system",
                "content": "You are a precise constraint generation assistant. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    raw = resp.choices[0].message.content or ""
    constraints = _parse_json_array(raw)

    # Tag each dynamic constraint with its originating step
    for c in constraints:
        c.setdefault("originating_step", step_index)
        c.setdefault("source", "dynamic")

    return constraints
