"""
Stage 2: Global Specification Extraction.

Extracts behavioral specifications from the static trace context (agent roles,
system prompts, tool schemas, task instruction) using GPT-4o.

Returns a list of Constraint dicts conforming to the AgentRx constraint schema.
These form C^G — the global constraints applied to every step.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EXTRACTION_MODEL, LLM_TEMPERATURE, PROMPTS_DIR, OPENAI_API_KEY

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _load_prompt(relative_path: str) -> str:
    full = os.path.join(PROMPTS_DIR, relative_path)
    with open(full, encoding="utf-8") as f:
        return f.read()


def _call_llm(system_prompt: str, user_content: str, model: str = EXTRACTION_MODEL) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return resp.choices[0].message.content or ""


def _parse_json_array(text: str) -> list[dict]:
    """Robustly extract a JSON array from LLM output."""
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        # Try to find array boundaries
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return []


# ── Extraction functions ──────────────────────────────────────────────────────

def _extract_role_constraints(agents: list[dict], task_instruction: str) -> list[dict]:
    template = _load_prompt("global_spec/role_constraints.txt")
    prompt = template.replace("{agents_json}", json.dumps(agents, ensure_ascii=False, indent=2))
    prompt = prompt.replace("{task_instruction}", task_instruction)

    # Split template into system / user sections at the AGENTS: line
    # The template IS the user prompt; use a fixed system message
    system = "You are a precise specification extraction assistant. Return only valid JSON."
    raw = _call_llm(system, prompt)
    constraints = _parse_json_array(raw)
    for c in constraints:
        c.setdefault("source", "role")
    return constraints


def _extract_tool_constraints(tool_schema: list[dict]) -> list[dict]:
    if not tool_schema:
        return []
    template = _load_prompt("global_spec/tool_constraints.txt")
    prompt = template.replace("{tool_schema_json}", json.dumps(tool_schema, ensure_ascii=False, indent=2))
    system = "You are a precise specification extraction assistant. Return only valid JSON."
    raw = _call_llm(system, prompt)
    constraints = _parse_json_array(raw)
    for c in constraints:
        c.setdefault("source", "tool_schema")
    return constraints


def _extract_protocol_constraints(agents: list[dict], task_instruction: str) -> list[dict]:
    template = _load_prompt("global_spec/protocol_constraints.txt")
    prompt = template.replace("{agents_json}", json.dumps(agents, ensure_ascii=False, indent=2))
    prompt = prompt.replace("{task_instruction}", task_instruction)
    system = "You are a precise specification extraction assistant. Return only valid JSON."
    raw = _call_llm(system, prompt)
    constraints = _parse_json_array(raw)
    for c in constraints:
        c.setdefault("source", "protocol")
    return constraints


def _extract_behavioral_constraints(agents: list[dict], task_instruction: str) -> list[dict]:
    template = _load_prompt("global_spec/behavioral_constraints.txt")
    prompt = template.replace("{agents_json}", json.dumps(agents, ensure_ascii=False, indent=2))
    prompt = prompt.replace("{task_instruction}", task_instruction)
    system = "You are a precise specification extraction assistant. Return only valid JSON."
    raw = _call_llm(system, prompt)
    constraints = _parse_json_array(raw)
    for c in constraints:
        c.setdefault("source", "behavioral")
    return constraints


def _infer_role_template_constraints(agents: list[dict]) -> list[dict]:
    """
    Fallback: when no system prompts are available, inject generic role-based
    constraint templates derived from common MAS roles.
    """
    ROLE_TEMPLATES: dict[str, list[dict]] = {
        "coder": [
            {
                "assertion_name": "coder_produces_code",
                "description": "Coder agent must produce executable code, not just descriptions",
                "agent_scope": "{agent_id}",
                "taxonomy_targets": ["DisobeyRoleSpec"],
                "constraint_type": "CAPABILITY",
                "check_hint": "Check that the Coder's output contains actual code (functions, classes, scripts), not just natural language instructions about what code to write.",
                "check_type": "nl_check",
                "source": "role_template",
            }
        ],
        "tester": [
            {
                "assertion_name": "tester_verifies_before_approval",
                "description": "Tester must run tests and report results before approving",
                "agent_scope": "{agent_id}",
                "taxonomy_targets": ["IncompleteVerification", "DisobeyRoleSpec"],
                "constraint_type": "CAPABILITY",
                "check_hint": "Check that before the Tester approves a solution, it actually executes or reviews test cases and reports concrete pass/fail results, not just assumed success.",
                "check_type": "nl_check",
                "source": "role_template",
            }
        ],
        "planner": [
            {
                "assertion_name": "planner_produces_structured_plan",
                "description": "Planner must produce a structured, step-by-step plan before delegation",
                "agent_scope": "{agent_id}",
                "taxonomy_targets": ["DisobeyRoleSpec", "IntentPlanMisalignment"],
                "constraint_type": "PROTOCOL",
                "check_hint": "Check that the Planner produces a concrete, enumerated plan of sub-tasks before handing off to other agents. Vague instructions or single-step delegation are violations.",
                "check_type": "nl_check",
                "source": "role_template",
            }
        ],
        "orchestrator": [
            {
                "assertion_name": "orchestrator_delegates_not_executes",
                "description": "Orchestrator must delegate tasks to sub-agents, not execute them directly",
                "agent_scope": "{agent_id}",
                "taxonomy_targets": ["DisobeyRoleSpec"],
                "constraint_type": "PROTOCOL",
                "check_hint": "Check that the Orchestrator issues delegation messages and does not itself perform domain actions (code writing, web browsing, file manipulation). If it directly executes a domain action, that is a violation.",
                "check_type": "nl_check",
                "source": "role_template",
            }
        ],
        "websurfer": [
            {
                "assertion_name": "websurfer_uses_browser_tools",
                "description": "WebSurfer must use browser navigation tools, not fabricate web content",
                "agent_scope": "{agent_id}",
                "taxonomy_targets": ["DisobeyRoleSpec", "Hallucination"],
                "constraint_type": "CAPABILITY",
                "check_hint": "Check that the WebSurfer agent invokes actual navigation/search tool calls (e.g., browse, click, search) to retrieve information. If it states web content without any tool call, that is a potential hallucination violation.",
                "check_type": "nl_check",
                "source": "role_template",
            }
        ],
    }

    constraints = []
    for agent in agents:
        role_lower = agent.get("role", "").lower()
        agent_id = agent.get("id", "*")
        for role_key, templates in ROLE_TEMPLATES.items():
            if role_key in role_lower:
                for t in templates:
                    c = dict(t)
                    c["assertion_name"] = c["assertion_name"].replace("{agent_id}", agent_id)
                    c["agent_scope"] = agent_id
                    c["description"] = c["description"].replace("{agent_id}", agent_id)
                    constraints.append(c)
    return constraints


# ── Public API ────────────────────────────────────────────────────────────────

def extract_global_specs(ir: dict) -> list[dict]:
    """
    Extract all global constraints C^G from a normalized trace IR.

    Runs 4 extraction calls in sequence (role, tool, protocol, behavioral).
    Falls back to role-template constraints when system prompts are empty.

    Returns a deduplicated list of constraint dicts.
    """
    agents = ir.get("agents", [])
    task = ir.get("task_instruction", "")
    tool_schema = ir.get("tool_schema", [])

    all_constraints: list[dict] = []

    # Check if we have actual system prompt content
    has_system_prompts = any(
        a.get("system_prompt", "").strip() for a in agents
    )

    if has_system_prompts or agents:
        all_constraints += _extract_role_constraints(agents, task)
        all_constraints += _extract_protocol_constraints(agents, task)
        all_constraints += _extract_behavioral_constraints(agents, task)

    if tool_schema:
        all_constraints += _extract_tool_constraints(tool_schema)

    # Always add role-template constraints as fallback / supplement
    template_constraints = _infer_role_template_constraints(agents)
    existing_names = {c.get("assertion_name") for c in all_constraints}
    for c in template_constraints:
        if c["assertion_name"] not in existing_names:
            all_constraints.append(c)
            existing_names.add(c["assertion_name"])

    return all_constraints
