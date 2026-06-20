"""
Convert raw MAST trace JSON files into the common Intermediate Representation (IR)
used by all downstream pipeline stages.

IR schema:
{
  "trajectory_id": str,
  "framework": str,               # AG2 | HyperAgent | MagenticOne | etc.
  "task_instruction": str,
  "agents": [
      {"id": str, "role": str, "system_prompt": str}
  ],
  "tool_schema": [...],           # list of tool dicts when available
  "steps": [
      {
          "step_index": int,
          "agent_id": str,
          "role": str,
          "content": str,
          "tool_calls": [...],    # list of {name, arguments}
          "tool_results": [...],  # list of {name, output, is_error}
      }
  ],
  "ground_truth": {               # from human/LLM annotation; may be None
      "culprit_agent": str | None,
      "decisive_step": int | None,
      "failure_mode": str | None,
      "failure_family": str | None,
      "all_failure_modes": [...], # all majority-voted modes: {raw, mode, family, votes}
  } | None
}
"""

from __future__ import annotations

import json
import re
import os
from pathlib import Path
from typing import Any


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    return json.dumps(val, ensure_ascii=False)


def _blank_step(index: int) -> dict:
    return {
        "step_index": index,
        "agent_id": "unknown",
        "role": "unknown",
        "content": "",
        "tool_calls": [],
        "tool_results": [],
    }


# ── MAD failure mode taxonomy mapping ─────────────────────────────────────────
# Maps the MAD dataset's old taxonomy labels (substring match) to MAST FC taxonomy.
# Multiple old labels map to the same canonical mode.

# ── Full dataset numeric taxonomy mapping ──────────────────────────────────────
# Maps mast_annotation numeric keys (used in MAD_full_dataset.json) to canonical modes.
# Based on the Generalizability round labels aligned with MAST FC taxonomy.
_MAST_NUMERIC_MAP: dict[str, tuple[str, str]] = {
    "1.1": ("DisobeyTaskSpec",        "FC1"),  # Disobey Task Specification
    "1.2": ("DisobeyRoleSpec",         "FC1"),  # Disobey Role Specification
    "1.3": ("UnawareOfStoppingCond",   "FC2"),  # Step Repetition
    "1.4": ("ContextLoss",             "FC3"),  # Loss of Conversation History
    "1.5": ("UnawareOfStoppingCond",   "FC2"),  # Unaware of Termination Conditions
    "2.1": ("UnawareOfStoppingCond",   "FC2"),  # Unbatched Repetitive Execution
    "2.2": ("InsufficientInfo",        "FC3"),  # Fail to ask for clarification
    "2.3": ("FailTaskSpec",            "FC3"),  # Task derailment
    "2.4": ("InsufficientInfo",        "FC3"),  # Information withholding
    "2.5": ("IgnoredOtherAgentInput",  "FC2"),  # Ignored other agents' input
    "2.6": ("ReasoningActionMismatch", "FC2"),  # Reasoning-action mismatch
    "3.1": ("PrematureTermination",    "FC2"),  # Premature Termination
    "3.2": ("NoVerification",          "FC3"),  # No or incomplete verification
    "3.3": ("IncorrectVerification",   "FC2"),  # Incorrect Verification
}

_MAD_MODE_MAP: list[tuple[str, str, str]] = [
    # (substring to match in label, canonical MAST mode, family)
    ("disobey task",            "DisobeyTaskSpec",          "FC1"),
    ("poor task constraint",    "DisobeyTaskSpec",          "FC1"),
    ("task constraint",         "DisobeyTaskSpec",          "FC1"),
    ("disobey role",            "DisobeyRoleSpec",          "FC1"),
    ("role specification",      "DisobeyRoleSpec",          "FC1"),
    ("ignoring suggestions",    "IgnoredOtherAgentInput",   "FC2"),
    ("ignored other",           "IgnoredOtherAgentInput",   "FC2"),
    ("inconsistency between",   "ReasoningActionMismatch",  "FC2"),
    ("reasoning and action",    "ReasoningActionMismatch",  "FC2"),
    ("reasoning-action",        "ReasoningActionMismatch",  "FC2"),
    ("premature termination",   "PrematureTermination",     "FC2"),
    ("ill specified termination","PrematureTermination",    "FC2"),
    ("unaware of stopping",     "UnawareOfStoppingCond",   "FC2"),
    ("unaware of termination",  "UnawareOfStoppingCond",   "FC2"),
    ("step repetition",         "UnawareOfStoppingCond",   "FC2"),
    ("backtracking interruption","UnawareOfStoppingCond",  "FC2"),
    ("incomplete verification", "IncompleteVerification",  "FC2"),
    ("lack of critical verif",  "IncompleteVerification",  "FC2"),
    ("critical verification",   "IncompleteVerification",  "FC2"),
    ("incorrect verification",  "IncorrectVerification",   "FC2"),
    ("no or incomplete verif",  "NoVerification",          "FC3"),
    ("lack of result verif",    "NoVerification",          "FC3"),
    ("no verification",         "NoVerification",          "FC3"),
    ("hallucination",           "Hallucination",           "FC3"),
    ("faulty reasoning",        "FaultyReasoning",         "FC3"),
    ("derailment from task",    "FailTaskSpec",            "FC3"),
    ("task derailment",         "FailTaskSpec",            "FC3"),
    ("fail task",               "FailTaskSpec",            "FC3"),
    ("conversation reset",      "ContextLoss",             "FC3"),
    ("loss of conversation",    "ContextLoss",             "FC3"),
    ("context loss",            "ContextLoss",             "FC3"),
    ("fail to elicit",          "InsufficientInfo",        "FC3"),
    ("fail to ask",             "InsufficientInfo",        "FC3"),
    ("waiting for known",       "InsufficientInfo",        "FC3"),
    ("insufficient info",       "InsufficientInfo",        "FC3"),
    ("information withol",      "InsufficientInfo",        "FC3"),
    ("withholding relevant",    "InsufficientInfo",        "FC3"),
    ("disagreement induced",    "PrematureTermination",    "FC2"),
    ("unbatched repetitive",    "UnawareOfStoppingCond",   "FC2"),
    ("undetected conversation", "InsufficientInfo",        "FC3"),
    ("conversation ambiguities","InsufficientInfo",        "FC3"),
]


def _map_mad_failure_mode(label: str) -> tuple[str, str]:
    """Map a raw MAD failure mode label to (canonical_mode, family)."""
    low = label.lower()
    for substr, mode, family in _MAD_MODE_MAP:
        if substr in low:
            return mode, family
    return "FailTaskSpec", "FC3"  # safe fallback


# ── MAD / HuggingFace format (actual schema) ──────────────────────────────────
# Actual MAD records look like:
# {
#   "round": "Round 1",
#   "mas_name": "AppWorld",          # framework name
#   "benchmark_name": "Test-C",
#   "trace_id": 0,
#   "trace": "<raw multiline conversation text>",
#   "annotations": [
#       {
#         "failure mode": "1.5 Unaware of stopping conditions\n\n<desc>",
#         "annotator_1": true,
#         "annotator_2": false,
#         "annotator_3": true
#       }, ...
#   ]
# }

def _parse_appworld_text(text: str) -> tuple[str, list[dict]]:
    """
    Parse AppWorld / GAIA plain-text traces.
    Boundaries: "Response from X Agent", "Message to X Agent", "Code Execution Output", etc.
    Task header: lines bracketed by asterisks.
    """
    task = ""
    header_match = re.search(
        r'\*{3,}.*?Task.*?\*{3,}\s*\n(.*?)(?=\n(?:Response|Message|Code|Entering|$))',
        text, re.DOTALL
    )
    if header_match:
        task = header_match.group(1).strip()

    boundary_pattern = re.compile(
        r'^((?:Response from|Message to|Code Execution Output|Entering \w[\w\s]* message loop)'
        r'.*?)$',
        re.MULTILINE
    )
    boundaries = [(m.start(), m.group(1).strip()) for m in boundary_pattern.finditer(text)]

    # Fallback: treat entire text as one step if no boundaries found
    if not boundaries:
        # Try extracting task from the whole text
        task = task or text.split('\n')[0].strip()
        return task, [{
            "step_index": 0,
            "agent_id": "unknown",
            "role": "assistant",
            "content": text.strip(),
            "tool_calls": [],
            "tool_results": [],
        }]

    steps = []
    for i, (start, label) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        body = text[start:end].strip()
        content_lines = body.split('\n')
        body = '\n'.join(content_lines[1:]).strip()

        resp_m = re.match(r'Response from (.+)', label)
        msg_m  = re.match(r'Message to (.+)', label)
        code_m = re.match(r'Code Execution', label)
        enter_m = re.match(r'Entering ([\w\s]+) message loop', label)

        if resp_m:
            agent_id, role = resp_m.group(1).strip(), "assistant"
        elif msg_m:
            agent_id, role = msg_m.group(1).strip(), "user"
        elif code_m:
            agent_id, role = "executor", "tool"
        elif enter_m:
            agent_id, role = enter_m.group(1).strip(), "system"
        else:
            agent_id, role = "unknown", "unknown"

        agent_id = re.sub(r'\s+Agent\s*$', '', agent_id, flags=re.IGNORECASE).strip()
        steps.append({
            "step_index": i,
            "agent_id": agent_id,
            "role": role,
            "content": body,
            "tool_calls": [],
            "tool_results": [],
        })

    return task, steps


def _parse_ag2_trajectory(traj: list) -> tuple[str, list[dict]]:
    """Parse AG2 trajectory: list of {content, role, name} dicts."""
    task = ""
    steps = []
    for i, msg in enumerate(traj):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(str(c) for c in content)
        content = _safe_str(content)
        role = _safe_str(msg.get("role", "user"))
        agent_id = _safe_str(msg.get("name") or msg.get("role", f"agent_{i}"))
        steps.append({
            "step_index": i,
            "agent_id": agent_id,
            "role": role,
            "content": content,
            "tool_calls": [],
            "tool_results": [],
        })
        if i == 0 and role == "user":
            task = content[:500]
    return task, steps


def _parse_hyperagent_log(log_lines: list) -> tuple[str, list[dict]]:
    """
    Parse HyperAgent trajectory: list of log strings.
    Lines look like: "HyperAgent_<id> - INFO - Planner's Response: ..."
    Groups by agent name (Planner, Navigator, Executor, etc.).
    """
    # Extract agent from message body after "INFO - "
    # e.g. "Planner's Response:", "Navigator->Planner:", "Inner-Navigator-Assistant's Response:"
    info_pattern = re.compile(r'HyperAgent_\S+\s+-\s+INFO\s+-\s+(.*)', re.DOTALL)
    agent_header = re.compile(
        r'^((?:Inner-)?(?:Planner|Navigator|Executor|Editor|Codebase Navigator|'
        r'Code Writer|Code Reviewer|HyperAgent))'
        r'(?:\'s Response|->[\w\s]+)?:',
        re.IGNORECASE
    )

    steps = []
    current_agent = "HyperAgent"
    current_lines: list[str] = []

    def flush():
        if current_lines:
            steps.append({
                "step_index": len(steps),
                "agent_id": current_agent,
                "role": "assistant",
                "content": "\n".join(current_lines),
                "tool_calls": [],
                "tool_results": [],
            })

    for raw_line in log_lines:
        if not isinstance(raw_line, str):
            continue
        m = info_pattern.match(raw_line)
        if m:
            msg = m.group(1)
            am = agent_header.match(msg)
            if am:
                agent = am.group(1).strip()
                # Normalize agent names
                if "Navigator" in agent:
                    agent = "Navigator"
                elif "Planner" in agent:
                    agent = "Planner"
                elif "Executor" in agent:
                    agent = "Executor"
                elif "Editor" in agent or "Writer" in agent:
                    agent = "Editor"
                if agent != current_agent:
                    flush()
                    current_agent = agent
                    current_lines = [msg]
                else:
                    current_lines.append(msg)
            else:
                current_lines.append(msg)
        else:
            current_lines.append(raw_line)

    flush()
    # Cap at 150 steps; the logs can be thousands of lines
    steps = steps[:150]
    return "", steps


def _parse_chatdev_text(text: str) -> tuple[str, list[dict]]:
    """
    Parse ChatDev log text.

    Actual MAD format uses timestamped INFO lines as boundaries:
      [2025-17-01 11:36:46 INFO] Chief Product Officer: **Role<->Role on : Phase, turn N**
      <content>

    Legacy format uses AI User: / AI Assistant: speaker markers.
    """
    task = ""
    task_m = re.search(r'\*\*task_prompt\*\*:\s*(.+?)(?=\n\n|\n\*\*)', text, re.DOTALL)
    if task_m:
        task = task_m.group(1).strip()

    # ── Actual MAD ChatDev format: [TIMESTAMP INFO] AgentName: **...**  ──────
    info_boundary = re.compile(
        r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} INFO\] ([\w ]+):\s*(.*)',
        re.MULTILINE,
    )
    info_matches = list(info_boundary.finditer(text))
    if info_matches:
        steps = []
        for i, m in enumerate(info_matches):
            agent_id = m.group(1).strip()
            # Skip pure noise agents
            if agent_id in ("HTTP Request",):
                continue
            end = info_matches[i + 1].start() if i + 1 < len(info_matches) else len(text)
            body = text[m.start():end]
            # Drop the header line itself
            body = body.split('\n', 1)[1].strip() if '\n' in body else ""
            role = "system" if agent_id == "System" else "assistant"
            steps.append({
                "step_index": len(steps),
                "agent_id": agent_id,
                "role": role,
                "content": body,
                "tool_calls": [],
                "tool_results": [],
            })
        return task, steps

    # ── Legacy format: AI User: / AI Assistant: markers ──────────────────────
    turn_pattern = re.compile(
        r'^(?:AI User:|AI Assistant:|Human Turn \d+:|Chatbot Turn \d+:)',
        re.MULTILINE,
    )
    boundaries = [(m.start(), m.group().strip()) for m in turn_pattern.finditer(text)]

    if not boundaries:
        return task, [{
            "step_index": 0,
            "agent_id": "ChatDev",
            "role": "assistant",
            "content": text.strip(),
            "tool_calls": [],
            "tool_results": [],
        }]

    steps = []
    for i, (start, label) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        body = text[start:end].strip()
        content_lines = body.split('\n', 1)
        body = content_lines[1].strip() if len(content_lines) > 1 else ""
        role = "user" if ("AI User" in label or "Human" in label) else "assistant"
        agent_id = "ChatDev"
        role_context = text[max(0, start - 200):start]
        role_m = re.search(r'\*\*([\w\s]+?)\s*<->', role_context)
        if role_m:
            agent_id = role_m.group(1).strip()
        steps.append({
            "step_index": i,
            "agent_id": agent_id,
            "role": role,
            "content": body,
            "tool_calls": [],
            "tool_results": [],
        })

    return task, steps


def _parse_metagpt_commlog(text: str) -> tuple[str, list[dict]]:
    """
    Parse MetaGPT Agent Communication Log (plain-text format).

    Format:
      === MetaGPT Agent Communication Log ... ===

      [TIMESTAMP] FROM: Human TO: {'<all>'}
      ACTION: ...
      CONTENT:
      <task text>
      ---...---

      [TIMESTAMP] NEW MESSAGES:

      AgentName:
      <content>
      ---...---
    """
    task = ""
    # Extract task from first FROM block's CONTENT section
    task_m = re.search(r'CONTENT:\s*\n(.*?)(?=\n-{10,})', text, re.DOTALL)
    if task_m:
        task = task_m.group(1).strip()

    # Split on dashed dividers
    blocks = re.split(r'\n-{10,}\n', text)
    steps = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # NEW MESSAGES block: "AgentName:\n<content>" after the timestamp header
        new_msg_m = re.search(r'\[\d{4}-\d{2}-\d{2}[^\]]+\]\s*NEW MESSAGES:\s*\n+([\w()]+):\s*\n(.*)',
                              block, re.DOTALL)
        if new_msg_m:
            agent_id = new_msg_m.group(1).strip()
            content  = new_msg_m.group(2).strip()
            steps.append({
                "step_index": len(steps),
                "agent_id": agent_id,
                "role": "assistant",
                "content": content,
                "tool_calls": [],
                "tool_results": [],
            })
            continue

        # FROM block: treat as a user/human turn
        from_m = re.search(r'\[\d{4}-\d{2}-\d{2}[^\]]+\]\s*FROM:\s*(\w+)', block)
        if from_m:
            agent_id = from_m.group(1).strip()
            content_m = re.search(r'CONTENT:\s*\n(.*)', block, re.DOTALL)
            content = content_m.group(1).strip() if content_m else block
            steps.append({
                "step_index": len(steps),
                "agent_id": agent_id,
                "role": "user" if agent_id.lower() == "human" else "assistant",
                "content": content,
                "tool_calls": [],
                "tool_results": [],
            })

    return task, steps


def _parse_separator_text(text: str) -> tuple[str, list[dict]]:
    """
    Parse MagenticOne/GAIA traces that use '---------- AgentName ----------' separators.
    Everything before the first separator is skipped (typically pip install preamble).
    """
    task = ""
    boundary = re.compile(r'^-{10,}\s*([\w][\w\s]*?)\s*-{10,}\s*$', re.MULTILINE)
    matches = list(boundary.finditer(text))

    if not matches:
        return task, []

    steps = []
    for i, m in enumerate(matches):
        agent_id = m.group(1).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()

        role = "user" if agent_id.lower() == "user" else "assistant"
        if i == 0 and agent_id.lower() == "user":
            task = body[:500]

        steps.append({
            "step_index": len(steps),
            "agent_id": agent_id,
            "role": role,
            "content": body,
            "tool_calls": [],
            "tool_results": [],
        })

    return task, steps


def _parse_metagpt_log(content: str) -> tuple[str, list[dict]]:
    """
    Parse MetaGPT log string.
    Format: "TIMESTAMP | INFO | metagpt.roles.role:_act:391 - Alice(SimpleCoder): to do ..."
    """
    task = ""
    # Lines starting with a timestamp + role info
    line_pattern = re.compile(
        r'^\d{4}-\d{2}-\d{2}.+?\|\s*INFO\s*\|.*?-\s*(.+?):\s*(.*)',
    )

    steps = []
    lines = content.split('\n')
    for i, line in enumerate(lines):
        m = line_pattern.match(line)
        if m:
            agent_id = m.group(1).strip()
            body = m.group(2).strip()
            # Accumulate following non-matched lines into this step
            j = i + 1
            extra = []
            while j < len(lines) and not line_pattern.match(lines[j]):
                extra.append(lines[j])
                j += 1
            if extra:
                body = body + "\n" + "\n".join(extra)

            steps.append({
                "step_index": len(steps),
                "agent_id": agent_id,
                "role": "assistant",
                "content": body,
                "tool_calls": [],
                "tool_results": [],
            })

    # Deduplicate consecutive same-agent steps by merging
    merged = []
    for s in steps:
        if merged and merged[-1]["agent_id"] == s["agent_id"]:
            merged[-1]["content"] += "\n" + s["content"]
        else:
            merged.append(s)
    for i, s in enumerate(merged):
        s["step_index"] = i

    return task, merged


def _route_plain_text(text: str) -> tuple[str, list[dict]]:
    """Route plain-text trace content to the correct sub-parser."""
    if "AI User:" in text or "AI Assistant:" in text:
        return _parse_chatdev_text(text)
    if re.search(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} INFO\]', text, re.MULTILINE):
        return _parse_chatdev_text(text)
    if "NEW MESSAGES:" in text and "FROM:" in text:
        return _parse_metagpt_commlog(text)
    if re.search(r'^-{10,}\s*[\w][\w\s]*\s*-{10,}', text, re.MULTILINE):
        return _parse_separator_text(text)
    return _parse_appworld_text(text)


def _normalize_mad_actual(raw: dict) -> dict:
    """Normalize the actual MAD dataset format (round/mas_name/trace/annotations)."""
    tid = str(raw.get("trace_id", "unknown"))
    framework = _safe_str(raw.get("mas_name", "unknown")).lower()
    trace_raw = raw.get("trace", "")

    task = ""
    steps: list[dict] = []

    if isinstance(trace_raw, str):
        stripped = trace_raw.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            # Embedded JSON string
            try:
                parsed = json.loads(trace_raw)
            except json.JSONDecodeError:
                parsed = None

            if parsed is None:
                # Broken JSON — route the same way as plain text
                task, steps = _route_plain_text(trace_raw)
            elif isinstance(parsed, dict) and "trajectory" in parsed:
                traj = parsed["trajectory"]
                ps = parsed.get("problem_statement", "")
                if isinstance(ps, list):
                    ps = " ".join(str(x) for x in ps)
                ps = _safe_str(ps)
                if traj and isinstance(traj[0], str):
                    # HyperAgent: list of log strings
                    _, steps = _parse_hyperagent_log(traj)
                    task = ps
                else:
                    # AG2: list of message dicts
                    _, steps = _parse_ag2_trajectory(traj)
                    task = ps or (steps[0]["content"][:200] if steps else "")
            elif isinstance(parsed, dict) and "content" in parsed and isinstance(parsed["content"], str):
                # MetaGPT log
                task, steps = _parse_metagpt_log(parsed["content"])
                if not task:
                    task = _safe_str(parsed.get("prompt", ""))
            elif isinstance(parsed, dict):
                # Generic JSON — try common keys
                task, steps = _parse_appworld_text(json.dumps(parsed, indent=2))
            else:
                task, steps = _parse_appworld_text(trace_raw)
        else:
            # Plain text
            task, steps = _route_plain_text(trace_raw)
    else:
        trace_text = _safe_str(trace_raw)
        task, steps = _parse_appworld_text(trace_text)

    # Build agents from unique agent_ids in steps
    agents: dict[str, dict] = {}
    for s in steps:
        aid = s["agent_id"]
        if aid not in agents:
            agents[aid] = {"id": aid, "role": s["role"], "system_prompt": ""}

    # Extract ground truth: find all failure modes voted by majority (≥2 of 3 annotators)
    annotations = raw.get("annotations", [])
    voted_modes: list[tuple[str, str, str, int]] = []  # (raw_label, canonical_mode, family, vote_count)
    for ann in annotations:
        votes = sum([
            bool(ann.get("annotator_1")),
            bool(ann.get("annotator_2")),
            bool(ann.get("annotator_3")),
        ])
        if votes >= 2:
            raw_label = ann.get("failure mode", "").split("\n")[0].strip()
            mode, family = _map_mad_failure_mode(raw_label)
            voted_modes.append((raw_label, mode, family, votes))

    # Sort so 3/3 modes precede 2/3 modes — preserves original order within each tier
    voted_modes.sort(key=lambda x: x[3], reverse=True)

    gt = None
    if voted_modes:
        # Primary failure mode = highest-voted mode (3/3 beats 2/3; ties keep taxonomy order)
        primary_raw, primary_mode, primary_family, _ = voted_modes[0]
        gt = {
            "culprit_agent": None,     # not annotated in MAD
            "decisive_step": None,     # not annotated in MAD
            "failure_mode": primary_mode,
            "failure_family": primary_family,
            "all_failure_modes": [
                {"raw": r, "mode": m, "family": f, "votes": v} for r, m, f, v in voted_modes
            ],
        }

    return {
        "trajectory_id": f"{framework}_{tid}",
        "framework": framework,
        "task_instruction": task,
        "agents": list(agents.values()),
        "tool_schema": [],
        "steps": steps,
        "ground_truth": gt,
    }


# ── Legacy / fallback MAD format (trajectory_snippet style) ──────────────────
# Kept for compatibility with other MAD export formats.

def _normalize_mad_snippet(raw: dict) -> dict:
    tid = raw.get("trajectory_id") or raw.get("id") or "unknown"
    task = _safe_str(raw.get("task_instruction") or raw.get("task") or "")

    snippet = raw.get("trajectory_snippet") or raw.get("messages") or []
    steps = []
    for i, msg in enumerate(snippet):
        step = _blank_step(i)
        step["role"] = _safe_str(msg.get("role", "unknown"))
        step["agent_id"] = _safe_str(msg.get("role", "unknown"))
        step["content"] = _safe_str(msg.get("content", ""))
        step["step_index"] = int(msg.get("index", i))
        steps.append(step)

    gt = None
    root_cause = raw.get("root_cause")
    failures = raw.get("failures") or []
    if root_cause or failures:
        rc_id = (root_cause or {}).get("failure_id")
        rc_failure = next((f for f in failures if f.get("failure_id") == rc_id), None) or {}
        decisive_step = rc_failure.get("step_number")
        culprit = rc_failure.get("failed_agent")
        mode_raw = rc_failure.get("failure_category") or rc_failure.get("failure_type") or ""
        mode, family = _map_mad_failure_mode(mode_raw)
        gt = {
            "culprit_agent": culprit,
            "decisive_step": decisive_step,
            "failure_mode": mode,
            "failure_family": family,
            "all_failure_modes": [],
        }

    seen_roles: dict[str, dict] = {}
    for msg in snippet:
        r = _safe_str(msg.get("role", "unknown"))
        if r not in seen_roles:
            seen_roles[r] = {"id": r, "role": r, "system_prompt": ""}

    return {
        "trajectory_id": tid,
        "framework": "mad",
        "task_instruction": task,
        "agents": list(seen_roles.values()),
        "tool_schema": [],
        "steps": steps,
        "ground_truth": gt,
    }


# ── GitHub trace formats ───────────────────────────────────────────────────────

def _detect_framework(raw: dict, source_path: str = "") -> str:
    for key in ("mas_name", "framework", "source", "system"):
        if key in raw and isinstance(raw[key], str):
            return raw[key].lower()

    path_lower = source_path.lower()
    for name in ("hyperagent", "ag2", "magenticone", "openmanus", "appworld",
                 "chatdev", "metagpt", "math_interventions", "programdev", "gaia"):
        if name in path_lower:
            return name

    if "trajectory_snippet" in raw:
        return "magentic_one"
    if "messages" in raw and isinstance(raw.get("messages"), list):
        first = raw["messages"][0] if raw["messages"] else {}
        if "agent" in first:
            return "ag2"
        if "role" in first:
            return "openai_style"
    return "unknown"


def _normalize_ag2(raw: dict, tid: str) -> dict:
    messages = raw.get("messages") or raw.get("conversation") or []
    agents: dict[str, dict] = {}
    steps = []
    for i, msg in enumerate(messages):
        agent_id = _safe_str(msg.get("name") or msg.get("agent") or msg.get("role", f"agent_{i}"))
        role = _safe_str(msg.get("role", agent_id))
        content = _safe_str(msg.get("content", ""))

        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or tc
            tool_calls.append({"name": fn.get("name", ""), "arguments": fn.get("arguments", {})})

        tool_results = []
        if msg.get("role") == "tool":
            tool_results.append({"name": _safe_str(msg.get("name", "")), "output": content, "is_error": False})

        steps.append({
            "step_index": i,
            "agent_id": agent_id,
            "role": role,
            "content": content,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        })

        if agent_id not in agents:
            agents[agent_id] = {"id": agent_id, "role": role, "system_prompt": _safe_str(raw.get("system_message", ""))}

    return {
        "trajectory_id": tid,
        "framework": "ag2",
        "task_instruction": _safe_str(raw.get("task") or raw.get("query") or ""),
        "agents": list(agents.values()),
        "tool_schema": raw.get("tools") or [],
        "steps": steps,
        "ground_truth": None,
    }


def _normalize_hyperagent(raw: dict, tid: str) -> dict:
    steps_raw = raw.get("steps") or raw.get("trajectory") or []
    agents: dict[str, dict] = {}
    steps = []

    for i, s in enumerate(steps_raw):
        agent_id = _safe_str(s.get("agent") or s.get("role", f"agent_{i}"))
        role = _safe_str(s.get("role", agent_id))
        content = _safe_str(s.get("content") or s.get("observation") or s.get("thought") or "")

        tool_calls = []
        if "action" in s:
            tool_calls.append({"name": _safe_str(s["action"]), "arguments": s.get("action_input", {})})
        for tc in s.get("tool_calls") or []:
            tool_calls.append({"name": tc.get("name", ""), "arguments": tc.get("arguments", {})})

        tool_results = []
        if "observation" in s and s.get("action"):
            tool_results.append({"name": _safe_str(s["action"]), "output": _safe_str(s["observation"]), "is_error": False})

        steps.append({
            "step_index": i,
            "agent_id": agent_id,
            "role": role,
            "content": content,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        })

        if agent_id not in agents:
            agents[agent_id] = {"id": agent_id, "role": role, "system_prompt": ""}

    return {
        "trajectory_id": tid,
        "framework": "hyperagent",
        "task_instruction": _safe_str(raw.get("task") or raw.get("issue") or raw.get("query") or ""),
        "agents": list(agents.values()),
        "tool_schema": raw.get("tools") or [],
        "steps": steps,
        "ground_truth": None,
    }


def _normalize_magentic_one(raw: dict, tid: str) -> dict:
    snippet = raw.get("trajectory_snippet") or raw.get("messages") or []
    agents: dict[str, dict] = {}
    steps = []

    for msg in snippet:
        idx = int(msg.get("index", len(steps)))
        role = _safe_str(msg.get("role", "unknown"))
        agent_id = role.split("(")[0].strip() if "(" in role else role
        content = _safe_str(msg.get("content", ""))
        steps.append({
            "step_index": idx,
            "agent_id": agent_id,
            "role": role,
            "content": content,
            "tool_calls": [],
            "tool_results": [],
        })
        if agent_id not in agents:
            agents[agent_id] = {"id": agent_id, "role": agent_id, "system_prompt": ""}

    gt = None
    failures = raw.get("failures") or []
    root_cause = raw.get("root_cause") or {}
    if failures:
        rc_id = root_cause.get("failure_id")
        rc = next((f for f in failures if f.get("failure_id") == rc_id), failures[0])
        mode_raw = rc.get("failure_category") or ""
        mode, family = _map_mad_failure_mode(mode_raw)
        gt = {
            "culprit_agent": rc.get("failed_agent"),
            "decisive_step": rc.get("step_number"),
            "failure_mode": mode,
            "failure_family": family,
            "all_failure_modes": [],
        }

    return {
        "trajectory_id": tid,
        "framework": "magentic_one",
        "task_instruction": _safe_str(raw.get("task_instruction") or raw.get("task") or ""),
        "agents": list(agents.values()),
        "tool_schema": [],
        "steps": steps,
        "ground_truth": gt,
    }


def _normalize_generic(raw: dict, tid: str, framework: str) -> dict:
    messages = (
        raw.get("messages")
        or raw.get("conversation")
        or raw.get("trajectory")
        or raw.get("trajectory_snippet")
        or []
    )
    agents: dict[str, dict] = {}
    steps = []

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        agent_id = _safe_str(msg.get("role") or msg.get("agent") or msg.get("name") or f"agent_{i}")
        content = _safe_str(msg.get("content") or msg.get("text") or "")
        idx = int(msg.get("index", i))
        steps.append({
            "step_index": idx,
            "agent_id": agent_id,
            "role": agent_id,
            "content": content,
            "tool_calls": [],
            "tool_results": [],
        })
        if agent_id not in agents:
            agents[agent_id] = {"id": agent_id, "role": agent_id, "system_prompt": ""}

    task = _safe_str(
        raw.get("task_instruction") or raw.get("task") or raw.get("query") or raw.get("question") or ""
    )

    return {
        "trajectory_id": tid,
        "framework": framework,
        "task_instruction": task,
        "agents": list(agents.values()),
        "tool_schema": raw.get("tools") or [],
        "steps": steps,
        "ground_truth": None,
    }


# ── MAD full dataset format (mast_annotation + trace dict) ───────────────────
# Records in MAD_full_dataset.json use a different structure:
#   {"mas_name": "HyperAgent", "trace": {"key": ..., "trajectory": "<str>"}, "mast_annotation": {...}}
# The trajectory string has a YAML-like header followed by indented log lines.

def _extract_problem_statement(trajectory: str) -> str:
    m = re.search(r'problem_statement:\s*\n(.*?)(?=\nother_data:|\ntrajectory:|\Z)', trajectory, re.DOTALL)
    if m:
        lines = m.group(1).split('\n')
        return '\n'.join(line.lstrip() for line in lines).strip()
    return ""


def _normalize_mad_full(raw: dict) -> dict:
    """Normalize MAD_full_dataset.json format (trace is a dict, mast_annotation for GT)."""
    tid = str(raw.get("trace_id", "unknown"))
    framework = _safe_str(raw.get("mas_name", "unknown")).lower()
    trace_dict = raw.get("trace", {})
    trajectory_str = _safe_str(trace_dict.get("trajectory", ""))

    task = _extract_problem_statement(trajectory_str)

    # Strip leading whitespace and skip the YAML header before the first INFO line
    lines = [line.lstrip() for line in trajectory_str.split('\n')]
    info_pat = re.compile(r'HyperAgent_\S+\s+-\s+INFO\s+-')
    first_info = next((i for i, l in enumerate(lines) if info_pat.match(l)), None)
    if first_info is not None:
        _, steps = _parse_hyperagent_log(lines[first_info:])
    else:
        # Trajectory content is not HyperAgent format — fall back to plain-text routing
        _, steps = _route_plain_text(trajectory_str)

    agents: dict[str, dict] = {}
    for s in steps:
        aid = s["agent_id"]
        if aid not in agents:
            agents[aid] = {"id": aid, "role": s["role"], "system_prompt": ""}

    # Ground truth from mast_annotation numeric keys
    mast_ann = raw.get("mast_annotation", {})
    active_modes = [(k, v) for k, v in mast_ann.items() if v == 1]
    gt = None
    if active_modes:
        all_modes = []
        for key, _ in active_modes:
            if key in _MAST_NUMERIC_MAP:
                mode, family = _MAST_NUMERIC_MAP[key]
                all_modes.append({"raw": key, "mode": mode, "family": family, "votes": 1})
        if all_modes:
            gt = {
                "culprit_agent": None,
                "decisive_step": None,
                "failure_mode": all_modes[0]["mode"],
                "failure_family": all_modes[0]["family"],
                "all_failure_modes": all_modes,
            }

    return {
        "trajectory_id": f"{framework}_{tid}",
        "framework": framework,
        "task_instruction": task,
        "agents": list(agents.values()),
        "tool_schema": [],
        "steps": steps,
        "ground_truth": gt,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def normalize_trace(raw: dict, source_path: str = "") -> dict:
    """
    Normalize a single raw MAST/MAD trace dict to the common IR format.

    Detects the format automatically:
    - Actual MAD dataset (round/mas_name/trace/annotations keys)
    - Legacy MAD snippet format (trajectory_snippet/failures keys)
    - AG2, HyperAgent, MagenticOne GitHub trace files
    - Generic fallback
    """
    # MAD full dataset format (mast_annotation + trace as dict)
    if "mast_annotation" in raw and isinstance(raw.get("trace"), dict):
        ir = _normalize_mad_full(raw)
        return ir

    # Actual MAD human-labelled dataset (downloaded from HuggingFace)
    if "annotations" in raw and "trace" in raw and isinstance(raw.get("trace"), str):
        ir = _normalize_mad_actual(raw)
        return ir

    framework = _detect_framework(raw, source_path)
    tid = (
        raw.get("trajectory_id")
        or raw.get("id")
        or Path(source_path).stem
        or "unknown"
    )

    if "trajectory_snippet" in raw or "failures" in raw:
        ir = _normalize_mad_snippet(raw)
        ir["framework"] = framework
    elif "ag2" in framework:
        ir = _normalize_ag2(raw, tid)
    elif "hyperagent" in framework:
        ir = _normalize_hyperagent(raw, tid)
    elif "magentic" in framework or "openmanus" in framework:
        ir = _normalize_magentic_one(raw, tid)
    else:
        ir = _normalize_generic(raw, tid, framework)

    ir["steps"].sort(key=lambda s: s["step_index"])
    return ir


def normalize_file(path: str) -> list[dict]:
    """Load a JSON file (single trace or list of traces) and normalize all."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = data if isinstance(data, list) else [data]
    return [normalize_trace(r, source_path=path) for r in records]


def normalize_mad_dataset(path: str) -> list[dict]:
    """Load MAD_full_dataset.json or MAD_human_labelled_dataset.json."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = data if isinstance(data, list) else list(data.values())
    return [normalize_trace(r, source_path=path) for r in records]
