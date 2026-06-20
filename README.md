# Specification-Based Fault Localization for LLM-Based Multi-Agent Systems

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20767030.svg)](https://doi.org/10.5281/zenodo.20767030)

Replication package for the CSE3000 Research Project thesis.

> **Mete Aksoy** — EEMCS, Delft University of Technology, 2026  
> Supervisors: B. Kulahcioglu Ozkan, A. Panichella, Z. Seyedghorban

📄 **[Read the paper](./RP_SPEC_FAULT_LOCALIZATION-FINAL.pdf)**

---

## Overview

This repository implements a six-stage specification-based fault localization (FL)
pipeline for LLM-based multi-agent systems (LLM-MAS). Given a failing agent
execution trace, the pipeline:

1. Normalizes the trace into a unified step-indexed IR
2. Extracts global constraints from system prompts and role schemas
3. Dynamically generates per-step constraints with GPT-4o
4. Evaluates each constraint against the step (GPT-4o-mini → `CLEAR_FAIL` / `PASS`)
5. Builds a violation log of all `CLEAR_FAIL` verdicts
6. Runs a fault attribution judge (GPT-4o) that reads the log and outputs the culprit agent, decisive step, and MAST failure mode

Evaluation is performed on the [MAST MAD dataset](https://huggingface.co/datasets/mcemri/MAD).

---

## Repository Structure

```
spec_fault_localization/
├── config.py                   # Model names, paths, pipeline parameters
├── requirements.txt
├── main.py                     # Single-trace entry point
│
├── src/
│   ├── spec_extractor.py       # Stage 2: global constraint extraction
│   ├── constraint_generator.py # Stage 2b: dynamic constraint generation
│   ├── constraint_evaluator.py # Stage 3: nl_check evaluation
│   ├── validation_log.py       # Stage 4: builds the violation log
│   ├── judge.py                # Stage 5: fault attribution judge
│   └── data/
│       ├── normalizer.py       # Stage 1: trace normalization (6 formats)
│       └── download_mast.py    # Downloads the MAST MAD dataset
│
├── evaluation/
│   ├── run_evaluation.py       # Main evaluation runner (all modes)
│   ├── metrics.py              # Strict / lenient accuracy computation
│   ├── merge_results.py        # Merges batch result files
│   ├── inspect_results.py      # Per-trace inspection utility
│   └── dashboard.py            # Summary dashboard
│
├── prompts/
│   ├── dynamic_spec/           # Prompt for dynamic constraint generation
│   ├── global_spec/            # Prompts for global constraint extraction
│   └── judge/                  # Prompts for baseline and full judge
│
└── results/                    # Pre-computed evaluation outputs (JSON)
```

---

## Setup

**Requirements:** Python 3.10+, an OpenAI API key.

```bash
# 1. Clone and install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 3. Download the MAST MAD dataset from HuggingFace
python src/data/download_mast.py --source huggingface
# Raw files are saved to data/raw/
```

---

## Running the Evaluation

```bash
# Full pipeline on MAD-Human (3 constraints/step)
python evaluation/run_evaluation.py --mode full

# Baseline (no specs, raw LLM judge)
python evaluation/run_evaluation.py --mode baseline

# HyperAgent-SWE traces only
python evaluation/run_evaluation.py --mode full --no-prefer_human --framework hyperagent

# Global constraints only (no dynamic generation)
python evaluation/run_evaluation.py --mode global_only

# Checklist ablation (taxonomy checklist, no violation log)
python evaluation/run_evaluation.py --mode checklist

# Save results to a specific file
python evaluation/run_evaluation.py --mode full --output results/my_run.json
```

**Key flags:**

| Flag | Description |
|---|---|
| `--mode` | `full` / `baseline` / `global_only` / `checklist` |
| `--framework hyperagent` | Filter to HyperAgent traces only |
| `--no-prefer_human` | Use the full MAD dataset instead of human-labelled subset |
| `--max_traces N` | Limit to first N traces |
| `--output PATH` | Output JSON path (default: auto-timestamped) |

### Controlling constraint count

Edit `MAX_DYNAMIC_CONSTRAINTS_PER_STEP` in `config.py` (default: `3`).
Setting it to `5` reproduces the 5c configuration from the paper.

---

## Pre-computed Results

The `results/` directory contains the JSON outputs used in the paper.

| File | Description |
|---|---|
| `eval_baseline_20260508_151245.json` | Baseline on MAD-Human |
| `eval_full_3c_merged.json` | Full pipeline 3c on MAD-Human |
| `eval_full_5c_merged.json` | Full pipeline 5c on MAD-Human |
| `eval_global_only.json` | Global-only on MAD-Human |
| `eval_checklist.json` | Checklist ablation on MAD-Human |
| `eval_baseline_hyperagent_swe.json` | Baseline run 1 on HyperAgent-SWE |
| `eval_baseline_hyperagent_run[2-10].json` | Baseline runs 2–10 on HyperAgent-SWE |
| `eval_full_hyperagent_swe.json` | Full pipeline 3c on HyperAgent-SWE |
| `eval_full_hyperagent_rq4_v2_corrected.json` | Full pipeline with constraint logs (RQ3 analysis) |
| `eval_global_only_hyperagent_swe.json` | Global-only on HyperAgent-SWE |
| `eval_checklist_hyperagent_swe.json` | Checklist ablation on HyperAgent-SWE |

---

## Models Used

| Role | Model |
|---|---|
| Global constraint extraction | `gpt-4o` |
| Dynamic constraint generation | `gpt-4o` |
| Constraint evaluation (nl_check) | `gpt-4o-mini` |
| Fault attribution judge | `gpt-4o` |

Models can be overridden via environment variables `EXTRACTION_MODEL`, `EVAL_MODEL`,
and `JUDGE_MODEL` (see `config.py`).

---

## MAST Taxonomy

The pipeline diagnoses failures according to the MAST taxonomy (3 families, 14 modes):

| Family | Failure Modes |
|---|---|
| FC1 Specification Issues | DisobeyTaskSpec, DisobeyRoleSpec |
| FC2 Inter-Agent Misalignment | IgnoredOtherAgentInput, ReasoningActionMismatch, PrematureTermination, UnawareOfStoppingCond, IncompleteVerification, IncorrectVerification |
| FC3 Task Verification Failures | NoVerification, Hallucination, FaultyReasoning, ContextLoss, FailTaskSpec, InsufficientInfo |

---

## Citation

If you use this code or dataset, please cite:

```
Mete Aksoy. Specification-Based Fault Localization for LLM-Based Multi-Agent Systems.
CSE3000 Research Project, Delft University of Technology, 2026.
```
