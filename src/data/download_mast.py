"""
Download the MAST MAD dataset from HuggingFace and cache locally.
Also fetches raw trace JSON files from the MAST GitHub repository.

Usage:
    python -m src.data.download_mast
    python -m src.data.download_mast --source huggingface   # only HF
    python -m src.data.download_mast --source github        # only GitHub traces
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import HUGGINGFACE_REPO_ID, MAD_FULL_FILE, MAD_HUMAN_FILE, RAW_DATA_DIR


# GitHub raw content base for MAST traces
MAST_GITHUB_API = "https://api.github.com/repos/multi-agent-systems-failure-taxonomy/MAST/contents/traces"
MAST_RAW_BASE = "https://raw.githubusercontent.com/multi-agent-systems-failure-taxonomy/MAST/main/traces"

# Subdirectories in the MAST traces folder we want to download
TRACE_SUBDIRS = [
    "AG2",
    "HyperAgent",
    "MagenticOne_GAIA",
    "OpenManus_GAIA",
    "AppWorld",
    "math_interventions",
    "programdev",
]


def download_huggingface(out_dir: str) -> dict:
    """Download MAD_full and MAD_human_labelled from HuggingFace."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for filename in [MAD_FULL_FILE, MAD_HUMAN_FILE]:
        dest = os.path.join(out_dir, filename)
        if os.path.exists(dest):
            print(f"  [skip] {filename} already exists")
            paths[filename] = dest
            continue
        print(f"  Downloading {filename} from HuggingFace …")
        src = hf_hub_download(
            repo_id=HUGGINGFACE_REPO_ID,
            filename=filename,
            repo_type="dataset",
        )
        import shutil
        shutil.copy(src, dest)
        print(f"  Saved → {dest}")
        paths[filename] = dest
    return paths


def _github_list_files(subdir: str) -> list[dict]:
    """List files in a MAST GitHub traces subdirectory via the GitHub API."""
    url = f"{MAST_GITHUB_API}/{subdir}"
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 403:
        print(f"  [warn] GitHub rate limit hit for {subdir}; try setting GITHUB_TOKEN env var")
        return []
    resp.raise_for_status()
    return [f for f in resp.json() if isinstance(f, dict) and f.get("name", "").endswith(".json")]


def download_github_traces(out_dir: str) -> list[str]:
    """Download JSON trace files from the MAST GitHub repository."""
    saved = []
    for subdir in TRACE_SUBDIRS:
        subdir_out = os.path.join(out_dir, "traces", subdir)
        os.makedirs(subdir_out, exist_ok=True)
        print(f"  Listing {subdir} …")
        try:
            files = _github_list_files(subdir)
        except Exception as e:
            print(f"  [warn] Could not list {subdir}: {e}")
            continue

        for f in files:
            dest = os.path.join(subdir_out, f["name"])
            if os.path.exists(dest):
                continue
            url = f"{MAST_RAW_BASE}/{subdir}/{f['name']}"
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                with open(dest, "w", encoding="utf-8") as fh:
                    fh.write(r.text)
                saved.append(dest)
                time.sleep(0.05)  # polite rate limiting
            except Exception as e:
                print(f"  [warn] Failed {f['name']}: {e}")

        print(f"  {subdir}: {len(files)} files found, {sum(1 for s in saved if subdir in s)} new saved")
    return saved


def load_mad(path: str) -> list[dict]:
    """Load a MAD JSON file and return the list of trace records."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # MAD files are either a list directly or wrapped in a key
    if isinstance(data, list):
        return data
    for key in ("data", "traces", "records"):
        if key in data:
            return data[key]
    raise ValueError(f"Unexpected MAD format in {path}")


def main():
    parser = argparse.ArgumentParser(description="Download MAST dataset")
    parser.add_argument("--source", choices=["huggingface", "github", "both"], default="both")
    parser.add_argument("--out_dir", default=RAW_DATA_DIR)
    args = parser.parse_args()

    print(f"Output directory: {args.out_dir}")

    if args.source in ("huggingface", "both"):
        print("\n── HuggingFace MAD dataset ──")
        hf_paths = download_huggingface(args.out_dir)
        for name, path in hf_paths.items():
            records = load_mad(path)
            print(f"  {name}: {len(records)} traces loaded")

    if args.source in ("github", "both"):
        print("\n── GitHub MAST trace files ──")
        saved = download_github_traces(args.out_dir)
        print(f"  Total new files saved: {len(saved)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
