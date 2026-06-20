"""
Visualization dashboard for spec-based FL ablation results.
Serves a Chart.js dashboard on http://localhost:8000
"""

import argparse
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# ── Result file config ────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

DATASET_CONFIGS = {
    "mad18": {
        "title": "MAST MAD · 18 traces (human-labelled) · GPT-4o judge",
        "configs": [
            ("Baseline",    "eval_baseline_lenient.json",      "baseline",   "#6c757d"),
            ("Checklist",   "eval_checklist_v2.json",           "checklist",  "#0d6efd"),
            ("Global Only", "eval_global_only.json",             "global_only","#fd7e14"),
            ("Full (5c)",   "eval_full_5c_merged.json",          "full",       "#198754"),
        ],
    },
    "hyperagent": {
        "title": "HyperAgent SWE-bench Lite · 14 failing traces (LLM-labelled) · GPT-4o judge",
        "configs": [
            ("Baseline",    "eval_baseline_hyperagent_swe.json",    "baseline",   "#6c757d"),
            ("Checklist",   "eval_checklist_hyperagent_swe.json",    "checklist",  "#0d6efd"),
            ("Global Only", "eval_global_only_hyperagent_swe.json",  "global_only","#fd7e14"),
            ("Full",        "eval_full_hyperagent_swe.json",          "full",       "#198754"),
        ],
    },
}

def parse_args():
    p = argparse.ArgumentParser(description="Ablation dashboard")
    p.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), default="mad18",
                   help="Which dataset to display (default: mad18)")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()

ARGS = parse_args()
SELECTED = DATASET_CONFIGS[ARGS.dataset]
CONFIGS = SELECTED["configs"]
DASHBOARD_TITLE = SELECTED["title"]

FAMILY_COLORS = {"FC1": "#dc3545", "FC2": "#fd7e14", "FC3": "#0d6efd", "UNKNOWN": "#adb5bd"}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_results():
    data = {}
    for label, fname, key, color in CONFIGS:
        path = os.path.join(RESULTS_DIR, fname)
        try:
            raw = json.load(open(path, encoding="utf-8"))
            data[label] = {"inner": raw[key], "color": color, "key": key}
        except Exception as e:
            print(f"Warning: could not load {fname}: {e}", file=sys.stderr)
    return data


def get_metrics(inner):
    m = inner["metrics"]
    return {
        "Strict Mode":   round(m["failure_mode_accuracy"] * 100, 1),
        "Lenient Mode":  round(m["failure_mode_accuracy_lenient"] * 100, 1),
        "Strict Family": round(m["failure_family_accuracy"] * 100, 1),
        "Lenient Family":round(m["failure_family_accuracy_lenient"] * 100, 1),
    }


def get_mode_distribution(inner):
    counts = {}
    for p in inner["predictions"]:
        mode = p.get("failure_mode") or "UNKNOWN"
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def build_per_trace_table(data):
    # ground_truths have no trajectory_id — join positionally with predictions.
    # Use the first loaded config as the source of trajectory ordering.
    first_inner = next(iter(data.values()))["inner"]
    # Build (tid, gt) pairs by zipping predictions (which carry _trajectory_id) with ground_truths
    tid_gt_pairs = []
    for pred, gt in zip(first_inner["predictions"], first_inner["ground_truths"]):
        tid = pred.get("_trajectory_id", str(len(tid_gt_pairs)))
        tid_gt_pairs.append((tid, {
            "mode":      gt.get("failure_mode", "?"),
            "family":    gt.get("failure_family", "?"),
            "all_modes": gt.get("all_failure_modes", []),
        }))

    # Index all configs' predictions by trajectory_id
    pred_by_config = {label: {} for label in data}
    for label, info in data.items():
        for p in info["inner"]["predictions"]:
            tid = p.get("_trajectory_id", "?")
            pred_by_config[label][tid] = {
                "mode":   p.get("failure_mode", "?"),
                "family": p.get("failure_family", "?"),
            }

    rows = []
    for tid, gt in tid_gt_pairs:
        row = {"tid": tid, "gt_mode": gt["mode"], "gt_family": gt["family"],
               "all_modes": gt["all_modes"], "preds": {}}
        for label in data:
            row["preds"][label] = pred_by_config[label].get(tid, {"mode": "—", "family": "—"})
        rows.append(row)
    return rows


def _split_pred(val):
    """Split pipe-separated judge outputs; handles multi-label responses."""
    return {v.strip().lower() for v in str(val).split("|") if v.strip()}


def mode_match(pred_mode, gt_mode, all_modes):
    preds = _split_pred(pred_mode)
    strict = gt_mode.strip().lower() in preds
    if all_modes:
        names = {(m["mode"] if isinstance(m, dict) else m).strip().lower() for m in all_modes}
        lenient = bool(preds & names)
    else:
        lenient = strict
    return strict, lenient


def family_match(pred_family, gt_family, all_modes):
    preds = _split_pred(pred_family)
    # Normalise to upper for family codes (FC1/FC2/FC3)
    preds_up = {p.upper() for p in preds}
    strict = gt_family.strip().upper() in preds_up
    if all_modes:
        families = {(m["family"] if isinstance(m, dict) else "").strip().upper() for m in all_modes}
        lenient = bool(preds_up & families)
    else:
        lenient = strict
    return strict, lenient


def compute_metrics_corrected(rows, labels):
    """Recompute summary metrics using pipe-aware matching (fixes metrics.py edge case)."""
    result = {}
    n = len(rows)
    for label in labels:
        strict_m = lenient_m = strict_f = lenient_f = 0
        for r in rows:
            pm, pf = r["preds"][label]["mode"], r["preds"][label]["family"]
            sm, lm = mode_match(pm, r["gt_mode"], r["all_modes"])
            sf, lf = family_match(pf, r["gt_family"], r["all_modes"])
            strict_m  += int(sm)
            lenient_m += int(lm)
            strict_f  += int(sf)
            lenient_f += int(lf)
        result[label] = {
            "Strict Mode":    round(strict_m  / n * 100, 1),
            "Lenient Mode":   round(lenient_m / n * 100, 1),
            "Strict Family":  round(strict_f  / n * 100, 1),
            "Lenient Family": round(lenient_f / n * 100, 1),
        }
    return result


# ── HTML generation ───────────────────────────────────────────────────────────

def build_html(data):
    dist_by_config    = {label: get_mode_distribution(info["inner"]) for label, info in data.items()}
    rows              = build_per_trace_table(data)
    # Use corrected metrics (pipe-aware matching) instead of raw JSON values
    metrics_by_config = compute_metrics_corrected(rows, list(data.keys()))

    labels     = list(data.keys())
    colors     = [data[l]["color"] for l in labels]
    metric_keys = ["Strict Mode", "Lenient Mode", "Strict Family", "Lenient Family"]

    # Bar chart datasets — one dataset per metric
    bar_datasets = []
    metric_colors = ["#dc3545", "#fd7e14", "#198754", "#0d6efd"]
    for i, mk in enumerate(metric_keys):
        bar_datasets.append({
            "label": mk,
            "data": [metrics_by_config[l][mk] for l in labels],
            "backgroundColor": metric_colors[i],
            "borderRadius": 4,
        })

    # Pie chart data per config
    all_modes_sorted = sorted({m for dist in dist_by_config.values() for m in dist})
    pie_colors = [
        "#dc3545","#fd7e14","#ffc107","#198754","#0d6efd","#6f42c1",
        "#d63384","#20c997","#0dcaf0","#adb5bd","#495057","#6c757d",
        "#343a40","#212529","#f8f9fa",
    ]
    mode_color_map = {m: pie_colors[i % len(pie_colors)] for i, m in enumerate(all_modes_sorted)}

    pie_charts_js = ""
    for label, info in data.items():
        dist  = dist_by_config[label]
        total = sum(dist.values())
        safe  = label.replace(" ", "_").replace("(", "").replace(")", "")
        modes = list(dist.keys())
        vals  = list(dist.values())
        pcts  = [round(v / total * 100, 1) for v in vals]
        c     = [mode_color_map.get(m, "#adb5bd") for m in modes]
        pie_charts_js += f"""
  new Chart(document.getElementById('pie_{safe}'), {{
    type: 'pie',
    data: {{
      labels: {json.dumps([f"{m} ({p}%)" for m, p in zip(modes, pcts)])},
      datasets: [{{ data: {json.dumps(vals)}, backgroundColor: {json.dumps(c)}, borderWidth: 1 }}]
    }},
    options: {{
      plugins: {{
        legend: {{ position: 'right', labels: {{ font: {{ size: 11 }} }} }},
        title: {{ display: true, text: '{label}', font: {{ size: 13, weight: 'bold' }} }}
      }}
    }}
  }});
"""

    # Per-trace table rows HTML
    family_badge = {
        "FC1": '<span class="badge fc1">FC1</span>',
        "FC2": '<span class="badge fc2">FC2</span>',
        "FC3": '<span class="badge fc3">FC3</span>',
    }

    table_rows_html = ""
    for row in rows:
        gt_mode   = row["gt_mode"]
        gt_family = row["gt_family"]
        all_modes = row["all_modes"]
        fb        = family_badge.get(gt_family, f'<span class="badge" style="background:#adb5bd">{gt_family}</span>')
        table_rows_html += f'<tr><td class="tid">{row["tid"]}</td><td>{fb} {gt_mode}</td>'
        for label in labels:
            pred   = row["preds"][label]
            pm, pf = pred["mode"], pred["family"]
            strict_m, lenient_m = mode_match(pm, gt_mode, all_modes)
            strict_f, _ = family_match(pf, gt_family, all_modes)
            if strict_m:
                cls = "correct"
            elif lenient_m:
                cls = "lenient"
            elif strict_f:
                cls = "family-only"
            else:
                cls = "wrong"
            # Pipe-separated judge outputs: show first value, full on hover
            pm_display = pm.split("|")[0].strip() if "|" in pm else pm
            pf_display = pf.split("|")[0].strip() if "|" in pf else pf
            multi = " ⋯" if "|" in pm else ""
            title = f' title="{pm}"' if "|" in pm else ""
            pfb = family_badge.get(pf_display, f'<span class="badge" style="border-color:#adb5bd;color:#adb5bd">{pf_display}</span>')
            table_rows_html += f'<td class="{cls}"{title}>{pfb} {pm_display}{multi}</td>'
        table_rows_html += "</tr>\n"

    # Summary metrics table
    metric_table_html = "<tr><th>Metric</th>" + "".join(f"<th>{l}</th>" for l in labels) + "</tr>\n"
    for mk in metric_keys:
        vals_row = [metrics_by_config[l][mk] for l in labels]
        best     = max(vals_row)
        metric_table_html += f"<tr><td><strong>{mk}</strong></td>"
        for v in vals_row:
            cls = "best-cell" if v == best else ""
            metric_table_html += f'<td class="{cls}">{v}%</td>'
        metric_table_html += "</tr>\n"

    pie_canvases = "".join(
        f'<div class="pie-wrap"><canvas id="pie_{l.replace(" ","_").replace("(","").replace(")","")}" width="380" height="260"></canvas></div>'
        for l in labels
    )

    col_headers = "".join(f"<th>{l}</th>" for l in labels)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Spec-Based FL, Ablation Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #f5f7fa; color: #212529; }}
  header {{ background: #1a1a2e; color: #fff; padding: 20px 32px; }}
  header h1 {{ font-size: 1.4rem; font-weight: 700; }}
  header p  {{ font-size: 0.85rem; color: #adb5bd; margin-top: 4px; }}
  main {{ max-width: 1300px; margin: 0 auto; padding: 28px 24px; }}
  section {{ background: #fff; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,.08);
             padding: 24px; margin-bottom: 28px; }}
  h2 {{ font-size: 1.05rem; font-weight: 700; margin-bottom: 18px; color: #1a1a2e; border-left: 4px solid #0d6efd; padding-left: 10px; }}
  .bar-wrap {{ position: relative; height: 340px; }}
  .pie-grid {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }}
  .pie-wrap {{ background: #f8f9fa; border-radius: 8px; padding: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.84rem; }}
  th {{ background: #1a1a2e; color: #fff; padding: 9px 12px; text-align: left; white-space: nowrap; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #dee2e6; vertical-align: top; }}
  tr:hover td {{ background: #f0f4ff; }}
  .tid {{ font-family: monospace; font-size: 0.78rem; color: #495057; }}
  .correct       {{ background: #00c04b; color: #fff; font-weight: 600; }}
  .lenient       {{ background: #f0a500; color: #fff; font-weight: 600; }}
  .family-only   {{ background: #7b4fff; color: #fff; font-weight: 600; }}
  .wrong         {{ background: #e8000d; color: #fff; font-weight: 600; }}
  .best-cell {{ background: #d1e7dd; font-weight: 700; }}
  .badge {{ display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 0.72rem; font-weight: 800; background: rgba(255,255,255,0.92); border: 2px solid; }}
  .fc1 {{ color: #c0000a; border-color: #c0000a; }}
  .fc2 {{ color: #b85c00; border-color: #b85c00; }}
  .fc3 {{ color: #0040cc; border-color: #0040cc; }}
  .legend {{ display: flex; gap: 20px; font-size: 0.8rem; margin-bottom: 14px; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
  .dot {{ width: 12px; height: 12px; border-radius: 2px; display: inline-block; }}
</style>
</head>
<body>
<header>
  <h1>Specification-Based Fault Localization, Ablation Dashboard</h1>
  <p>{DASHBOARD_TITLE} · Comparing 4 pipeline configurations</p>
</header>
<main>

<section>
  <h2>Accuracy Metrics by Configuration</h2>
  <div class="bar-wrap"><canvas id="barChart"></canvas></div>
</section>

<section>
  <h2>Summary Table</h2>
  <table>
    {metric_table_html}
  </table>
</section>

<section>
  <h2>Predicted Failure Mode Distribution</h2>
  <div class="pie-grid">{pie_canvases}</div>
</section>

<section>
  <h2>Per-Trace Predictions vs Ground Truth</h2>
  <div class="legend">
    <span><span class="dot" style="background:#00c04b"></span> Strict mode match</span>
    <span><span class="dot" style="background:#f0a500"></span> Lenient mode match (in co-labels)</span>
    <span><span class="dot" style="background:#7b4fff"></span> Family correct, mode wrong</span>
    <span><span class="dot" style="background:#e8000d"></span> Completely wrong</span>
  </div>
  <div style="overflow-x:auto">
  <table>
    <tr><th>Trace</th><th>Ground Truth</th>{col_headers}</tr>
    {table_rows_html}
  </table>
  </div>
</section>

</main>
<script>
// Bar chart
new Chart(document.getElementById('barChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(labels)},
    datasets: {json.dumps(bar_datasets)}
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'top' }},
      tooltip: {{ callbacks: {{ label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y + '%' }} }}
    }},
    scales: {{
      y: {{
        beginAtZero: true,
        max: 100,
        ticks: {{ callback: v => v + '%' }},
        title: {{ display: true, text: 'Accuracy (%)' }}
      }}
    }}
  }}
}});

// Pie charts
{pie_charts_js}
</script>
</body>
</html>
"""


# ── HTTP server ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    html = ""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(Handler.html.encode("utf-8"))

    def log_message(self, fmt, *args):
        pass  # suppress per-request noise


def main():
    data = load_results()
    if not data:
        print("No result files could be loaded. Check the results/ directory.")
        sys.exit(1)

    Handler.html = build_html(data)
    port = ARGS.port
    server = HTTPServer(("", port), Handler)
    print(f"Dashboard running at http://localhost:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
