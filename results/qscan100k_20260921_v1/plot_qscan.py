#!/usr/bin/env python3
"""Render frozen Q-scan aggregates without loading a model or original dataset.

Usage: python plot_qscan.py --input-dir . --output-dir figures
The output directory must not exist. PNGs are 400 dpi; SVG text is editable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

WIDTH = 184 / 25.4
INK = "#23313B"
BLUE = "#24618A"
EMPTY = "#D9DDE1"
PROXY_NOTE = "FROZEN FORWARD PROXY — NOT FRESH EMX"
PANELS = {
    "lp_ls": (20, 20, (0.5, 3.0), (0.5, 3.0), r"Target $L_p$ (nH)", r"Target $L_s$ (nH)"),
    "lp_k": (20, 20, (0.5, 3.0), (0.0, 0.8), r"Target $L_p$ (nH)", r"Target $|k|$"),
    "ls_k": (20, 20, (0.5, 3.0), (0.0, 0.8), r"Target $L_s$ (nH)", r"Target $|k|$"),
    "q_selected_k": (11, 20, (9.5, 20.5), (0.0, 0.8), r"Selected target $Q$", r"Target $|k|$"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integer(value, label):
    result = int(value)
    if float(value) != result:
        raise ValueError(f"{label} is not integral: {value}")
    return result


def flag(value):
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0"}:
        return False
    raise ValueError(f"Invalid boolean: {value!r}")


def read_sources(root):
    summary_path = root / "SUMMARY.json"
    bins_path = root / "hotspot_bins.csv"
    missing = [str(p) for p in [summary_path, bins_path] if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required frozen inputs are absent: " + ", ".join(missing))
    summary = json.loads(summary_path.read_text())
    n = integer(summary["n_requests"], "n_requests")
    if n != 10000:
        raise ValueError(f"Expected frozen T10000 denominator, got {n}")
    if integer(summary["n_candidates"], "n_candidates") != n * 11:
        raise ValueError("Expected 11 Q candidates per request")
    failed = integer(summary["n_failed_requests"], "n_failed_requests")
    if not 0 <= failed <= n:
        raise ValueError("Invalid failure count")
    for required in ["model_id", "selection_score", "weight_sha256", "target_sha256"]:
        if not summary.get(required):
            raise ValueError(f"Missing summary field: {required}")
    for required in ["weight_sha256", "target_sha256"]:
        if len(str(summary[required])) != 64:
            raise ValueError(f"Malformed {required}")
    joint = {t: integer(summary["joint_hits"][str(t)], f"joint{t}") for t in [5, 10, 15, 20]}
    if not 0 <= joint[5] <= joint[10] <= joint[15] <= joint[20] <= n - failed:
        raise ValueError("Joint hit counts are inconsistent")
    features = {k: integer(summary["feature_hits_10pct"][k], k) for k in ["lp_nh", "ls_nh", "q_scalar", "k_abs"]}
    if any(not joint[10] <= value <= n for value in features.values()):
        raise ValueError("Feature counts do not reconcile with joint hits")
    rows = list(csv.DictReader(bins_path.open(newline="")))
    if set(row["projection"] for row in rows) != set(PANELS):
        raise ValueError("Unexpected projection set")
    matrices = {}
    for key, (nx, ny, xspan, yspan, _, _) in PANELS.items():
        selected = [row for row in rows if row["projection"] == key]
        if len(selected) != nx * ny:
            raise ValueError(f"{key}: expected every structural bin")
        n_total = np.zeros((ny, nx), dtype=int)
        n_hit = np.zeros_like(n_total)
        seen = set()
        xe = np.linspace(*xspan, nx + 1)
        ye = np.linspace(*yspan, ny + 1)
        for row in selected:
            ix, iy = integer(row["bin_x"], "bin_x"), integer(row["bin_y"], "bin_y")
            if (ix, iy) in seen or not (0 <= ix < nx and 0 <= iy < ny):
                raise ValueError(f"{key}: duplicate/out-of-range bin")
            seen.add((ix, iy))
            for field, expected in [("x_lower", xe[ix]), ("x_upper", xe[ix + 1]), ("y_lower", ye[iy]), ("y_upper", ye[iy + 1])]:
                if not math.isclose(float(row[field]), expected, abs_tol=1e-10, rel_tol=1e-9):
                    raise ValueError(f"{key}: wrong {field}")
            count = integer(row["n_total"], "n_total")
            hit = integer(row["n_hit"], "n_hit")
            if not 0 <= hit <= count:
                raise ValueError(f"{key}: invalid counts")
            if flag(row["empty"]) != (count == 0):
                raise ValueError(f"{key}: wrong empty flag")
            if flag(row["low_N"]) != (0 < count < 10):
                raise ValueError(f"{key}: wrong low_N flag")
            if count and not math.isclose(float(row["hit_fraction"]), hit / count, abs_tol=1e-9):
                raise ValueError(f"{key}: incorrect hit fraction")
            n_total[iy, ix], n_hit[iy, ix] = count, hit
        expected_total = n
        if key == "q_selected_k" and n_total.sum() == n - failed:
            expected_total = n - failed
        if n_total.sum() != expected_total or n_hit.sum() != joint[10]:
            raise ValueError(f"{key}: totals do not reconcile with SUMMARY")
        matrices[key] = dict(count=n_total, hit=n_hit, xe=xe, ye=ye)
    return summary, joint, features, matrices, {p.name: sha(p) for p in [summary_path, bins_path]}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.labelsize": 8.5, "axes.titlesize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.65, "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def save(fig, out, name):
    # Deliberately keep the physical canvas width; no bbox_inches='tight'.
    for ext in ["svg", "png"]:
        fig.savefig(out / f"{name}.{ext}", dpi=400, metadata={"Creator": "Frozen Q-scan aggregate renderer"} if ext == "svg" else None)
    plt.close(fig)


def map_panel(ax, matrix, key, *, counts=False, vmax=100):
    _, _, xspan, yspan, xlabel, ylabel = PANELS[key]
    count, hit = matrix["count"], matrix["hit"]
    vals = count.astype(float) if counts else np.divide(100 * hit, count, out=np.zeros_like(count, dtype=float), where=count > 0)
    cmap = plt.get_cmap("cividis" if counts else "viridis").copy()
    cmap.set_bad(EMPTY)
    mesh = ax.pcolormesh(matrix["xe"], matrix["ye"], np.ma.masked_where(count == 0, vals), cmap=cmap, norm=Normalize(0, vmax), shading="flat", rasterized=False)
    sparse = (count > 0) & (count < 10)
    xc = (matrix["xe"][:-1] + matrix["xe"][1:]) / 2
    yc = (matrix["ye"][:-1] + matrix["ye"][1:]) / 2
    xx, yy = np.meshgrid(xc, yc)
    ax.scatter(xx[sparse], yy[sparse], marker=".", s=6, c="#171717", linewidths=0)
    ax.set(xlim=xspan, ylim=yspan, xlabel=xlabel, ylabel=ylabel)
    ax.set_box_aspect(1)
    if key == "q_selected_k":
        ax.set_xticks([10, 12, 14, 16, 18, 20])
    else:
        ax.set_xticks([0.5, 1.5, 2.5, 3.0])
    ax.set_yticks([0.5, 1.5, 2.5, 3.0] if key == "lp_ls" else [0.0, 0.2, 0.4, 0.6, 0.8])
    return mesh


def map_legend(fig, y):
    handles = [Patch(facecolor=EMPTY, edgecolor="none", label="Empty bin (N = 0)"), Line2D([0], [0], color="#171717", marker=".", ls="none", markersize=4, label="Low count (0 < N < 10)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=2, frameon=False, fontsize=7.8, handlelength=1.3, columnspacing=2)


def target_maps(out, summary, matrices, counts=False):
    fig = plt.figure(figsize=(WIDTH, 84 / 25.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.055], left=0.075, right=0.925, top=0.78, bottom=0.26, wspace=0.43)
    keys = ["lp_ls", "lp_k", "ls_k"]
    vmax = max(int(matrices[key]["count"].max()) for key in keys) if counts else 100
    for i, key in enumerate(keys):
        ax = fig.add_subplot(gs[0, i])
        mesh = map_panel(ax, matrices[key], key, counts=counts, vmax=vmax)
        ax.set_title([r"(a) $L_p$–$L_s$", r"(b) $L_p$–$|k|$", r"(c) $L_s$–$|k|$"][i], loc="left", pad=7)
    cb = fig.colorbar(mesh, cax=fig.add_subplot(gs[0, 3]))
    cb.ax.tick_params(labelsize=8)
    cb.set_label("Requests per bin" if counts else "Joint ±10% hit rate (%)", fontsize=8.5)
    if not counts:
        cb.set_ticks([0, 25, 50, 75, 100])
    fig.text(0.075, 0.94, "Q-scan 100k: target sampling counts" if counts else "Q-scan 100k: target-space joint hit rates", fontsize=11, weight="bold", va="top")
    fig.text(0.075, 0.865, "10,000 fixed requests · Q = 10…20 · one selected candidate per request", fontsize=8.4, va="top")
    map_legend(fig, 0.11)
    fig.text(0.5, 0.055, PROXY_NOTE, fontsize=7.5, weight="bold", ha="center")
    save(fig, out, "Qscan100k_target_counts" if counts else "Qscan100k_target_hotspots")


def selected_q_map(out, summary, matrices):
    fig = plt.figure(figsize=(WIDTH, 96 / 25.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.045, 1, 0.045], left=0.08, right=0.935, top=0.79, bottom=0.26, wspace=0.34)
    matrix = matrices["q_selected_k"]
    for i, counts in enumerate([False, True]):
        ax = fig.add_subplot(gs[0, i * 2])
        mesh = map_panel(ax, matrix, "q_selected_k", counts=counts, vmax=max(1, int(matrix["count"].max())) if counts else 100)
        if counts:
            # The paired maps share their y coordinate; avoid a repeated ylabel
            # colliding with the first panel's adjacent colorbar.
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title("(b) Selected-request count" if counts else "(a) Joint ±10% hit rate", loc="left", pad=7)
        cb = fig.colorbar(mesh, cax=fig.add_subplot(gs[0, i * 2 + 1]))
        cb.set_label("Requests per bin" if counts else "Hit rate (%)", fontsize=8.5)
        cb.ax.tick_params(labelsize=8)
        if not counts:
            cb.set_ticks([0, 25, 50, 75, 100])
    fig.text(0.08, 0.945, "Q-scan 100k: selected Q and target coupling", fontsize=11, weight="bold", va="top")
    fig.text(0.08, 0.877, "Selected target Q is chosen by the model; it is not independently sampled.", fontsize=8.3, va="top")
    map_legend(fig, 0.115)
    unassigned = summary["n_requests"] - int(matrix["count"].sum())
    fig.text(0.5, 0.075, f"Original denominator: 10,000 requests · failed/unassigned Q: {unassigned:,}", ha="center", fontsize=7.8)
    fig.text(0.5, 0.032, PROXY_NOTE, fontsize=7.5, weight="bold", ha="center")
    save(fig, out, "SelectedQ_K_hotspot")


def performance(out, summary, joint, features):
    n = summary["n_requests"]
    fig = plt.figure(figsize=(WIDTH, 100 / 25.4))
    fig.text(0.075, 0.94, "Q-scan 100k: selected-candidate performance", fontsize=11, weight="bold", va="top")
    fig.text(0.075, 0.874, "10,000 fixed requests · 110,000 candidates · Q = 10…20", fontsize=8.5, va="top")
    fig.text(0.075, 0.715, f"{100 * joint[10] / n:.2f}%", fontsize=32, weight="bold", color=BLUE)
    fig.text(0.075, 0.64, "Joint ±10% hit rate", fontsize=10, weight="bold")
    fig.text(0.075, 0.581, f"{joint[10]:,} / {n:,} requests", fontsize=9)
    for y, t in zip([0.46, 0.388, 0.316], [5, 15, 20]):
        fig.text(0.075, y, f"Joint ±{t}%", fontsize=8.7)
        fig.text(0.367, y, f"{100 * joint[t] / n:.2f}%", fontsize=10, weight="bold", ha="right")
    fig.text(0.075, 0.229, f"Failed requests retained: {summary['n_failed_requests']:,}", fontsize=8)
    ax = fig.add_axes([0.51, 0.305, 0.4, 0.44])
    labels = [r"$L_p$", r"$L_s$", r"$Q_{\mathrm{scalar}}$", r"$|k|$"]
    values = np.array(list(features.values())) / n * 100
    y = np.arange(4)
    ax.barh(y, np.full(4, 100), color="#EEF1F3", height=0.5, zorder=1)
    ax.barh(y, values, color=BLUE, height=0.5, zorder=2)
    for ypos, value in zip(y, values):
        if value >= 30:
            ax.text(value - 2, ypos, f"{value:.2f}%", ha="right", va="center", color="white", fontsize=9, weight="bold")
        else:
            ax.text(value + 2, ypos, f"{value:.2f}%", ha="left", va="center", color=INK, fontsize=9, weight="bold")
    ax.set(yticks=y, yticklabels=labels, xlim=(0, 100), ylim=(3.55, -0.55), xlabel="Feature ±10% hit rate (%)")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_title("Individual feature hit rates", loc="left", fontsize=9.5, pad=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.grid(axis="x", color="#DDE2E5", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.text(0.075, 0.147, "Symmetric target-relative errors for all four features, including Q.", fontsize=8)
    fig.text(0.075, 0.098, "Candidate selection uses a different, fixed-span normalized error score.", fontsize=8)
    fig.text(0.5, 0.043, PROXY_NOTE, fontsize=7.5, weight="bold", ha="center")
    save(fig, out, "Qscan100k_performance_summary")


def captions(summary, joint, hashes, matrices):
    n = summary["n_requests"]
    score = json.dumps(summary["selection_score"], ensure_ascii=False, sort_keys=True)
    unassigned = n - int(matrices["q_selected_k"]["count"].sum())
    return f"""# Q-scan 100k figure captions and provenance

## Shared evaluation contract

These figures describe the fixed finite frame of {n:,} three-input requests (Lp, Ls, |k|), each evaluated with integer target Q = 10, 11, ..., 20 ({summary['n_candidates']:,} candidate evaluations). One candidate is selected per request using the frozen score stated in SUMMARY.json: `{score}`. Selection by fixed-span normalized error and the subsequent target-relative tolerance hit statistics are different operations and must not be identified as the same error measure. A joint hit at tolerance t requires every feature to satisfy abs(prediction - target) / abs(target) <= t, without a denominator floor. Q is evaluated symmetrically around the selected target Q; excessive Q can therefore miss. The scalar physical Q label is Qmin = min(Qp, Qs), which does not make the present exact-Q request a minimum-Q requirement. Target-relative |k| errors become sensitive near zero (the minimum target |k| in this frame is 0.01545946987287989).

The original model is `{summary['model_id']}`. Its historical source-table / gradient-training / validation / test row counts are 100,000 / 78,891 / 9,740 / 11,369. The historical seed20260713 training used a minimum-Q objective and a +2 training guardband; the present symmetric-Q scanning and evaluation do not relabel or retrain that historical objective, and no extra training guardband is added at inference. No causal training-data-size improvement is claimed. Every response used in these figures is a frozen-forward surrogate prediction. **FROZEN FORWARD PROXY, NOT FRESH EMX.** The figures do not establish manufacturability or physical accuracy.

The {n:,}-request denominator retains all {summary['n_failed_requests']:,} failed requests as joint misses. These are descriptive finite-frame rates; no i.i.d. deployment-population confidence intervals are claimed. The only variation displayed in the maps is target position (or the selected-Q stratum); there is no experimentally varied treatment. Model weights, inference/selection rule, target frame, error definition, and bin edges are fixed. Low-count markings communicate local support, not statistical significance.

## Qscan100k_target_hotspots

Joint ±10% symmetric target-relative hit rates after Q scanning, projected onto (a) target Lp–Ls, (b) target Lp–|k|, and (c) target Ls–|k|. Each panel aggregates all {n:,} original requests with the same {joint[10]:,} joint hits ({100 * joint[10] / n:.2f}%). Fixed bins use 20 subdivisions per axis, Lp and Ls in [0.5, 3.0] nH and |k| in [0, 0.8]. Color shows the count-weighted bin rate on a shared 0–100% scale. Gray denotes an empty bin; a dot denotes 0 < N < 10. The omitted target dimension and selected Q are marginalized, so local projected rates do not define a four-dimensional feasibility boundary. Failed requests remain in the target-bin denominator.

## Qscan100k_target_counts

Numbers of original fixed-frame requests in the same three target projections and identical bins. All panels share one linear count scale and each sums to {n:,}. Gray denotes N = 0; dots denote 0 < N < 10. This companion figure exposes sampling support behind the hit-rate map; it is not a second performance measure.

## SelectedQ_K_hotspot

(a) Joint ±10% symmetric target-relative hit rate and (b) request count against model-selected target Q and the original target |k|. Q is discrete, using 11 bins centered at integers 10 through 20 with boundaries 9.5 through 20.5; |k| uses the same 20 fixed bins in [0, 0.8]. Selected Q is an outcome of the fixed candidate-ranking procedure, not an independently random input. The strata therefore describe selection-conditioned support and performance; differences cannot be interpreted as a causal Q effect. {unassigned:,} requests have no assignable selected Q and are absent from this projection, while all remain in the overall {n:,}-request performance denominator. Empty and low-count encodings match the target maps.

## Qscan100k_performance_summary

Selected-candidate performance on all {n:,} requests. The large number is the joint ±10% hit rate ({joint[10]:,}/{n:,}); individual bars show the four feature-wise ±10% rates using the same original denominator. Secondary joint rates use ±5%, ±15%, and ±20% tolerance. These are intersected four-feature hit counts, not mean error or 1-minus-loss. Both Q overshoot and shortfall are penalized in this evaluation. The fixed-span selection score is not the relative-error statistic shown here.

## Reproduction and source identity

Run `python plot_qscan.py --input-dir INPUT_DIR --output-dir NEW_FIGURE_DIR` in an environment with NumPy and Matplotlib. NEW_FIGURE_DIR must not exist. Only SUMMARY.json and hotspot_bins.csv are consumed; no model loading, training, inference, data generation, network access, or EMX is performed. Exports are 184 mm wide, SVG with editable text and PNG at 400 dpi. Each map's rows and derived matrices are checked against the input summary before drawing. Source code and output hashes are recorded in RENDER_RECEIPT.json and SHA256SUMS.

- SUMMARY.json SHA-256: `{hashes['SUMMARY.json']}`
- hotspot_bins.csv SHA-256: `{hashes['hotspot_bins.csv']}`
- Frozen model weights SHA-256: `{summary['weight_sha256']}`
- Frozen target SHA-256: `{summary['target_sha256']}`
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure", choices=["all", "selected-q"], default="all")
    args = parser.parse_args()
    root = args.input_dir.resolve()
    out = (args.output_dir or root / "figures").resolve()
    summary, joint, features, matrices, hashes = read_sources(root)
    out.mkdir(parents=True, exist_ok=False)
    try:
        style()
        if args.figure == "all":
            target_maps(out, summary, matrices)
            target_maps(out, summary, matrices, counts=True)
        selected_q_map(out, summary, matrices)
        if args.figure == "all":
            performance(out, summary, joint, features)
        (out / "CAPTIONS_EN.md").write_text(captions(summary, joint, hashes, matrices))
        outputs = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
        receipt = {
            "status": "RENDER_COMPLETE", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "inputs_sha256": hashes, "script_sha256": sha(Path(__file__)),
            "model_id": summary["model_id"], "model_weights_sha256": summary["weight_sha256"],
            "target_sha256": summary["target_sha256"], "n_requests": summary["n_requests"],
            "n_candidates": summary["n_candidates"], "n_failed_requests": summary["n_failed_requests"],
            "joint_hits": joint, "feature_hits_10pct": features,
            "png_dpi": 400, "width_mm": 184, "matplotlib_version": matplotlib.__version__,
            "numpy_version": np.__version__, "evidence_class": "FROZEN_FORWARD_PROXY_NOT_FRESH_EMX",
            "source_checks": "all bins, boundaries, flags, rates, summary counts, and original denominators reconciled",
            "visual_review": "PENDING_EXTERNAL_INSPECTION", "output_sha256": outputs,
        }
        (out / "RENDER_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
        files = sorted(p for p in out.iterdir() if p.is_file())
        (out / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.name}\n" for p in files))
        print(json.dumps({"status": receipt["status"], "output_dir": str(out), "n_figures": 4 if args.figure == "all" else 1, "n_requests": summary["n_requests"]}))
    except Exception as exc:
        (out / "FAILURE.json").write_text(json.dumps({"status": "FAIL", "error": repr(exc), "timestamp_utc": datetime.now(timezone.utc).isoformat(), "inputs_sha256": hashes}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
