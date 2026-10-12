#!/usr/bin/env python3
"""Terminal-gated legacy10k-recipe D100k evaluation; never trains or runs EM.

Callable: postprocess(run_dir, config). CLI: --run-dir PATH --config JSON.
Required config bindings {path, sha256}: loader_path, target_10000_csv,
target_8000_csv. common_test_csv is optional, but a supplied binding must pass.
The caller invokes this once after releasing the completed training run lock.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np

MODEL_ID = "legacyrecipe_D100k_seed20260711"
FEATURES = ["lp_nh", "ls_nh", "q_scalar", "k_abs"]
SPANS = np.array([2.5, 2.5, 20., .8])
Q = np.arange(10, 21)
SCORE = "sqrt(mean(((proxy-target)/[2.5,2.5,20,0.8])**2)); symmetric Q; exact ties choose lower Q"
INPUTS = ["input__lp_nh_center", "input__ls_nh_center", "input__q_center", "input__k_abs_center"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def json_write(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def csv_write(path, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def pin_check(binding):
    if not {"path", "sha256"} <= set(binding):
        raise ValueError("Binding requires path and sha256")
    path = Path(binding["path"]).expanduser().resolve()
    if sha(path) != binding["sha256"]:
        raise ValueError("SHA-256 mismatch: " + str(path))
    return path


def rows_read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_targets(path, expected):
    rows = rows_read(path)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} frozen rows, got {len(rows)}")
    ids = [r["target_id"] for r in rows]
    if len(set(ids)) != expected:
        raise ValueError("Frozen target IDs must be unique")
    names = (["target_lp_nh", "target_ls_nh", "target_k_abs"] if expected == 10000
             else ["target_Lp_nH", "target_Ls_nH", "target_K_abs"])
    triplets = np.array([[float(r[c]) for c in names] for r in rows])
    if not np.isfinite(triplets).all() or (triplets <= 0).any():
        raise ValueError("Targets must be finite and strictly positive")
    if any("frequency_ghz" in r and float(r["frequency_ghz"]) != 15 for r in rows):
        raise ValueError("Only frozen 15 GHz targets are in scope")
    return ids, triplets


def score_symmetric(proxy, targets):
    return np.sqrt(np.mean(((proxy - targets) / SPANS) ** 2, axis=-1))


def select_candidates(geometry, proxy, targets):
    scores = score_symmetric(proxy, targets)
    finite = np.isfinite(geometry).all(-1) & np.isfinite(proxy).all(-1) & np.isfinite(scores)
    chosen = np.argmin(np.where(finite, scores, np.inf), axis=1)
    return scores, finite, chosen


def batch_inverse(module, model, requests):
    norm = model.model["normalization"]
    raw = module._predict((requests - norm["x_mean"]) / norm["x_scale"], model.model["inverse_weights"], model.model["inverse_biases"])
    standardized = module._project_inverse_geometry(raw, norm["geometry_lower"], norm["geometry_upper"], mode="independent_sigmoid", normalization=norm, topology_contract=model.model["topology_contract"])
    geometry = standardized * norm["y_scale"] + norm["y_mean"]
    proxy = module._predict(standardized, model.model["forward_weights"], model.model["forward_biases"]) * norm["x_scale"] + norm["x_mean"]
    return geometry, proxy


def load_model(run, loader_path, summary_path, weights_path):
    summary = json.loads(summary_path.read_text())
    a = summary["arguments"]
    checks = {
        "seed": int(a["seed"]) == 20260711,
        "independent_sigmoid": a["inverse_geometry_projection"] == "independent_sigmoid",
        "exact_Q": a["q_target_semantics"] == "exact",
        "no_Q_margin": float(a.get("q_minimum_margin_physical", 0)) == 0,
        "anchor_0p01": math.isclose(float(a["geometry_anchor_weight"]), .01, rel_tol=0, abs_tol=1e-12),
        "mse_loss": a["response_loss_family"] == "mse",
        "declared_range_scaling": a["response_loss_scaling"] == "declared_range",
        "topology_weight_zero": float(a["topology_feasibility_weight"]) == 0,
    }
    if not all(checks.values()):
        raise ValueError("Recipe mismatch: " + repr(checks))
    manifest = {
        "schema": "rfic_frozen_inverse_mlp_bundle_v1", "model_id": MODEL_ID,
        "model_seed": 20260711, "target_frequency_ghz": 15., "q_target_semantics": "exact",
        "architecture": {"forward_surrogate": [10, 256, 256, 256, 4], "inverse_mlp": [4, 256, 256, 256, 10]},
        "artifacts": {"model_summary": {"path": summary_path.name, "sha256": sha(summary_path)}, "model_weights": {"path": weights_path.name, "sha256": sha(weights_path)}},
        "scientific_boundary": "Private new legacy-recipe D100k model. SELF_PROXY is not physical accuracy or fresh EMX.",
    }
    manifest_path = run / "model_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing private manifest conflicts; do not overwrite")
    else:
        json_write(manifest_path, manifest)
    spec = importlib.util.spec_from_file_location("legacyrecipe_frozen_loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.FrozenInverseModel.load(run)
    if model.model["projection_mode"] != "independent_sigmoid":
        raise ValueError("Actual weight decoder is not independent_sigmoid")
    checks["actual_forward_layers"] = list(module._layer_widths(model.model["forward_weights"])) == [10, 256, 256, 256, 4]
    checks["actual_inverse_layers"] = list(module._layer_widths(model.model["inverse_weights"])) == [4, 256, 256, 256, 10]
    if not all(checks.values()):
        raise ValueError("Loaded model architecture mismatch")
    return module, model, checks


def grid_rows(target, hit):
    projections = [
        ("lp_ls", 0, 1, np.linspace(.5, 3, 21), np.linspace(.5, 3, 21), "Target Lp (nH)", "Target Ls (nH)"),
        ("lp_k", 0, 3, np.linspace(.5, 3, 21), np.linspace(0, .8, 21), "Target Lp (nH)", "Target |k|"),
        ("ls_k", 1, 3, np.linspace(.5, 3, 21), np.linspace(0, .8, 21), "Target Ls (nH)", "Target |k|"),
        ("q_selected_k", 2, 3, np.arange(9.5, 21.5), np.linspace(0, .8, 21), "Selected target Q", "Target |k|"),
    ]
    rows, matrices = [], {}
    for name, xi, yi, xe, ye, xl, yl in projections:
        counts = np.histogram2d(target[:, xi], target[:, yi], bins=(xe, ye))[0].astype(int)
        hits = np.histogram2d(target[hit, xi], target[hit, yi], bins=(xe, ye))[0].astype(int)
        if counts.sum() != len(target) or hits.sum() != hit.sum():
            raise ValueError("Fixed target bin bounds dropped rows: " + name)
        matrices[name] = (xe, ye, counts.T, hits.T, xl, yl)
        for ix in range(len(xe)-1):
            for iy in range(len(ye)-1):
                count, yes = int(counts[ix, iy]), int(hits[ix, iy])
                rows.append(dict(projection=name, x_label=xl, y_label=yl, bin_x=ix, bin_y=iy, x_lower=xe[ix], x_upper=xe[ix+1], y_lower=ye[iy], y_upper=ye[iy+1], n_total=count, n_hit=yes, hit_fraction=yes/count if count else "", empty=count == 0, low_N=0 < count < 10))
    return rows, matrices


def finite_number(x):
    return float(x) if np.isfinite(x) else None


def errors_table(truth, prediction, valid):
    result = []
    residual = prediction-truth
    relative = np.divide(abs(residual), abs(truth), out=np.full_like(residual, np.nan), where=truth != 0)
    for j, feature in enumerate(FEATURES):
        finite = np.isfinite(residual[:, j])
        values = residual[finite, j]
        rel = relative[np.isfinite(relative[:, j]), j]
        result.append(dict(feature=feature, unit="nH" if j < 2 else "dimensionless", original_denominator=len(truth), n_finite=int(finite.sum()), MAE=finite_number(np.mean(abs(values))) if len(values) else None, RMSE=finite_number(np.sqrt(np.mean(values**2))) if len(values) else None, P95_absolute_error=finite_number(np.percentile(abs(values), 95)) if len(values) else None, mean_relative_error_pct=finite_number(100*np.mean(rel)) if len(rel) else None, P95_relative_error_pct=finite_number(100*np.percentile(rel, 95)) if len(rel) else None, hits_10pct=int((valid & (relative[:, j] <= .1)).sum())))
    return result, relative


def plot_results(out, frame, summary, matrices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5, "svg.fonttype": "none", "axes.linewidth": .65})
    title = f"Legacy-recipe D100k, seed11 | {frame}"
    footer = "SELF_PROXY — NOT PHYSICAL ACCURACY / FRESH EMX"
    def draw(ax, matrix, counts=False, vmax=100):
        xe, ye, n, hit, xl, yl = matrix
        value = n if counts else np.divide(100*hit, n, out=np.zeros_like(n, dtype=float), where=n > 0)
        cmap = plt.get_cmap("cividis" if counts else "viridis").copy()
        cmap.set_bad("#d9dde1")
        m = ax.pcolormesh(xe, ye, np.ma.masked_where(n == 0, value), cmap=cmap, vmin=0, vmax=vmax, shading="flat")
        xx, yy = np.meshgrid((xe[:-1]+xe[1:])/2, (ye[:-1]+ye[1:])/2)
        low = (n > 0) & (n < 10)
        ax.scatter(xx[low], yy[low], s=3, c="black", marker=".")
        ax.set(xlabel=xl, ylabel=yl)
        return m
    def save(fig, name):
        fig.savefig(out / (name+".png"), dpi=300)
        fig.savefig(out / (name+".svg"))
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(184/25.4, 125/25.4))
    fig.subplots_adjust(left=.09, right=.89, bottom=.15, top=.84, wspace=.47, hspace=.56)
    count_max = max(matrices[k][2].max() for k in ["lp_ls", "lp_k", "ls_k"])
    for column, key in enumerate(["lp_ls", "lp_k", "ls_k"]):
        for row in [0, 1]:
            mesh = draw(axes[row, column], matrices[key], counts=row == 1, vmax=max(1, count_max) if row else 100)
            axes[row, column].set_title(("Hit rate" if row == 0 else "Request count") + " · " + key.replace("_", "–"), fontsize=8.5)
            if column == 2:
                cb = fig.colorbar(mesh, cax=fig.add_axes([.915, .565 if row == 0 else .15, .015, .27]))
                cb.set_label("Joint ±10% (%)" if row == 0 else "N per bin", fontsize=8)
    fig.suptitle(title, fontsize=11, y=.96)
    fig.text(.5, .905, f"N={summary['n_requests']:,}; {summary['joint_hits']['10']:,} joint ±10% hits; Q=10…20", ha="center", fontsize=8.5)
    fig.text(.5, .055, "Gray = empty; dot = 0 < N < 10. Fixed 20×20 bins; failures retained.", ha="center", fontsize=7.5)
    fig.text(.5, .025, footer, ha="center", fontsize=7.5, weight="bold")
    save(fig, "target_hotspots_and_counts")
    fig, axes = plt.subplots(1, 2, figsize=(184/25.4, 90/25.4))
    fig.subplots_adjust(left=.09, right=.86, top=.77, bottom=.25, wspace=.48)
    for i in [0, 1]:
        mesh = draw(axes[i], matrices["q_selected_k"], counts=bool(i), vmax=max(1, matrices["q_selected_k"][2].max()) if i else 100)
        if i:
            axes[i].set_ylabel("")
        cb = fig.colorbar(mesh, ax=axes[i], fraction=.045, pad=.04)
        cb.ax.tick_params(labelsize=7)
        axes[i].set_title("Selected-Q count" if i else "Joint ±10% (%)", fontsize=9)
    fig.suptitle(title + " | selected Q", fontsize=10.5, y=.96)
    fig.text(.5, .865, "Model-selected Q is a conditional stratum, not an independently sampled target.", ha="center", fontsize=8)
    fig.text(.5, .12, "Gray = empty; dot = 0 < N < 10. Q integer bins 10…20; |k| fixed [0, 0.8].", ha="center", fontsize=7.5)
    fig.text(.5, .055, footer, ha="center", fontsize=7.5, weight="bold")
    save(fig, "selected_Q_K_hotspot")
    fig, ax = plt.subplots(figsize=(184/25.4, 88/25.4))
    fig.subplots_adjust(left=.13, right=.94, top=.68, bottom=.27)
    vals = [100*summary["feature_hits_10pct"][k]/summary["n_requests"] for k in FEATURES]
    ax.barh(FEATURES, vals, color="#24618a")
    ax.invert_yaxis()
    ax.set(xlim=(0, 110), xlabel="Feature-wise symmetric ±10% hit rate (%)")
    ax.set_xticks([0, 25, 50, 75, 100])
    for i, value in enumerate(vals):
        ax.text(value+1, i, f"{value:.2f}%", va="center", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=11, y=.96)
    fig.text(.5, .795, f"Joint ±10%: {100*summary['joint_hits']['10']/summary['n_requests']:.2f}% ({summary['joint_hits']['10']:,}/{summary['n_requests']:,})", ha="center", fontsize=16, weight="bold", color="#24618a")
    fig.text(.5, .145, " · ".join(f"Joint ±{t}%: {100*summary['joint_hits'][str(t)]/summary['n_requests']:.2f}%" for t in [5, 15, 20]), ha="center", fontsize=8.5)
    fig.text(.5, .09, "Exact symmetric Q; fixed-span selection score differs from relative hit statistics.", ha="center", fontsize=7.5)
    fig.text(.5, .035, footer, ha="center", fontsize=7.5, weight="bold")
    save(fig, "performance_summary")


def evaluate_scan(module, model, source, expected, out, frame):
    ids, triplets = read_targets(source, expected)
    out.mkdir()
    targets = np.empty((expected, 11, 4))
    targets[:, :, 0] = triplets[:, 0, None]
    targets[:, :, 1] = triplets[:, 1, None]
    targets[:, :, 2] = Q[None, :]
    targets[:, :, 3] = triplets[:, 2, None]
    flat = targets.reshape(-1, 4)
    if not (np.all(flat >= model.support_lower) and np.all(flat <= model.support_upper)):
        raise ValueError("Frozen target frame is outside declared model support; no clipping")
    geometry = np.empty((len(flat), 10))
    proxy = np.empty((len(flat), 4))
    for start in range(0, len(flat), 2048):
        geometry[start:start+2048], proxy[start:start+2048] = batch_inverse(module, model, flat[start:start+2048])
    geometry = geometry.reshape(expected, 11, 10)
    proxy = proxy.reshape(expected, 11, 4)
    scores, finite, chosen = select_candidates(geometry, proxy, targets)
    index = np.arange(expected)
    st, sp, sg = targets[index, chosen], proxy[index, chosen], geometry[index, chosen]
    norm = model.model["normalization"]
    lo = norm["geometry_lower"]*norm["y_scale"]+norm["y_mean"]
    hi = norm["geometry_upper"]*norm["y_scale"]+norm["y_mean"]
    geometry_ok = np.isfinite(sg).all(1) & (sg >= lo-1e-9).all(1) & (sg <= hi+1e-9).all(1)
    valid = finite[index, chosen] & geometry_ok & (sp[:, :3] > 0).all(1) & (sp[:, 3] > 0) & (sp[:, 3] < 1)
    error_rows, relative = errors_table(st, sp, valid)
    hits = {str(t): valid & (relative <= t/100).all(1) for t in [5, 10, 15, 20]}
    with (out / "all_candidates.npz").open("xb") as f:
        np.savez_compressed(f, target_ids=np.asarray(ids), requests=targets, geometry=geometry, proxy=proxy, scores=scores, finite=finite, selected_index=chosen)
    selected = []
    for i, target_id in enumerate(ids):
        row = dict(target_id=target_id, model_id=MODEL_ID, selected_q=int(Q[chosen[i]]), status="COMPLETE" if finite[i, chosen[i]] else "FAILED_NO_FINITE_CANDIDATE", proxy_valid=bool(valid[i]), geometry_bounds_valid=bool(geometry_ok[i]), selection_nrmse=finite_number(scores[i, chosen[i]]), evidence="SELF_PROXY", fresh_EMX_status="NOT_RUN")
        for prefix, values in [("target", st[i]), ("proxy", sp[i]), ("signed_error", sp[i]-st[i]), ("relative_error_pct", 100*relative[i])]:
            row.update({prefix+"_"+name: finite_number(value) for name, value in zip(FEATURES, values)})
        row.update({name: finite_number(value) for name, value in zip(model.geometry_columns, sg[i])})
        row.update({"joint_hit_"+t+"pct": bool(hit[i]) for t, hit in hits.items()})
        selected.append(row)
    csv_write(out / "selected_results.csv", selected)
    csv_write(out / "physical_error_summary.csv", error_rows)
    csv_write(out / "selected_Q_counts.csv", [dict(q=int(q), n_selected=int((Q[chosen] == q).sum()), n_hit_10pct=int(((Q[chosen] == q) & hits["10"]).sum())) for q in Q])
    bins, matrices = grid_rows(st, hits["10"])
    csv_write(out / "hotspot_bins.csv", bins)
    summary = dict(status="COMPLETE", model_id=MODEL_ID, frame=frame, n_requests=expected, n_candidates=expected*11, n_finite_candidates=int(finite.sum()), n_failed_requests=int((~finite[index, chosen]).sum()), n_proxy_invalid_requests=int((~valid).sum()), joint_hits={t: int(v.sum()) for t, v in hits.items()}, feature_hits_10pct={r["feature"]: r["hits_10pct"] for r in error_rows}, selection_score=SCORE, target_sha256=sha(source), model_training_Q_semantics="exact symmetric; zero Q minimum margin", evaluation_Q_semantics="exact symmetric; no postprocessing guardband", physical_errors=error_rows, evidence="SELF_PROXY_NOT_PHYSICAL_ACCURACY", fresh_EMX="NOT_RUN", failed_request_Q_assignment="All-nonfinite requests retain Q=10 sentinel for bin bookkeeping; never counted as hits.")
    json_write(out / "SUMMARY.json", summary)
    plot_results(out, frame, summary, matrices)
    return summary


def evaluate_forward(module, model, source, out):
    data = rows_read(source)
    if len(data) != 11369:
        raise ValueError("Common test must retain exactly 11369 cached EM-label rows")
    identities = [row["source_row_index_0based"] for row in data]
    if len(set(identities)) != 11369:
        raise ValueError("Duplicate common-test source identity")
    geometry = np.array([[float(row[k]) for k in model.geometry_columns] for row in data])
    truth = np.array([[float(row[k]) for k in INPUTS] for row in data])
    if not np.isfinite(geometry).all() or not np.isfinite(truth).all():
        raise ValueError("Nonfinite original common-test cache")
    norm = model.model["normalization"]
    predictions = []
    for start in range(0, len(data), 2048):
        x = (geometry[start:start+2048]-norm["y_mean"])/norm["y_scale"]
        predictions.append(module._predict(x, model.model["forward_weights"], model.model["forward_biases"])*norm["x_scale"]+norm["x_mean"])
    prediction = np.vstack(predictions)
    valid = np.isfinite(prediction).all(1)
    rows = []
    for source_row, t, p, ok in zip(data, truth, prediction, valid):
        row = dict(source_row_index_0based=source_row["source_row_index_0based"], evaluation=source_row.get("evaluation", ""), touchstone_sha256=source_row.get("touchstone_sha256", ""), prediction_finite=bool(ok), evidence="EXISTING_REAL_EM_LABEL_HOLDOUT_NOT_FRESH_EM")
        row.update({"truth_"+k: float(v) for k, v in zip(FEATURES, t)})
        row.update({"prediction_"+k: finite_number(v) for k, v in zip(FEATURES, p)})
        rows.append(row)
    out.mkdir()
    csv_write(out / "predictions.csv", rows)
    errors, _ = errors_table(truth, prediction, valid)
    csv_write(out / "errors.csv", errors)
    result = dict(status="COMPLETE", n=11369, n_failed=int((~valid).sum()), source_sha256=sha(source), errors=errors, evidence="EXISTING_REAL_EM_LABEL_HOLDOUT_NOT_FRESH_EM", test_used_for_model_selection=False)
    json_write(out / "SUMMARY.json", result)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(184/25.4, 130/25.4), layout="constrained")
    for j, ax in enumerate(axes.flat):
        ax.scatter(truth[valid, j], prediction[valid, j], s=1, alpha=.18, color="#24618a", rasterized=True)
        lo, hi = np.min(truth[:, j]), np.max(truth[:, j])
        ax.plot([lo, hi], [lo, hi], color="#303030", ls="--", lw=.8)
        ax.set(xlabel="Cached EM label: "+FEATURES[j], ylabel="Forward prediction", title=f"MAE = {errors[j]['MAE']:.4g}" if errors[j]["MAE"] is not None else "No finite predictions")
    fig.suptitle("Legacy-recipe D100k seed11 | common 11,369 held-out EM labels\nExisting labels; no fresh EMX; test not used for selection", fontsize=10)
    for ext in ["png", "svg"]:
        fig.savefig(out / ("forward_scatter."+ext), dpi=300)
    plt.close(fig)
    return result


def postprocess(run_dir, config):
    run = Path(run_dir).expanduser().resolve()
    terminal_path = run / "ENTRY_TERMINAL.json"
    if not terminal_path.is_file():
        return {"status": "NOT_READY", "reason": "training terminal absent", "inference_calls": 0}
    terminal = json.loads(terminal_path.read_text())
    if terminal.get("status") != "COMPLETE":
        return {"status": "NOT_READY", "reason": "training terminal is not COMPLETE", "inference_calls": 0}
    summary_path = run / "physical_feature_tandem_inverse_summary.json"
    weights_path = run / "physical_feature_tandem_inverse_weights.npz"
    if terminal.get("summary_sha256") != sha(summary_path):
        raise ValueError("Training terminal summary SHA mismatch")
    frozen_summary = json.loads(summary_path.read_text())
    if frozen_summary.get("weights_npz_sha256") != sha(weights_path):
        raise ValueError("Training summary weights_npz_sha256 mismatch")
    if terminal.get("weights_sha256") and terminal["weights_sha256"] != sha(weights_path):
        raise ValueError("Training terminal weights SHA mismatch")
    paths = {key: pin_check(config[key]) for key in ["loader_path", "target_10000_csv", "target_8000_csv"]}
    if config.get("common_test_csv"):
        paths["common_test_csv"] = pin_check(config["common_test_csv"])
    identity = dict(terminal_sha256=sha(terminal_path), summary_sha256=sha(summary_path), weights_sha256=sha(weights_path), inputs={k: dict(path=str(p), sha256=sha(p)) for k, p in paths.items()}, postprocess_sha256=sha(__file__))
    out = run / "analysis"
    with (run / ".postprocess.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "COMPLETE.json").exists():
            done = json.loads((out / "COMPLETE.json").read_text())
            if done["identity"] != identity:
                raise ValueError("Existing analysis identity differs; no silent reuse")
            for rel, digest in done["outputs"].items():
                if sha(out / rel) != digest:
                    raise ValueError("Existing analysis output hash mismatch: " + rel)
            return dict(status="REUSED_COMPLETE", complete=str(out / "COMPLETE.json"), inference_calls=0)
        if out.exists():
            raise FileExistsError("Unclosed analysis exists; preserve failure and use a newly authorized run")
        out.mkdir()
        json_write(out / "STARTED.json", dict(utc=now(), pid=os.getpid(), identity=identity))
        try:
            module, model, checks = load_model(run, paths["loader_path"], summary_path, weights_path)
            json_write(out / "MODEL_CHECK.json", dict(status="PASS", checks=checks, manifest_sha256=sha(run / "model_manifest.json")))
            scans = {}
            for name, key, n in [("T10000", "target_10000_csv", 10000), ("application8000", "target_8000_csv", 8000)]:
                scans[name] = evaluate_scan(module, model, paths[key], n, out / name, name)
            forward = (evaluate_forward(module, model, paths["common_test_csv"], out / "common_test11369") if "common_test_csv" in paths else dict(status="NOT_AVAILABLE", reason="No pinned common-test cache supplied", inference_calls=0))
            caption = ("# Legacy-recipe D100k, seed20260711\n\nF/I: three 256-wide GELU hidden layers; independent-sigmoid geometry decoder; exact symmetric Q training. Original100K source/gradient-training/validation/test rows: 100000/78891/9740/11369. New normalizer is fitted by the training entry on its training split. This postprocessor never fits a normalizer.\n\nThe saved T10000 and application8000 request IDs and Lp/Ls/|k| values are reused separately; original random Q is ignored. Each request evaluates 11 integer Q targets (10 through 20). Selection uses " + SCORE + ". The selected candidate is a joint t% hit only when every abs(proxy-target)/abs(target)<=t/100, with finite decoded geometry within model bounds and positive Lp/Ls/Q and 0<|k|<1. There is no one-sided Q shortfall and no denominator floor. Small |k| magnifies relative errors. All failed requests remain in denominators; all-nonfinite requests use a documented Q=10 bin sentinel and cannot hit. MAE/RMSE/P95 describe available finite values and list their separate finite denominator.\n\nTarget projections use identical fixed 20x20 bins over Lp/Ls [0.5,3] nH and |k| [0,0.8]. Selected Q uses 11 integer-centered bins [9.5,20.5]. Gray means empty; dots mark 0<N<10. Selected Q is an inference outcome, not independent random sampling. These finite-frame descriptive rates have no claimed deployment-population confidence interval. Source heatmap rows and model/input SHA-256 identities accompany the figures. Model, decoder, target identities, metric, ranking, and bins are held fixed within each figure; no causal training-size comparison is made.\n\nSELF_PROXY is not physical accuracy. No fresh EMX is performed. The optional common11369 test scores the frozen forward network against existing EM labels and is not used for model selection. Those historical EM labels are distinct from Q-scan proxy targets.\n\nReproduce only from this closed model using postprocess.py --run-dir RUN --config CONFIG; valid COMPLETE is reused, never re-inferred. Outputs are PNG at 300 dpi and SVG. Previews require human visual inspection before publication.\n")
            with (out / "CAPTIONS_EN.md").open("x") as f:
                f.write(caption)
            concise = {name: {k: value[k] for k in ["n_requests", "n_candidates", "n_failed_requests", "n_proxy_invalid_requests", "joint_hits", "feature_hits_10pct"]} for name, value in scans.items()}
            json_write(out / "SUMMARY.json", dict(model_id=MODEL_ID, status="COMPLETE", scans=concise, common_test=forward, fresh_EMX="NOT_RUN", evidence="SELF_PROXY_NOT_PHYSICAL_ACCURACY", visual_review="PENDING_HUMAN_INSPECTION"))
            outputs = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
            result = dict(status="COMPLETE", completed_utc=now(), identity=identity, outputs=outputs, scans=concise, common_test_status=forward["status"], new_candidate_evaluations=198000, fresh_EMX="NOT_RUN", visual_review="PENDING_HUMAN_INSPECTION")
            json_write(out / "COMPLETE.json", result)
            with (out / "SHA256SUMS").open("x") as f:
                for rel, digest in sorted({**outputs, "COMPLETE.json": sha(out / "COMPLETE.json")}.items()):
                    f.write(digest+"  "+rel+"\n")
            return result
        except BaseException as exc:
            if not (out / "FAILURE.json").exists():
                json_write(out / "FAILURE.json", dict(status="FAIL", utc=now(), error=repr(exc), identity=identity))
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    a = p.parse_args()
    result = postprocess(a.run_dir, json.loads(a.config.read_text()))
    print(json.dumps({k: result[k] for k in ["status", "scans", "common_test_status", "new_candidate_evaluations"] if k in result}, ensure_ascii=False))
