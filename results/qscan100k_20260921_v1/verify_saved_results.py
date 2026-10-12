#!/usr/bin/env python3
"""Independent arithmetic verification of saved Q-scan outputs; no inference."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
FEATURES = ("lp_nh", "ls_nh", "q_scalar", "k_abs")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def columns(rows, names):
    return np.array([[float(row[name]) for name in names] for row in rows])


def main():
    contract = json.loads((ROOT / "CONTRACT.json").read_text())
    summary = json.loads((ROOT / "SUMMARY.json").read_text())
    source_path = Path(next(x["path"] for x in contract["inputs"] if x["path"].endswith("random10000_targets.csv")))
    weights_path = Path(next(x["path"] for x in contract["inputs"] if x["path"].endswith("weights.npz")))
    source, selected, bins = (read_csv(p) for p in (source_path, ROOT / "selected_10000_results.csv", ROOT / "hotspot_bins.csv"))
    checks = {}

    def check(name, value):
        checks[name] = bool(value)
        if not checks[name]:
            raise AssertionError(name)

    for path in (source_path, weights_path):
        check("input_sha_" + path.name, digest(path) == next(x["sha256"] for x in contract["inputs"] if x["path"] == str(path)))
    for name in ("CONTRACT.json", "all_candidates.npz", "selected_10000_results.csv", "hotspot_bins.csv"):
        check("output_sha_" + name, digest(ROOT / name) == summary["outputs"][name])
    with np.load(ROOT / "all_candidates.npz", allow_pickle=False) as saved:
        data = {name: saved[name] for name in saved.files}
    requests, proxy, geometry = data["requests"], data["proxy"], data["geometry"]
    check("exact_shapes", requests.shape == proxy.shape == (10000, 11, 4) and geometry.shape == (10000, 11, 10))
    source_ids = [row["target_id"] for row in source]
    check("exact_ids_and_order", len(set(source_ids)) == 10000 and source_ids == data["target_ids"].tolist() == [row["target_id"] for row in selected])
    source_triplets = columns(source, ("target_lp_nh", "target_ls_nh", "target_k_abs"))
    check("source_triplets_unchanged_all_candidates", np.array_equal(requests[:, :, [0, 1, 3]], np.broadcast_to(source_triplets[:, None, :], (10000, 11, 3))))
    check("q_grid_exact", np.array_equal(requests[:, :, 2], np.broadcast_to(np.arange(10, 21), (10000, 11))))
    check("original_q_archived", np.array_equal(data["original_random_Q"], columns(source, ("target_q_min",))[:, 0]))
    check("scales_exact", contract["spans"] == [2.5, 2.5, 20.0, 0.8])
    residual = proxy - requests
    squares = [(residual[:, :, j] / span) ** 2 for j, span in enumerate((2.5, 2.5, 20.0, 0.8))]
    scores = np.sqrt((squares[0] + squares[1] + squares[2] + squares[3]) / 4.0)
    check("all_110000_scores_exact", np.array_equal(scores, data["scores"]))
    finite = np.isfinite(scores) & np.isfinite(proxy).all(axis=2) & np.isfinite(geometry).all(axis=2)
    check("all_finite_masks_exact", np.array_equal(finite, data["finite"]))
    check("finite_candidate_count", int(finite.sum()) == summary["n_finite_candidates"] == 110000)
    chosen = np.argmin(np.where(finite, scores, np.inf), axis=1)
    q = chosen + 10
    check("all_argmins_exact", np.array_equal(chosen, data["selected_index"]))
    check("selected_q_exact", np.array_equal(q, columns(selected, ("selected_q",))[:, 0]))
    check("candidate_ids_exact", [r["selected_candidate_id"] for r in selected] == [f"{identity}_q{value:02d}" for identity, value in zip(source_ids, q)])
    row = np.arange(10000)
    target, observed, geom = requests[row, chosen], proxy[row, chosen], geometry[row, chosen]
    check("selected_targets_exact", np.array_equal(target, columns(selected, ["target_" + name for name in FEATURES])))
    check("selected_proxy_exact", np.array_equal(observed, columns(selected, ["proxy_" + name for name in FEATURES])))
    check("selected_scores_exact", np.array_equal(scores[row, chosen], columns(selected, ("selection_nrmse",))[:, 0]))
    # Decode stored standardized bounds to physical geometry units; no network is executed.
    with np.load(weights_path, allow_pickle=False) as weights:
        lower = weights["normalization__geometry_lower"] * weights["normalization__y_scale"] + weights["normalization__y_mean"]
        upper = weights["normalization__geometry_upper"] * weights["normalization__y_scale"] + weights["normalization__y_mean"]
    geometry_ok = np.isfinite(geom).all(axis=1) & (geom >= lower - 1e-9).all(axis=1) & (geom <= upper + 1e-9).all(axis=1)
    valid = finite[row, chosen] & geometry_ok & (observed[:, :3] > 0).all(axis=1) & (observed[:, 3] > 0) & (observed[:, 3] < 1)
    check("selected_geometry_valid_masks", np.array_equal(geometry_ok, [r["geometry_bounds_valid"] == "True" for r in selected]))
    check("selected_proxy_valid_masks", np.array_equal(valid, [r["proxy_valid"] == "True" for r in selected]))
    check("proxy_valid_count", int(valid.sum()) == summary["n_proxy_valid_requests"] == 10000)
    signed = observed - target
    relative = np.abs(signed) / np.abs(target)
    for prefix, expected in (("signed_error", signed), ("absolute_error", np.abs(signed)), ("relative_error_pct", relative * 100.0)):
        check(prefix + "_all_four_symmetric_exact", np.array_equal(expected, columns(selected, [prefix + "_" + name for name in FEATURES])))
    joint = {}
    masks = {}
    for percent in (5, 10, 15, 20):
        mask = valid & (relative <= percent / 100.0).all(axis=1)
        masks[percent] = mask
        joint[str(percent)] = int(mask.sum())
        check("joint_" + str(percent) + "_row_flags", np.array_equal(mask, [r[f"joint_hit_{percent}pct"] == "True" for r in selected]))
    check("joint_counts_exact", joint == summary["joint_hits"])
    feature_hits = {name: int((valid & (relative[:, j] <= .1)).sum()) for j, name in enumerate(FEATURES)}
    check("feature_10pct_counts_exact", feature_hits == summary["feature_hits_10pct"])
    axes = {"lp_nh": target[:, 0], "ls_nh": target[:, 1], "k_abs": target[:, 3], "selected_q": q}
    projections = {"lp_ls": ("lp_nh", "ls_nh"), "lp_k": ("lp_nh", "k_abs"), "ls_k": ("ls_nh", "k_abs"), "q_selected_k": ("selected_q", "k_abs")}
    grids = {}
    check("bin_rows_exact", len(bins) == 1420 and {b["projection"] for b in bins} == set(projections))
    for name, (x, y) in projections.items():
        edges = [np.array(contract["binning"][key]) for key in (x, y)]
        counts = np.histogram2d(axes[x], axes[y], bins=edges)[0].astype(int)
        hits = np.histogram2d(axes[x][masks[10]], axes[y][masks[10]], bins=edges)[0].astype(int)
        records = [b for b in bins if b["projection"] == name]
        check(name + "_bin_keys", len(records) == counts.size and len({(r["bin_x"], r["bin_y"]) for r in records}) == counts.size)
        for item in records:
            i, j = int(item["bin_x"]), int(item["bin_y"])
            n, h = int(counts[i, j]), int(hits[i, j])
            assert int(item["n_total"]) == n and int(item["n_hit"]) == h
            assert [float(item[k]) for k in ("x_lower", "x_upper", "y_lower", "y_upper")] == [edges[0][i], edges[0][i + 1], edges[1][j], edges[1][j + 1]]
            assert item["empty"] == str(n == 0) and item["low_N"] == str(0 < n < contract["low_N"])
            assert (item["hit_fraction"] == "") if n == 0 else (float(item["hit_fraction"]) == h / n)
        check(name + "_every_bin_count_fraction_bounds_flags", True)
        grids[name] = dict(n_total=int(counts.sum()), n_hit=int(hits.sum()), n_empty=int((counts == 0).sum()), n_low=int(((counts > 0) & (counts < 10)).sum()))
    check("all_grid_summaries_exact", grids == summary["grids"])
    check("all_grid_denominators_10000", all(x["n_total"] == 10000 and x["n_hit"] == joint["10"] for x in grids.values()))
    receipt = dict(schema="independent_saved_qscan_arithmetic_verification.v1", status="PASS", completed_utc=datetime.now(timezone.utc).isoformat(), script_sha256=digest(__file__), inputs={str(p): digest(p) for p in (ROOT / "CONTRACT.json", ROOT / "SUMMARY.json", ROOT / "all_candidates.npz", ROOT / "selected_10000_results.csv", ROOT / "hotspot_bins.csv", source_path, weights_path)}, checks=checks, n_checks=len(checks), joint_hits=joint, feature_hits_10pct=feature_hits, grids=grids, maximum_score_absolute_difference=float(np.max(np.abs(scores - data["scores"]))), selected_q_mismatch_count=int((q != columns(selected, ("selected_q",))[:, 0]).sum()), q_above_target_count=int((signed[:, 2] > 0).sum()), q_below_target_count=int((signed[:, 2] < 0).sum()), scope="Saved-array arithmetic only; no inference, training, EMX, plotting, or assessment of numerical backend warnings. Weights read only for decoder-bound constants.")
    with (ROOT / "VERIFICATION.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: receipt[k] for k in ("status", "n_checks", "joint_hits", "feature_hits_10pct", "maximum_score_absolute_difference", "selected_q_mismatch_count", "q_above_target_count", "q_below_target_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
