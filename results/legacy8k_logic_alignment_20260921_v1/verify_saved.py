#!/usr/bin/env python3
"""Verify saved same-8K arithmetic; sample at most 33 non-BLAS candidates."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import warnings
from datetime import datetime, timezone

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "4"
sys.dont_write_bytecode = True
import numpy as np

OUT = Path(__file__).resolve().parent
FEATURES = ["lp_nh", "ls_nh", "q_scalar", "k_abs"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def matrix(rows, columns):
    return np.array([[float(row[key]) for key in columns] for row in rows])


def score(proxy, target):
    delta = proxy - target
    return np.sqrt(((delta[..., 0] / 2.5) ** 2 + (delta[..., 1] / 2.5) ** 2 + (delta[..., 2] / 20) ** 2 + (delta[..., 3] / .8) ** 2) / 4)


def main():
    contract = json.loads((OUT / "CONTRACT.json").read_text())
    summary = json.loads((OUT / "SUMMARY.json").read_text())
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            raise AssertionError(name)

    for source in contract["inputs"]:
        check("input_sha:" + source["path"], sha(source["path"]) == source["sha256"])
    for name, expected in summary["outputs"].items():
        check("output_sha:" + name, sha(OUT / name) == expected)
    check("contract_sha", sha(OUT / "CONTRACT.json") == summary["contract_sha256"])
    source_paths = [Path(row["path"]) for row in contract["inputs"]]
    source = read_csv(next(p for p in source_paths if p.name == "application8k_qscan_v1.csv"))
    old = read_csv(next(p for p in source_paths if p.name == "selected_best_q_per_target.csv"))
    new = read_csv(OUT / "original100k_selected_same8000.csv")
    with np.load(OUT / "all_original100k_88000_candidates.npz", allow_pickle=False) as archive:
        z = {key: archive[key] for key in archive.files}
    req, proxy, geometry = z["requests"], z["proxy"], z["geometry"]
    ids = [r["target_id"] for r in source]
    check("8000_exact_unique_ids_same_order", len(ids) == len(set(ids)) == 8000 and ids == [r["target_id"] for r in old] == [r["target_id"] for r in new] == z["target_ids"].tolist())
    check("saved_shapes", req.shape == proxy.shape == (8000, 11, 4) and geometry.shape == (8000, 11, 10))
    triplets = matrix(source, ["target_Lp_nH", "target_Ls_nH", "target_K_abs"])
    check("new_source_triplets_exact_all_candidates", np.array_equal(req[:, :, [0, 1, 3]], np.broadcast_to(triplets[:, None, :], (8000, 11, 3))))
    old_triplets = matrix(old, ["target_Lp_nH", "target_Ls_nH", "target_K_abs"])
    check("old_triplets_agree_with_csv_precision", np.allclose(old_triplets, triplets, rtol=0, atol=5.1e-12))
    check("q_grid_exact", contract["q_values"] == list(range(10, 21)) and np.array_equal(req[:, :, 2], np.broadcast_to(np.arange(10, 21), (8000, 11))))
    scores = score(proxy, req)
    check("88000_scores_exact", np.array_equal(scores, z["scores"]))
    finite = np.isfinite(scores) & np.isfinite(proxy).all(axis=2) & np.isfinite(geometry).all(axis=2)
    check("all_candidates_finite", bool(finite.all()) == summary["all_candidates_finite"] == True)
    check("finite_masks_exact", np.array_equal(finite, z["finite"]))
    chosen = np.argmin(np.where(finite, scores, np.inf), axis=1)
    check("all_argmins_exact", np.array_equal(chosen, z["selected_index"]))
    check("selected_q_exact", np.array_equal(chosen + 10, matrix(new, ["selected_q"])[:, 0]))
    rows = np.arange(8000)
    target, observed = req[rows, chosen], proxy[rows, chosen]
    check("selected_target_exact", np.array_equal(target, matrix(new, ["target_" + k for k in FEATURES])))
    check("selected_proxy_exact", np.array_equal(observed, matrix(new, ["proxy_" + k for k in FEATURES])))
    check("selected_score_exact", np.array_equal(scores[rows, chosen], matrix(new, ["selection_score"])[:, 0]))
    relative = np.abs(observed - target) / target
    check("selected_symmetric_relative_errors_exact", np.array_equal(relative * 100, matrix(new, ["error_relative_pct_" + k for k in FEATURES])))
    old_target = matrix(old, ["target_Lp_nH", "target_Ls_nH", "selected_Q", "target_K_abs"])
    old_proxy = matrix(old, ["proxy_Lp_nH", "proxy_Ls_nH", "proxy_Q", "proxy_K_abs"])
    old_relative = np.abs(old_proxy - old_target) / old_target
    old_scores = score(old_proxy, old_target)
    check("old_selected_scores_csv_precision", np.allclose(old_scores, matrix(old, ["selection_score"])[:, 0], rtol=1e-8, atol=1e-11))
    old_hits, new_hits = {}, {}
    for percent in (5, 10, 15, 20, 25):
        a, b = (old_relative <= percent / 100).all(axis=1), (relative <= percent / 100).all(axis=1)
        old_hits[str(percent)], new_hits[str(percent)] = int(a.sum()), int(b.sum())
        check(f"old_hit_flags_{percent}", np.array_equal(a, [r[f"all_four_within_{percent}pct"] == "True" for r in old]))
        check(f"new_hit_flags_{percent}", np.array_equal(b, [r[f"joint_hit_{percent}pct"] == "True" for r in new]))
    old_feature = {key: int((old_relative[:, j] <= .1).sum()) for j, key in enumerate(FEATURES)}
    new_feature = {key: int((relative[:, j] <= .1).sum()) for j, key in enumerate(FEATURES)}
    check("old_hits_summary", old_hits == summary["old_hits"])
    check("new_hits_summary", new_hits == summary["new_hits"])
    check("old_feature_counts_summary", old_feature == summary["old_feature_hits_10pct"])
    check("new_feature_counts_summary", new_feature == summary["new_feature_hits_10pct"])
    both = int(((old_relative <= .1).all(axis=1) & (relative <= .1).all(axis=1)).sum())
    check("both_models_hit10", both == summary["both_models_hit10"])
    any_q = int(((np.abs(proxy - req) / req <= .1).all(axis=2) & finite).any(axis=1).sum())
    check("anyQhit_posthoc_count", any_q == summary["new_any_Q_hits_10pct_posthoc_not_selected"])
    comparison = read_csv(OUT / "same8000_descriptive_comparison.csv")
    for line, counts, per_feature in zip(comparison, (old_hits, new_hits), (old_feature, new_feature)):
        check(line["model"] + "_comparison_counts", int(line["target_count"]) == 8000 and all(int(line[f"joint_hit_{p}pct"]) == n for p, n in counts.items()) and all(int(line[k + "_hits10"]) == n for k, n in per_feature.items()))
    # Limited alternative arithmetic check; 3 requests x 11 Q only.
    loader = next(p for p in source_paths if p.name == "frozen_mlp.py")
    bundle = next(p.parent for p in source_paths if p.name == "model_manifest.json")
    spec = importlib.util.spec_from_file_location("same8k_bounded_numeric_loader", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.FrozenInverseModel.load(bundle)
    n = model.model["normalization"]
    physical_lower = n["geometry_lower"] * n["y_scale"] + n["y_mean"]
    physical_upper = n["geometry_upper"] * n["y_scale"] + n["y_mean"]
    selected_g = geometry[rows, chosen]
    bounds = (selected_g >= physical_lower - 1e-9).all(axis=1) & (selected_g <= physical_upper + 1e-9).all(axis=1)
    valid = finite[rows, chosen] & bounds & (observed[:, :3] > 0).all(axis=1) & (observed[:, 3] > 0) & (observed[:, 3] < 1)
    check("selected_geometry_bounds_count", int(bounds.sum()) == summary["selected_geometry_bounds_valid"])
    check("selected_proxy_valid_count", int(valid.sum()) == summary["selected_physical_proxy_valid"])
    indices = np.array([0, 3999, 7999])
    sample = req[indices].reshape(-1, 4)

    def alternative(values, weights, biases):
        for j, (weight, bias) in enumerate(zip(weights, biases)):
            values = np.einsum("ij,jk->ik", values, weight, optimize=False) + bias
            if j != len(weights) - 1:
                values = module._gelu(values)
        return values

    with warnings.catch_warnings(record=True) as caught, np.errstate(all="raise"):
        warnings.simplefilter("always")
        raw = alternative((sample - n["x_mean"]) / n["x_scale"], model.model["inverse_weights"], model.model["inverse_biases"])
        gnorm = module._project_inverse_geometry(raw, n["geometry_lower"], n["geometry_upper"], mode=model.model["projection_mode"], normalization=n, topology_contract=model.model["topology_contract"])
        g = gnorm * n["y_scale"] + n["y_mean"]
        p = alternative(gnorm, model.model["forward_weights"], model.model["forward_biases"]) * n["x_scale"] + n["x_mean"]
    expected_g, expected_p = geometry[indices].reshape(-1, 10), proxy[indices].reshape(-1, 4)
    check("33_nonBLAS_geometries", np.allclose(g, expected_g, rtol=1e-11, atol=1e-11))
    check("33_nonBLAS_proxy_values", np.allclose(p, expected_p, rtol=1e-11, atol=1e-11))
    check("three_nonBLAS_Q_selections", np.array_equal(np.argmin(score(p, sample).reshape(3, 11), axis=1), chosen[indices]))
    numeric = dict(sampled_request_indices=indices.tolist(), candidate_count=33, method="einsum optimize=False with original GELU/decoder; np.errstate(all=raise)", warnings=[str(w.message) for w in caught], max_abs_geometry_difference=float(np.max(np.abs(g - expected_g))), max_abs_proxy_difference=float(np.max(np.abs(p - expected_p))), original_warning="Original run emitted NumPy matmul divide/overflow/invalid warnings; preserved as observed, not independently explained.", interpretation="All 88000 saved candidates finite; no divergence beyond 1e-11 tolerance found in 33 sampled non-BLAS candidates. This is not a full alternative-backend inference rerun or proof of warning root cause.")
    receipt = dict(schema="same_legacy8k_saved_results_verification.v1", status="PASS", created_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__), contract_sha256=sha(OUT / "CONTRACT.json"), summary_sha256=sha(OUT / "SUMMARY.json"), inputs=contract["inputs"], output_pins=summary["outputs"], checks=checks, check_count=len(checks), n_targets=8000, n_candidates=88000, score_max_abs_difference=float(np.max(np.abs(scores - z["scores"]))), selected_q_mismatch_count=int((chosen != z["selected_index"]).sum()), old_triplet_max_abs_csv_rounding_difference=float(np.max(np.abs(old_triplets - triplets))), old_score_max_abs_csv_rounding_difference=float(np.max(np.abs(old_scores - matrix(old, ["selection_score"])[:, 0]))), old_hits=old_hits, new_hits=new_hits, old_feature_hits_10pct=old_feature, new_feature_hits_10pct=new_feature, both_hit10=both, anyQhit10_posthoc=any_q, numeric_backend_sample=numeric, scope="Saved-array/CSV verification plus 33 bounded alternative arithmetic candidates; no full inference rerun, training, physical solver, remote access or old-result modification.")
    with (OUT / "VERIFICATION.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: receipt[k] for k in ("status", "check_count", "score_max_abs_difference", "old_hits", "new_hits", "numeric_backend_sample")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
