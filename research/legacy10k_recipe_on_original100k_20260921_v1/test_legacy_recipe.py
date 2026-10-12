#!/usr/bin/env python3
"""Targeted synthetic checks only; no dataset reads, training or remote calls."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import sys
import time
import types
import warnings

sys.dont_write_bytecode = True
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "4"
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
W = OUT.parent
TOOLKIT = ROOT / "reports/frozen_mlp_synthesis_toolkit_20260804/toolkit/rfic_transformer_inverse_synthesis_toolkit_v1/model"
OLD = ROOT / "reports/model_training_monday_20260720/multifrequency_real10k_pilot_20260720/model_seed_20260711/A_center_15ghz_4d/physical_feature_tandem_inverse_summary.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    spec = importlib.util.spec_from_file_location("targeted_legacy_wrapper", OUT / "train_legacy_recipe.py")
    wrapper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    paths = dict(core=W / "datasize_20260921_v1/train_original100k_nested.py", trainer=TOOLKIT / "trainer_runtime.py", splitter=ROOT / "github_worktrees/eucap15-mlp-capacity-20260909/rfic_transformer_inverse_design/model_splitting.py", original_summary=TOOLKIT / "physical_feature_tandem_inverse_summary.json", dataset_adapter=W / "dataset_size_811_20260921_v1/train_dataset_sizes.py")
    cfg = {"sources": {k: dict(path=str(p), sha256=digest(p)) for k, p in paths.items()}}
    dataset = wrapper.modules(cfg)
    runtime = dataset.c.load_runtime()
    legacy = json.loads(OLD.read_text())
    # Only contract metadata is read; the mother CSV and real index lists are not read.
    metadata_path = W / "dataset_size_811_20260921_v1/data/DATASET_SIZE_CONTRACT.json"
    mother = json.loads(metadata_path.read_text())["mother"]["path"]
    args = runtime._parse_args(wrapper.recipe(legacy, mother, OUT / "synthetic_not_launched"))
    checks = {}
    details = {}

    def check(name, condition, detail=None):
        checks[name] = bool(condition)
        if detail is not None:
            details[name] = detail
        if not checks[name]:
            raise AssertionError(name)

    for key, value in legacy["arguments"].items():
        expected = mother if key == "training_csv" else str(OUT / "synthetic_not_launched") if key == "out_dir" else value
        actual = getattr(args, key)
        if isinstance(actual, Path):
            actual = str(actual)
        # Historical strict JSON serializes disabled infinite thresholds as null.
        if key in ("max_forward_test_rmse", "max_tandem_response_test_rmse") and value is None and actual == float("inf"):
            actual = None
        check("legacy_argument:" + key, actual == expected)
    expected = wrapper.assert_recipe(args)
    check("wrapper_assert_recipe", bool(expected))
    check("initial_paths_empty", all(getattr(args, side + "_" + key) is None for side in ("forward", "inverse") for key in ("initial_weights", "initial_weights_sha256", "initial_summary", "initial_summary_sha256")))
    check("new_semantics_explicit", args.q_target_semantics == "exact" and args.q_minimum_margin_physical == 0 and args.response_semantic_loss_weights == "" and args.inverse_checkpoint_selection == "training_objective" and args.inverse_geometry_projection == "independent_sigmoid" and not args.enforce_power_line_port_ground_overlap)
    rng = np.random.default_rng(8123)
    x = rng.normal(size=(4, 4))
    y = rng.normal(size=(4, 10))
    matrix = dict(x=x.copy(), y=y.copy(), split_x=x.copy(), source_indices=list(range(4)), source_evaluations=list("abcd"), source_geometry_identities=list("ABCD"))
    split = {"train": np.array([0, 1]), "validation": np.array([2]), "test": np.array([3])}
    baseline = runtime._normalize(matrix, split, 1e-12)
    altered = dict(matrix, x=x.copy(), y=y.copy())
    altered["x"][2:] += 1e9
    altered["y"][2:] -= 1e9
    normalized = runtime._normalize(altered, split, 1e-12)
    check("train_only_normalizer_ignores_holdout_outliers", all(np.array_equal(baseline["normalization"][k], normalized["normalization"][k]) for k in baseline["normalization"]))
    check("train_normalized_arrays_unchanged", np.array_equal(baseline["x"][:2], normalized["x"][:2]) and np.array_equal(baseline["y"][:2], normalized["y"][:2]))
    names = legacy["input_columns"]
    response = runtime._configure_response_loss(baseline, names, names, dict(physical_cell_lower=[.5, .5, 5, 0], physical_cell_upper=[3, 3, 25, .8]), args)
    check("response_is_mse_exact_state_none", response["family"] == "mse" and response["q_target_semantics"] == "exact" and baseline["response_loss_state"] is None and response["relative_mse"] is None and set(response["target_semantics"].values()) == {"exact"})
    spans = baseline["normalization"]["response_loss_physical_spans"]
    weights = baseline["normalization"]["response_loss_dimension_weights"]
    scale = baseline["normalization"]["x_scale"]
    check("physical_spans_exact", np.array_equal(spans, [2.5, 2.5, 20, .8]))
    check("declared_range_dimension_weights", np.array_equal(weights, (scale / spans) ** 2 / np.mean((scale / spans) ** 2)))
    truth = np.zeros((1, 4))
    positive = truth.copy(); positive[0, 2] = 2 / scale[2]
    loss_p, grad_p = runtime._response_loss_and_gradient(positive, truth, weights, None, apply_target_semantics=True)
    loss_m, grad_m = runtime._response_loss_and_gradient(-positive, truth, weights, None, apply_target_semantics=True)
    loss_0, grad_0 = runtime._response_loss_and_gradient(truth, truth, weights, None, apply_target_semantics=True)
    check("plus_minus_Q_equal_positive_loss_opposite_gradient", loss_p == loss_m and loss_p > 0 and np.array_equal(grad_p, -grad_m) and grad_p[0, 2] > 0)
    check("exact_Q_zero_only_at_target", loss_0 == 0 and np.count_nonzero(grad_0) == 0)
    raw = rng.normal(size=(3, 10)); lower = -np.arange(1, 11); upper = np.arange(1, 11)
    decoded, derivative, jacobian = runtime._project_inverse_geometry(raw, lower, upper, mode=args.inverse_geometry_projection)
    sigmoid = 1 / (1 + np.exp(-np.clip(raw, -40, 40)))
    check("independent_decoder_formula", jacobian is None and np.array_equal(decoded, lower + (upper - lower) * sigmoid) and np.array_equal(derivative, (upper - lower) * sigmoid * (1 - sigmoid)))
    check("independent_decoder_finite_inside_envelope", np.isfinite(decoded).all() and np.isfinite(derivative).all() and (derivative > 0).all() and (decoded > lower).all() and (decoded < upper).all())
    # Tiny differentiable pass only, no Adam step or training loop is invoked.
    iw, ib = runtime._init_mlp(4, 10, [6], rng)
    fw, fb = runtime._init_mlp(10, 4, [5], rng)
    frozen = [a.copy() for a in fw + fb]
    inputs = rng.normal(size=(2, 4))
    raw, activations, preacts = runtime._forward_with_cache(inputs, iw, ib)
    geom, deriv, _ = runtime._project_inverse_geometry(raw, -np.ones(10), np.ones(10), mode="independent_sigmoid")
    observed, _, forward_pre = runtime._forward_with_cache(geom, fw, fb)
    loss, gout = runtime._response_loss_and_gradient(observed, inputs, weights, None, apply_target_semantics=True)
    ggeometry = runtime._backward_input_gradient(gout, fw, forward_pre)
    giw, gib, _ = runtime._backward(ggeometry * deriv, iw, activations, preacts, 0.)
    check("response_gradient_reaches_inverse", np.isfinite(loss) and all(np.isfinite(a).all() for a in giw + gib) and sum(np.linalg.norm(a) for a in giw + gib) > 0)
    check("frozen_F_unmodified_by_response_backprop", all(np.array_equal(a, b) for a, b in zip(frozen, fw + fb)))
    tree = ast.parse(inspect.getsource(runtime._train_inverse))
    adam_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_adam_step"]
    check("actual_inverse_runtime_adam_targets_only_I", len(adam_calls) == 1 and [ast.unparse(a) for a in adam_calls[0].args[:2]] == ["inverse_weights", "inverse_biases"])
    check("callback_F_positional_indices_3_4", list(inspect.signature(runtime._train_inverse).parameters)[3:5] == ["forward_weights", "forward_biases"])
    # Exercise observer with synthetic stubs, capture every write in memory.
    class VirtualOut:
        def __truediv__(self, name):
            return self
        def open(self, *unused, **kwargs):
            return io.StringIO()
    captured = []
    original_atomic = wrapper.atomic
    wrapper.atomic = lambda path, value: captured.append(value)
    try:
        def stub_forward(*unused, **kwargs):
            return ([{"stub_only": True}], {"optimizer_updates": 0})
        def stub_inverse(data, wi, bi, wf, bf, parsed):
            wi[0][0, 0] += .01
            return ([{"stub_only": True}], {"optimizer_updates": 0})
        fake = types.SimpleNamespace(_adam_step=lambda *a: None, _train_forward=stub_forward, _train_inverse=stub_inverse)
        wrapper.observe(fake, VirtualOut(), time.time() + 60)
        fake._train_inverse({}, iw, ib, fw, fb, args)
        check("observer_accepts_I_changes_F_frozen", captured[-1]["forward_frozen_verified"] is True)
        def bad_inverse(data, wi, bi, wf, bf, parsed):
            wf[0][0, 0] += 1.
            return ([{"stub_only": True}], {"optimizer_updates": 0})
        bad = types.SimpleNamespace(_adam_step=lambda *a: None, _train_forward=stub_forward, _train_inverse=bad_inverse)
        wrapper.observe(bad, VirtualOut(), time.time() + 60)
        rejected = False
        try:
            bad._train_inverse({}, iw, ib, [a.copy() for a in fw], [a.copy() for a in fb], args)
        except AssertionError:
            rejected = True
        check("observer_rejects_F_mutation", rejected)
    finally:
        wrapper.atomic = original_atomic
    result = dict(schema="legacy_recipe_targeted_checks.v1", status="PASS", utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started, script_sha256=digest(__file__), wrapper_sha256=digest(OUT / "train_legacy_recipe.py"), legacy_summary_sha256=digest(OLD), sources=cfg["sources"], checks=checks, check_count=len(checks), synthetic_normalizer_rows=4, synthetic_gradient_rows=2, optimizer_updates_executed=0, training_loops_executed=0, real_dataset_rows_read=0, scope="Parser arguments, symmetric MSE, independent decoder, synthetic gradient and train-only normalizer, static inverse optimizer target, in-memory observer stubs; no real training, dataset scans, native tools, remote access or whole suite.", details=details)
    with (OUT / "TARGETED_CHECKS.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "check_count", "elapsed_seconds", "wrapper_sha256", "real_dataset_rows_read", "optimizer_updates_executed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
