# Automatic completed-model evaluation

Call `postprocess(run_dir, config)` after training exits and releases its lock, or run:

```text
python postprocess.py --run-dir RUN_DIR --config POSTPROCESS_CONFIG.json
```

Required config fields `loader_path`, `target_10000_csv`, and `target_8000_csv` each contain `{ "path": "...", "sha256": "..." }`. Optional `common_test_csv` has the same binding form. A missing common-test binding records `NOT_AVAILABLE`; a supplied but invalid binding fails closed. Only the known cache is consumed, never the 100k mother CSV. Geometry/test cache private source paths are omitted from chart data.

`ENTRY_TERMINAL.json` must have `status: COMPLETE` and `summary_sha256`, optionally `weights_sha256`. Summary and weights reside directly in RUN_DIR under the standard physical_feature_tandem_inverse filenames. Before loading, the code validates exact-Q/zero-margin/anchor0.01/seed11/independent-sigmoid arguments. The private loader manifest declares—and the loader checks—F 10→256→256→256→4 and I 4→256→256→256→10.

Outputs are under new `RUN_DIR/analysis`: separate `T10000` and `application8000` contain all 11 candidate arrays, selected rows, per-feature errors, source bins, selected-Q counts, and six PNG/SVG previews. `common_test11369` contains frozen-forward predictions against existing EM labels, errors, and a scatter preview. The root contains readable `SUMMARY.json`, English captions, source identity, `COMPLETE.json`, and `SHA256SUMS`. All scan charts identify SELF_PROXY, not physical accuracy. No minimum-Q evaluator is called.

No polling, training, redraw, physical execution, or network is implemented. The caller's existing completion queue triggers this function. An absent/incomplete training terminal returns NOT_READY without creating analysis. An identical valid COMPLETE is verified and reused without inference. Partial/failed analysis is never overwritten; its FAILURE evidence must be reviewed before a new authorized run. Use the same pinned postprocess source on retry.

The preview renderer is deterministic from the completed model and frozen source rows but human visual review remains pending until actual outputs exist. The four bounded unit checks use synthetic arrays and temporary metadata only. The initial test run passed three checks; the fourth failed because macOS resolves `/var` to `/private/var`. The test's expected path was corrected to its resolved identity; the production binding code was already correct. This was a test assertion issue, not model inference or a data failure.
