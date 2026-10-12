# Fixed additional seeds 17 and 29 — new scratch repetitions

**Actual update — 2026-09-21 15:36:40 UTC:** 8/12 new repetitions are
COMPLETE in the original queue. Large17, Large29 and D50k29 finished training
and automatic evaluation and were received with all model and five CSV SHA
bindings verified. Each completed 3600 F and 3600 I updates. Validation-best
F/I checkpoints are 3520/920, 3500/2620 and 3520/1500, respectively.
Their SELF / COMMON hits are 1139/919, 1366/1117 and 1102/913, each over the
unchanged 10000 targets with zero request failures. These results do not
establish physical superiority or a complete three-seed family comparison.
The four outstanding jobs are Small17/29 and D10k17/29; ordinary queue
dispatch remains authoritative. No repeated training, inference or native
simulation was invoked while receiving these results.

**Actual update — 2026-09-21 15:17:20 UTC:** the same queue is RUNNING with
5/12 repetitions complete. D50k seed17 finished training at 15:13:19 UTC;
D100k seed29 finished at 15:15:22 UTC. Both then completed automatic
evaluation, were received locally, and passed model/ENTRY/summary and all
five evaluation CSV SHA checks. Both F/I stages completed 3600 updates.
D50k17 selected F3520/I3080 and has 1283 SELF / 961 COMMON hits;
D100k29 selected F3600/I2160 and has 1283 SELF / 1067 COMMON hits.
All denominators remain the same T10000, request failures zero; forward test
is the same 11369 held-out geometries. These are proxy diagnostics, not EM.
The corresponding new runs are under `received/{runs,analysis}/dataset/`.
No old result or model has been replaced and no family aggregate is ready yet.

**Actual update — 2026-09-21 13:23:54 UTC:** D100k seed17 also completed
training and automatic evaluation, bringing the original queue to 3/12
completed repetitions. This is a newly initialized asymmetric F/I pair,
not historical D100k weights. Both stages completed 3600 updates; validation
selected F3580/I3220. SELF/COMMON hits are 1348/1095 of the same 10000 targets
(zero request failures), evaluated once. Received run/evaluation identities
and all five CSV pins passed. New paper curves are at
`paper_curves/seed17/paper/D100k/a18e3714861f1827/`, COMPLETE SHA
`96701b5b0fee0d9b0fb3d7b3244b429a444b2a96e27a33260a573e1776cdff0a`.
The original worker-only observer 59833 returned normally; read-only observer
35371 now waits for the same original queue PID3720718 to finish, without
controlling training or native simulation. The next job is dispatched by the
existing queue, not by repeated AI submissions.

**Actual update — 2026-09-21 13:18:42 UTC:** the original queue remains
RUNNING; Medium seed17 and seed29 have completed both training and automatic
evaluation (2/12 repetitions). Their original outputs were received under
`received/runs/architecture/seed{17,29}/Medium/` and
`received/analysis/architecture/seed{17,29}/Medium/`. Remote/local completion,
model identities and all five evaluation CSV SHA values matched. No repeated
training or inference was run for transport or plotting.

| Seed | F/I actual updates | Validation-best F/I update | SELF hits / 10000 | COMMON hits / 10000 |
|---|---|---|---|---|
| 17 | 3600 / 3600 | 3580 / 3580 | 1267 | 1013 |
| 29 | 3600 / 3600 | 3560 / 2000 | 1332 | 1069 |

Both use the same 78,891/9,740/11,369 original split; common evaluation
remains the original seed20260713 forward proxy, not EM truth. Request
failures are zero. The original `REVIEW_REQUIRED_VALIDATION_ONLY` and
`eligible_for_model_success_claim=false` flags are preserved: completed
training is not physical qualification. Other arms remain running/queued;
full-family three-seed summaries must wait for their real terminal results.
The dated launch-only observation below is retained as history.

**Actual update — 2026-09-21 11:20:08 UTC:** the parent deployed these exact
sources and launched MARS queue PID 3720718 at 11:18:30 UTC. Five jobs have
performed real forward updates; seven remain queued. Source preparation and
commands below are retained as provenance, not an instruction to relaunch.
See `mars_launch_receipts/OBSERVATION_20260921T112008Z.json` and the three actual
launch/config receipts. The original primary arms and paired64 owner are not
restarted. Queue logs and each job's training/evaluation logs are below the
recorded remote `runtime_v1` root. Completion and multi-seed statistics are
still pending; do not interpret a launch as a completed repetition.

This private, local delivery **does not start training**. The parent execution
owner performs the one MARS launch after checking current resources.

The user-provided sequence is 17, 29, 43. Seeds 17 and 29 are fixed before seeing
their outcomes, adding two runs to primary seed 20260713. All twelve new F/I
pairs start from random initialization, including D100k. No primary model,
source, helper, original split or saved result is overwritten.

## Frozen scientific conditions

- Dataset family: D10k=8,000/1,000/1,000; D50k=40,000/5,000/5,000;
  D100k=78,891/9,740/11,369 train/validation/test. Exact existing nested indices.
  F 10→256→256→128→4, I 4→512→512→256→10.
- Architecture family: Small/Medium/Large use three 128/256/512-wide hidden
  layers in both F and I, each on the exact original D100k split.
- Original GELU, decoder, minimum-Q loss with its original training margin,
  Adam, sampling and validation selection remain in the hash-pinned runtime.
  Only the model seed changes from each corresponding primary arm. Fixed
  split/refinement/robustness seeds remain original. Each F and I has 3,600
  optimizer updates; no short run is reported as a completed repetition.
- Each normalizer is fitted by the original function on that arm's train only.
  F trains first, then its selected weights freeze; the original differentiable
  geometry-to-F path remains in I training. Training emits validation-only
  results, not test results. Test/T10000 cannot change these settings.
- The queue schedules both Large jobs first, then Medium, D100k, D50k, Small,
  D10k. This order is prospective scheduling, not outcome-based selection.

## One ordinary queue, no watcher

`repeat_study.py` imports the preserved architecture, dataset and core adapters.
It adds no optimizer, decoder or alternate model implementation. Source SHA
checks occur before each new process. At most five workers, four BLAS threads
per worker, nice 19: at most 20 requested research CPU threads. `--workers 1`
through `--workers 5` accommodates lower resource admission; the frozen config
records the chosen concurrency. There are no native simulator calls.

Each worker trains one pair, waits for its process to exit, then invokes the
existing evaluator once. That fixed evaluation uses original common test
11,369 and frozen T10000 with the original seed20260713 common F. SELF_PROXY
and COMMON_PROXY remain separate. New D100k cannot use the old prediction
shortcut. Completion seed is checked against actual ENTRY and summary.
The existing common-test CSV is read and SHA-checked; no repeated full mother
scan is used for evaluation.

Each new result includes `target_hit_counts.csv` (two small rows), original
error/hotspot/prediction tables, their SHA pins, and `COMPLETE.json` with explicit
family, model seed and training identity. All twelve queue jobs terminate before
one `aggregate_repeats.py` callback. That callback uses primary plus new results;
missing/failed repetitions do not become zero-valued observations. A callback
failure is preserved and is not automatically retried. The other family is not
scientifically blocked by a missing family.

## Deadline and recovery boundary

Launch fixes a 24-hour deadline. After it, no new job or optimizer stage starts.
An already-running stage finishes normally; original runtime then saves its
completed-stage checkpoint. This is **not a hard SIGKILL deadline** and drain
time is reported. If F finishes after the cutoff, I does not start; the run is
`PARTIAL_DEADLINE_STAGE_BOUNDARY`, not COMPLETE. Evaluation of a completed pair
may finish after the dispatch cutoff. All failures remain on disk; no automatic
retry, second controller or resetting of primary budgets.

The only supported explicit `resume-job` uses the existing exact completed-stage
checkpoint. It does not restore mid-stage Adam moments or sampler RNG. An
interrupted, uncheckpointed stage would restart that stage, not continue from
its last displayed optimizer update. The owner must inspect the retained error
and authorize/record any new continuation time window before that command.
The whole queue is deliberately not automatically re-launched. COMPLETE jobs
are never retrained; an existing evaluation directory is never overwritten.

## MARS deployment and single launch (parent owner only)

Deploy these new files to an otherwise new code directory:
`/volumes/research-localdata/ywang3652/eucap_paper_repeats_20260921_v1/`
(`repeat_study.py`, `aggregate_repeats.py`; tests/docs optional).
Use its new `runtime_v1` subdirectory for all outputs. No original large data,
weights or runtime source copy is required. Commands below configure and then
launch; **they have not been run by this subtask**.

```sh
REPEAT_CODE=/volumes/research-localdata/ywang3652/eucap_paper_repeats_20260921_v1
REPEAT_PYTHON=/volumes/research-localdata/ywang3652/rfic-transformer-inverse-design-mars56-s4p-20260702/.venv/bin/python
"$REPEAT_PYTHON" "$REPEAT_CODE/repeat_study.py" configure \
  --dataset-root /volumes/research-localdata/ywang3652/eucap_paper_dataset_size_811_20260921_v1 \
  --architecture-root /volumes/research-localdata/ywang3652/eucap_paper_architecture_100k_20260921_v1 \
  --out-root "$REPEAT_CODE/runtime_v1" \
  --aggregate-script "$REPEAT_CODE/aggregate_repeats.py" --workers 5
"$REPEAT_PYTHON" "$REPEAT_CODE/repeat_study.py" launch \
  --config "$REPEAT_CODE/runtime_v1/REPEAT_CONFIG.json"
```

The launcher preserves the exact venv interpreter path (no symlink resolution)
and checks numpy/matplotlib using it before spawning. Exclusive launch intent,
queue lock, per-job attempt and run lock prevent accidental repeat submission.
`REPEAT_CONFIG.json`, `LAUNCH_INTENT.json`, `LAUNCH_RECEIPT.json`, `RUN_ALL.json`
and per-job `RESULT.json` provide the owner ordinary-program state.

Paths beneath `runtime_v1`:

```text
runs/<dataset|architecture>/seed<17|29>/<arm>/ENTRY_*.json
runs/<dataset|architecture>/seed<17|29>/<arm>/PROGRESS.json
runs/<dataset|architecture>/seed<17|29>/<arm>/stage_checkpoints/
analysis_outputs/<family>/seed<seed>/results/<arm>/COMPLETE.json
jobs/<family>__<arm>__seed<seed>/RESULT.json
aggregate_v1/
```

Explicit recovery syntax, only after evidence review and a recorded new budget:

```sh
"$REPEAT_PYTHON" "$REPEAT_CODE/repeat_study.py" resume-job \
  --config "$REPEAT_CODE/runtime_v1/REPEAT_CONFIG.json" \
  --job architecture__Large__seed17 --deadline-epoch ACTUAL_AUTHORIZED_DEADLINE
```

This manual stage recovery does not silently repeat a failed evaluation. A
separate result-recovery decision is required if its no-clobber evaluation
directory already exists. No action in this package touches EMX/production.

## Local evidence

Three targeted plumbing tests passed in 0.055 s using actual hash-pinned source:
all twelve seed/architecture argument combinations; scratch initialization and
validation-only training; tiny synthetic train-only normalization; new D100k
not original; actual trained seed mismatch rejected; deadline blocks new stage
without modifying original return values. No real training, model inference,
100K CSV read, target evaluation, simulator, MARS access or production control
was performed. Synthetic checks are not research outcomes.
