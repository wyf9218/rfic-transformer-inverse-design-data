# Original100K full-tandem width comparison

This is a new, isolated experiment required by the 2026-09-21 unified paper goal. It does not replace, stop or modify the already-running asymmetric D10k/D50k dataset-size arms. The earlier 6329 development comparison is not reused as a result for these arms.

| Arm | F | I | F parameters | I parameters |
|---|---|---|---:|---:|
| Small | 10→128→128→128→4 | 4→128→128→128→10 | 34948 | 34954 |
| Medium | 10→256→256→256→4 | 4→256→256→256→10 | 135428 | 135434 |
| Large | 10→512→512→512→4 | 4→512→512→512→10 | 532996 | 533002 |

All three use the same original100K mother and **78891 train / 9740 validation / 11369 test** original physical-cell partitions. This is the historical approximately 8:1:1 split, not an invented exact 80/10/10 split. All normalizers fit the same 78891 train rows. Seed is 20260713. Each arm starts its own F from scratch, trains F for 3600 optimizer updates, then trains its own I for 3600 updates through frozen F with geometry gradients retained.

Only hidden-layer width changes. GELU, original hard-feasible decoder, original relative-response loss, minimum-Q semantics with **training loss margin +2**, Adam/batch/weight-decay/schedules, validation cadence20 and original validation-based checkpoint selection are unchanged. The +2 margin is not added to inference targets. EMX is not in the training loop.

## Entry and portable binding

From this directory, after the parent owner checks current research resources:

```sh
python run_all.py run --contract <actual-private-DATASET_SIZE_CONTRACT.json> --arms Small Medium Large --workers 3 --threads 4
```

The parent deploys this ordinary program once. It spawns at most 3 research children with `nice -n 19`, each limiting OPENBLAS/OMP/MKL/VECLIB/NUMEXPR to4 threads. It is not a simulator/controller and does not modify native production. It neither stops nor resubmits the existing dataset-size runs.

Use `EUCAP_DATASET_ADAPTER` to bind the existing frozen `dataset_size_811_20260921_v1/train_dataset_sizes.py`. Its `EUCAP_DATASET_CORE`, `EUCAP_DATASIZE_TRAINER`, `EUCAP_DATASIZE_SPLITTER` and `EUCAP_DATASIZE_SUMMARY` bindings remain exact SHA-pinned copies. The real remote contract already used by the dataset-size runs is reused, not recreated with another split. Private files stay private; no credentials or PDK content are copied.

## Progress, completion and recovery

Each new `runs/{Small,Medium,Large}` receives its own immutable `ENTRY_IDENTITY.json`, real `ENTRY_STARTED.json`, append-only console log, original stage checkpoints, original history/summary/NPZ names, and terminal receipt. This wrapper only imports old code; it never edits old or active source files.

The new-arm-only observer writes `PROGRESS.json` and append-only `progress_events.jsonl` after the original Adam call succeeds: first optimizer update, every120 updates, and stage completion. It records actual stage/update count, elapsed time and a small observed parameter value. At stage completion it also records the original final validation history entry and validation-selected best update. This is observation, not a new stopping rule or optimizer. It is not retrofitted into the running D10k/D50k jobs.

Consume weights only after `ENTRY_TERMINAL.json` is COMPLETE and the summary SHA matches; then verify the summary's NPZ SHA. The final NPZ contains validation-selected best F/I with the matching embedded normalizer, not necessarily last-step weights. Last optimizer update counts are in actual history/budget records. No claim of a full last-step Adam checkpoint is made.

If a runner truly exits, an explicit same-code/same-contract `run_all.py resume ...` reuses completed arms and completed stages only. Healthy owners are protected by runner/arm advisory locks. The historical trainer does not support mid-stage Adam/RNG continuation: an interrupted unfinished stage restarts that stage. Do not terminate healthy training to test recovery. Failed independent arms are preserved; no infinite automatic retry.

After all requested arms complete, `completed_training/ARCHITECTURE_VALIDATION_STATUS.csv` is built from their real summaries. It is not the common-test or T10k result. The evaluation agent separately performs one fixed common11369-test forward evaluation and one inference on the already-frozen T10k, with SELF_PROXY and COMMON_PROXY explicitly separated. Neither constitutes fresh EMX. A single seed produces no invented seed mean/error bar.

## Actual local checks

Three targeted checks passed in0.032 seconds on2026-09-21: width-only argument differences; architecture parameter counts; the observer's synthetic three Adam updates compared bitwise with the untouched original Adam call, including parameters and all optimizer state. No actual model training, model inference, remote access or native solver was run by these checks. Existing data/split qualification was directly reused and not rebuilt.
