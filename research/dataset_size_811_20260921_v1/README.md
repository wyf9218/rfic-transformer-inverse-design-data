# Original-100K dataset-size training adapter

This supersedes the unlaunched `../datasize_20260921_v1` external-holdout study. That directory remains preserved. Here **10K / 50K / 100K denote dataset totals, not gradient-training counts**.

| Arm | Gradient train | Validation | Reserved arm test |
|---|---:|---:|---:|
| D10k | 8000 | 1000 | 1000 |
| D50k | 40000 | 5000 | 5000 |
| D100k | 78891 | 9740 | 11369 |

The smaller subsets are nested within each original frozen partition. D100k preserves the actual historical split, not an invented exact 80/10/10 split. The common forward test is the original 11369-row test; its subsequent evaluation is separate from validation-only training. No original test row enters any new arm's train or validation. Historical usage remains historical; this is not the final independent 10K fresh-EMX validation.

## Scientific recipe and protected artifacts

`train_dataset_sizes.py` wraps the exact SHA-pinned historical `trainer_runtime.py`, with only explicit data/split selection and new output plumbing. F is 10→256→256→128→4; I is 4→512→512→256→10; seed 20260713. Original hard-feasible decoder, normalization floor, GELU, relative-response loss, physical-cell-tail checkpoint selection and **minimum-Q semantics with training margin +2** are retained. This is not the later symmetric four-target recipe. Both stages retain 3600 optimizer updates and the historical batch settings.

Each normalizer is fit only on that arm's gradient-training rows. F trains first; the unchanged I training differentiates through frozen F without updating F. Old weights, normalizer, summaries, split and outputs are not overwritten. New run folders are no-clobber; an exact completed arm is never retrained by resume.

D100k historical weights may be reused only through the parent's explicit matching audit. Default launch starts D10k and D50k; it does not wait for that audit and does not automatically claim a fresh D100k run.

## Commands

Run from this directory, using the qualified `data/DATASET_SIZE_CONTRACT.json`:

```sh
python train_dataset_sizes.py check --contract data/DATASET_SIZE_CONTRACT.json
python run_all.py run --contract data/DATASET_SIZE_CONTRACT.json --arms D10k D50k --workers 2 --threads 4
```

The ordinary runner uses `nice -n 19`; each child sets OPENBLAS/OMP/MKL/VECLIB/NUMEXPR threads to 4. At most 3 workers are accepted (12 requested BLAS threads). It does not access MARS itself, request native licenses, invoke simulators, or modify production. Resource admission remains the executing owner's responsibility; the parent alone launches this research queue.

After an actual interruption, first confirm no healthy owner is active, then:

```sh
python run_all.py resume --contract data/DATASET_SIZE_CONTRACT.json --arms D10k D50k --workers 2 --threads 4
```

The runner and each arm have their own advisory process locks. Resume requires the exact contract and code identities. **Historical stage checkpoints preserve completed F/I stages only. There is no mid-stage Adam/RNG checkpoint. An interrupted unfinished stage restarts that stage; it must not be reported as exact update-level resumption.** Do not stop a healthy process merely to exercise resume. No deterministic error is retried automatically.

An explicitly authorized fresh D100k can use `train_dataset_sizes.py run --contract ... --arm D100k` in its own arm folder. Do not modify an active runner's arm list or outputs.

## Portable private runtime

Deploy unchanged small files and bind their real remote paths:

- `EUCAP_DATASET_CORE`: preserved `../datasize_20260921_v1/train_original100k_nested.py` helper; it is imported for utilities only, never invokes its external-holdout launch.
- `EUCAP_DATASIZE_TRAINER`: exact original `trainer_runtime.py`.
- `EUCAP_DATASIZE_SPLITTER`: exact `model_splitting.py` pinned by the core.
- `EUCAP_DATASIZE_SUMMARY`: exact original seed13 summary, used to recover scientific arguments.

The parent may issue a remote release mapping of contract file paths. Dataset/index/qualification content hashes and original scientific identity remain explicit. Do not use public `real10k_model_contract.json` or copy credentials/PDK contents. The dataset and paths remain private. No default remote paths are guessed.

## Outputs and completion

Each `runs/D10k` or `runs/D50k` contains `ENTRY_IDENTITY.json`, `ENTRY_STARTED.json`, append-only `console.log`, historical `stage_checkpoints/`, and the original trainer's summary/NPZ/history/predictions filenames. Consume weights only after `ENTRY_TERMINAL.json` says COMPLETE and its summary SHA matches; then verify the summary's NPZ SHA. Startup receipt records actual Python/NumPy versions.

After all requested arms finish, the normal program writes `completed_training/TRAINING_VALIDATION_STATUS.csv`. It is only a training/validation status table: validation subsets differ, so these values alone are not a common-test ranking. It does not perform inference, test evaluation or fresh EMX. The separately implemented common-11369 forward evaluation and frozen common-F inverse evaluation must keep their own explicit evidence labels; inverse self-proxy is not physical EM validation.

## Checks actually completed

Input identity/index check passed against the real qualified contract. Three targeted checks passed on 2026-09-21 07:21 UTC: actual matrix position mapping, partition/cell isolation and train-only normalization; original scientific arguments; rejection of crossed/non-nested indices. Runtime 1.816 s. No formal training, optimizer update, new inference, server connection or EM was performed by these checks. The nine earlier external-adapter checks are not repeated or relabeled as new training.
