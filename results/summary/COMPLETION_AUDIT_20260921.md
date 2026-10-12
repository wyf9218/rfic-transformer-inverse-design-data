# Bounded EuCAP training and figure task — completion audit

Audit time: 2026-09-21 17:42 UTC. Scope: the active `goal-objective.md` in attachment
`d34d81af-0750-404a-8ecf-b04007020726`, read before this continuation. This is a
requirement-to-evidence check, not another experiment or re-audit of the raw pool.
Paths below are relative to this report root (`../` from this file).

| Requirement | Actual inspected evidence and result |
|---|---|
| Reuse original100K; 15GHz, no new training database | `DATASET_LOCATIONS.md`, `DATASET_SUMMARY.csv`, `dataset_size_811_20260921_v1/data/DATASET_SIZE_CONTRACT.json`: exact mother SHA fe94d15b…, 100000 original unique geometries; four finite labels per row; original rows already satisfy target K≤0.8. No new database generation. |
| D10k/D50k/D100k mean total counts, nested original partitions | Same `data/QUALIFICATION.json`, `DATASET_SIZE_INDICES.json`, `SOURCE_ROW_MEMBERSHIP.csv`: 8000/1000/1000, 40000/5000/5000, 78891/9740/11369. Prefix PCG64 selection within original partitions, seeds2026092104/05/06; original11369 common test disjoint from every train/validation. Geometry-disjoint evidence is retained; no new universal family-independence claim. |
| Fixed architecture for dataset experiment, own train-only normalizers, scratch small models | `D100K_REUSE.json`, primary run ENTRY/summary, repeat run ENTRY/summary and evaluated weight bindings. Asymmetric F10→256→256→128→4, I4→512→512→256→10 retained because this family was already launched. Historical seed20260713 reused exactly; seeds17/29 D100k are genuine new scratch models. |
| Separate full-tandem Small/Medium/Large on same100K | Primary and repeat model/evaluation identities: both networks have three hidden layers of128/256/512. Each trains its own F then freezes F weights while preserving geometry derivatives during I. Same recipe/decoder/GELU/minimum-Q semantics, split and3600F+3600I updates. No6329 experiment substituted. |
| Actual bounded, duplicate-safe execution; first seed then repetitions | Primary five new runs completed, original D100k reused; repeat `received/FINAL_RUN_ALL.json` contains12 COMPLETE jobs, aggregate returncode0, finish17:31:19.527353UTC. Its SHA is349d77e9…. Queue3720718 and final workers4036297/4038167 absent at17:32 read; no signals. The original five-worker×four-thread/nice19/24h contract remains recorded. No auto retry or extra seed added. |
| Checkpoints, logs, selected best, costs and actual updates | Primary `trained_models/` roots and repeat `received/runs/<family>/seedN/<arm>/`: NPZ with bound normalization, stage_checkpoints, console.log, history, progress events, ENTRY and summary. All17 new F/I pairs completed3600/3600. All18 primary/repeat curve groups and cost tables delivered; best is not necessarily last. Resume support is completed-stage reuse, not an assertion of mid-update Adam/RNG replay. |
| Common holdout and frozen T10000 evaluated once per new model | Primary six `analysis_received/<arm>/COMPLETE.json` and twelve repeat `received/analysis/.../COMPLETE.json`: N11369 common holdout, N10000 identical targets, fixed frame SHA6aac6890…, target SHAa71e4328…. Final transfers verified weights/summary/ENTRY and five CSVs; automatic aggregates bind all18 receipts. Saved original T10000 inference reused. No test labels used for checkpoint/seed choice. |
| Target-coordinate hotspots and honest denominators | `p0_target_hotspots_20260921_v1/`, primary `paper_publication_v2_env_recovery/{dataset,architecture}/`: both Lp–Ls and Qmin–K,20×20 common grids,0..1 rate scale, empty/low-N explicit, per-bin numerator/denominator/failure CSV. Same10000 diagnostic targets across models, not30000 independent targets. Full four-indicator hit, not a two-dimensional feasibility claim. |
| Paired physical comparison, no10000-EM substitution | `paired64_20260921_v1/owner_terminal_20260921T1230Z/RECEIVED_TERMINAL.json`, final snapshot0063, `paper_paired64_v1/snapshot_0063/` and `final64_error_summary_v1/`: all64 same frozen target/geometry/config pairs identity-verified,8both-hit/1proxy-only/0EM-only/55both-miss;0pending/chain failure. Three plots, error tables and signed residuals. No interpolation; unverified9936 not failures. Old3 cases and7298 different-model NO-GO results excluded from numerical comparison. |
| Historical target/result discovery rather than assuming only3 cases | `dataset_size_811_20260921_v1/historical_experiment_census_20260921_v1/EXPERIMENT_CENSUS.json`: original centered-LHS fixed10k distinguished from current uniform targets; target-K filter8000, historical7298 identity pairs, numerical gate retained. No unauthorized numerical release or mixed experiments. |
| Three-seed aggregate only after completion | Repeat `received/aggregate_v1/dataset/d14358c5cefdd731/` and `architecture/b0e1ba535efa6985/`: each exactly9 models, seeds20260713/17/29, per-seed rows and mean±sampleSD(ddof1), notCI. All source pins and10 outputs per family checked. Two new dataset figures and earlier architecture figures inspected; architecture CSV-only portability replot actually executed once. |
| Distribution and staged data formation explanation | `paper_distribution_v1/`: original unweighted four-field histograms,N100000 each, composition CSV and captions. Existing generator evidence distinguishes3590bootstrap/96410main, LHS bootstrap, three-mode1.2M proposal pool, five-quota120K acquisition, final capacity-aware selection. Exact final proposal-mode membership unknown; no pureLHS/active-learning fiction. Distribution PNG inspected at final audit. |
| Separate accurate motivation and workflow; paper typography | Primary ZIP `P3_methods/Fig1/` contains separate editable motivation sources; `paper_methods_v1/` contains actual F/I/decoder/freeze/one-shot flow, CSV nodes/edges and English definitions. No EM inside gradient update, noQscan or candidate search added. SVG/vector+PNG, actual184.273mm formats,≥8pt text; visual records preserved. No generative scientific image. |
| Deliver tables, scripts, figures, captions, status, preserve private originals | Primary ZIP191files plus new supplement346files, each exact allowlist/manifest/SHA/CRC. Full local model/prediction/log roots remain available; ZIPs intentionally exclude raw100K, weights, PDK and credentials. No public push or historical overwrite. Current README/summary supersede dated running-state notes without rewriting old receipts. |

## Completed artifacts

- Primary: `/Users/wyf/Downloads/EuCAP_Primary_Comparisons_and_Paired64_20260921.zip`,
  SHA256 `f293df1451557030a2e764c2807d4efc7ea84f1d77e7b7e9bead957efc0221d4`.
- Three-seed supplement: `/Users/wyf/Downloads/EuCAP_ThreeSeed_Comparisons_Supplement_20260921.zip`,
  SHA256 `636493055f53f9e63a15e53859775ba4a9c337474bdd1bf722ac4bf7b3b6710c`.
- Final RUN_ALL SHA256 `349d77e93e205cce8fac92f7c0b8fec2103d409511875deea7ddcfb4509cc1c0`.
- Dataset aggregate COMPLETE SHA256 `b928d11eb18b0aa890d0caf460e91a14f19c535daf9e02bc39ce0c3ea7654493`.
- Architecture aggregate COMPLETE SHA256 `2aaff00091c50b35d4c86bf5c280e494ae5867a93d96b18a8749fb326ab262f8`.

## Scientific limits retained, not missing figure-task work

SELF_PROXY and the fixed original D100k common F are not physical truth; common F
is self-referential for the historical D100k seed. Q is a minimum requirement;
relative Lp/Ls/K tolerance is10%, and the training +2 Q margin is not an inference
requirement. Three seeds condition on fixed subsets; training and validation
populations both vary with dataset total. Historical environment/timing is
UNKNOWN; no causal speedup or universal scaling/physical-winner claim is made.
The saved T10000 is a development diagnostic, not a final blind physical test.
Only64 exact candidates have paired fresh EM; sparse maps cannot prove full
physical coverage. Full100K physical recertification, exact proposal-membership
recovery, advisor approval and publication acceptance are outside this bounded
goal. No required figure/training deliverable remains open.
