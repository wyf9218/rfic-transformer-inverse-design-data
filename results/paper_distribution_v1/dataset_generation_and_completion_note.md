# Original100K generation and selection evidence

This note concerns the existing historical100K dataset, not the later 6329 development dataset, the public real10K model, or the current production ledger. No additional data completion was performed in this analysis. Only existing records and cached histograms/censuses were reused. The present |k| ≤ 0.8 plotting rule removes zero rows, and prior geometry deduplication already found 100000 unique rows. No EM data, candidate geometry, training run or server process was started or changed.

## Source identity and scope

Local table: `/Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/old100k_baseline_recovery_20260914_v1/source_v1/physical_feature_inverse_training_table.csv`.

Original remote reference: `/volumes/research-localdata/ywang3652/current_foundry_campaign_20260728/first100k_v1/real_emx_round_v2/model_checkpoint_1_n100000/training_table/physical_feature_inverse_training_table.csv`.

SHA-256: `fe94d15b0d083330fe7a64d3e059d083a5aa32d73394c85fc4b881e9904e68a2`.

The frozen census confirms 100000 source rows, 100000 unique 10D geometry identities, and 100000 sets of finite 15 GHz labels. The original model uses 78891 train, 9740 validation and 11369 test rows. This analysis does not change that split or imply that all100K rows participated in gradient training.

The center-frequency training table contains 15 GHz labels. Original raw S4P acquisition was configured for 5–60 GHz in 0.5 GHz increments (111 points), with four signal ports and grounded auxiliary conductors. Existing parsed-frequency evidence covers 68570 members; the other31430 were not individually reverified during the prior paper export. Therefore the 111-point configuration is not presented as a new complete100K per-file frequency audit. Original TSMC65, port, grounded-frame and extraction versions are retained. Present-day layout changes are not retroactively attributed to this dataset.

## Historical stages (not new data completion)

1. **Bootstrap contribution.** The selected100K contains3590 rows from the historical bootstrap4096. The bootstrap queue summary names `latin_hypercube_random_cd`, seed2026072802:4096 generated/evaluated candidates, one LHS round,411 geometry projections and no pre-completion rejection. The attempted-candidate maximum40960 is a budget, not an observed attempt count. Geometry legality and a0.005 μm manufacturing grid were applied in the historical generator. This only proves the bootstrap's method; it does not make the entire100K a pure LHS dataset.
2. **Main-production contribution.** The remaining96410 selected rows come from `first100k/real_emx_round_v2`. The launched120000-row queue is bound by SHA-256 `caf54d6d394722ab70e8d283da294728a6c196ab16de1574917b0755eb7e05a1`. Existing commands record a1,200,000-candidate pool,1,101,182 retained after geometry feasibility and98,818 rejected. The subsequent seed20260710 acquisition mix requested36590 coarse-4D,30082 rare-marginal,29328 pairwise-gap,12000 random-exploration and12000 geometry-diversity proposals. These add to120000 candidate slots; they are not final100K source contributions or per-strategy real-EM success counts. The queue summary records unique production fingerprints and source candidate IDs under its original1e-6 μm quantization. No new identity precision is introduced here.
3. **Upstream proposal pool and scoring — evidence recovered 2026-09-21.** The original saved `candidate_physical_feature_prediction_summary.json` records 1,200,000 proposals: 600,000 `local_sparse_target_perturbation`, 240,000 `local_rare_marginal_perturbation`, and 360,000 `global_latin_hypercube`; seed 20260710 and LHS optimization `random-cd`. The 840,000 local proposals were anchored to the 4,056-row real-EM seed pool. The main routine used `knn_idw_real_seed_surrogate_v1` for acquisition scoring (8 neighbors, inverse-distance power 2). These counts describe the **upstream proposal pool**, not the 96,410 main-production members selected into the original100K. The saved rebound-v2 command log explicitly consumes this same original `adaptive_queue/predictions/candidate_physical_feature_predictions.csv`, closing the source-level link rather than inferring it from the later parent complement. Those predictions are not EM labels. The entire100K is neither pure LHS nor a set of independent uniform random geometries. The original generation-script executable hash is not available in this small evidence set; the original execution summary establishes the named modes, arguments and seed, not an independently rerun reproduction.
4. **Final selection.** The historical accepted parent pool contained120810 entries. The100K checkpoint was selected using `four_dimensional_capacity_aware_water_fill`, seed20260710, with deterministic within-cell sampling and original source-order output. Its historical response-space grid was4×4×4×4=256 cells, with103 occupied cells. The historical overall uniformity result was FAIL. "Balanced" in a filename is not evidence of uniform physical-space coverage or superiority to ordinary DOE.

The old selection domain of Lp/Ls0.5–3 nH, Q5–25 and|k|0–0.8 explains how the historical100K was composed. These are historical facts, not newly imposed Lp, Q or half-SRF filters. The observed distribution is kept as is. Existing density does not demonstrate feasibility of every combination in the bounding box. Parent-pool, bootstrap and selected-source counts must not be added together.

## Evidence files and outstanding limits

All paths below are relative to `/Users/wyf/Documents/模拟变压器AI反向建模/reports/eucap15ghz_20260908T220300Z/old100k_baseline_recovery_20260914_v1/`:

- `paper100k_unfiltered_20260918_v2/SAMPLING_EVIDENCE_CN.md` (SHA-256 `f47e5c221322852f52c045edaf695406d6adcb659ca67e7696781d7ccc88da87`).
- `paper100k_unfiltered_20260918_v2/SOURCE_AND_SPLIT_COUNTS.csv` (SHA-256 `5a502ba3eac1d6bf2ac40f3769e3bc2ce643211598579173378e07ca377f5d2d`).
- `paper100k_unfiltered_20260918_v2/original_metadata/bootstrap_queue_summary.json`, `launch_manifest.json`, and `adaptive_queue_commands.log`.
- `compatibility_v1/balanced_physical_feature_checkpoint_summary.json`.

Not all original rows have complete saved physical-qualification evidence in the offline recovery. Finite-label descriptive plots remain useful, but should not be captioned100000 newly certified/strict-passing geometries. Cached old strict/SRF flags are preserved and are not used to reinstate a half-SRF gate. No missing label or qualification is filled from proxy output.

## Recovered source-level evidence and attribution limits

The following already retrieved files are under `reports/eucap_paper_figures_overnight_20260921_v1/datasize_20260921_v1/source_parent/provenance/`:

- `1_candidate_physical_feature_prediction_summary.json`, SHA-256 `c23b37a43cdad18fb8ca5d952ee781eb6eacae39be83dc1ed25597b4f59b2991`: actual July 28 candidate-generation mode counts, seed 20260710, `random-cd`, 4,056 seed rows, KNN/IDW settings and the original candidate CSV path.
- `2_mars56_s4p_candidate_queue_provenance_summary.json`, SHA-256 `16263bc28c75783ecb80781fce3434644fdb57dbde09f6cc68b7b8efdfb25d07`: links the 120,000-row selection to the launched queue; records exact five-arm quotas, geometry/source-ID uniqueness and queue SHA `caf54d6d394722ab70e8d283da294728a6c196ab16de1574917b0755eb7e05a1`.
- `3_mars56_grounded_s4p_candidate_queue_summary.json`, SHA-256 `938e77c0b1cb8c5f8c8875b1530be0ff0c5c8eb8dbf6df36c25095352c449c92`: confirms materialization, geometry screening and acquisition provenance. Its PASS is queue provenance/analytical screening, not 120,000 successful EM solves.

The exact parent-to-mother join was already completed in `datasize_20260921_v1/complement_audit_v1/COMPLEMENT_AUDIT.json`: parent 120,810 = original mother 100,000 + bootstrap remainder 466 + main remainder 20,344, with exact geometry/evaluation/S4P/label matching for all 100,000 mother members. Parent sources are 4,056 bootstrap-derived and 116,754 main-production entries. Therefore the previously recorded selected contributions 3,590 + 96,410 are consistent with actual membership, not an addition of parent and child pools.

The later queue-mode join classified **only the 20,344 main-complement records**. Its 2,573 global-LHS / 17,771 local counts are not the selected original100K counts and are deliberately not used in the captions or distribution composition. The original100K's exact selected per-generation-mode and per-acquisition-arm contributions remain **UNKNOWN** in the present cached evidence; they are not estimated from proposal quotas. No new member-level scan was made to obtain this note.

The upstream generation modes and seed are now known from the recovered original summary. Still missing are its original executable/source hash, exact selected100K per-mode contributions and complete old sample-level physical-qualification census. No claim of a sampling-method advantage, completion-induced coverage gain or new fresh-EM validation follows from Fig. A1/A2. The model-training queues running separately do not change these frozen data-distribution figures.

## Short English figure note

“Historical data formation: constrained LHS bootstrap; mixed local-perturbation/global-LHS proposals with real-seed KNN/IDW acquisition; recorded EM-response labeling; four-dimensional capacity-aware water-fill selection. The final 100,000 geometries are not a pure LHS sample; histograms are unweighted empirical counts.”

This note may be placed in the figure caption instead of shrinking the plot labels. The already exported histogram counts, bins, plotted points and images remain unchanged.
