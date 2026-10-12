# Medium: completed original-100K full-tandem arm

Model `architecture_Medium_seed20260713`: forward 10–256–256–256–4 and inverse 4–256–256–256–10; 135,428 and 135,434 parameters including biases. Dataset total 100,000; gradient train 78,891, validation 9,740, reserved test 11,369. Both networks trained from scratch, with their own train-fitted normalization and validation-only checkpoint selection. Forward weights are frozen during inverse training, without cutting the gradient with respect to geometry.

Both stages actually executed 3,600 optimizer updates and 14,200,380 repeated row draws. The saved validation-selected checkpoints are **F update 3,560 and I update 1,740**, not the last update. Recorded wall times: F 1,840.1683 s; I 3,687.6503 s; whole entry 5,533.976876 s. Whole entry includes loading, validation and saving; these are not CPU-seconds or a controlled speed benchmark.

## Existing EM-label forward evaluation

The scatter panels compare predictions with existing 15 GHz EM labels for the same 11,369-row original test partition used by the other arms; every prediction is finite. The black diagonal denotes equality. Lp/Ls units are nH; Qmin and |k| are dimensionless. This is `EXISTING_REAL_EM_LABEL_HOLDOUT_NOT_FRESH_EM`, not a simulation of newly generated geometries.

| Descriptor | MAE | RMSE | P95 absolute error (linear quantile) |
|---|---:|---:|---:|
| Lp (nH) | 0.08679058084530811 | 0.12570064080176063 | 0.28645093610147726 |
| Ls (nH) | 0.06127076507835047 | 0.10639234369340236 | 0.26289062830015986 |
| Qmin | 0.31668583460021577 | 0.4262578075978691 | 0.8948828762203552 |
| K_abs | 0.014033090884042808 | 0.02137661273491395 | 0.051612781575756814 |

## Frozen T10k inverse diagnostics

The two-column target-coordinate maps use all 10,000 identical saved requests, fixed 20×20 bins and the same 0–1 color scale. SELF_PROXY: 1,120/10,000 joint hits (11.20%); common frozen original seed20260713 forward: 1,019/10,000 (10.19%). Both retain all requests, with zero inference failures. Hatched bins contain 1–9 requests; a two-dimensional projection does not establish the complete four-dimensional feasible region.

Joint hit means absolute relative Lp/Ls/|k| errors each at most 10%, and achieved Qmin at least the requested minimum. No Q scan, inference +2 margin, candidate repair or EM-based replacement is used. The common forward provides a shared diagnostic reference, not EM truth. These outcomes do not establish a physical accuracy ranking. The separate paired64 EM cohort belongs to the original asymmetric D100k model, not this Medium arm.

Source receipt SHA-256: `b383afee7eb74434948cf8a93c723b2e309a04f30e2438b3431b5b44e5c8d801`.
Weights: `d02ffc42cd75c184e1ec063f60440387a410df0abe2650f7880a7738ffb88352`.
Summary: `01d591f9ad68b9a6ecce56481ad570904974ae50dbbd1f705f47aaece96393cc`.
Target and common-test identities are retained in COMPLETE.json. The two PNG diagnostic figures were visually checked after transfer; final paper-family layout is exported separately only after all three width arms are complete.
