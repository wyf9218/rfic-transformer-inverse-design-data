# Small: first completed width-only arm

Model `architecture_Small_seed20260713`: F10–128–128–128–4 and I4–128–128–128–10, with34,948 and34,954 parameters including biases. The original100,000-row dataset and78,891/9,740/11,369 split are shared with the other width arms. Training was from scratch; each stage completed3,600 updates. F was frozen before I training, while the geometry gradient through F remained enabled. The validation-selected F/I updates were3,560/3,520. This is not the historical asymmetric D100k reference or the6329-development experiment.

**Forward figure caption.** Predictions of the completed Small forward network against existing15GHz held-out EM labels(N=11,369). All predictions are finite. Lp and Ls are in nH; Qmin and coupling magnitude are dimensionless. The diagonal denotes agreement. MAE/RMSE/P95 use all held-out rows, without clipping or selecting examples. The test set did not select the checkpoint. These are existing EM labels, not fresh EM validation of inverse-generated geometries. Full-precision metrics and individual predictions are supplied in the adjacent CSV files.

**Inverse figure caption.** Joint-hit fractions on the same saved T10k frame(N=10,000), binned by target coordinates on the frozen20×20 Lp–Ls and Qmin–coupling grids. SELF_PROXY gives1,369/10,000 hits(13.69%); the independently fixed original seed20260713 common-forward diagnostic gives1,050/10,000(10.50%). The two evaluators score the identical generated geometry for each request. Neither score is EM truth. Joint success requires Lp,Ls and coupling magnitude within10% target-relative error and achieved Qmin at least the requested minimum. One-shot inference has no Q scan or local search. Empty bins are undefined; n<10 bins are marked. All failed requests, if any, stay in their target-bin denominators(here0 request failures). The frame has prior diagnostic exposure and is not a new final blind test.

Observed own/common disagreement motivates retaining both columns; it does not, without paired EM, establish which surrogate is physically correct. Do not rank three widths before their complete common-protocol results exist. This arm has no paired fresh EM results; the separately authorized64 candidates belong to the original D100k model and cannot be relabeled Small.

The recorded whole-entry time is2,659.99732s, including loading, training, validation and serialization. F/I monotonic stage times are925.4103256870294s and1,729.4390210190322s; neither is CPU-seconds. Each stage draws14,200,380 real training rows across repeated epochs, not that many unique geometries. Curves and cost sources are in `../../training_curves_cost_v1/Small/7f18caf9db438301/`.

Model weights SHA256: `ab641c846cc03653bc308e73166b3383960c695e44d8da86bc426a18d0d2d03d`.
Summary SHA256: `fa1229c9d5b677c9c9cc5a11e370c709d1c0fec24f8ac514ee8957ecb69caa76`.
Evaluation receipt SHA256: `5fe14c12ae2166f5a65d05739d117669bf3b93f3394616253e8c2a4dadd35539`.
Curve/cost receipt SHA256: `c9a0cd1f5e1dcf747449998ccf84571d652f3769c5e950b3c6984f101fb7b4db`.
Transport checks verified8 evaluation outputs, the model/summary/history binding and8 curve/cost outputs. Source receipts remain unchanged. Final paper-size exports are separate from these full diagnostic previews.
