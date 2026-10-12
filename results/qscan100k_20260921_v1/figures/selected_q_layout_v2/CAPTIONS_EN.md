# Q-scan 100k figure captions and provenance

## Shared evaluation contract

These figures describe the fixed finite frame of 10,000 three-input requests (Lp, Ls, |k|), each evaluated with integer target Q = 10, 11, ..., 20 (110,000 candidate evaluations). One candidate is selected per request using the frozen score stated in SUMMARY.json: `"sqrt(mean(((F(I([Lp,Ls,q,abs(k)]))-[Lp,Ls,q,abs(k)])/[2.5,2.5,20,0.8])**2)); symmetric Q; ascending-q exact ties"`. Selection by fixed-span normalized error and the subsequent target-relative tolerance hit statistics are different operations and must not be identified as the same error measure. A joint hit at tolerance t requires every feature to satisfy abs(prediction - target) / abs(target) <= t, without a denominator floor. Q is evaluated symmetrically around the selected target Q; excessive Q can therefore miss. The scalar physical Q label is Qmin = min(Qp, Qs), which does not make the present exact-Q request a minimum-Q requirement. Target-relative |k| errors become sensitive near zero (the minimum target |k| in this frame is 0.01545946987287989).

The original model is `current_foundry_qmin_response_only_seed20260713`. Its historical source-table / gradient-training / validation / test row counts are 100,000 / 78,891 / 9,740 / 11,369. The historical seed20260713 training used a minimum-Q objective and a +2 training guardband; the present symmetric-Q scanning and evaluation do not relabel or retrain that historical objective, and no extra training guardband is added at inference. No causal training-data-size improvement is claimed. Every response used in these figures is a frozen-forward surrogate prediction. **FROZEN FORWARD PROXY, NOT FRESH EMX.** The figures do not establish manufacturability or physical accuracy.

The 10,000-request denominator retains all 0 failed requests as joint misses. These are descriptive finite-frame rates; no i.i.d. deployment-population confidence intervals are claimed. The only variation displayed in the maps is target position (or the selected-Q stratum); there is no experimentally varied treatment. Model weights, inference/selection rule, target frame, error definition, and bin edges are fixed. Low-count markings communicate local support, not statistical significance.

## Qscan100k_target_hotspots

Joint ±10% symmetric target-relative hit rates after Q scanning, projected onto (a) target Lp–Ls, (b) target Lp–|k|, and (c) target Ls–|k|. Each panel aggregates all 10,000 original requests with the same 2,073 joint hits (20.73%). Fixed bins use 20 subdivisions per axis, Lp and Ls in [0.5, 3.0] nH and |k| in [0, 0.8]. Color shows the count-weighted bin rate on a shared 0–100% scale. Gray denotes an empty bin; a dot denotes 0 < N < 10. The omitted target dimension and selected Q are marginalized, so local projected rates do not define a four-dimensional feasibility boundary. Failed requests remain in the target-bin denominator.

## Qscan100k_target_counts

Numbers of original fixed-frame requests in the same three target projections and identical bins. All panels share one linear count scale and each sums to 10,000. Gray denotes N = 0; dots denote 0 < N < 10. This companion figure exposes sampling support behind the hit-rate map; it is not a second performance measure.

## SelectedQ_K_hotspot

(a) Joint ±10% symmetric target-relative hit rate and (b) request count against model-selected target Q and the original target |k|. Q is discrete, using 11 bins centered at integers 10 through 20 with boundaries 9.5 through 20.5; |k| uses the same 20 fixed bins in [0, 0.8]. Selected Q is an outcome of the fixed candidate-ranking procedure, not an independently random input. The strata therefore describe selection-conditioned support and performance; differences cannot be interpreted as a causal Q effect. 0 requests have no assignable selected Q and are absent from this projection, while all remain in the overall 10,000-request performance denominator. Empty and low-count encodings match the target maps.

## Qscan100k_performance_summary

Selected-candidate performance on all 10,000 requests. The large number is the joint ±10% hit rate (2,073/10,000); individual bars show the four feature-wise ±10% rates using the same original denominator. Secondary joint rates use ±5%, ±15%, and ±20% tolerance. These are intersected four-feature hit counts, not mean error or 1-minus-loss. Both Q overshoot and shortfall are penalized in this evaluation. The fixed-span selection score is not the relative-error statistic shown here.

## Reproduction and source identity

Run `python plot_qscan.py --input-dir INPUT_DIR --output-dir NEW_FIGURE_DIR` in an environment with NumPy and Matplotlib. NEW_FIGURE_DIR must not exist. Only SUMMARY.json and hotspot_bins.csv are consumed; no model loading, training, inference, data generation, network access, or EMX is performed. Exports are 184 mm wide, SVG with editable text and PNG at 400 dpi. Each map's rows and derived matrices are checked against the input summary before drawing. Source code and output hashes are recorded in RENDER_RECEIPT.json and SHA256SUMS.

- SUMMARY.json SHA-256: `7106d5fca5f8f1431d8980aa39a462d42bc2cbabd565be1b3afa531ff2804e18`
- hotspot_bins.csv SHA-256: `a82599dea08bd8eca8c38accdb2dc2d354d386cc712a08e4df4f661be034355b`
- Frozen model weights SHA-256: `c61a93d56136ad21686863aae72984c49eb0d33ce766923c8ec2d03d9c3177dd`
- Frozen target SHA-256: `a71e43284e1e629d14892d6e0ff72a4c2fbdf64616784884aa854bdd6984b223`
