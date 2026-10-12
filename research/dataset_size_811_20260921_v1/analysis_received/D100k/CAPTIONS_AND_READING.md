# Original D100k: completed evaluation figure captions

## Forward prediction against existing held-out EM labels

**Caption.** Predictions of the frozen original100K forward model (seed20260713) against existing15GHz electromagnetic labels on the common held-out set (N=11,369 unique source geometries;11,369 finite predictions,0 failures). The source dataset contains100,000 rows, split into78,891 gradient-training,9,740 validation and11,369 test rows. Panels show primary inductance Lp(nH), secondary inductance Ls(nH), minimum winding quality factor Qmin(dimensionless), and coupling magnitude |k|(dimensionless). The diagonal denotes ideal prediction. Predictions were computed only after model selection was fixed; these labels are not fresh EM simulations of inverse-generated geometries. Points are rasterized inside a vector-axes SVG to limit file size; the PNG is a conventional numerical preview.

Files: `common_test_forward_scatter.svg/.png`, `common_test_forward_predictions.csv`, `common_test_forward_errors.csv`. MAE, RMSE and the linearly interpolated95th percentile of absolute error use all11,369 finite rows; no outlier removal or range clipping is applied to the statistics. Original geometry and labels are unchanged.

| Feature | Unit | MAE | RMSE | P95 absolute error |
|---|---|---:|---:|---:|
| Lp | nH |0.0776500162|0.114731915|0.260103189|
| Ls | nH |0.0515458822|0.0869913515|0.202728619|
| Qmin | dimensionless |0.282031862|0.366911449|0.737044287|
| Coupling magnitude | dimensionless |0.0117340844|0.0172087102|0.0351123550|

The CSV retains full precision. These four different-unit errors are not averaged into an unlabeled accuracy percentage. The histogram/hotspot results do not establish that every target combination is physically attainable.

## Target-coordinate inverse-design diagnostic

**Caption.** Target-coordinate joint-hit rates for the original one-shot inverse model on the saved uniform T10k frame(seed20260921,N=10,000). Lp–Ls and Qmin–|k| are two-dimensional projections of the same four-dimensional requests. Each20×20 bin records hits divided by all requests assigned by their target coordinates. Joint success requires each absolute relative Lp,Ls and|k| residual to be at most10%, and achieved Qmin to meet or exceed the requested minimum. Q is not an exact-match target; the training-only+2 margin is not added to this inference criterion. Empty bins are undefined and low-count bins(n<10) are identified; no interpolation creates unsupported regions. Overall1,420/10,000 requests meet the proxy criterion(14.20%).

Files: `target_coordinate_hotspots.svg/.png`, `target_coordinate_hotspots.csv`, `request_results.csv`. Saved original inverse predictions were reused; this export performed0 new inverse calls. SELF_PROXY and the declared COMMON_PROXY coincide for this reference arm because its own F is the common evaluator. Neither column is fresh EMX validation. This historically inspected T10k frame is a development diagnostic, not a new blind final test.

## Identity and comparison limits

- Model: `current_foundry_qmin_response_only_seed20260713`; weights SHA256 `c61a93d56136ad21686863aae72984c49eb0d33ce766923c8ec2d03d9c3177dd`.
- Common-test index SHA256 `5c22ced2ec926dad716708afcb2da313f0d29ebe2d0a627670913a4c69dccdd6`.
- T10k SHA256 `a71e43284e1e629d14892d6e0ff72a4c2fbdf64616784884aa854bdd6984b223`.
- `COMPLETE.json` SHA256 `77753749147e63148b8d7b680b9c557d0beb15decf17ce909ac5eb6528ec12dc` binds all8 received result files. Their hashes were checked after transport; the forward-scatter PNG was visually reviewed.
- The original asymmetric architecture is the D10k/D50k/D100k size-experiment reference, not the new uniform-width Medium model. A single seed and unrecorded historical runtime environment do not support a population-level training-size or runtime advantage claim.
- Corresponding inverse-generated fresh EM results are kept in the separate frozen paired64 cohort; never substitute the held-out dataset labels for those candidate-bound results.
