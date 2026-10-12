# Legacy-recipe D100k, seed20260711

F/I: three 256-wide GELU hidden layers; independent-sigmoid geometry decoder; exact symmetric Q training. Original100K source/gradient-training/validation/test rows: 100000/78891/9740/11369. New normalizer is fitted by the training entry on its training split. This postprocessor never fits a normalizer.

The saved T10000 and application8000 request IDs and Lp/Ls/|k| values are reused separately; original random Q is ignored. Each request evaluates 11 integer Q targets (10 through 20). Selection uses sqrt(mean(((proxy-target)/[2.5,2.5,20,0.8])**2)); symmetric Q; exact ties choose lower Q. The selected candidate is a joint t% hit only when every abs(proxy-target)/abs(target)<=t/100, with finite decoded geometry within model bounds and positive Lp/Ls/Q and 0<|k|<1. There is no one-sided Q shortfall and no denominator floor. Small |k| magnifies relative errors. All failed requests remain in denominators; all-nonfinite requests use a documented Q=10 bin sentinel and cannot hit. MAE/RMSE/P95 describe available finite values and list their separate finite denominator.

Target projections use identical fixed 20x20 bins over Lp/Ls [0.5,3] nH and |k| [0,0.8]. Selected Q uses 11 integer-centered bins [9.5,20.5]. Gray means empty; dots mark 0<N<10. Selected Q is an inference outcome, not independent random sampling. These finite-frame descriptive rates have no claimed deployment-population confidence interval. Source heatmap rows and model/input SHA-256 identities accompany the figures. Model, decoder, target identities, metric, ranking, and bins are held fixed within each figure; no causal training-size comparison is made.

SELF_PROXY is not physical accuracy. No fresh EMX is performed. The optional common11369 test scores the frozen forward network against existing EM labels and is not used for model selection. Those historical EM labels are distinct from Q-scan proxy targets.

Reproduce only from this closed model using postprocess.py --run-dir RUN --config CONFIG; valid COMPLETE is reused, never re-inferred. Outputs are PNG at 300 dpi and SVG. Previews require human visual inspection before publication.
