# Visual review — 2026-09-21

PASS: both the newly plotted forward_learning_curve.png and the unchanged downloaded ../forward_scatter.png were inspected. Labels are readable and not clipped; training and validation use solid/dashed lines as well as different colors. SVGs were not independently rendered. This note supplements the frozen manifest's render-time PENDING field.

Model: legacyrecipe_D100k_seed20260711, the 2026-09-21 recorded-recipe reconstruction; not the original seed20260713 model. Source/train/validation/test: 100000/78891/9740/11369. The learning curve is saved feature-balanced normalized RMSE, not a percentage. Best forward checkpoint is update 1200; a completed fixed budget does not establish convergence. Scatter uses already-computed predictions on 11369 held-out EM labels, not fresh EMX on inverse-generated geometries.

No new model inference, training or simulation. Remote actions only read/download four existing files. History SHA matches the model summary; summary and both scatter hashes match the previously received terminal receipt.
