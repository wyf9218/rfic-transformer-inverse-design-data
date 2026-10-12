# Dataset-size experiment — current user definition

D10k / D50k / D100k denote **total dataset size**, including training,
validation and test, not gradient-training rows. This supersedes the earlier
external-holdout draft; those draft files remain historical and are not run.

Use only the original pinned 100000-row mother table and its historical
78891 / 9740 / 11369 partition. Subsample each partition independently using
fixed nested indices: D10k = 8000 / 1000 / 1000; D50k = 40000 / 5000 / 5000;
D100k retains all original rows in their original partitions. Preserve the
actual 15 GHz labels and original |k| <= 0.8 scope, not the 6329 dataset.

Common forward evaluation is the original 11369-row test partition, with no
test-based checkpoint/model selection. Subset-specific tests are named
separately. Common inverse targets are the already frozen uniform10000 frame;
do not generate it again. Each arm's own F score is SELF_PROXY. The explicitly
frozen original seed20260713 forward is the shared COMMON_PROXY diagnostic,
never physical truth. All arms retain the original minimum-Q objective.

Primary hotspots bin **target coordinates**, with hit_count / target_count,
identical grids and [0,1] color scales; empty and sparse bins are distinguished.
Old response-coordinate occupancy figures are retained as historical
descriptions, not relabeled as target hit-rate maps. Unpaired historical3 versus
random10000 IoU is not a meaningful global validation result and is excluded
from the new primary comparison. The authorized random64 paired EM cohort is
separate and must not block training or be called 10000 physical validations.

Missing small models start from scratch; normalization is train-only per arm.
Reuse the original 100K architecture/decoder/loss/optimizer/update and
checkpoint-selection protocol. Keep historical weights and splits unchanged.
One fixed-seed comparison first; additional runs only after the primary result.
