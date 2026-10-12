# 15 GHz transformer research assets — private companion

Prepared for Professor Liu and authorized successors, 12 October 2026.

**Start with the English [handover guide](https://github.com/wyf9218/rfic-transformer-inverse-design/blob/main/docs/handover/HANDOVER_EN.md) and [executable quickstart](https://github.com/wyf9218/rfic-transformer-inverse-design/blob/main/docs/handover/QUICKSTART.md).** The public code repository and this private repository are both required. Access must be granted to each recipient's own GitHub account; never share passwords or tokens.

This repository contains the exact original 100,000-row dataset as a compressed CSV; original split and nested dataset indices; original100K, real10k and symmetric-recipe100K model bundles; completed dataset-size and network-width experiments; saved predictions, figures, histories, stage checkpoints, configuration, frozen training sources and result receipts. The English guide distinguishes incompatible experiments and explains limitations.

The source table SHA-256 after decompression is `fe94d15b0d083330fe7a64d3e059d083a5aa32d73394c85fc4b881e9904e68a2`. Original split: 78,891 training / 9,740 validation / 11,369 test. No new data, training, or simulation was created during publication.

- `ASSET_MANIFEST.json`: exact source → delivered path mapping, file sizes and hashes. Original private paths are historical references, not commands to execute on a new machine.
- `SHA256SUMS.txt`: transfer checksums.
- `dataset/`: original100k.csv.gz; use the public extraction tool to preserve exact identity.
- `models/`: three explicitly named bundles with bound normalizers.
- `research/`: original research sources and completed experiment artifacts. Do not execute archived launch scripts blindly: historical deadlines, output paths and controller context remain in these files.
- `results/`: saved tables, paper figure sources and paired64 final summaries/receipts.
- `replot/`: unchanged CSV-only reproduction scripts.
- `runtime/`: frozen split helper.
- `historical_notes/`: detailed prior notes; the current English guide supersedes their in-person handover requirement.

## Scope and access boundary

Full raw S4P/GDS storage, PDK/process decks, license files and authentication material are **not bundled**. Raw-artifact references are retained in the dataset and receipts; access requires the institution's approved storage and licensed environment. Result receipts alone are not a substitute for the absent raw files. This transfer does not assert that a successor currently has that access.

The public package supports local proxy inference. Native physical mode is not accepted for unattended use: the archived GUI backend lacks the currently required operating-point QA/DRC evidence producer and fails before dispatch. No MARS process was altered for this handover.

This private research archive grants no public redistribution permission for proprietary datasets, geometry, or institution-controlled material. Obtain the project owner's permission before sharing it beyond the authorized research team.
