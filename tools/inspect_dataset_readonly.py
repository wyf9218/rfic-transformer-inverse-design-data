"""Bounded handover exercise; no model, simulator, dataset edits or recount."""
import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset', required=True, type=Path)
p.add_argument('--rows', type=int, default=1)
a = p.parse_args()
if not 1 <= a.rows <= 10:
    p.error('--rows must be 1..10')
path = a.dataset.resolve(strict=True)
with path.open(newline='') as f:
    reader = csv.DictReader(f)
    required = ['row_index', 'evaluation', 'touchstone_path', 'touchstone_sha256',
                'input__lp_nh_center', 'input__ls_nh_center', 'input__q_center',
                'input__k_abs_center']
    if not set(required).issubset(reader.fieldnames or []):
        raise SystemExit('Dataset schema differs; do not substitute another table.')
    geometry = [k for k in reader.fieldnames if k.startswith('geom__')]
    if len(geometry) != 10:
        raise SystemExit('Expected the documented 10-dimensional geometry schema.')
    records = [{k: row[k] for k in required + geometry}
               for _, row in zip(range(a.rows), reader)]
print(json.dumps({'dataset': str(path), 'bytes': path.stat().st_size,
    'geometry_fields_in_csv_order': geometry,
    'units': {'geometry': 'um', 'Lp': 'nH', 'Ls': 'nH', 'Q': 'dimensionless',
              'K_abs': 'dimensionless'},
    'rows_read': len(records), 'total_rows': 'NOT_RECOUNTED',
    's4p_files_opened': 0, 'records': records}, ensure_ascii=False, indent=2))
