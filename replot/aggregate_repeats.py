#!/usr/bin/env python3
"""Once-only closed-table aggregation, with no inference, training, or watcher.

python aggregate_repeats.py --manifest REPEAT_CONFIG.json --out aggregate_v1
--check-only validates available small tables and lists missing arms, without plots
or any three-seed completion claim. --primary-root relocates received primary
tables explicitly; recorded source identities are never rewritten.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

SEEDS = (20260713, 17, 29)
FAMILIES = {'dataset': ('D10k', 'D50k', 'D100k'), 'architecture': ('Small', 'Medium', 'Large')}
PARTS = ('train', 'validation', 'test')
COUNTS = {'D10k': (8000, 1000, 1000), 'D50k': (40000, 5000, 5000),
          **{a: (78891, 9740, 11369) for a in ('D100k', 'Small', 'Medium', 'Large')}}
FEATURES = ('lp_nh', 'ls_nh', 'q_min', 'k_abs')
METRICS = ('MAE', 'RMSE', 'P95_absolute_error_linear')
EVIDENCE = ('SELF_PROXY', 'COMMON_PROXY_ORIGINAL_SEED20260713')
TARGET_SHA = 'a71e43284e1e629d14892d6e0ff72a4c2fbdf64616784884aa854bdd6984b223'
TEST_SHA = '5c22ced2ec926dad716708afcb2da313f0d29ebe2d0a627670913a4c69dccdd6'
FRAME_SHA = '6aac68901d58180d5fef3eb52064052cdde9fe0730d18121b86ade99bb500e48'
ORIGINAL_WEIGHTS = 'c61a93d56136ad21686863aae72984c49eb0d33ce766923c8ec2d03d9c3177dd'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_json(pin):
    raw = Path(pin['path']).read_bytes()
    if digest(raw) != pin['sha256']:
        raise ValueError('SHA mismatch: ' + pin['path'])
    return json.loads(raw)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def checked_table(root, receipt, name):
    raw = (root / name).read_bytes()
    binding = receipt['outputs'][name]
    expected = binding['sha256'] if isinstance(binding, dict) else binding
    if digest(raw) != expected:
        raise ValueError('SHA mismatch: ' + str(root / name))
    return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))), expected


def check_counts(c):
    n, h, f = (int(c[k]) for k in ('original_N', 'hits', 'request_failures'))
    if n != 10000 or min(h, f) < 0 or h + f > n:
        raise ValueError('Original target count or failure accounting changed')
    if not math.isclose(float(c['fraction']), h / n, rel_tol=0, abs_tol=1e-12):
        raise ValueError('Stored joint-hit fraction disagrees with counts')


def load_arm(root, family, arm, seed):
    """Only receipt, 4-row error CSV and bound saved hit/bin CSV are read."""
    root = Path(root)
    raw = (root / 'COMPLETE.json').read_bytes()
    r = json.loads(raw)
    if r.get('status') != 'COMPLETE':
        raise ValueError('Evaluation terminal is not COMPLETE')
    primary = seed == SEEDS[0]
    expected_family = 'original' if primary and arm == 'D100k' else family
    if r['arm'] != arm or r['comparison_family'] != expected_family:
        raise ValueError('Arm/family mismatch')
    if tuple(r['partition_counts'][p] for p in PARTS) != COUNTS[arm]:
        raise ValueError('Actual partition counts changed')
    if (r['target_sha256'], r['common_test_indices_sha256']) != (TARGET_SHA, TEST_SHA):
        raise ValueError('Frozen target or common-test identity changed')
    if r['frame_contract']['sha256'] != FRAME_SHA:
        raise ValueError('Common original-forward frame changed')
    if (r['target_N'], r['common_forward_test_N']) != (10000, 11369):
        raise ValueError('Evaluation denominators changed')
    if r['fresh_EMX'] != 'NOT_RUN' or r['test_used_for_model_selection'] is not False:
        raise ValueError('Evidence class or test-use policy changed')
    if primary:
        if not r['model_id'].endswith('seed20260713'):
            raise ValueError('Primary identity is not seed20260713')
    else:
        if r.get('model_seed') != seed or r.get('kind') != 'new-trained':
            raise ValueError('Repeat must be an explicit genuinely new seed model')
        if r.get('repeat_job_id') != f'{family}__{arm}__seed{seed}':
            raise ValueError('Repeat job identity mismatch')
        training = checked_json(r['training_identity'])
        if training.get('model_seed', training.get('seed')) != seed or training.get('arm') != arm:
            raise ValueError('Seed/arm disagree with bound training identity')
    mi = r['model_identity']
    if primary and arm == 'D100k':
        if mi['weights_sha256'] != ORIGINAL_WEIGHTS:
            raise ValueError('Historical original D100k changed')
    else:
        width = {'Small': 128, 'Medium': 256, 'Large': 512}.get(arm)
        f = [10, width, width, width, 4] if width else [10, 256, 256, 128, 4]
        i = [4, width, width, width, 10] if width else [4, 512, 512, 256, 10]
        if mi.get('forward') != f or mi.get('inverse') != i:
            raise ValueError('Declared F/I architecture changed')
    errors, error_sha = checked_table(root, r, 'common_test_forward_errors.csv')
    if len(errors) != 4 or {v['feature'] for v in errors} != set(FEATURES):
        raise ValueError('Expected one error row per descriptor')
    for e in errors:
        if e['arm'] != arm:
            raise ValueError('Saved error arm differs')
        n, finite, failed = (int(e[k]) for k in ('original_N', 'finite_N', 'failed_N'))
        if n != 11369 or min(finite, failed) < 0 or finite + failed != n:
            raise ValueError('Forward finite/failure counts do not reconcile')
        for metric in METRICS:
            if finite == 0 and e[metric] == '':
                continue
            if not math.isfinite(float(e[metric])) or float(e[metric]) < 0:
                raise ValueError('Invalid saved error metric')
    if set(r['proxy_counts']) != set(EVIDENCE):
        raise ValueError('Expected SELF and original-fixed COMMON only')
    for c in r['proxy_counts'].values():
        check_counts(c)
    pins = {'receipt': digest(raw), 'common_test_forward_errors.csv': error_sha}
    if 'target_hit_counts.csv' in r['outputs']:
        saved, hit_sha = checked_table(root, r, 'target_hit_counts.csv')
        if len(saved) != 2 or {x['evidence_class'] for x in saved} != set(EVIDENCE):
            raise ValueError('Expected two saved hit rows')
        for x in saved:
            check_counts(x)
            c = r['proxy_counts'][x['evidence_class']]
            if any(float(x[k]) != float(c[k]) for k in ('original_N', 'hits', 'fraction', 'request_failures')):
                raise ValueError('Bound saved hit table differs from completion')
        pins['target_hit_counts.csv'] = hit_sha
    elif primary:
        # Archived primary exports have no separate 2-row hit table. Validate
        # their saved target-bin arithmetic against the immutable count receipt.
        bins, bin_sha = checked_table(root, r, 'target_coordinate_hotspots.csv')
        if len(bins) != 1600:
            raise ValueError('Archived primary target-bin table is incomplete')
        for evidence in EVIDENCE:
            c = r['proxy_counts'][evidence]
            for projection in ('lp_ls', 'q_k'):
                rows = [v for v in bins if v['evidence_class'] == evidence and v['projection'] == projection]
                keys = {(int(v['bin_x']), int(v['bin_y'])) for v in rows}
                if len(rows) != 400 or keys != {(x, y) for x in range(20) for y in range(20)}:
                    raise ValueError('Archived grid missing/duplicate bins')
                if any(v['arm'] != arm or v['coordinate_basis'] != 'REQUEST_TARGET_NOT_RESPONSE' for v in rows):
                    raise ValueError('Wrong archived coordinate basis or arm')
                n = sum(int(v['total_requests']) for v in rows)
                h = sum(int(v['hit_requests']) for v in rows)
                f = sum(int(v['failed_requests']) for v in rows)
                saved = c['projections'][projection]
                if saved['outside'] != 0 or (n, h, f) != (10000, c['hits'], c['request_failures']):
                    raise ValueError('Primary projection cannot verify full original count')
                if (n, h) != (saved['inside'], saved['hits_inside']):
                    raise ValueError('Archived bins disagree with saved projection counts')
        pins['target_coordinate_hotspots.csv'] = bin_sha
    else:
        raise ValueError('New repeat must include bound target_hit_counts.csv')
    base = {'family': family, 'arm': arm, 'model_seed': seed,
            'dataset_total_rows': sum(COUNTS[arm]),
            **{p + '_rows': n for p, n in zip(PARTS, COUNTS[arm])},
            'model_id': r['model_id'], 'weights_sha256': mi['weights_sha256'],
            'receipt_path': str(root / 'COMPLETE.json'), 'receipt_sha256': pins['receipt']}
    erows = [{**base, **{k: e[k] for k in ('feature', 'unit', 'original_N', 'finite_N', 'failed_N')},
              **{k: e[k] for k in METRICS}, 'evidence_class': 'EXISTING_REAL_EM_LABEL_HOLDOUT_NOT_FRESH_EM'} for e in errors]
    hrows = [{**base, 'evidence_class': ev, 'original_N': 10000,
              **{k: r['proxy_counts'][ev][k] for k in ('hits', 'fraction', 'request_failures')},
              'target_sha256': TARGET_SHA, 'fresh_EMX': 'NOT_RUN'} for ev in EVIDENCE]
    return {'errors': erows, 'hits': hrows, 'pins': pins,
            'dataset_contract_sha256': r['dataset_contract_sha256']}


def resolve_manifest(path, primary_override=None):
    cfg = read_json(path)
    primary_cfg = checked_json(cfg['primary_evaluation_config'])
    root = Path(primary_override or cfg.get('primary_analysis_root') or primary_cfg['output_root'])
    jobs = cfg['jobs']
    expected = {(family, arm, seed) for family, arms in FAMILIES.items() for arm in arms for seed in (17, 29)}
    actual = {(j['family'], j['arm'], int(j['model_seed'])) for j in jobs}
    if len(jobs) != 12 or actual != expected:
        raise ValueError('Manifest must declare the exact twelve new repeat jobs')
    specs = []
    for family, arms in FAMILIES.items():
        for arm in arms:
            primary_dir = root / 'results' / arm if (root / 'results').is_dir() else root / arm
            specs.append((family, arm, SEEDS[0], primary_dir, None))
            for seed in (17, 29):
                job = next(j for j in jobs if (j['family'], j['arm'], int(j['model_seed'])) == (family, arm, seed))
                specs.append((family, arm, seed, Path(job['eval_dir']), Path(job['run_dir'])))
    return cfg, specs


def collect(specs):
    loaded, missing = {}, []
    for family, arm, seed, root, run in specs:
        key = (family, arm, seed)
        if not (root / 'COMPLETE.json').exists():
            reason = 'EVALUATION_COMPLETE_MISSING'
            if run is not None and (run / 'ENTRY_TERMINAL.json').is_file():
                terminal = read_json(run / 'ENTRY_TERMINAL.json')
                reason = 'TRAINING_' + str(terminal.get('status', 'UNKNOWN')) + '_EVALUATION_MISSING'
            missing.append({'family': family, 'arm': arm, 'model_seed': seed, 'status': 'NOT_COMPLETE', 'reason': reason, 'expected_receipt': str(root / 'COMPLETE.json')})
            continue
        try:
            loaded[key] = load_arm(root, family, arm, seed)
        except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
            missing.append({'family': family, 'arm': arm, 'model_seed': seed, 'status': 'INVALID', 'reason': str(exc), 'expected_receipt': str(root / 'COMPLETE.json')})
    return loaded, missing


def aggregate_family(family, loaded):
    required = {(family, arm, seed) for arm in FAMILIES[family] for seed in SEEDS}
    if not required.issubset(loaded):
        raise ValueError('NOT_COMPLETE: all nine arm/seed evaluations required')
    if len({loaded[k]['dataset_contract_sha256'] for k in required}) != 1:
        raise ValueError('Family dataset contracts differ')
    errors = [r for arm in FAMILIES[family] for seed in SEEDS for r in loaded[family, arm, seed]['errors']]
    hits = [r for arm in FAMILIES[family] for seed in SEEDS for r in loaded[family, arm, seed]['hits']]
    means_e, means_h = [], []
    for arm in FAMILIES[family]:
        for feature in FEATURES:
            rows = [r for r in errors if r['arm'] == arm and r['feature'] == feature]
            for metric in METRICS:
                values = [float(r[metric]) for r in rows if r[metric] != '']
                valid = len(values) == 3
                means_e.append({'family': family, 'arm': arm, 'feature': feature, 'metric': metric,
                                'unit': rows[0]['unit'], 'required_seed_count': 3, 'finite_seed_count': len(values),
                                'mean': statistics.mean(values) if valid else '',
                                'sample_SD_ddof1': statistics.stdev(values) if valid else '',
                                'status': 'COMPLETE' if valid else 'NOT_ESTIMABLE_MISSING_METRIC',
                                'common_test_N_per_seed': 11369, 'seeds': '20260713;17;29'})
        for ev in EVIDENCE:
            values = [float(r['fraction']) for r in hits if r['arm'] == arm and r['evidence_class'] == ev]
            means_h.append({'family': family, 'arm': arm, 'evidence_class': ev, 'required_seed_count': 3,
                            'mean': statistics.mean(values), 'sample_SD_ddof1': statistics.stdev(values),
                            'original_target_N_per_seed': 10000, 'unique_frozen_targets': 10000,
                            'seeds': '20260713;17;29', 'fresh_EMX': 'NOT_RUN'})
    return errors, hits, means_e, means_h


def draw(family, errors, hits, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.titlesize': 9.5, 'axes.labelsize': 9,
                         'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                         'legend.fontsize': 8.5, 'svg.fonttype': 'none'})
    arms = FAMILIES[family]
    width = 184.2734722222222 / 25.4
    label = 'Dataset totals' if family == 'dataset' else 'Full-tandem hidden width'
    colors = ('#1f5d8c', '#b76a15')
    fig, axes = plt.subplots(2, 2, figsize=(width, 118.11 / 25.4), layout='constrained')
    for ax, feature in zip(axes.flat, FEATURES):
        for j, metric in enumerate(('MAE', 'RMSE')):
            rows = [next(r for r in errors if r['arm'] == arm and r['feature'] == feature and r['metric'] == metric) for arm in arms]
            x = [i + (j - .5) * .15 for i in range(3)]
            y = [float(r['mean']) if r['mean'] != '' else float('nan') for r in rows]
            sd = [float(r['sample_SD_ddof1']) if r['sample_SD_ddof1'] != '' else float('nan') for r in rows]
            ax.errorbar(x, y, yerr=sd, fmt='o-', lw=1, ms=3, capsize=3, color=colors[j], label=metric)
        ax.set_xticks(range(3), arms)
        ax.set_title({'lp_nh': 'Lp', 'ls_nh': 'Ls', 'q_min': 'Qmin', 'k_abs': '|k|'}[feature])
        ax.set_ylabel('Error (nH)' if feature in ('lp_nh', 'ls_nh') else 'Error (dimensionless)')
        ax.grid(alpha=.2)
    axes.flat[0].legend(frameon=False)
    fig.suptitle(label + ': forward EM-label test errors\nThree-seed mean ± sample SD; 11,369 common geometries', fontsize=10)
    for ext in ('svg', 'png'):
        fig.savefig(dest / ('forward_MAE_RMSE_mean_SD.' + ext), dpi=400)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(width, 76.2 / 25.4), layout='constrained')
    error_limits = []
    for j, ev in enumerate(EVIDENCE):
        rows = [next(r for r in hits if r['arm'] == arm and r['evidence_class'] == ev) for arm in arms]
        error_limits.extend(v for r in rows for v in (r['mean'] - r['sample_SD_ddof1'], r['mean'] + r['sample_SD_ddof1']))
        ax.errorbar([i + (j - .5) * .08 for i in range(3)], [r['mean'] for r in rows],
                    yerr=[r['sample_SD_ddof1'] for r in rows], fmt='o-', lw=1, ms=3,
                    capsize=3, color=colors[j], label='SELF_PROXY' if j == 0 else 'COMMON_PROXY')
    ax.set_xticks(range(3), arms)
    ax.set_ylim(min(0, min(error_limits) - .01), max(1, max(error_limits) + .01))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel('Joint target-hit fraction')
    ax.set_title(label + ': mean ± sample SD across three seeds\nSame 10,000 requests per seed; proxy only', fontsize=10)
    ax.legend(frameon=False, loc='upper right')
    ax.grid(alpha=.2)
    for ext in ('svg', 'png'):
        fig.savefig(dest / ('joint_hit_mean_SD.' + ext), dpi=400)
    plt.close(fig)


def caption(family):
    return f'''# {family}: three-seed results

Points and error bars are the arithmetic mean and sample standard deviation (denominator n−1) across the exact model seeds {{20260713, 17, 29}}, n=3. Bars are not confidence intervals and the repeated evaluations do not create 30,000 independent target requests. The same frozen 10,000 targets and original 11,369-geometry held-out EM-label set are used for each seed. Per-seed finite/failure counts and original values remain in the source CSVs. Missing arms or failed evaluations prevent this family from being released as a complete three-seed comparison.

Forward panels show MAE and RMSE in each descriptor's units, using each model's embedded train-only normalizer. These are predictions against existing EM labels, not fresh EM of newly generated geometries. P95 values are retained in the source tables; the mean of three seed-specific P95 values is not a pooled-error P95. Each family is separate; no ranking across dataset and width families is implied.

Joint target hits require the original saved numerical/decoder validity, no more than 10% relative errors in Lp, Ls and |k|, and Qmin at least its requested minimum. There is no Q scan or added +2 training margin at inference. Failure requests remain in each 10,000 denominator. SELF_PROXY uses each arm's own F. COMMON_PROXY always uses the original seed20260713 F, not a newly selected F per seed. Neither is physical ground truth; the original D100k seed20260713 is scored by its own F in both cases, which limits independence of COMMON_PROXY.

Dataset totals correspond to actual train/validation counts 8,000/1,000; 40,000/5,000; and 78,891/9,740. Thus the dataset family changes both training and validation populations. D10k/D50k subset indices are fixed across seeds: these standard deviations describe training randomness conditional on the fixed mother dataset, fixed subsets and fixed common forward reference, not subset-sampling uncertainty or a deployment-population confidence interval. The original D100k seed20260713 is historical selected-checkpoint reuse with unknown historical timing/environment; added seed17/29 D100k models are newly trained. Width models all use the same original100K partition and vary both complete F/I hidden widths together. Three seeds quantify only this limited training-seed variability, not universal physical superiority. No timing means or causal speed claims are supplied.

Figure width is 184.273 mm, all text is at least 8.5 pt, SVG text is editable, and PNGs are 400 dpi. No inference, training, raw100K reread, new EM, primary-hotspot redraw or pooled per-target recomputation is performed by this callback. Figure rendering awaits visual review after actual completion.
'''


def run(manifest, out, primary_override=None, check_only=False):
    cfg, specs = resolve_manifest(manifest, primary_override)
    loaded, missing = collect(specs)
    overview = {'schema': 'three_seed_closed_table_aggregation.v1', 'seeds': list(SEEDS),
                'available_evaluations': len(loaded), 'expected_evaluations': 18,
                'missing_or_invalid': missing, 'check_only': check_only, 'families': {}}
    for family, arms in FAMILIES.items():
        absent = [r for r in missing if r['family'] == family]
        if absent:
            overview['families'][family] = {'status': 'NOT_COMPLETE', 'missing_or_invalid': len(absent)}
            continue
        if check_only:
            overview['families'][family] = {'status': 'INPUTS_COMPLETE_NOT_AGGREGATED'}
            continue
        errors, hits, means_e, means_h = aggregate_family(family, loaded)
        identity = {'family': family, 'seeds': list(SEEDS), 'manifest_sha256': sha(manifest),
                    'script_sha256': sha(__file__), 'inputs': {f'{a}__seed{s}': loaded[family, a, s]['pins'] for a in arms for s in SEEDS}}
        fingerprint = digest(json.dumps(identity, sort_keys=True).encode())
        dest = Path(out) / family / fingerprint[:16]
        if (dest / 'COMPLETE.json').exists():
            done = read_json(dest / 'COMPLETE.json')
            if done['identity'] != identity or any(sha(dest / n) != h for n, h in done['outputs'].items()):
                raise ValueError('Existing aggregate output differs; preserved without overwrite')
            overview['families'][family] = {'status': 'REUSED_COMPLETE', 'path': str(dest)}
            continue
        dest.mkdir(parents=True, exist_ok=False)
        write_json(dest / 'STARTED.json', {'identity': identity, 'utc': datetime.now(timezone.utc).isoformat()})
        write_csv(dest / 'per_seed_forward.csv', errors)
        write_csv(dest / 'per_seed_target_hits.csv', hits)
        write_csv(dest / 'forward_mean_sample_SD.csv', means_e)
        write_csv(dest / 'target_hit_mean_sample_SD.csv', means_h)
        (dest / 'CAPTIONS.md').write_text(caption(family))
        try:
            draw(family, means_e, means_h, dest)
        except Exception as exc:
            write_json(dest / 'FAILED.json', {'status': 'PLOT_FAILED_SOURCES_PRESERVED', 'reason': str(exc)})
            raise
        files = {p.name: sha(p) for p in dest.iterdir() if p.is_file()}
        write_json(dest / 'COMPLETE.json', {'status': 'GENERATED_PENDING_VISUAL_REVIEW', 'identity': identity,
                   'utc': datetime.now(timezone.utc).isoformat(), 'outputs': files, 'replicate_count': 3,
                   'uncertainty': 'SAMPLE_STANDARD_DEVIATION_DDOF1_NOT_CI', 'new_inference': 0,
                   'new_training': 0, 'new_EMX': 0, 'unique_target_requests': 10000})
        overview['families'][family] = {'status': 'GENERATED_PENDING_VISUAL_REVIEW', 'path': str(dest)}
    if not check_only and missing:
        # Small immutable missing-state cut, not a monitor and not a partial mean.
        fingerprint = digest(json.dumps(overview, sort_keys=True).encode())
        dest = Path(out) / 'incomplete' / fingerprint[:16]
        if not dest.exists():
            dest.mkdir(parents=True)
            write_json(dest / 'NOT_COMPLETE.json', overview)
            write_csv(dest / 'missing_items.csv', missing)
    return overview


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--primary-root', type=Path)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    print(json.dumps(run(a.manifest, a.out, a.primary_root, a.check_only), indent=2))


if __name__ == '__main__':
    main()
