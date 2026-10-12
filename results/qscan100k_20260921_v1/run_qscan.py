#!/usr/bin/env python3
"""Original seed13 D100k, preserved T10000 triplets, external symmetric Q scan.

Local inference only. Original weights, decoder, normalizer and targets read-only.
No remote, training, physical execution, repair, label filtering or target redraw.
"""
import csv
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import numpy as np

OUT = Path(__file__).resolve().parent
W = OUT.parent
ROOT = W.parents[1]
TOOLKIT = ROOT / 'reports/frozen_mlp_synthesis_toolkit_20260804/toolkit/rfic_transformer_inverse_synthesis_toolkit_v1'
BUNDLE = TOOLKIT / 'model'
LOADER = TOOLKIT / 'src/rfic_transformer_inverse_design/synthesis/frozen_mlp.py'
TARGET = W / 'figure_data/random10000_targets.csv'
FEATURES = ['lp_nh', 'ls_nh', 'q_scalar', 'k_abs']
SPANS = np.array([2.5, 2.5, 20., .8])
Q = np.arange(10, 21)
EXPECTED = {
    LOADER: 'd962de6f923234d5e9702556a1df16ff63d84df59c2684f3b46e469f2fe39eb9',
    BUNDLE / 'model_manifest.json': '37a47545bc94193f4d2ef0afed9697fcd0b1b2a5f579aeaf118330d526fb396c',
    BUNDLE / 'physical_feature_tandem_inverse_summary.json': 'f05c4c0b3937bf9107f95d38d14679995aedc936071852ac7696f9838c5d43c1',
    BUNDLE / 'physical_feature_tandem_inverse_weights.npz': 'c61a93d56136ad21686863aae72984c49eb0d33ce766923c8ec2d03d9c3177dd',
    TARGET: 'a71e43284e1e629d14892d6e0ff72a4c2fbdf64616784884aa854bdd6984b223',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def predict(module, model, requests):
    n = model.model['normalization']
    raw = module._predict((requests - n['x_mean']) / n['x_scale'],
                          model.model['inverse_weights'], model.model['inverse_biases'])
    standardized_geometry = module._project_inverse_geometry(
        raw, n['geometry_lower'], n['geometry_upper'], mode=model.model['projection_mode'],
        normalization=n, topology_contract=model.model['topology_contract'])
    geometry = standardized_geometry * n['y_scale'] + n['y_mean']
    proxy = module._predict(standardized_geometry, model.model['forward_weights'],
                            model.model['forward_biases']) * n['x_scale'] + n['x_mean']
    return geometry, proxy


def score_symmetric(proxy, requests):
    return np.sqrt(np.mean(((proxy - requests) / SPANS) ** 2, axis=-1))


def execute():
    if (OUT / 'STARTED.json').exists():
        raise FileExistsError('Existing run: reuse outputs or inspect failure; no silent rerun')
    started = utc()
    tick = time.monotonic()
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise ValueError('Input identity mismatch: ' + str(path))
    with TARGET.open(encoding='utf-8-sig', newline='') as f:
        source = list(csv.DictReader(f))
    ids = [r['target_id'] for r in source]
    triplets = np.array([[float(r['target_lp_nh']), float(r['target_ls_nh']),
                          float(r['target_k_abs'])] for r in source])
    original_q = np.array([float(r['target_q_min']) for r in source])
    assert len(source) == len(set(ids)) == len(np.unique(triplets, axis=0)) == 10000
    assert np.isfinite(triplets).all() and all(float(r['frequency_ghz']) == 15 for r in source)
    spec = importlib.util.spec_from_file_location('original100k_loader_qscan', LOADER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.FrozenInverseModel.load(BUNDLE)
    assert model.model_id == 'current_foundry_qmin_response_only_seed20260713'
    requests = np.empty((10000, 11, 4))
    requests[:, :, 0] = triplets[:, 0, None]
    requests[:, :, 1] = triplets[:, 1, None]
    requests[:, :, 2] = Q[None, :]
    requests[:, :, 3] = triplets[:, 2, None]
    flat = requests.reshape(-1, 4)
    assert np.all(flat >= model.support_lower) and np.all(flat <= model.support_upper)
    score_description = 'sqrt(mean(((F(I([Lp,Ls,q,abs(k)]))-[Lp,Ls,q,abs(k)])/[2.5,2.5,20,0.8])**2)); symmetric Q; ascending-q exact ties'
    contract = dict(
        schema='original100k_external_symmetric_qscan.v1', created_utc=started,
        inputs=[dict(path=str(p), sha256=s) for p, s in EXPECTED.items()],
        script_sha256=sha(__file__), model_id=model.model_id, model_seed=model.model_seed,
        frequency_ghz=15, source_rows=100000, train_rows=78891, validation_rows=9740, test_rows=11369,
        model_training_Q_semantics='minimum; margin=2 in original training, unchanged',
        external_evaluation_Q_semantics='exact symmetric error relative to each scanned q, NOT minimum',
        n_requests=10000, n_candidates=110000, q_values=Q.tolist(),
        target_use='Reuse exact source IDs and Lp/Ls/K. Source random Q archived but NOT used; no target redraw.',
        target_range_min=triplets.min(axis=0).tolist(), target_range_max=triplets.max(axis=0).tolist(),
        selection_score=score_description, feature_order=FEATURES, spans=SPANS.tolist(),
        selection_failure='Nonfinite scores or geometry receive infinite score; all-failed requests remain denominator. No replacement.',
        hit_rule='all four abs(proxy-target)/abs(target)<=threshold AND finite original-decoder bounds and physical proxy validity; symmetric Q; no margin',
        physical_proxy_validity='Lp,Ls,Q positive; 0<K_abs<1; not DRC/EM or SRF qualification',
        geometry_handling='original floating decoder output, no grid/repair/GDS regeneration',
        hit_thresholds=[.05, .10, .15, .20], low_N=10,
        binning={'lp_nh': np.linspace(.5, 3, 21).tolist(), 'ls_nh': np.linspace(.5, 3, 21).tolist(),
                 'k_abs': np.linspace(0, .8, 21).tolist(), 'selected_q': np.arange(9.5, 21.5).tolist()},
        evidence='SELF_PROXY_NOT_FRESH_EMX', fresh_emx='NOT_RUN_FOR_QSCAN_CANDIDATES',
        old64='Remains paired to prior one-shot geometries, not automatically evidence for selected scan geometries',
        inference_batch_rows=2048, numpy=np.__version__, python=sys.version,
        runtime='local; native server untouched; CPU thread cap 4')
    write_json(OUT / 'CONTRACT.json', contract)
    write_json(OUT / 'STARTED.json', dict(utc=started, pid=os.getpid(), contract_sha256=sha(OUT / 'CONTRACT.json')))

    # Actual original API vs batch implementation checks, no replacement model.
    probe_indices = [0, 5, 10, 55000, 109999]
    pg, pp = predict(module, model, flat[probe_indices])
    equivalence = []
    geom_names = [k.removeprefix('geom__') for k in model.geometry_columns]
    for j, index in enumerate(probe_indices):
        t = flat[index]
        result = model.infer(module.PhysicalTarget('check_' + str(index), *t))
        g = np.array([result.geometry_um[k] for k in geom_names])
        p = np.array([result.proxy_diagnostic[k] for k in ['Lp_nH', 'Ls_nH', 'Q_proxy', 'K_abs_proxy']])
        np.testing.assert_allclose(pg[j], g, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(pp[j], p, rtol=1e-11, atol=1e-11)
        equivalence.append(dict(flat_index=index, geometry_max_abs=float(np.max(abs(pg[j]-g))),
                                proxy_max_abs=float(np.max(abs(pp[j]-p)))))
    # Both signs of Q residual receive equal scores; exact tie picks first/lower Q.
    test_target = np.array([[1., 1., 15., .5], [1., 1., 15., .5]])
    test_proxy = test_target.copy(); test_proxy[:, 2] += np.array([-1., 1.])
    np.testing.assert_array_equal(score_symmetric(test_proxy, test_target), [.025, .025])
    assert int(Q[np.argmin(np.ones(11))]) == 10
    write_json(OUT / 'IMPLEMENTATION_CHECK.json', dict(status='PASS', original_api_equivalence=equivalence,
        symmetric_Q_sign_check=True, low_Q_exact_tie=True, no_one_sided_shortfall=True))

    geom = np.empty((len(flat), 10)); proxy = np.empty((len(flat), 4))
    for start in range(0, len(flat), 2048):
        end = min(start + 2048, len(flat))
        geom[start:end], proxy[start:end] = predict(module, model, flat[start:end])
        if start % 20480 == 0:
            print(json.dumps(dict(stage='Q_SCAN_INFERENCE', completed=end, total=len(flat))), flush=True)
    geom = geom.reshape(10000, 11, 10); proxy = proxy.reshape(10000, 11, 4)
    scores = score_symmetric(proxy, requests)
    finite = np.isfinite(geom).all(axis=2) & np.isfinite(proxy).all(axis=2) & np.isfinite(scores)
    ranked = np.where(finite, scores, np.inf)
    chosen = np.argmin(ranked, axis=1)
    rows = np.arange(10000)
    selected_t, selected_p, selected_g = requests[rows, chosen], proxy[rows, chosen], geom[rows, chosen]
    n = model.model['normalization']
    lower = n['geometry_lower'] * n['y_scale'] + n['y_mean']
    upper = n['geometry_upper'] * n['y_scale'] + n['y_mean']
    geometry_ok = np.isfinite(selected_g).all(axis=1) & (selected_g >= lower-1e-9).all(axis=1) & (selected_g <= upper+1e-9).all(axis=1)
    valid = finite[rows, chosen] & geometry_ok & (selected_p[:, :3] > 0).all(axis=1) & (selected_p[:, 3] > 0) & (selected_p[:, 3] < 1)
    residual = selected_p - selected_t
    relative = abs(residual) / abs(selected_t)
    hits = {str(int(round(t*100))): valid & (relative <= t).all(axis=1) for t in [.05,.10,.15,.20]}
    # Store all actual candidates; retained arrays permit verification without repeat inference.
    with (OUT / 'all_candidates.npz').open('xb') as f:
        np.savez_compressed(f, target_ids=np.array(ids), requests=requests, geometry=geom, proxy=proxy,
                            scores=scores, finite=finite, selected_index=chosen, original_random_Q=original_q)
    fields = ['target_id','q_target','selected','finite','selection_nrmse'] + ['target_'+k for k in FEATURES] + ['proxy_'+k for k in FEATURES] + ['geometry_'+k for k in geom_names]
    with gzip.open(OUT / 'all_110000_candidates.csv.gz', 'xt', encoding='utf-8', newline='') as f:
        writer = csv.writer(f); writer.writerow(fields)
        for i, target_id in enumerate(ids):
            for j, q in enumerate(Q):
                writer.writerow([target_id,int(q),j==chosen[i],bool(finite[i,j]),scores[i,j],*requests[i,j],*proxy[i,j],*geom[i,j]])
    selected_rows = []
    for i, target_id in enumerate(ids):
        row = dict(target_id=target_id, frequency_ghz=15, selected_q=int(Q[chosen[i]]),
                   source_random_Q_ignored=float(original_q[i]), selected_candidate_id=f'{target_id}_q{Q[chosen[i]]:02d}',
                   geometry_sha256=module._vector_digest(selected_g[i]), candidate_finite=bool(finite[i, chosen[i]]),
                   geometry_bounds_valid=bool(geometry_ok[i]), proxy_valid=bool(valid[i]),
                   status='COMPLETE' if finite[i, chosen[i]] else 'FAILED_NO_FINITE_CANDIDATE',
                   selection_nrmse=float(scores[i,chosen[i]]), evidence='SELF_PROXY', fresh_EMX_status='NOT_RUN')
        for prefix, values in [('target',selected_t[i]),('proxy',selected_p[i]),('signed_error',residual[i]),('absolute_error',abs(residual[i])),('relative_error_pct',relative[i]*100)]:
            row.update({prefix+'_'+k:float(v) for k,v in zip(FEATURES,values)})
        row.update({'geometry_'+k:float(v) for k,v in zip(geom_names,selected_g[i])})
        row.update({'joint_hit_'+t+'pct':bool(v[i]) for t,v in hits.items()})
        selected_rows.append(row)
    write_csv(OUT / 'selected_10000_results.csv', selected_rows)
    bins = []; coverage = {}
    projections = [('lp_ls',0,1,np.linspace(.5,3,21),np.linspace(.5,3,21),'Target Lp (nH)','Target Ls (nH)'),
                   ('lp_k',0,3,np.linspace(.5,3,21),np.linspace(0,.8,21),'Target Lp (nH)','Target |k|'),
                   ('ls_k',1,3,np.linspace(.5,3,21),np.linspace(0,.8,21),'Target Ls (nH)','Target |k|'),
                   ('q_selected_k',2,3,np.arange(9.5,21.5),np.linspace(0,.8,21),'Selected Q target','Target |k|')]
    for name, xi, yi, xe, ye, xlabel, ylabel in projections:
        counts = np.histogram2d(selected_t[:,xi], selected_t[:,yi], bins=(xe,ye))[0].astype(int)
        hc = np.histogram2d(selected_t[hits['10'],xi], selected_t[hits['10'],yi], bins=(xe,ye))[0].astype(int)
        assert counts.sum() == 10000 and hc.sum() == hits['10'].sum()
        coverage[name] = dict(n_total=int(counts.sum()), n_hit=int(hc.sum()), n_empty=int((counts==0).sum()),
                              n_low=int(((counts>0)&(counts<10)).sum()))
        for ix in range(len(xe)-1):
            for iy in range(len(ye)-1):
                count, hit = int(counts[ix,iy]), int(hc[ix,iy])
                bins.append(dict(projection=name,x_label=xlabel,y_label=ylabel,bin_x=ix,bin_y=iy,
                    x_lower=xe[ix],x_upper=xe[ix+1],y_lower=ye[iy],y_upper=ye[iy+1],n_total=count,n_hit=hit,
                    hit_fraction=hit/count if count else '',empty=count==0,low_N=0<count<10))
    write_csv(OUT / 'hotspot_bins.csv', bins)
    error_rows = []
    for j, feature in enumerate(FEATURES):
        values = residual[np.isfinite(residual[:,j]),j]
        rel = relative[np.isfinite(relative[:,j]),j]
        error_rows.append(dict(feature=feature,unit='nH' if j<2 else 'dimensionless',n=int(len(values)),
            MAE=float(np.mean(abs(values))),RMSE=float(np.sqrt(np.mean(values**2))),
            P95_absolute_error=float(np.percentile(abs(values),95,method='linear')),
            mean_relative_error_pct=float(np.mean(rel)*100),P95_relative_error_pct=float(np.percentile(rel,95,method='linear')*100),
            hits_10pct=int((valid & (relative[:,j]<=.1)).sum()),original_denominator=10000))
    write_csv(OUT / 'physical_error_summary.csv', error_rows)
    write_csv(OUT / 'selected_Q_counts.csv', [dict(q=int(q),n_selected=int((Q[chosen]==q).sum()),
             n_hit_10pct=int(((Q[chosen]==q)&hits['10']).sum())) for q in Q])
    summary = dict(status='COMPLETE',completed_utc=utc(),elapsed_seconds=time.monotonic()-tick,
        n_requests=10000,n_candidates=110000,n_finite_candidates=int(finite.sum()),
        n_failed_requests=int((~finite[rows,chosen]).sum()),n_proxy_valid_requests=int(valid.sum()),
        n_proxy_invalid_requests=int((~valid).sum()),joint_hits={t:int(v.sum()) for t,v in hits.items()},
        feature_hits_10pct={row['feature']:row['hits_10pct'] for row in error_rows},
        model_id=model.model_id,model_seed=model.model_seed,selection_score=score_description,
        weight_sha256=EXPECTED[BUNDLE / 'physical_feature_tandem_inverse_weights.npz'],target_sha256=EXPECTED[TARGET],
        model_training_Q_semantics='minimum with margin2; unchanged weights',
        evaluation_Q_semantics='symmetric scanned Q target, not a minimum',
        dataset_total=100000,gradient_training_rows=78891,validation_rows=9740,test_rows=11369,
        frame='same 10000 saved IDs and Lp/Ls/K; ignored source Q replaced by declared Q scan',
        target_triplet_min=triplets.min(axis=0).tolist(),target_triplet_max=triplets.max(axis=0).tolist(),
        proxy_only=True,fresh_emx_new=0,old64_not_reused_as_scan_validation=True,
        grids=coverage,physical_errors=error_rows,
        inputs=contract['inputs'],contract_sha256=sha(OUT / 'CONTRACT.json'),script_sha256=sha(__file__))
    summary['outputs']={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix not in ['.py'] and p.name!='.run.lock'}
    write_json(OUT / 'SUMMARY.json',summary)
    print(json.dumps({k:summary[k] for k in ['status','elapsed_seconds','n_requests','n_candidates','n_failed_requests','n_proxy_invalid_requests','joint_hits','feature_hits_10pct']}),flush=True)


if __name__ == '__main__':
    with (OUT / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute()
        except BaseException as exc:
            if not (OUT / 'FAILED.json').exists():
                write_json(OUT / 'FAILED.json',dict(utc=utc(),error=type(exc).__name__,message=str(exc)))
            raise
