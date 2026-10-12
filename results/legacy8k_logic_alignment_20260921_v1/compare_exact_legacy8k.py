#!/usr/bin/env python3
"""Same saved application8k, old results reused, original100k inferred once.

Descriptive own-proxy comparison, NOT a controlled dataset-size experiment.
No target generation, training, network, physical solver or old-output writes.
"""
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
W = OUT.parent
ROOT = W.parents[1]
HELPER = W / 'qscan100k_20260921_v1/run_qscan.py'
TARGET = ROOT / 'reports/application8k_qscan_v1/application8k_qscan_v1.csv'
OLD_RESULTS = Path('/Users/wyf/Desktop/Q_Optimized_8000_Advisor_Report/csv/selected_best_q_per_target.csv')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def main():
    h = load_module('saved_original100k_qscan_helpers', HELPER)
    if (OUT / 'STARTED.json').exists():
        raise FileExistsError('Run already started: do not repeat inference')
    pins = {
        HELPER: '729d99b62990bc90a3e324832ee2916a0f4b9c1625ed0550102064d5de6a2dac',
        TARGET: 'c12835ab65f898f4b72e2471b125e68fe81edfe6221541ed166c1e21b569a346',
        OLD_RESULTS: 'e4d5b40e62f212952df345ef1078903b595c56b3bb5eafab8c864fb7f3693825',
    }
    pins.update({p:s for p,s in h.EXPECTED.items() if p != h.TARGET})
    for p,s in pins.items():
        assert h.sha(p) == s, str(p)
    source, old = read_csv(TARGET), read_csv(OLD_RESULTS)
    ids = np.array([r['target_id'] for r in source])
    assert len(ids) == len(set(ids)) == 8000
    assert list(ids) == [r['target_id'] for r in old]
    names = ['target_Lp_nH','target_Ls_nH','target_K_abs']
    triplets = np.array([[float(r[k]) for k in names] for r in source])
    old_triplets = np.array([[float(r[k]) for k in names] for r in old])
    # Historical selected CSV uses 12 significant digits; canonical targets 12 decimals.
    np.testing.assert_allclose(triplets, old_triplets, rtol=0, atol=5.1e-12)
    old_t = np.array([[float(r[k]) for k in ['target_Lp_nH','target_Ls_nH','selected_Q','target_K_abs']] for r in old])
    old_p = np.array([[float(r[k]) for k in ['proxy_Lp_nH','proxy_Ls_nH','proxy_Q','proxy_K_abs']] for r in old])
    old_rel = abs(old_p-old_t)/old_t
    thresholds = [5,10,15,20,25]
    old_hits = {str(t): int((old_rel <= t/100).all(axis=1).sum()) for t in thresholds}
    assert old_hits == {'5':4945,'10':6468,'15':7371,'20':7812,'25':7943}
    for t in thresholds:
        flags = np.array([r[f'all_four_within_{t}pct'] == 'True' for r in old])
        np.testing.assert_array_equal(flags, (old_rel <= t/100).all(axis=1))
    np.testing.assert_allclose(h.score_symmetric(old_p, old_t),
        [float(r['selection_score']) for r in old], rtol=1e-8, atol=1e-11)
    module = load_module('original100k_loader_same_legacy8k', h.LOADER)
    model = module.FrozenInverseModel.load(h.BUNDLE)
    assert model.model_id == 'current_foundry_qmin_response_only_seed20260713'
    req = np.empty((8000,11,4))
    req[:,:,0] = triplets[:,0,None]
    req[:,:,1] = triplets[:,1,None]
    req[:,:,2] = h.Q[None,:]
    req[:,:,3] = triplets[:,2,None]
    flat = req.reshape(-1,4)
    assert np.all(flat >= model.support_lower) and np.all(flat <= model.support_upper)
    contract = dict(created_utc=h.utc(), n_targets=8000, n_candidates=88000,
        inputs=[dict(path=str(p),sha256=s) for p,s in pins.items()],
        script_sha256=h.sha(__file__), old_model='real10k_center15ghz_seed20260711',
        new_model=model.model_id, target_rule='Exact saved application8k IDs and triplets; no redraw or replacements',
        score='sqrt(mean(((proxy-target)/[2.5,2.5,20,0.8])**2))',
        q_values=h.Q.tolist(), tie_rule='ascending Q, first argmin',
        hit_rule='After selection: all four abs(proxy-target)/target <= tolerance; symmetric Q, no minimum/margin',
        evidence='DESCRIPTIVE_SELF_PROXY_NOT_FRESH_EMX',
        caveat='Same targets/scan/score/hit rule, but different models, training, decoder and own forward evaluator; not a causal data-size comparison',
        original100k_training_Q='minimum with margin2; unchanged',
        failure_policy='Nonfinite candidates excluded from selection; requests remain denominator; no replacements',
        original_API_batch_equivalence='Reuse qscan100k_20260921_v1/IMPLEMENTATION_CHECK.json and VERIFICATION.json; same helper/weights',
        execution='local CPU cap4; no remote access, training or physical execution')
    h.write_json(OUT/'CONTRACT.json', contract)
    h.write_json(OUT/'STARTED.json', dict(started_utc=h.utc(), contract_sha256=h.sha(OUT/'CONTRACT.json')))
    tick = time.monotonic()
    geom, pred = np.empty((len(flat),10)), np.empty((len(flat),4))
    for start in range(0,len(flat),2048):
        stop = min(start+2048,len(flat))
        geom[start:stop],pred[start:stop] = h.predict(module, model, flat[start:stop])
    elapsed = time.monotonic()-tick
    geom, pred = geom.reshape(8000,11,10), pred.reshape(8000,11,4)
    scores = h.score_symmetric(pred,req)
    finite = np.isfinite(geom).all(axis=2) & np.isfinite(pred).all(axis=2) & np.isfinite(scores)
    chosen = np.argmin(np.where(finite,scores,np.inf),axis=1)
    rows = np.arange(8000)
    t,p,g = req[rows,chosen],pred[rows,chosen],geom[rows,chosen]
    rel = abs(p-t)/t
    hits = {str(tol): (rel <= tol/100).all(axis=1) & finite[rows,chosen] for tol in thresholds}
    with (OUT/'all_original100k_88000_candidates.npz').open('xb') as f:
        np.savez_compressed(f,target_ids=ids,requests=req,proxy=pred,geometry=geom,scores=scores,finite=finite,selected_index=chosen)
    selected = []
    for i,target_id in enumerate(ids):
        r = dict(target_id=target_id,selected_q=int(t[i,2]),selection_score=float(scores[i,chosen[i]]),
            evidence='SELF_PROXY',fresh_EMX='NOT_RUN',finite=bool(finite[i,chosen[i]]))
        for prefix,values in [('target',t[i]),('proxy',p[i]),('error_relative_pct',rel[i]*100)]:
            r.update({prefix+'_'+k:float(v) for k,v in zip(h.FEATURES,values)})
        r.update({f'joint_hit_{tol}pct':bool(hits[str(tol)][i]) for tol in thresholds})
        r['old_real10k_hit_10pct'] = bool((old_rel[i]<=.1).all())
        selected.append(r)
    h.write_csv(OUT/'original100k_selected_same8000.csv', selected)
    comparison = []
    for label, rt, rp, rh, split in [
        ('historical_real10k',old_t,old_p,old_hits,[10000,7871,1227,902]),
        ('original100k_seed13',t,p,{k:int(v.sum()) for k,v in hits.items()},[100000,78891,9740,11369])]:
        r = dict(model=label,target_count=8000,dataset_total=split[0],train_count=split[1],validation_count=split[2],test_count=split[3],evidence='SELF_PROXY')
        r.update({f'joint_hit_{tol}pct':rh[str(tol)] for tol in thresholds})
        for j,k in enumerate(h.FEATURES):
            e = rp[:,j]-rt[:,j]
            r.update({k+'_hits10':int((abs(e)/rt[:,j]<=.1).sum()),k+'_MAE':float(abs(e).mean()),k+'_RMSE':float(np.sqrt(np.mean(e*e)))})
        comparison.append(r)
    h.write_csv(OUT/'same8000_descriptive_comparison.csv', comparison)
    norm = model.model['normalization']
    lower = norm['geometry_lower']*norm['y_scale']+norm['y_mean']
    upper = norm['geometry_upper']*norm['y_scale']+norm['y_mean']
    geometry_valid = (g>=lower-1e-9).all(axis=1)&(g<=upper+1e-9).all(axis=1)
    physical_proxy_valid = (p[:,:3]>0).all(axis=1)&(p[:,3]>0)&(p[:,3]<1)
    all_q_hits = (abs(pred-req)/req <= .1).all(axis=2)&finite
    summary = dict(status='COMPLETE',completed_utc=h.utc(), inference_seconds=elapsed,
        n_targets=8000,n_new_candidate_inferences=88000, n_old_candidate_inferences=0,
        all_candidates_finite=bool(finite.all()), failed_requests=int((~finite[rows,chosen]).sum()),
        selected_geometry_bounds_valid=int(geometry_valid.sum()), selected_physical_proxy_valid=int(physical_proxy_valid.sum()),
        old_hits=old_hits,new_hits={k:int(v.sum()) for k,v in hits.items()},
        new_any_Q_hits_10pct_posthoc_not_selected=int(all_q_hits.any(axis=1).sum()),
        old_feature_hits_10pct=dict(zip(h.FEATURES,(old_rel<=.1).sum(axis=0).astype(int).tolist())),
        new_feature_hits_10pct=dict(zip(h.FEATURES,(rel<=.1).sum(axis=0).astype(int).tolist())),
        both_models_hit10=int(((old_rel<=.1).all(axis=1)&hits['10']).sum()),
        unchanged_methods=True,new_training_runs=0,new_physical_runs=0,
        comparison_scope=contract['caveat'], contract_sha256=h.sha(OUT/'CONTRACT.json'),
        outputs={name:h.sha(OUT/name) for name in ['all_original100k_88000_candidates.npz','original100k_selected_same8000.csv','same8000_descriptive_comparison.csv']})
    h.write_json(OUT/'SUMMARY.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
