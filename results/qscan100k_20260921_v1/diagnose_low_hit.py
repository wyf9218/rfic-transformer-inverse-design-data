"""Saved-candidate diagnostic only: no new model inference, training or EMX."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'diagnostic_20260921_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    summary = json.loads((ROOT/'SUMMARY.json').read_text())
    assert sha(ROOT/'all_candidates.npz') == summary['outputs']['all_candidates.npz']
    a = np.load(ROOT/'all_candidates.npz', allow_pickle=False)
    target, proxy, geom = a['requests'], a['proxy'], a['geometry']
    index = a['selected_index']; rows = np.arange(len(index))
    weights_path = next(Path(x['path']) for x in summary['inputs'] if x['path'].endswith('physical_feature_tandem_inverse_weights.npz'))
    with np.load(weights_path, allow_pickle=False) as weights:
        lower = weights['normalization__geometry_lower']*weights['normalization__y_scale']+weights['normalization__y_mean']
        upper = weights['normalization__geometry_upper']*weights['normalization__y_scale']+weights['normalization__y_mean']
    valid = np.isfinite(geom).all(2)&np.isfinite(proxy).all(2)&(geom>=lower-1e-9).all(2)&(geom<=upper+1e-9).all(2)&(proxy[:,:,:3]>0).all(2)&(proxy[:,:,3]>0)&(proxy[:,:,3]<1)
    relative = abs(proxy-target)/abs(target)
    feature_pass = relative <= .1
    candidate_hit = valid & feature_pass.all(2)
    selected = candidate_hit[rows,index]
    any_hit = candidate_hit.any(1)
    chosen_features = feature_pass[rows,index]
    assert selected.sum() == 2073
    assert int(valid.sum()) == 110000
    lp,ls,k = target[:,0,0],target[:,0,1],target[:,0,3]
    ratio = ls/lp
    old = (lp>=.5)&(lp<=1.5)&(ls>=.5)&(ls<=1.5)&(k>=.5)&(k<=.8)&(ratio>=.67)&(ratio<=1.5)
    ratio_ok = (ratio>=.67)&(ratio<=1.5)
    def group(name, mask):
        count=int(mask.sum()); hit=int(selected[mask].sum())
        return dict(group=name,n=count,hits=hit,percent=100*hit/count if count else None,any_Q_hit_count=int(any_hit[mask].sum()))
    groups=[group('all',np.ones(10000,dtype=bool)),group('inside_old_target_domain',old),group('outside_old_target_domain',~old),
            group('Ls_over_Lp_0p67_to_1p5',ratio_ok),group('outside_Ls_over_Lp_0p67_to_1p5',~ratio_ok),
            group('both_L_at_most_1p5', (lp<=1.5)&(ls<=1.5)),group('any_L_above_1p5',(lp>1.5)|(ls>1.5))]
    fields=['Lp','Ls','Q','K']
    single_only={f:int(((~chosen_features[:,j])&np.delete(chosen_features,j,axis=1).all(1)).sum()) for j,f in enumerate(fields)}
    patterns=[]
    for bits in range(16):
        mask=(~chosen_features == np.array([bool(bits&(1<<j)) for j in range(4)])).all(1)
        patterns.append(dict(failed_features=[fields[j] for j in range(4) if bits&(1<<j)],n=int(mask.sum())))
    assert sum(p['n'] for p in patterns)==10000
    examples=[]
    for row in np.flatnonzero(any_hit&~selected)[:3]:
        selected_j=int(index[row]); alternatives=np.flatnonzero(candidate_hit[row]); alternative_j=int(alternatives[np.argmin(a['scores'][row,alternatives])])
        examples.append(dict(target_id=str(a['target_ids'][row]),selected_Q=int(target[row,selected_j,2]),
            selected_score=float(a['scores'][row,selected_j]),selected_relative_error_pct=(relative[row,selected_j]*100).tolist(),
            passing_Q=int(target[row,alternative_j,2]),passing_score=float(a['scores'][row,alternative_j]),
            passing_relative_error_pct=(relative[row,alternative_j]*100).tolist()))
    result=dict(status='COMPLETE',utc=datetime.now(timezone.utc).isoformat(),source_npz_sha256=sha(ROOT/'all_candidates.npz'),
        source_summary_sha256=sha(ROOT/'SUMMARY.json'),script_sha256=sha(Path(__file__)),n_requests=10000,n_candidates=110000,
        mutually_exclusive_outcomes=dict(selected_hits=int(selected.sum()),passing_candidate_not_selected=int((any_hit&~selected).sum()),
                                        no_passing_candidate_among_saved_11=int((~any_hit).sum())),
        any_Q_joint_hits=int(any_hit.sum()),groups=groups,
        feature_pass_counts={f:int(chosen_features[:,j].sum()) for j,f in enumerate(fields)},single_feature_only_failures=single_only,
        exclusive_failure_patterns=patterns,Q_overshoot_above10pct=int((proxy[rows,index,2]>1.1*target[rows,index,2]).sum()),
        Q_undershoot_below10pct=int((proxy[rows,index,2]<.9*target[rows,index,2]).sum()),
        passing_Lp_Ls_K_count=int(chosen_features[:,[0,1,3]].all(1).sum()),selection_tradeoff_examples=examples,
        qualification='POST_HOC_DIAGNOSTIC_ONLY; selection/rules/figures unchanged. Any-Q bound is only over these 11 saved proxy candidates, not physical feasibility or achievable best design.',
        comparison='Old-domain slice is a different finite target sample and model from historical 8000; not a paired causal model comparison.',
        no_new_inference=True,no_new_EMX=True)
    (OUT/'DIAGNOSTIC.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    cards=[
        dict(id='candidate-outcomes',title='已保存11个候选的命中与选择',queries=[dict(id='saved-candidates',source=dict(label='100K Q扫描保存结果',
             caveats=[result['qualification']]),rows=[dict(outcome=k,n=v) for k,v in result['mutually_exclusive_outcomes'].items()],columns=['outcome','n'])]),
        dict(id='target-domain',title='目标范围内外的描述性切片',queries=[dict(id='target-slices',source=dict(label='同一10000请求的固定范围切片',
             filters=['15 GHz','每请求整数Q=10..20','四项相对误差均不超过10%'],caveats=[result['comparison']]),rows=groups,columns=['group','n','hits','percent'])]),
        dict(id='q-and-feature-errors',title='所选候选的各指标及Q超差',queries=[dict(id='feature-diagnostic',source=dict(label='所选10000候选的双侧误差',
             caveats=['原权重训练为minimum-Q/+2，本次仅作对称Q外部评价。以下是误差分解，不是训练目标的因果贡献。','所有结果是SELF_PROXY，不是EMX物理精度。']),
             rows=[dict(metric=k,n=v) for k,v in {**result['feature_pass_counts'],'Q_only_fail':single_only['Q'],'Q_above_plus10pct':result['Q_overshoot_above10pct'],'Q_below_minus10pct':result['Q_undershoot_below10pct']}.items()],columns=['metric','n'])])]
    (OUT/'sources.json').write_text(json.dumps(dict(schemaVersion=1,items=cards),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','mutually_exclusive_outcomes','groups','single_feature_only_failures','Q_overshoot_above10pct','Q_undershoot_below10pct']},ensure_ascii=False))


if __name__=='__main__':
    main()
