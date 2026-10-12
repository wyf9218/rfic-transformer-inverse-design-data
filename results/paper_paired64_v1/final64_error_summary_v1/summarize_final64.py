#!/usr/bin/env python3
"""Deterministic arithmetic on the pinned final64 CSV; no EM/inference/extraction."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

CSV_SHA='feee0b8e67118d2145cce3b629aaca6416cc466423e8b3e472872dab369ebbe9'
RECEIPT_SHA='5e63cae021ab72d620040182f0555815aad3ac910c87407111868501b42de375'
FEATURES=('lp_nh','ls_nh','q_min','k_abs')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def write_csv(path,rows):
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def p95(values):
    """Linear quantile (Hyndman-Fan type7), at h=(N-1)*.95."""
    x=sorted(values);h=(len(x)-1)*.95;lo=math.floor(h);hi=math.ceil(h)
    return x[lo]+(h-lo)*(x[hi]-x[lo])
def criterion(value,target,feature):
    return value>=target if feature=='q_min' else abs(value-target)/abs(target)<=.1
def stats(values):
    n=len(values)
    return {'N':n,'mean_signed_residual':math.fsum(values)/n,
            'MAE':math.fsum(abs(v) for v in values)/n,
            'RMSE':math.sqrt(math.fsum(v*v for v in values)/n),
            'P95_absolute_residual_linear':p95([abs(v) for v in values]),
            'min_signed_residual':min(values),'max_signed_residual':max(values)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    source=a.source.absolute();receipt=source.parent/'ANALYSIS_RECEIPT.json';out=a.out.absolute()
    if sha(source)!=CSV_SHA or sha(receipt)!=RECEIPT_SHA:raise ValueError('Wrong final64 source identity')
    meta=json.loads(receipt.read_text())
    if meta['original_denominator']!=64 or meta['state_counts']['FRESH_EMX_VERIFIED']!=64:
        raise ValueError('This summary requires final 64/64 physical closure')
    if any(v for k,v in meta['state_counts'].items() if k!='FRESH_EMX_VERIFIED'):raise ValueError('Unclosed or failed candidate remains')
    with source.open(newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=64 or any(len({r[k] for r in rows})!=64 for k in ('target_id','candidate_id','geometry_digest_sha256')):
        raise ValueError('Expected unique 64 target/candidate/geometry identities')
    residuals=[];discordant=[];count={'both_hit':0,'proxy_only_hit':0,'emx_only_hit':0,'both_miss':0}
    for row in rows:
        if row['state']!='FRESH_EMX_VERIFIED':raise ValueError('Not a verified fresh candidate')
        ph=[];eh=[];feature_rows=[]
        for feature in FEATURES:
            t,pv,em=(float(row[prefix+'_'+feature]) for prefix in ('target','proxy','emx'))
            if not all(math.isfinite(v) for v in (t,pv,em)):raise ValueError('Missing/nonfinite paired metric')
            if feature!='q_min' and t==0:raise ValueError('Zero target: relative criterion needs original special rule')
            pp,ep=criterion(pv,t,feature),criterion(em,t,feature);ph.append(pp);eh.append(ep)
            raw={'target_id':row['target_id'],'geometry_digest_sha256':row['geometry_digest_sha256'],
                 'feature':feature,'unit':'nH' if feature in ('lp_nh','ls_nh') else 'dimensionless',
                 'target':t,'proxy':pv,'emx':em,'emx_minus_target':em-t,'proxy_minus_target':pv-t,
                 'emx_minus_proxy':em-pv,'signed_identity_residual':(em-t)-((pv-t)+(em-pv)),
                 'proxy_target_absolute_relative_error':abs(pv-t)/abs(t) if feature!='q_min' else '',
                 'emx_target_absolute_relative_error':abs(em-t)/abs(t) if feature!='q_min' else '',
                 'proxy_Q_shortfall':max(t-pv,0) if feature=='q_min' else '',
                 'emx_Q_shortfall':max(t-em,0) if feature=='q_min' else '',
                 'proxy_criterion_pass':pp,'emx_criterion_pass':ep,
                 'criterion':'achieved_Qmin >= requested_Qmin' if feature=='q_min' else 'absolute_target_relative_error <= 0.10',
                 'result_sha256':row['result_sha256'],'feature_receipt_sha256':row['feature_receipt_sha256']}
            if not math.isclose(raw['signed_identity_residual'],0,abs_tol=1e-14):raise ValueError('Signed residual identity failed')
            residuals.append(raw);feature_rows.append(raw)
        proxy_hit,emx_hit=all(ph),all(eh)
        if proxy_hit!=(row['proxy_hit'].lower()=='true') or emx_hit!=(row['emx_hit'].lower()=='true'):
            raise ValueError('Unchanged hit rule disagrees with source')
        key='both_hit' if proxy_hit and emx_hit else 'proxy_only_hit' if proxy_hit else 'emx_only_hit' if emx_hit else 'both_miss'
        count[key]+=1
        if proxy_hit and not emx_hit:
            for item in feature_rows:discordant.append({**item,'criterion_flipped':item['proxy_criterion_pass'] and not item['emx_criterion_pass']})
    if count!=meta['same_candidate_contingency'] or count['proxy_only_hit']!=1:raise ValueError('Final paired contingency mismatch')
    summary=[]
    for feature in FEATURES:
        data=[r for r in residuals if r['feature']==feature]
        for field,label in [('emx_minus_target','EM_MINUS_TARGET'),('proxy_minus_target','PROXY_MINUS_TARGET'),('emx_minus_proxy','EM_MINUS_PROXY')]:
            summary.append({'comparison':label,'feature':feature,'unit':data[0]['unit'],
                            'metric_definition':'symmetric signed physical residual; P95 of absolute residual',
                            **stats([r[field] for r in data]),'source_csv_sha256':CSV_SHA})
        if feature=='q_min':
            for prefix in ('emx','proxy'):
                summary.append({'comparison':prefix.upper()+'_Q_SHORTFALL','feature':feature,'unit':'dimensionless',
                                'metric_definition':'max(requested_Qmin-achieved_Qmin,0); minimum-Q acceptance',
                                **stats([r[prefix+'_Q_shortfall'] for r in data]),'source_csv_sha256':CSV_SHA})
    # All validation occurs before authoring. Never overwrite a previous cut.
    out.mkdir(parents=True,exist_ok=True)
    names=['physical_error_summary.csv','signed_residuals.csv','proxy_only_hit_case.csv','RESULT_PARAGRAPH.md','SUMMARY.json','SHA256SUMS']
    if any((out/n).exists() for n in names):raise FileExistsError('Existing final64 summary is preserved')
    write_csv(out/names[0],summary);write_csv(out/names[1],residuals);write_csv(out/names[2],discordant)
    bykey={(r['comparison'],r['feature']):r for r in summary}
    vals=lambda comp:', '.join(f"{bykey[comp,f]['MAE']:.6g}" for f in FEATURES)
    flips=[r for r in discordant if r['criterion_flipped']]
    fliptext='; '.join((f"{r['feature']}: proxy absolute relative error {100*r['proxy_target_absolute_relative_error']:.6g}% to EM {100*r['emx_target_absolute_relative_error']:.6g}%" if r['feature']!='q_min' else f"Qmin: target {r['target']:.9g}, proxy {r['proxy']:.9g}, EM {r['emx']:.9g}") for r in flips)
    text=(f"All 64 preselected target–candidate pairs had verified fresh-EMX results (zero pending or physical-chain failures). "
          f"Using the unchanged joint criterion (absolute target-relative errors no greater than 10% for Lp, Ls and |k|, and achieved Qmin at least its requested minimum), "
          f"the proxy met 9/64 targets (14.0625%) and EM met 8/64 (12.5%): eight joint hits, one proxy-only hit and 55 joint misses. "
          f"The sole proxy-only hit was {discordant[0]['target_id']}; the flipped criterion was {fliptext}. "
          f"Across the same 64 pairs, EM–target MAEs for (Lp, Ls, Qmin, |k|) were ({vals('EM_MINUS_TARGET')}); "
          f"EM–proxy MAEs were ({vals('EM_MINUS_PROXY')}). Lp/Ls are in nH; Qmin and |k| are dimensionless. "
          f"The Qmin symmetric residual is descriptive, not the lower-bound acceptance penalty; mean one-sided EM Q shortfall was "
          f"{bykey['EMX_Q_SHORTFALL','q_min']['MAE']:.6g}, with P95 {bykey['EMX_Q_SHORTFALL','q_min']['P95_absolute_residual_linear']:.6g}. "
          "These are paired descriptive results for this frozen 64-target subset, not a final independent 10,000-request physical test or evidence of a causal model improvement. "
          "Signed residuals obey EM−target = (proxy−target) + (EM−proxy); absolute component sizes are not causal contribution fractions.\n\n"
          f"Source: `{source}` (SHA-256 `{CSV_SHA}`), final analysis receipt `{RECEIPT_SHA}`. "
          "P95 is the linear quantile of absolute residuals at h=(N−1)×0.95. Original targets, candidates, Q semantics and thresholds were not changed. "
          "No new extraction, inference, simulation or figure was produced.\n")
    with (out/names[3]).open('x') as f:f.write(text)
    receipt_out={'schema':'paired64_physical_residual_summary.v1','status':'COMPLETE_OFFLINE_ARITHMETIC',
                 'source_csv':{'path':str(source),'sha256':CSV_SHA},'source_analysis_receipt':{'path':str(receipt),'sha256':RECEIPT_SHA},
                 'script':{'path':str(Path(__file__).absolute()),'sha256':sha(__file__)},'N':64,'finite_N_per_metric':64,
                 'source_observation_utc':meta['utc'],'contingency':count,'proxy_only_hit_target':discordant[0]['target_id'],
                 'flipped_criteria':[r['feature'] for r in flips],'p95_method':'linear/type7; percentile of absolute residual',
                 'new_inference':0,'new_training':0,'new_emx':0,'re_extraction':0,
                 'outputs':{n:{'path':str(out/n),'sha256':sha(out/n)} for n in names[:4]}}
    write_json(out/names[4],receipt_out)
    with (out/'SHA256SUMS').open('x') as f:
        for n in names[:5]:f.write(sha(out/n)+'  '+n+'\n')
        f.write(sha(__file__)+'  '+Path(__file__).name+'\n')
    print(json.dumps({'N':64,'contingency':count,'proxy_only_hit_target':discordant[0]['target_id'],
                      'flipped_criteria':[r['feature'] for r in flips],'summary_sha256':sha(out/'SUMMARY.json')}))


if __name__=='__main__':main()
