#!/usr/bin/env python3
"""Render only the two frozen mean/SD CSVs through the unchanged draw function.

No manifest traversal, remote paths, model loading, evaluation or aggregation.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

AGGREGATOR_SHA='f5a4ca3efd184019795ec6a28c8f400040b3211a554990224404a257a658eb2c'
FAMILIES={'dataset':('D10k','D50k','D100k'),'architecture':('Small','Medium','Large')}
SEEDS=[20260713,17,29]
FILES=('forward_mean_sample_SD.csv','target_hit_mean_sample_SD.csv')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def load_inputs(cut,completion_sha):
    require(sha(cut/'COMPLETE.json')==completion_sha,'Closed COMPLETE SHA mismatch')
    receipt=json.loads((cut/'COMPLETE.json').read_text());identity=receipt['identity'];family=identity['family']
    require(receipt['status']=='GENERATED_PENDING_VISUAL_REVIEW' and family in FAMILIES and
            receipt['replicate_count']==3 and identity['seeds']==SEEDS and
            receipt['uncertainty']=='SAMPLE_STANDARD_DEVIATION_DDOF1_NOT_CI' and
            receipt['unique_target_requests']==10000,'Not an exact closed three-seed family')
    require(identity['script_sha256']==AGGREGATOR_SHA,'Unexpected originating aggregator')
    expected={f'{a}__seed{s}' for a in FAMILIES[family] for s in SEEDS}
    require(set(identity['inputs'])==expected,'Aggregate input identities do not bind all nine models')
    tables=[]
    for name in FILES:
        require(sha(cut/name)==receipt['outputs'][name],'Bound CSV SHA mismatch: '+name)
        with (cut/name).open(newline='') as f:tables.append(list(csv.DictReader(f)))
    errors,hits=tables
    expected_e={(a,f,m) for a in FAMILIES[family] for f in ('lp_nh','ls_nh','q_min','k_abs')
                for m in ('MAE','RMSE','P95_absolute_error_linear')}
    require(len(errors)==36 and {(r['arm'],r['feature'],r['metric']) for r in errors}==expected_e,'Mean-error table incomplete/duplicated')
    expected_h={(a,e) for a in FAMILIES[family] for e in ('SELF_PROXY','COMMON_PROXY_ORIGINAL_SEED20260713')}
    require(len(hits)==6 and {(r['arm'],r['evidence_class']) for r in hits}==expected_h,'Mean-hit table incomplete/duplicated')
    for row in errors+hits:
        require(row['family']==family and row['seeds']=='20260713;17;29' and int(row['required_seed_count'])==3,'CSV family or seed scope changed')
        if row in errors:
            require(int(row['common_test_N_per_seed'])==11369,'Common-test denominator changed')
            if row['status']=='NOT_ESTIMABLE_MISSING_METRIC':
                require(int(row['finite_seed_count'])<3 and row['mean']==row['sample_SD_ddof1']=='','Missing metric cannot become a plotted value')
                continue
            require(row['status']=='COMPLETE' and int(row['finite_seed_count'])==3,'Invalid complete error row')
        else:
            require(int(row['original_target_N_per_seed'])==int(row['unique_frozen_targets'])==10000 and
                    row['fresh_EMX']=='NOT_RUN','Hit denominator or evidence class changed')
        # Convert saved numeric cells for the preserved draw function, not new
        # statistics. Missing forward values remain blank and render as NaN.
        for key in ('mean','sample_SD_ddof1'):
            row[key]=float(row[key]);require(math.isfinite(row[key]) and row[key]>=0,'Invalid saved mean/SD')
        if row in hits:require(row['mean']<=1,'Hit fraction above one')
    return family,errors,hits,receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cut',type=Path,required=True)
    p.add_argument('--complete-sha',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--aggregator',type=Path,default=Path(__file__).resolve().parent/'aggregate_repeats.py')
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('No-clobber: use a NEW replot output directory')
    family,errors,hits,receipt=load_inputs(a.cut,a.complete_sha)
    require(sha(a.aggregator)==AGGREGATOR_SHA,'Preserved drawing source SHA changed')
    spec=importlib.util.spec_from_file_location('preserved_repeat_csv_draw',a.aggregator)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    a.out.mkdir(parents=True,exist_ok=False)
    source={'family':family,'completion_sha256':a.complete_sha,'source_CSV_sha256':{n:sha(a.cut/n) for n in FILES},
            'aggregator_sha256':AGGREGATOR_SHA,'entry_sha256':sha(__file__),'new_statistics':0,'new_evaluation':0,'remote_calls':0}
    (a.out/'SOURCE.json').write_text(json.dumps(source,indent=2)+'\n')
    try:module.draw(family,errors,hits,a.out)
    except Exception as exc:
        (a.out/'FAILED.json').write_text(json.dumps({'status':'RENDER_FAILED_PRESERVED','reason':str(exc)})+'\n');raise
    outputs=[f'{name}.{ext}' for name in ('forward_MAE_RMSE_mean_SD','joint_hit_mean_SD') for ext in ('svg','png')]
    done={'status':'RENDERED_PENDING_VISUAL_REVIEW','source':source,'outputs':{n:sha(a.out/n) for n in outputs},
          'sample_SD_not_CI':True,'new_inference_training_EM':0}
    (a.out/'RENDER_COMPLETE.json').write_text(json.dumps(done,indent=2)+'\n')
    print(json.dumps({'status':done['status'],'out':str(a.out.absolute()),'new_statistics':0}))


if __name__=='__main__':main()
