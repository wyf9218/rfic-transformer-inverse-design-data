#!/usr/bin/env python3
"""Ordinary bounded three-architecture queue; 3 arms x 4 BLAS threads maximum."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import train_architectures as a


def aggregate(arms):
    rows=[]
    for arm in arms:
        out=a.STUDY/'runs'/arm;t=a.c.read(out/'ENTRY_TERMINAL.json');sp=out/'physical_feature_tandem_inverse_summary.json'
        if t['status']!='COMPLETE' or a.c.sha(sp)!=t['summary_sha256']:raise ValueError('No exact terminal summary for '+arm)
        s=a.c.read(sp);b=s['optimizer_budget_contract']['realized']
        if not b['exact_update_budget_pass']:raise ValueError('Incomplete training budget')
        for feature in s['input_columns']:
            rows.append({'arm':arm,'width':a.WIDTHS[arm],**a.architecture(arm),'train':78891,'validation':9740,'test_reserved':11369,
              'feature':feature,'forward_validation_mae':s['validation_metrics']['per_feature_forward_proxy_physical_mae'][feature],
              'inverse_validation_original_semantics_mae':s['validation_metrics']['per_feature_physical_mae'][feature],
              'forward_updates':b['forward_optimizer_updates'],'inverse_updates':b['inverse_optimizer_updates'],
              'weights_path':s['weights_npz'],'weights_sha256':s['weights_npz_sha256'],'summary_path':str(sp),
              'summary_sha256':t['summary_sha256'],'test_evaluation':'NOT_RUN','inverse_evidence':'SELF_PROXY','fresh_emx':'NOT_RUN'})
    out=a.STUDY/'completed_training';out.mkdir(exist_ok=False)
    p=out/'ARCHITECTURE_VALIDATION_STATUS.csv'
    with p.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    a.c.write_atomic(out/'COMPLETION.json',{'utc':a.c.utc(),'arms':arms,'csv_sha256':a.c.sha(p),'new_inference':0,'new_emx':0})
    return str(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['run','resume','aggregate']);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--arms',nargs='+',choices=list(a.WIDTHS),default=list(a.WIDTHS))
    p.add_argument('--workers',type=int,choices=[1,2,3],default=3);p.add_argument('--threads',type=int,choices=[1,2,3,4],default=4)
    args=p.parse_args()
    if len(set(args.arms))!=len(args.arms):raise ValueError('Duplicate architecture requested')
    a.d.load_contract(args.contract)
    if args.mode=='aggregate':print(aggregate(args.arms));return
    journal=a.STUDY/'RUN_ALL.json'
    with (a.STUDY/'.run_all.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Existing architecture runner owns lock')
        identity={'contract_sha256':a.c.sha(args.contract),'arms':args.arms,'wrapper_sha256':a.c.sha(a.__file__),'runner_sha256':a.c.sha(__file__)}
        if journal.exists():
            prev=a.c.read(journal)
            if args.mode!='resume' or prev['identity']!=identity:raise ValueError('Exact explicit resume required')
            if prev['status']=='COMPLETE':print('Already complete; not repeated');return
        elif args.mode=='resume':raise ValueError('No prior runner journal')
        state={'utc':a.c.utc(),'pid':os.getpid(),'identity':identity,'status':'RUNNING','workers':args.workers,
               'threads_per_arm':args.threads,'nice':19,'results':{}}
        a.c.write_atomic(journal,state)
        def worker(arm):
            mode='resume' if (a.STUDY/'runs'/arm).exists() else 'run'
            if args.mode=='run' and mode=='resume':raise ValueError('Existing arm requires exact explicit resume')
            env=dict(os.environ)
            for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]=str(args.threads)
            cmd=['nice','-n','19',sys.executable,str(a.STUDY/'train_architectures.py'),mode,'--contract',str(args.contract.resolve()),'--arm',arm]
            start=time.monotonic()
            with (a.STUDY/(arm+'_launcher.log')).open('a') as log:proc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            return {'returncode':proc.returncode,'elapsed_seconds':time.monotonic()-start,'command':cmd}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(worker,arm):arm for arm in args.arms}
            for future in as_completed(futures):
                arm=futures[future]
                try:state['results'][arm]=future.result()
                except BaseException as exc:state['results'][arm]={'returncode':-1,'error':str(exc)}
                a.c.write_atomic(journal,state)
        if any(r['returncode']!=0 for r in state['results'].values()):
            state.update(status='FAILED_PRESERVED_NO_AUTOMATIC_RETRY',finished_utc=a.c.utc());a.c.write_atomic(journal,state);raise SystemExit(2)
        try:dest=aggregate(args.arms)
        except BaseException as exc:
            state.update(status='TRAINED_CALLBACK_FAILED',error=str(exc));a.c.write_atomic(journal,state);raise
        state.update(status='COMPLETE',finished_utc=a.c.utc(),results_path=dest);a.c.write_atomic(journal,state);print(dest)


if __name__=='__main__':main()
