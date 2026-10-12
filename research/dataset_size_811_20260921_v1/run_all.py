#!/usr/bin/env python3
"""Bounded ordinary-program runner; default starts only missing D10k/D50k arms."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import train_dataset_sizes as a


def completed_callback(arms):
    """Read actual completed summaries only; no new inference or test evaluation."""
    rows=[]
    for arm in arms:
        out=a.STUDY/'runs'/arm
        t=a.c.read(out/'ENTRY_TERMINAL.json')
        p=out/'physical_feature_tandem_inverse_summary.json'
        if t['status']!='COMPLETE' or a.c.sha(p)!=t['summary_sha256']:
            raise ValueError('Exact completed summary is required')
        s=a.c.read(p);realized=s['optimizer_budget_contract']['realized']
        if not realized['exact_update_budget_pass']:raise ValueError('Training budget incomplete')
        vm=s['validation_metrics']
        for feature in s['input_columns']:
            rows.append({'arm':arm,'dataset_total_rows':sum(a.ARMS[arm]),'gradient_train_rows':a.ARMS[arm][0],
              'validation_rows':a.ARMS[arm][1],'reserved_arm_test_rows':a.ARMS[arm][2],'feature':feature,
              'forward_validation_mae':vm['per_feature_forward_proxy_physical_mae'][feature],
              'inverse_validation_original_semantics_mae':vm['per_feature_physical_mae'][feature],
              'forward_updates':realized['forward_optimizer_updates'],'inverse_updates':realized['inverse_optimizer_updates'],
              'summary_path':str(p),'summary_sha256':t['summary_sha256'],'weights_path':s['weights_npz'],
              'weights_sha256':s['weights_npz_sha256'],'training_history_path':s['history_csv'],
              'comparison_limit':'Different nested validation subsets; common-test comparison requires separate fixed evaluation',
              'inverse_evidence':'SELF_PROXY; original minimum-Q error semantics','test_evaluation':'NOT_RUN','fresh_emx':'NOT_RUN'})
    dest=a.STUDY/'completed_training'
    if dest.exists():
        receipt=a.c.read(dest/'COMPLETION.json')
        p=dest/'TRAINING_VALIDATION_STATUS.csv'
        if receipt['arms']==arms and a.c.sha(p)==receipt['csv_sha256']:
            return str(dest) # callback resume after a completed atomic delivery, no overwrite
        raise FileExistsError('Existing callback output has another identity')
    dest.mkdir(exist_ok=False)
    p=dest/'TRAINING_VALIDATION_STATUS.csv'
    with p.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    a.c.write_atomic(dest/'COMPLETION.json',{'utc':a.c.utc(),'arms':arms,'csv_sha256':a.c.sha(p),
                      'actual_completed_training_arms':len(arms),'new_evaluation':False,'fresh_emx':'NOT_RUN'})
    return str(dest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['run','resume','aggregate']);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--arms',nargs='+',choices=list(a.ARMS),default=['D10k','D50k'])
    p.add_argument('--workers',type=int,choices=[1,2,3],default=2)
    p.add_argument('--threads',type=int,choices=[1,2,3,4],default=4);args=p.parse_args()
    if len(set(args.arms))!=len(args.arms):raise ValueError('Duplicate arm requested')
    a.load_contract(args.contract)
    if args.mode=='aggregate':print(completed_callback(args.arms));return
    journal=a.STUDY/'RUN_ALL.json'
    with (a.STUDY/'.run_all.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another runner owns this study; no duplicate')
        fingerprint={'contract_sha256':a.c.sha(args.contract),'arms':args.arms,
                     'adapter_sha256':a.c.sha(a.__file__),'runner_sha256':a.c.sha(__file__)}
        if journal.exists():
            prev=a.c.read(journal)
            if args.mode!='resume' or prev['identity']!=fingerprint:raise ValueError('Exact explicit resume required')
            if prev['status']=='COMPLETE':print('Already complete; no repeated work');return
        elif args.mode=='resume':raise ValueError('No existing runner journal')
        state={'identity':fingerprint,'utc':a.c.utc(),'pid':os.getpid(),'status':'RUNNING','results':{},
               'workers':args.workers,'threads_per_arm':args.threads,'nice':19,'native_simulation':False}
        a.c.write_atomic(journal,state)
        def worker(arm):
            mode='resume' if (a.STUDY/'runs'/arm).exists() else 'run'
            if args.mode=='run' and mode=='resume':raise ValueError('Existing arm requires explicit resume')
            env=dict(os.environ)
            for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
                env[key]=str(args.threads)
            cmd=['nice','-n','19',sys.executable,str(a.STUDY/'train_dataset_sizes.py'),mode,
                 '--contract',str(args.contract.resolve()),'--arm',arm]
            start=time.monotonic()
            with (a.STUDY/(arm+'_launcher.log')).open('a') as log:
                proc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            return {'returncode':proc.returncode,'elapsed_seconds':time.monotonic()-start,'command':cmd}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(worker,arm):arm for arm in args.arms}
            for future in as_completed(futures):
                arm=futures[future]
                try:state['results'][arm]=future.result()
                except BaseException as exc:state['results'][arm]={'returncode':-1,'error':str(exc)}
                a.c.write_atomic(journal,state)
        if any(r['returncode']!=0 for r in state['results'].values()):
            state.update(status='FAILED_PRESERVED_NO_AUTOMATIC_RETRY',finished_utc=a.c.utc());a.c.write_atomic(journal,state)
            raise SystemExit(2)
        try:dest=completed_callback(args.arms)
        except BaseException as exc:
            state.update(status='TRAINED_CALLBACK_FAILED',error=str(exc));a.c.write_atomic(journal,state);raise
        state.update(status='COMPLETE',finished_utc=a.c.utc(),comparison_path=dest);a.c.write_atomic(journal,state)
        print(dest)


if __name__=='__main__':main()
