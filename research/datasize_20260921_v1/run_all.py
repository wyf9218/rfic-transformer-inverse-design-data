#!/usr/bin/env python3
"""One bounded three-arm training runner and completed-results callback.

No native simulator calls. No automatic retry of a deterministic failure.
Run only after the parent releases a genuinely qualified external holdout.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import train_original100k_nested as a


def aggregate(study):
    rows=[]
    cohorts=set()
    for n in a.COUNTS:
        out=study/'runs'/f'train_{n}'
        terminal=a.read(out/'ENTRY_TERMINAL.json')
        if terminal.get('status')!='COMPLETE':
            raise ValueError('Comparison callback requires all three real terminal runs')
        summary_path=out/'physical_feature_tandem_inverse_summary.json'
        if a.sha(summary_path)!=terminal['summary_sha256']:
            raise ValueError('Summary changed after terminal receipt')
        summary=a.read(summary_path)
        identity=a.read(out/'ENTRY_IDENTITY.json')
        audit=summary['split_audit']
        if audit['row_counts']['train']!=n:
            raise ValueError('Actual gradient training count mismatch')
        cohorts.add(tuple(audit['global_split_index_sha256'][k] for k in ('validation','test')))
        realized=summary['optimizer_budget_contract']['realized']
        if not realized['exact_update_budget_pass']:
            raise ValueError('Exact optimizer budget not completed')
        vm=summary['validation_metrics']
        for column in summary['input_columns']:
            rows.append({'train_geometries':n,'validation_geometries':audit['row_counts']['validation'],
              'test_geometries_reserved':audit['row_counts']['test'],'feature':column,
              'forward_mae_physical':vm['per_feature_forward_proxy_physical_mae'][column],
              'inverse_response_mae_physical_original_semantics':vm['per_feature_physical_mae'][column],
              'inverse_error_semantics':'minimum-Q shortfall; other features absolute',
              'forward_validation_range_rmse':vm['forward_proxy']['validation_range_normalized_rmse'],
              'inverse_validation_response_range_rmse':vm['tandem_inverse']['validation_response_range_normalized_rmse'],
              'forward_actual_updates':realized['forward_optimizer_updates'],
              'inverse_actual_updates':realized['inverse_optimizer_updates'],
              'forward_best_update':summary['best_optimizer_updates']['forward_proxy'],
              'inverse_best_update':summary['best_optimizer_updates']['tandem_inverse'],
              'validation_global_indices_sha256':audit['global_split_index_sha256']['validation'],
              'test_global_indices_sha256':audit['global_split_index_sha256']['test'],
              'weights_path':summary['weights_npz'],'weights_sha256':summary['weights_npz_sha256'],
              'summary_path':str(summary_path),'summary_sha256':terminal['summary_sha256'],
              'validation_predictions_path':summary['validation_predictions_csv'],
              'validation_predictions_sha256':summary['validation_predictions_csv_sha256'],
              'training_history_path':summary['history_csv'],'training_history_sha256':summary['history_csv_sha256'],
              'forward_evidence':'EXISTING_EM_LABELS_EXTERNAL_RETROSPECTIVE_VALIDATION',
              'inverse_evidence':'OWN_FROZEN_FORWARD_PROXY_NOT_NEW_EMX',
              'test_evaluation':'NOT_RUN','fresh_emx':'NOT_RUN','independence_scope':identity['independence_scope'],
              'historical_exposure_status':identity['historical_exposure_status'],
              'family_lineage_status':identity['family_lineage_status']})
    if len(cohorts)!=1:
        raise ValueError('Arms do not share exact validation/test indices')
    dest=study/'completed_comparison'
    dest.mkdir(exist_ok=False)
    csv_path=dest/'TRAIN_SIZE_VALIDATION_COMPARISON.csv'
    with csv_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    a.write_atomic(dest/'COMPLETION.json',{'utc':a.utc(),'status':'ALL_THREE_ARMS_TRAINED_AND_AGGREGATED',
                 'train_sizes':list(a.COUNTS),'csv_rows':len(rows),'csv_sha256':a.sha(csv_path),
                 'comparison':'same mother source, nested trains, common retrospective external validation',
                 'fresh_emx':'NOT_RUN','test_evaluation':'NOT_RUN','new_inference_in_callback':0})
    return str(dest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['run','resume','aggregate'])
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--workers',type=int,choices=[1,2,3],default=1)
    parser.add_argument('--threads',type=int,choices=[1,2,3,4],default=4)
    args=parser.parse_args()
    if args.mode=='aggregate':
        print(aggregate(a.STUDY));return
    # One input gate before workers exist. Children also verify the pins they consume.
    contract,indices,qualification=a.load_contract(args.contract)
    journal=a.STUDY/'RUN_ALL.json'
    with (a.STUDY/'.run_all.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('An existing runner owns this study; no duplicate')
        contract_sha=a.sha(args.contract)
        if journal.exists():
            previous=a.read(journal)
            if args.mode!='resume' or previous['contract_sha256']!=contract_sha:
                raise ValueError('Existing run_all identity requires explicit exact resume')
            if previous.get('status')=='COMPLETE':
                print('Already complete; no work repeated');return
        elif args.mode=='resume':
            raise ValueError('No existing runner journal to resume')
        state={'schema':'original100k_three_arm_runner.v1','started_utc':a.utc(),'pid':os.getpid(),
               'status':'RUNNING','contract_sha256':contract_sha,'workers':args.workers,
               'blas_threads_per_arm':args.threads,'nice':19,'arms':{},'native_simulation':False}
        a.write_atomic(journal,state)
        def worker(n):
            mode='resume' if (a.STUDY/'runs'/f'train_{n}').exists() else 'run'
            if args.mode=='run' and mode=='resume':
                raise ValueError('An arm already exists; no-clobber run refused')
            env=dict(os.environ)
            for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
                env[key]=str(args.threads)
            command=['nice','-n','19',sys.executable,str(a.STUDY/'train_original100k_nested.py'),mode,
                     '--contract',str(args.contract.resolve()),'--arm',str(n)]
            start=time.monotonic()
            with (a.STUDY/f'arm_{n}_launcher.log').open('a') as log:
                process=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            return {'returncode':process.returncode,'elapsed_seconds':time.monotonic()-start,'command':command}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(worker,n):n for n in a.COUNTS}
            for future in as_completed(futures):
                n=futures[future]
                try:state['arms'][str(n)]=future.result()
                except BaseException as exc:state['arms'][str(n)]={'error':str(exc),'returncode':-1}
                a.write_atomic(journal,state)
        if any(r['returncode']!=0 for r in state['arms'].values()):
            state.update(status='FAILED_PRESERVED_NO_AUTOMATIC_RETRY',finished_utc=a.utc())
            a.write_atomic(journal,state)
            raise SystemExit(2)
        try:
            destination=aggregate(a.STUDY)
        except BaseException as exc:
            state.update(status='TRAINING_FINISHED_CALLBACK_FAILED',error=str(exc),finished_utc=a.utc())
            a.write_atomic(journal,state)
            raise
        state.update(status='COMPLETE',finished_utc=a.utc(),comparison_path=destination)
        a.write_atomic(journal,state)
        print(destination)


if __name__=='__main__':
    main()
