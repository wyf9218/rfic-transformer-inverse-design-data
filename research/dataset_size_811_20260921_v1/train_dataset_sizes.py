#!/usr/bin/env python3
"""Original 100K dataset TOTAL-size arms; no original output is modified.

Only data selection/split plumbing changes. The pinned historical NumPy trainer
provides the networks, decoder, loss, F/I training, validation selection and save.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

STUDY=Path(__file__).resolve().parent
CORE=Path(os.environ.get('EUCAP_DATASET_CORE',str(STUDY.parent/'datasize_20260921_v1/train_original100k_nested.py')))
spec=importlib.util.spec_from_file_location('historical_runtime_adapter_core',CORE)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
SCHEMA='original100k_nested_dataset_totals.v1'
ARMS={'D10k':(8000,1000,1000),'D50k':(40000,5000,5000),'D100k':(78891,9740,11369)}
PARTS=('train','validation','test')


def validate_indices(indices,counts=ARMS,mother_rows=100000):
    if set(indices)!=set(counts)|{'common_test'}:
        raise ValueError('Explicit D10k/D50k/D100k and common_test are required')
    previous=None
    for arm,sizes in counts.items():
        mapping=indices[arm]
        if set(mapping)!=set(PARTS):raise ValueError('Every arm requires all three partitions')
        seen=set()
        for part,n in zip(PARTS,sizes):
            values=mapping[part]
            if not isinstance(values,list) or len(values)!=n or any(type(i) is not int for i in values):
                raise ValueError('Wrong actual partition size or non-integer indices')
            current=set(values)
            if len(current)!=n or min(current)<0 or max(current)>=mother_rows or current&seen:
                raise ValueError('Duplicate, out-of-range or crossed partition indices')
            if previous is not None and not set(previous[part])<=current:
                raise ValueError('Dataset subsets must be nested within original partitions')
            seen|=current
        previous=mapping
    full=indices[list(counts)[-1]]
    if set().union(*(set(full[p]) for p in PARTS))!=set(range(mother_rows)):
        raise ValueError('D100k must preserve the entire original mother table')
    if indices['common_test']!=full['test']:
        raise ValueError('Common test must exactly equal the original test, including order')


def load_contract(path):
    contract=c.read(path)
    if contract.get('schema')!=SCHEMA or contract['mother']['rows']!=100000 or contract['mother']['sha256']!=c.MOTHER_SHA:
        raise ValueError('Wrong original100K dataset identity')
    q=c.read(c.checked_pin(contract['qualification']))
    if q.get('status')!='PASS':raise ValueError('Original split qualification is not PASS')
    for k,v in {'mother_csv_sha256':contract['mother']['sha256'],'indices_sha256':contract['indices']['sha256']}.items():
        if q.get(k)!=v:raise ValueError('Qualification does not bind '+k)
    for key in ['original_split_exact','nested_within_original_partitions','geometry_disjoint_train_validation_test','common_test_is_original_test','mother_dataset_exact']:
        if q.get('checks',{}).get(key) is not True:raise ValueError('Missing split evidence: '+key)
    indices=c.read(c.checked_pin(contract['indices']));validate_indices(indices)
    c.checked_pin(contract['mother'])
    return contract,indices,q


def select_matrix(full,indices,arm):
    if full['count']!=100000 or full['source_indices']!=list(range(100000)):
        raise ValueError('Historical finite-row build changed original position mapping')
    chosen=sorted(set().union(*(set(indices[arm][p]) for p in PARTS)))
    ids=[full['source_geometry_identities'][i] for i in chosen]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate geometry in selected dataset')
    out=dict(full);out['count']=len(chosen)
    for key in ['x','y','split_x']:out[key]=np.asarray(full[key])[chosen].copy()
    for key in ['source_indices','source_evaluations','source_geometry_identities']:out[key]=[full[key][i] for i in chosen]
    out['global_mother_indices']=chosen
    return out


def explicit_split(matrix,indices,arm,contract_sha):
    positions={j:i for i,j in enumerate(matrix['global_mother_indices'])}
    split={p:np.array([positions[i] for i in indices[arm][p]],dtype=np.int64) for p in PARTS}
    lower,upper=np.array([.5,.5,5,0]),np.array([3,3,25,.8])
    values=np.asarray(matrix['split_x']);norm=(values-lower)/(upper-lower)
    if np.any(~np.isfinite(norm)) or np.any(norm < -1e-12) or np.any(norm > 1+1e-12):
        raise ValueError('Original checkpoint cell domain exceeded; do not clip labels')
    cells=np.clip(np.floor(np.clip(norm,0,1)*4).astype(int),0,3)
    ids=[':'.join(map(str,row)) for row in cells]
    cell_ids={p:sorted({ids[i] for i in split[p]}) for p in PARTS}
    overlap=(set(cell_ids['train'])&set(cell_ids['validation']))|(set(cell_ids['train'])&set(cell_ids['test']))|(set(cell_ids['validation'])&set(cell_ids['test']))
    if overlap:raise ValueError('Original grouped split unexpectedly crosses physical cells')
    audit={'split_mode':'physical_cell_grouped','physical_cell_grouped':True,'physical_cell_range_source':'explicit',
           'physical_cell_bins_per_dimension':4,'physical_cell_lower':lower.tolist(),'physical_cell_upper':upper.tolist(),
           'cell_ids':cell_ids,'physical_cell_overlap_count':0,'row_counts':{p:len(split[p]) for p in PARTS},
           'split_index_sha256':{p:c.index_sha(split[p]) for p in PARTS},
           'global_split_index_sha256':{p:c.index_sha(indices[arm][p]) for p in PARTS},
           'all_rows_assigned_once':True,'explicit_indices':True,'original_partition_preserved':True,
           'study_contract_sha256':contract_sha,'adapter_sha256':c.sha(__file__),
           'dataset_total_rows':sum(len(split[p]) for p in PARTS),
           'common_test_global_index_sha256':c.index_sha(indices['common_test']),
           'boundary':'Nested dataset totals within original frozen physical-cell partitions; historical use is retained.'}
    return split,audit


def install_adapter(runtime,indices,arm,contract_sha):
    original_build=runtime._build_matrix
    original_parse=runtime._parse_args
    def build(rows,input_columns,geometry_columns,split_reference_columns):
        return select_matrix(original_build(rows,input_columns,geometry_columns,split_reference_columns),indices,arm)
    def split(matrix,args):return explicit_split(matrix,indices,arm,contract_sha)
    def parse(argv):
        args=original_parse(argv)
        args.split_mode='physical_cell_grouped'
        total=sum(len(indices[arm][p]) for p in PARTS)
        args.validation_fraction=len(indices[arm]['validation'])/total
        args.test_fraction=len(indices[arm]['test'])/total
        return args
    runtime._build_matrix=build;runtime._split_indices=split;runtime._parse_args=parse


def trainer_argv(contract,out):
    # Reuse every original seed13 scientific argument. Only input selection,
    # output, validation-only final reporting and exact stage checkpoints differ.
    return c.trainer_argv({'combined_csv':contract['mother']},out)


def launch(contract_path,contract,indices,q,arm,resume=False):
    out=STUDY/'runs'/arm
    if out.is_symlink() or (out.parent.exists() and out.parent.resolve()!=STUDY/'runs'):
        raise ValueError('No symlink output target allowed')
    contract_sha=c.sha(contract_path)
    identity={'schema':'original100k_dataset_total_arm.v1','arm':arm,'dataset_total_rows':sum(ARMS[arm]),
              'actual_train_rows':ARMS[arm][0],'validation_rows':ARMS[arm][1],'test_rows_reserved':ARMS[arm][2],
              'contract_sha256':contract_sha,'adapter_sha256':c.sha(__file__),'core_sha256':c.sha(CORE),
              'runtime_sha256':c.RUNTIME_SHA,'splitter_sha256':c.SPLITTER_SHA,
              'global_index_sha256':{p:c.index_sha(indices[arm][p]) for p in PARTS},
              'common_test_global_index_sha256':c.index_sha(indices['common_test']),
              'old_model_outputs_untouched':True,'fresh_emx':'NOT_RUN','original_split_preserved':True}
    if out.exists() and not resume:raise FileExistsError('No-clobber: '+str(out))
    if resume:
        if not out.is_dir() or c.read(out/'ENTRY_IDENTITY.json')!=identity:raise ValueError('Exact existing identity required to resume')
        if (out/'ENTRY_TERMINAL.json').exists() and c.read(out/'ENTRY_TERMINAL.json').get('status')=='COMPLETE':
            print('Already complete; no repeated work: '+str(out));return
    else:
        out.mkdir(parents=True,exist_ok=False);c.write_atomic(out/'ENTRY_IDENTITY.json',identity)
    with (out/'.run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Live arm owner exists; duplicate refused')
        runtime=c.load_runtime();install_adapter(runtime,indices,arm,contract_sha)
        argv=trainer_argv(contract,out)
        c.write_atomic(out/'ENTRY_STARTED.json',{'utc':c.utc(),'pid':os.getpid(),'argv':argv,
                       'resume_completed_stages_only':resume,'numpy_version':np.__version__,'python_version':sys.version})
        try:
            with (out/'console.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                code=runtime.main(argv)
            summary_path=out/'physical_feature_tandem_inverse_summary.json';summary=c.read(summary_path)
            complete=summary.get('execution_status')=='PASS'
            c.write_atomic(out/'ENTRY_TERMINAL.json',{'utc':c.utc(),'status':'COMPLETE' if complete else 'FAILED',
                 'returncode':code,'summary_sha256':c.sha(summary_path),'dataset_total_rows':sum(ARMS[arm]),
                 'actual_train_rows':ARMS[arm][0],'validation_rows':ARMS[arm][1],'test_rows_reserved':ARMS[arm][2],
                 'test_evaluation':'NOT_RUN','fresh_emx':'NOT_RUN'})
            if not complete:raise RuntimeError('Training execution did not report PASS')
        except BaseException as exc:
            name='ENTRY_ERROR_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'
            c.write_atomic(out/name,{'utc':c.utc(),'type':type(exc).__name__,'error':str(exc),
                                   'resume_boundary':'Completed stages only; no live Adam/RNG recovery'})
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['check','run','resume']);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--arm',choices=list(ARMS),default='D10k');args=p.parse_args()
    contract,indices,q=load_contract(args.contract)
    if args.mode=='check':
        print(json.dumps({'status':'PASS_INPUT_IDENTITIES_AND_NESTED_DATASET_TOTALS','counts':ARMS,
                         'common_test_rows':len(indices['common_test']),'training_started':False}));return
    launch(args.contract,contract,indices,q,args.arm,resume=args.mode=='resume')


if __name__=='__main__':main()
