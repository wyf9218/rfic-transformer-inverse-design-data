#!/usr/bin/env python3
"""Three width-only complete F/I arms on the exact original D100k split.

Imports the frozen dataset adapter read-only; never patches files used by an
active run. The observer runs after successful Adam updates and changes no math.
"""
import argparse
import contextlib
from datetime import datetime,timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

STUDY=Path(__file__).resolve().parent
DATASET_ADAPTER=Path(os.environ.get('EUCAP_DATASET_ADAPTER',str(STUDY.parent/'dataset_size_811_20260921_v1/train_dataset_sizes.py')))
spec=importlib.util.spec_from_file_location('frozen_dataset_adapter_for_architecture',DATASET_ADAPTER)
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
c=d.c
WIDTHS={'Small':128,'Medium':256,'Large':512}


def architecture_argv(contract,out,arm):
    width=WIDTHS[arm]
    # argparse's final occurrence overrides only widths. All scientific recipe
    # arguments, including the original minimum-Q loss, come from seed13.
    return d.trainer_argv(contract,out)+[
        '--forward-depth','3','--forward-width',str(width),'--forward-hidden-widths',','.join([str(width)]*3),
        '--inverse-depth','3','--inverse-width',str(width),'--inverse-hidden-widths',','.join([str(width)]*3)]


def architecture(arm):
    n=WIDTHS[arm]
    return {'forward':[10,n,n,n,4],'inverse':[4,n,n,n,10],
            'forward_parameters':2*n*n+17*n+4,'inverse_parameters':2*n*n+17*n+10}


def install_progress_observer(runtime,out):
    """Observe only successful original optimizer calls; preserve call/return."""
    raw_step=runtime._adam_step
    progress={'stage':'not_started','actual_optimizer_updates':0,'stage_started_monotonic':None}
    def emit(status,extra=None):
        payload={'utc':c.utc(),'pid':os.getpid(),'stage':progress['stage'],'status':status,
                 'actual_optimizer_updates':progress['actual_optimizer_updates'],'target_optimizer_updates':3600,
                 'stage_elapsed_seconds':None if progress['stage_started_monotonic'] is None else time.monotonic()-progress['stage_started_monotonic'],
                 'observer':'after_original_adam_step; no optimizer or gradient change','fresh_emx':'NOT_RUN'}
        if extra:payload.update(extra)
        c.write_atomic(out/'PROGRESS.json',payload)
        with (out/'progress_events.jsonl').open('a') as stream:
            stream.write(json.dumps(payload,allow_nan=False)+'\n')
    def observed_step(weights,biases,grad_weights,grad_biases,state,learning_rate):
        value=raw_step(weights,biases,grad_weights,grad_biases,state,learning_rate)
        progress['actual_optimizer_updates']=int(state['step'])
        if state['step']==1 or state['step']%120==0:
            emit('TRAINING',{'observed_parameter_0_0':float(weights[0][0,0])})
        return value
    runtime._adam_step=observed_step
    def stage_wrapper(raw,name):
        def wrapped(*args,**kwargs):
            progress.update(stage=name,actual_optimizer_updates=0,stage_started_monotonic=time.monotonic())
            emit('STAGE_STARTED')
            result=raw(*args,**kwargs)
            history,best=result
            emit('STAGE_TRAINING_COMPLETE',{'best_optimizer_update':int(best.get('optimizer_updates',0)),
                                          'last_saved_validation':history[-1] if history else None})
            return result
        return wrapped
    runtime._train_forward=stage_wrapper(runtime._train_forward,'forward_training')
    runtime._train_inverse=stage_wrapper(runtime._train_inverse,'inverse_training')
    return progress


def launch(contract_path,arm,resume=False):
    contract,indices,q=d.load_contract(contract_path)
    out=STUDY/'runs'/arm
    if out.is_symlink() or (out.parent.exists() and out.parent.resolve()!=STUDY/'runs'):
        raise ValueError('No symlink output target allowed')
    identity={'schema':'original100k_width_only_full_tandem.v1','arm':arm,'architecture':architecture(arm),
      'model_seed':20260713,'dataset_rows':100000,'partition_rows':{'train':78891,'validation':9740,'test':11369},
      'dataset_contract_sha256':c.sha(contract_path),'wrapper_sha256':c.sha(__file__),
      'dataset_adapter_sha256':c.sha(DATASET_ADAPTER),'core_sha256':c.sha(d.CORE),'runtime_sha256':c.RUNTIME_SHA,
      'global_split_index_sha256':{p:c.index_sha(indices['D100k'][p]) for p in d.PARTS},
      'scientific_variable':'shared hidden width of complete F and I; own F is trained then frozen',
      'fresh_emx':'NOT_RUN','existing_models_modified':False}
    if out.exists() and not resume:raise FileExistsError('No-clobber: '+str(out))
    if resume:
        if not out.is_dir() or c.read(out/'ENTRY_IDENTITY.json')!=identity:raise ValueError('Exact architecture identity required to resume')
        if (out/'ENTRY_TERMINAL.json').exists() and c.read(out/'ENTRY_TERMINAL.json').get('status')=='COMPLETE':
            print('Already complete; not repeated: '+str(out));return
    else:
        out.mkdir(parents=True,exist_ok=False);c.write_atomic(out/'ENTRY_IDENTITY.json',identity)
    with (out/'.run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Live architecture owner exists; no duplicate')
        runtime=c.load_runtime();d.install_adapter(runtime,indices,'D100k',c.sha(contract_path))
        install_progress_observer(runtime,out)
        argv=architecture_argv(contract,out,arm)
        c.write_atomic(out/'ENTRY_STARTED.json',{'utc':c.utc(),'pid':os.getpid(),'argv':argv,
                       'python_version':sys.version,'numpy_version':d.np.__version__,'resume_completed_stages_only':resume})
        try:
            with (out/'console.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                code=runtime.main(argv)
            sp=out/'physical_feature_tandem_inverse_summary.json';summary=c.read(sp)
            complete=summary.get('execution_status')=='PASS'
            c.write_atomic(out/'ENTRY_TERMINAL.json',{'utc':c.utc(),'status':'COMPLETE' if complete else 'FAILED',
              'returncode':code,'summary_sha256':c.sha(sp),'architecture':architecture(arm),
              'actual_train_rows':78891,'validation_rows':9740,'test_rows_reserved':11369,
              'test_evaluation':'NOT_RUN','fresh_emx':'NOT_RUN'})
            if not complete:raise RuntimeError('Original runtime did not report execution PASS')
        except BaseException as exc:
            c.write_atomic(out/('ENTRY_ERROR_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),
                           {'utc':c.utc(),'type':type(exc).__name__,'error':str(exc),'no_automatic_retry':True})
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['check','run','resume']);p.add_argument('--contract',required=True,type=Path)
    p.add_argument('--arm',choices=list(WIDTHS),default='Small');args=p.parse_args()
    if args.mode=='check':
        contract,indices,q=d.load_contract(args.contract)
        runtime=c.load_runtime();d.install_adapter(runtime,indices,'D100k',c.sha(args.contract))
        parsed=runtime._parse_args(architecture_argv(contract,Path('/NOT_CREATED/check'),args.arm))
        print(json.dumps({'status':'PASS_INPUTS_AND_WIDTH_ARGUMENTS','arm':args.arm,'architecture':architecture(args.arm),
          'seed':parsed.seed,'forward_budget':parsed.forward_max_optimizer_updates,'inverse_budget':parsed.inverse_max_optimizer_updates,
          'training_started':False}));return
    launch(args.contract,args.arm,resume=args.mode=='resume')


if __name__=='__main__':main()
