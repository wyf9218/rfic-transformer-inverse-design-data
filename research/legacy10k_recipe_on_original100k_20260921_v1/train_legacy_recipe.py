#!/usr/bin/env python3
"""Recorded legacy10k training recipe, original100k data; no native simulation.

Uses the existing hash-pinned trainer/dataset adapter unchanged. This is a
historical recipe reconstruction, not recovery of original trainer byte identity.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def utc(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def checked(pin):
    p = Path(pin['path'])
    if sha(p) != pin['sha256']: raise ValueError('Input identity changed: '+str(p))
    return p
def exclusive(p,value):
    with Path(p).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
def atomic(p,value):
    tmp=Path(str(p)+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(p)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def modules(cfg):
    keys={'EUCAP_DATASET_CORE':'core','EUCAP_DATASIZE_TRAINER':'trainer',
          'EUCAP_DATASIZE_SPLITTER':'splitter','EUCAP_DATASIZE_SUMMARY':'original_summary'}
    for env,key in keys.items():os.environ[env]=str(checked(cfg['sources'][key]))
    return module('legacy_recipe_dataset_adapter',checked(cfg['sources']['dataset_adapter']))


def recipe(legacy, mother, out):
    args=dict(legacy['arguments'])
    args.update(training_csv=str(mother),out_dir=str(out),evaluation_mode='validation_only',
        inverse_geometry_projection='independent_sigmoid',inverse_checkpoint_selection='training_objective',
        q_target_semantics='exact',q_minimum_margin_physical=0.0,response_semantic_loss_weights='',
        forward_initialization_mode='random',inverse_initialization_mode='random',
        enforce_power_line_port_ground_overlap=False)
    for side in ('forward','inverse'):
        for key in ('initial_weights','initial_summary','initial_weights_sha256','initial_summary_sha256'):
            args[side+'_'+key]=None
    argv=[]
    for key,value in args.items():
        if value is None or value is False:continue
        argv.append('--'+key.replace('_','-'))
        if value is not True:argv.append(str(value))
    return argv


def assert_recipe(args):
    expect={'seed':20260711,'forward_depth':3,'inverse_depth':3,'forward_width':256,'inverse_width':256,
        'forward_hidden_widths':None,'inverse_hidden_widths':None,'batch_size':1024,'learning_rate':.001,
        'weight_decay':1e-6,'response_weight':1.,'geometry_anchor_weight':.01,'topology_feasibility_weight':0.,
        'forward_max_optimizer_updates':1200,'inverse_max_optimizer_updates':1200,'validation_every_optimizer_updates':20,
        'response_loss_family':'mse','response_loss_scaling':'declared_range','q_target_semantics':'exact',
        'q_minimum_margin_physical':0.,'response_semantic_loss_weights':'','training_batch_sampler':'row_uniform',
        'response_weight_schedule':'warmup_ramp_adaptive_ema','response_schedule_domain':'optimizer_update',
        'response_warmup_fraction':.05,'response_ramp_fraction':.25,'response_adaptive_ema_decay':.95,
        'response_adaptive_min_multiplier':.25,'response_adaptive_max_multiplier':4.,
        'inverse_geometry_projection':'independent_sigmoid','inverse_checkpoint_selection':'training_objective',
        'local_refinement_steps':0,'evaluation_mode':'validation_only','stage_checkpoint_mode':'resume_exact',
        'forward_initialization_mode':'random','inverse_initialization_mode':'random',
        'enforce_power_line_port_ground_overlap':False}
    for k,v in expect.items():
        if getattr(args,k,None)!=v:raise ValueError(f'Recipe mismatch {k}: {getattr(args,k,None)!r} != {v!r}')
    for side in ('forward','inverse'):
        for key in ('initial_weights','initial_summary','initial_weights_sha256','initial_summary_sha256'):
            if getattr(args,side+'_'+key,None) is not None:raise ValueError('Scratch initialization required')
    return expect


def observe(runtime,out,deadline):
    """Same after-Adam observer as existing architecture adapter; budget is1200."""
    original=runtime._adam_step
    state={'stage':'not_started','actual_optimizer_updates':0,'stage_started':None}
    def emit(status,extra=None):
        row=dict(utc=utc(),pid=os.getpid(),stage=state['stage'],status=status,
            actual_optimizer_updates=state['actual_optimizer_updates'],target_optimizer_updates=1200,
            stage_elapsed_seconds=0 if state['stage_started'] is None else time.monotonic()-state['stage_started'])
        if extra:row.update(extra)
        atomic(out/'PROGRESS.json',row)
        with (out/'progress_events.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    def step(w,b,gw,gb,s,lr):
        value=original(w,b,gw,gb,s,lr);state['actual_optimizer_updates']=int(s['step'])
        if s['step']==1 or s['step']%120==0:emit('TRAINING',{'observed_parameter_0_0':float(w[0][0,0])})
        return value
    runtime._adam_step=step
    for name,stage in [('_train_forward','forward_training'),('_train_inverse','inverse_training')]:
        raw=getattr(runtime,name)
        def wrapped(*args,_raw=raw,_stage=stage,**kwargs):
            if time.time()>deadline:raise RuntimeError('STAGE_BOUNDARY_DEADLINE: completed stage retained, no automatic retry')
            state.update(stage=_stage,actual_optimizer_updates=0,stage_started=time.monotonic());emit('STAGE_STARTED')
            frozen=None
            if _stage=='inverse_training':frozen=[x.copy() for x in args[3]+args[4]]
            result=_raw(*args,**kwargs)
            if frozen is not None:
                import numpy as np
                for x,y in zip(frozen,args[3]+args[4]):np.testing.assert_array_equal(x,y)
            hist,best=result
            emit('STAGE_TRAINING_COMPLETE',dict(best_optimizer_update=int(best['optimizer_updates']),
                last_validation=hist[-1],forward_frozen_verified=frozen is not None))
            return result
        setattr(runtime,name,wrapped)


def train(cfg,config_path,resume=False):
    d=modules(cfg);c=d.c;cp=checked(cfg['dataset_contract'])
    contract,indices,_=d.load_contract(cp)
    legacy=read(checked(cfg['legacy_summary']));out=ROOT/'run'
    identity=dict(schema='legacy10k_recipe_on_original100k.v1',config_sha256=sha(config_path),
        wrapper_sha256=sha(__file__),legacy_summary=cfg['legacy_summary'],sources=cfg['sources'],
        source_rows=100000,partition_rows={'train':78891,'validation':9740,'test':11369},
        global_split_index_sha256={p:c.index_sha(indices['D100k'][p]) for p in d.PARTS},
        training_reproduction='RECORDED_RECIPE_RECONSTRUCTION_NOT_ORIGINAL_TRAINER_BYTES',
        initialization='SCRATCH_F_AND_I',dataset_original_split_preserved=True,old_outputs_untouched=True)
    if out.is_symlink():raise ValueError('Symlink output forbidden')
    if resume:
        if read(out/'ENTRY_IDENTITY.json')!=identity:raise ValueError('Exact existing identity required')
        if (out/'ENTRY_TERMINAL.json').exists() and read(out/'ENTRY_TERMINAL.json').get('status')=='COMPLETE':return out
    else:
        out.mkdir(exist_ok=False);exclusive(out/'ENTRY_IDENTITY.json',identity)
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime=c.load_runtime();d.install_adapter(runtime,indices,'D100k',sha(cp))
        argv=recipe(legacy,contract['mother']['path'],out); parsed=runtime._parse_args(argv);assert_recipe(parsed)
        exclusive(out/('ENTRY_RESUME_'+str(time.time_ns())+'.json' if resume else 'ENTRY_STARTED.json'),
            dict(utc=utc(),pid=os.getpid(),argv=argv,numpy_version=d.np.__version__,python_version=sys.version,
                completed_stage_resume_only=True,midstage_optimizer_resume=False))
        observe(runtime,out,float(cfg['deadline_epoch']))
        try:
            with (out/'console.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                code=runtime.main(argv)
            sp=out/'physical_feature_tandem_inverse_summary.json';summary=read(sp)
            if summary['execution_status']!='PASS' or not summary['optimizer_budget_contract']['realized']['exact_update_budget_pass']:
                raise RuntimeError('Training budget incomplete')
            assert_recipe(argparse.Namespace(**summary['arguments']))
            atomic(out/'ENTRY_TERMINAL.json',dict(status='COMPLETE',utc=utc(),returncode=code,
                summary_sha256=sha(sp),actual_train_rows=78891,validation_rows=9740,test_rows_reserved=11369,
                test_evaluation='NOT_RUN',fresh_emx='NOT_RUN'))
        except BaseException as exc:
            err=dict(status='FAILED_PRESERVED_NO_AUTOMATIC_RETRY',utc=utc(),error=str(exc),type=type(exc).__name__)
            exclusive(out/('ENTRY_ERROR_'+str(time.time_ns())+'.json'),err);atomic(out/'ENTRY_TERMINAL.json',err);raise
    return out


def launch(cfg,path):
    if (ROOT/'LAUNCH_INTENT.json').exists():raise FileExistsError('Existing launch intent: inspect; no duplicate')
    if cfg['threads']!=4:raise ValueError('Thread budget changed')
    cpus=os.cpu_count();load=os.getloadavg();free=shutil.disk_usage(ROOT).free
    mem={x.split(':')[0]:int(x.split(':')[1].split()[0])*1024 for x in Path('/proc/meminfo').read_text().splitlines()}
    if cpus<8 or load[0]>cpus-8 or mem['MemAvailable']<8*1024**3 or free<5*1024**3:
        raise RuntimeError('RESOURCE_WAIT: need4 CPUs plus reserve, 8GiB available RAM and5GiB disk; no job started')
    cmd=['nice','-n','19',sys.executable,'-B',str(Path(__file__).resolve()),'run','--config',str(path.resolve())]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='4'
    checked(cfg['postprocess_script'])
    exclusive(ROOT/'LAUNCH_INTENT.json',dict(utc=utc(),command=cmd,config_sha256=sha(path),
        logical_cpus=cpus,load=load,available_memory=mem['MemAvailable'],disk_free=free,
        threads=4,concurrent_training_jobs=1,native_simulation=False))
    with (ROOT/'queue.log').open('x') as log:
        proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
    receipt=dict(utc=utc(),status='STARTED',pid=proc.pid,process_start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19],log=str(ROOT/'queue.log'))
    exclusive(ROOT/'LAUNCH_RECEIPT.json',receipt);print(json.dumps(receipt))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','launch','run','resume']);p.add_argument('--config',type=Path,required=True)
    opt=p.parse_args();cfg=read(opt.config)
    if cfg['schema']!='legacy10k_recipe_on_original100k.v1':raise ValueError('Wrong contract')
    if opt.mode=='launch':launch(cfg,opt.config);return
    if opt.mode=='check':
        d=modules(cfg);rt=d.c.load_runtime();old=read(checked(cfg['legacy_summary']))
        args=rt._parse_args(recipe(old,read(checked(cfg['dataset_contract']))['mother']['path'],ROOT/'run'))
        expected=assert_recipe(args)
        result=dict(status='PASS_RECIPE_ARGUMENTS',utc=utc(),wrapper_sha256=sha(__file__),checks=expected)
        exclusive(ROOT/'CHECK_ARGUMENTS.json',result);print(json.dumps(result));return
    with (ROOT/'.queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic(ROOT/'RUN_STATE.json',dict(status='TRAINING',utc=utc(),pid=os.getpid()))
        try:
            resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
            run=train(cfg,opt.config,resume=opt.mode=='resume')
            atomic(ROOT/'RUN_STATE.json',dict(status='EVALUATING_FROZEN_CHECKPOINT',utc=utc(),pid=os.getpid()))
            post=module('legacy_recipe_completion_postprocess',checked(cfg['postprocess_script']))
            result=post.postprocess(run,cfg['evaluation'])
            if result.get('status') not in ('COMPLETE','REUSED_COMPLETE'):
                raise RuntimeError('Postprocess did not complete: '+str(result.get('status')))
            atomic(ROOT/'RUN_STATE.json',dict(status='COMPLETE',utc=utc(),pid=os.getpid(),result=result))
        except BaseException as exc:
            atomic(ROOT/'RUN_STATE.json',dict(status='FAILED_PRESERVED',utc=utc(),pid=os.getpid(),error=str(exc),no_automatic_retry=True));raise


if __name__=='__main__':main()
