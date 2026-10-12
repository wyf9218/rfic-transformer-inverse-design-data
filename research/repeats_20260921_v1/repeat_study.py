#!/usr/bin/env python3
"""Prospectively fixed seed17/29 repetitions, using preserved primary code.

This is a thin entry around the existing explicit-index trainer and completed-
arm evaluator. No original/active file is changed. 24h stops new dispatch and
stage entry; an in-flight optimizer stage drains to its original checkpoint.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).absolute().parent
SEEDS = (17, 29)
PARTS = ('train', 'validation', 'test')
SCHEMA = 'original100k_repeat_seeds.v1'
ENTRY_SCHEMA = 'original100k_repeat_arm.v1'
PINS = {
    'architecture_adapter': '159adcff89f2590b7750af0452fa8919b4d00d468e394ca655c960446b1e2b73',
    'dataset_adapter': 'bd59a6b6d0188d67e1c0f5ae702ce7654e9c6a2f0870625054fd032a8580af55',
    'core': 'ddb5088d9143ac9bd9b870fd4bef1cbdff93ef36c7f0e0176db3fefac03bed49',
    'trainer': 'fbfda2023f862af2f0b185e41c4bf899f97cb79b0d0d763d20501c03525e8a4f',
    'splitter': '9db4b7f1ed9435f73e95e49c558003bdaaf153579c5c44f04fb779aa31e0ef8a',
    'original_summary': 'f05c4c0b3937bf9107f95d38d14679995aedc936071852ac7696f9838c5d43c1',
    'evaluator': '240d7a4419856ec6b17a5bb504ba9e6587bf12a56ecc0342caab879732642381',
    'evaluator_base': '5d713a19b21bdc1c1875531edec11707aa66615f9a88447f5f6c9f54cf04b557',
}


def utc(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()
def pin(p): return {'path': str(Path(p).absolute()), 'sha256': sha(p)}
def checked(v):
    if sha(v['path']) != v['sha256']: raise ValueError('Source identity changed: ' + v['path'])
    return Path(v['path'])
def atomic(p, v):
    p = Path(p); tmp = p.with_name(p.name + '.part')
    with tmp.open('w') as f:
        json.dump(v, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(p)
def exclusive(p, v):
    with Path(p).open('x') as f: json.dump(v, f, indent=2, allow_nan=False); f.write('\n')
def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m
def environment(threads=4):
    env = dict(os.environ)
    for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[k] = str(threads)
    return env


def configure(args):
    dr, ar, root = args.dataset_root.absolute(), args.architecture_root.absolute(), args.out_root.absolute()
    if root.exists(): raise FileExistsError('New repeat root required: ' + str(root))
    paths = {
        'architecture_adapter': ar / 'train_architectures.py',
        'dataset_adapter': dr / 'train_dataset_sizes.py',
        'core': dr / 'runtime_inputs/train_original100k_nested.py',
        'trainer': dr / 'runtime_inputs/trainer_runtime.py',
        'splitter': dr / 'runtime_inputs/model_splitting.py',
        'original_summary': dr / 'runtime_inputs/physical_feature_tandem_inverse_summary.json',
        'evaluator': dr / 'analysis/evaluate_completed_arm.py',
        'evaluator_base': dr / 'analysis/evaluate_arm_once.py',
    }
    sources = {k: pin(p) for k, p in paths.items()}
    for k, expected in PINS.items():
        if sources[k]['sha256'] != expected: raise ValueError('Preserved source changed: ' + k)
    template = pin(dr / 'analysis/COMPLETION_CONFIG.json')
    ec = read(template['path']); checked(ec['dataset_contract'])
    sources.update(entry=pin(__file__), aggregator=pin(args.aggregate_script))
    counts = {'D10k': (8000, 1000, 1000), 'D50k': (40000, 5000, 5000), 'D100k': (78891, 9740, 11369)}
    jobs = []
    # Long arms first is scheduling only, fixed before observing their results.
    for family, arm in [('architecture', 'Large'), ('architecture', 'Medium'), ('dataset', 'D100k'),
                        ('dataset', 'D50k'), ('architecture', 'Small'), ('dataset', 'D10k')]:
        for seed in SEEDS:
            partition = arm if family == 'dataset' else 'D100k'
            jobs.append({'job_id': f'{family}__{arm}__seed{seed}', 'kind': 'new-trained',
                         'family': family, 'arm': arm, 'model_seed': seed, 'partition': partition,
                         'partition_counts': dict(zip(PARTS, counts[partition])),
                         'run_dir': str(root / 'runs' / family / f'seed{seed}' / arm),
                         'eval_dir': str(root / 'analysis_outputs' / family / f'seed{seed}' / 'results' / arm)})
    cfg = {'schema': SCHEMA, 'created_utc': utc(), 'root': str(root), 'sources': sources,
           'primary_evaluation_config': template, 'primary_analysis_root': ec['output_root'],
           'dataset_contract': ec['dataset_contract'], 'jobs': jobs, 'seeds': list(SEEDS),
           'primary_seed': 20260713, 'workers': args.workers, 'threads': 4, 'nice': 19,
           'max_hours': 24, 'deadline_policy': 'stop new dispatch/stage entry; drain active stage, never SIGKILL',
           'selection': 'validation only; fixed T10000/common11369 results cannot alter training or select seeds',
           'resume': 'Explicit exact completed-stage checkpoint only; no mid-stage Adam/RNG resume; no automatic retry',
           'scientific_boundaries': 'Dataset and architecture families separate; commonF original seed20260713; no fresh EM'}
    root.mkdir(parents=True, exist_ok=False); exclusive(root / 'REPEAT_CONFIG.json', cfg)
    print(json.dumps({'status': 'CONFIGURED_NOT_STARTED', 'config': pin(root / 'REPEAT_CONFIG.json'), 'jobs': len(jobs)}))


def load_config(path):
    cfg = read(path)
    if cfg.get('schema') != SCHEMA or cfg.get('seeds') != list(SEEDS) or len(cfg.get('jobs', [])) != 12:
        raise ValueError('Wrong frozen repeat study')
    if not 1 <= cfg['workers'] <= 5 or cfg['threads'] != 4: raise ValueError('CPU envelope exceeded')
    for v in cfg['sources'].values(): checked(v)
    checked(cfg['primary_evaluation_config']); checked(cfg['dataset_contract'])
    cfg['_config_path'] = str(Path(path).absolute())
    return cfg


def training_modules(cfg):
    envkeys = {'EUCAP_DATASET_ADAPTER': 'dataset_adapter', 'EUCAP_DATASET_CORE': 'core',
               'EUCAP_DATASIZE_TRAINER': 'trainer', 'EUCAP_DATASIZE_SPLITTER': 'splitter',
               'EUCAP_DATASIZE_SUMMARY': 'original_summary'}
    for envkey, key in envkeys.items(): os.environ[envkey] = cfg['sources'][key]['path']
    return module(cfg['sources']['architecture_adapter']['path'], 'preserved_repeat_architecture')


def select_job(cfg, job_id):
    job = next(j for j in cfg['jobs'] if j['job_id'] == job_id)
    if job['kind'] != 'new-trained' or job['model_seed'] not in SEEDS: raise ValueError('Repeats must train new weights')
    if job['job_id'] != f"{job['family']}__{job['arm']}__seed{job['model_seed']}": raise ValueError('Repeat identity mismatch')
    return job


def training_argv(a, contract, job):
    out = Path(job['run_dir'])
    argv = (a.architecture_argv(contract, out, job['arm']) if job['family'] == 'architecture'
            else a.d.trainer_argv(contract, out))
    return argv + ['--seed', str(job['model_seed'])]


def check_recipe(args, job):
    expected = {'seed': job['model_seed'], 'evaluation_mode': 'validation_only',
                'forward_max_optimizer_updates': 3600, 'inverse_max_optimizer_updates': 3600,
                'q_target_semantics': 'minimum', 'inverse_geometry_projection': 'hard_feasible_topology_v1',
                'stage_checkpoint_mode': 'resume_exact', 'forward_initial_weights': None,
                'inverse_initial_weights': None, 'forward_initial_summary': None, 'inverse_initial_summary': None,
                'forward_initialization_mode': 'random', 'inverse_initialization_mode': 'random'}
    for key, value in expected.items():
        if getattr(args, key, None) != value: raise ValueError(f'Unexpected {key}: {getattr(args,key,None)}')


class StageDeadline(Exception): pass


def install_deadline(runtime, deadline):
    # Original main saves a completed F stage before entering I. Refusing the
    # next stage here preserves that checkpoint and never changes an update.
    for name in ('_train_forward', '_train_inverse'):
        raw = getattr(runtime, name)
        def wrapped(*args, _raw=raw, _name=name, **kwargs):
            if time.time() >= deadline: raise StageDeadline('Deadline before ' + _name)
            return _raw(*args, **kwargs)
        setattr(runtime, name, wrapped)


def train(cfg, job, deadline, resume=False):
    a = training_modules(cfg); d, c = a.d, a.c
    contract, indices, _ = d.load_contract(Path(cfg['dataset_contract']['path']))
    out = Path(job['run_dir']); partition = job['partition']
    architecture = (a.architecture(job['arm']) if job['family'] == 'architecture'
                    else {'forward': [10, 256, 256, 128, 4], 'inverse': [4, 512, 512, 256, 10],
                          'forward_parameters': 102020, 'inverse_parameters': 399114})
    identity = {'schema': ENTRY_SCHEMA, **job, 'architecture': architecture,
                'dataset_contract_sha256': cfg['dataset_contract']['sha256'],
                'global_split_index_sha256': {p: c.index_sha(indices[partition][p]) for p in PARTS},
                'sources': cfg['sources'], 'initialization': 'SCRATCH_F_AND_I',
                'common_test_in_training': False, 'fresh_emx': 'NOT_RUN'}
    if out.is_symlink(): raise ValueError('Symlink output forbidden')
    if not resume:
        out.mkdir(parents=True, exist_ok=False); exclusive(out / 'ENTRY_IDENTITY.json', identity)
    elif read(out / 'ENTRY_IDENTITY.json') != identity:
        raise ValueError('Exact repeat identity required for completed-stage resume')
    if (out / 'ENTRY_TERMINAL.json').exists() and read(out / 'ENTRY_TERMINAL.json').get('status') == 'COMPLETE':
        return {'status': 'ALREADY_COMPLETE_NOT_REPEATED'}
    with (out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime = c.load_runtime(); d.install_adapter(runtime, indices, partition, cfg['dataset_contract']['sha256'])
        argv = training_argv(a, contract, job); check_recipe(runtime._parse_args(argv), job)
        a.install_progress_observer(runtime, out); install_deadline(runtime, deadline)
        started = {'utc': utc(), 'pid': os.getpid(), 'argv': argv, 'model_seed': job['model_seed'],
                   'python_version': sys.version, 'numpy_version': d.np.__version__, 'deadline_epoch': deadline,
                   'resume_completed_stages_only': resume, 'midstage_adam_resume': False}
        startname = 'ENTRY_RESUME_' + str(time.time_ns()) + '.json' if resume else 'ENTRY_STARTED.json'
        exclusive(out / startname, started)
        try:
            with (out / 'console.log').open('a') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                code = runtime.main(argv)
            sp = out / 'physical_feature_tandem_inverse_summary.json'; summary = read(sp)
            exact = summary['optimizer_budget_contract']['realized']['exact_update_budget_pass']
            if summary.get('execution_status') != 'PASS' or not exact: raise RuntimeError('Training not completed at exact budget')
            check_recipe(runtime._parse_args(argv), job)
            atomic(out / 'ENTRY_TERMINAL.json', {'status': 'COMPLETE', 'utc': utc(), 'returncode': code,
                  'summary_sha256': sha(sp), 'model_seed': job['model_seed'], 'partition_counts': job['partition_counts'],
                  'test_evaluation': 'NOT_RUN', 'fresh_emx': 'NOT_RUN', 'deadline_drain_seconds': max(0, time.time()-deadline)})
            return {'status': 'COMPLETE'}
        except BaseException as exc:
            status = 'PARTIAL_DEADLINE_STAGE_BOUNDARY' if isinstance(exc, StageDeadline) else 'FAILED_PRESERVED'
            failure = {'utc': utc(), 'status': status, 'error': str(exc), 'type': type(exc).__name__,
                       'no_automatic_retry': True, 'completed_stage_resume_only': True}
            exclusive(out / ('ENTRY_ERROR_' + str(time.time_ns()) + '.json'), failure)
            atomic(out / 'ENTRY_TERMINAL.json', failure)
            raise


def repeat_gate(e, job, contract_sha, indices):
    root = Path(job['run_dir']); tp = root / 'ENTRY_TERMINAL.json'
    if not tp.exists(): return {'status': 'WAITING', 'reason': 'NO_TERMINAL'}
    terminal, ident = read(tp), read(root / 'ENTRY_IDENTITY.json')
    if terminal.get('status') != 'COMPLETE': return {'status': 'TRAINING_FAILED', 'reason': terminal.get('status')}
    if ident.get('schema') != ENTRY_SCHEMA or ident.get('kind') != 'new-trained': raise ValueError('Not a new trained repetition')
    for key in ('arm', 'family', 'model_seed', 'job_id', 'partition'):
        if ident.get(key) != job[key]: raise ValueError('Entry repeat identity mismatch: ' + key)
    if ident['dataset_contract_sha256'] != contract_sha: raise ValueError('Repeat contract mismatch')
    if ident['global_split_index_sha256'] != {p: e.b.ish(indices[job['partition']][p]) for p in PARTS}: raise ValueError('Repeat split mismatch')
    sp = root / 'physical_feature_tandem_inverse_summary.json'; e.b.checked(sp, terminal['summary_sha256'])
    summary = read(sp)
    # Seed derives from actual trained summary and ENTRY, never a label/default.
    if summary['arguments']['seed'] != ident['model_seed'] or terminal['model_seed'] != ident['model_seed']:
        raise ValueError('Actual trained seed mismatch')
    if summary['arguments']['evaluation_mode'] != 'validation_only': raise ValueError('Unexpected training test use')
    if not summary['optimizer_budget_contract']['realized']['exact_update_budget_pass']: raise ValueError('Incomplete budget')
    try:
        with (root / '.run.lock').open('r') as lock: fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError: return {'status': 'WAITING', 'reason': 'TRAINER_LOCK_HELD'}
    return {'status': 'READY', 'kind': job['family'], 'terminal': pin(tp), 'identity': pin(root/'ENTRY_IDENTITY.json'),
            'summary': str(sp), 'summary_sha256': terminal['summary_sha256'], 'partition': job['partition']}


def evaluation_job(job):
    if job['kind'] != 'new-trained' or job['model_seed'] not in SEEDS: raise ValueError('No original-prediction shortcut allowed')
    if job['family'] not in ('dataset', 'architecture'): raise ValueError('Unknown comparison family')
    return {'name': job['arm'], 'kind': job['family'], 'run_dir': job['run_dir']}


def evaluate(cfg, job):
    e = module(cfg['sources']['evaluator']['path'], 'preserved_repeat_evaluator')
    evalcfg = read(cfg['primary_evaluation_config']['path'])
    evalcfg.update(output_root=str(Path(job['eval_dir']).parents[1]), plots='off', _config_path=cfg['_config_path'])
    arithmetic_job = evaluation_job(job)
    e.inspect_terminal = lambda ignored, contract_sha, indices: repeat_gate(e, job, contract_sha, indices)
    def forbidden(*args, **kwargs): raise RuntimeError('Historical prediction shortcut prohibited for every new seed, including D100k')
    e.reuse_original_predictions = forbidden
    original_tables = e.data_tables
    def tables_with_small_counts(result, fr, frame, out, arithmetic_job):
        counts = original_tables(result, fr, frame, out, arithmetic_job)
        e.b.write_csv(out/'target_hit_counts.csv', [
            {'evidence_class': key, **{k: value[k] for k in ('original_N','hits','fraction','request_failures')}}
            for key, value in counts.items()])
        return counts
    e.data_tables = tables_with_small_counts
    common_table = e.b.common_test_table
    def preserved_common_test(contract, indices):
        previous = e.b.A; e.b.A = Path(cfg['primary_analysis_root'])
        try:
            if not (e.b.A/'common_test_source/SOURCE.json').is_file(): raise ValueError('Pinned primary common-test cache missing; no new mother scan')
            return common_table(contract, indices)
        finally: e.b.A = previous
    e.b.common_test_table = preserved_common_test
    # Queue calls this after the trainer subprocess has exited and released lock.
    result = e.evaluate(evalcfg, arithmetic_job)
    if result['status'] != 'COMPLETE': raise RuntimeError('Evaluation not ready: ' + json.dumps(result))
    dest = Path(job['eval_dir']) / 'COMPLETE.json'; done = read(dest)
    done.update(kind='new-trained', model_seed=read(Path(job['run_dir'])/'ENTRY_IDENTITY.json')['model_seed'],
                repeat_job_id=job['job_id'], training_identity=pin(Path(job['run_dir'])/'ENTRY_IDENTITY.json'),
                comparison_boundary='Prospectively fixed seed17/29 repetitions; dataset and architecture families separate. '
                    'New scratch F/I weights including D100k. Frozen original seed20260713 commonF; SELF/COMMON proxy, not fresh EM. '
                    'Test and T10000 not used for training, selection, or seed choice.')
    atomic(dest, done)
    return {'status': 'COMPLETE', 'receipt': pin(dest), 'job_id': job['job_id']}


def launch(cfg):
    root = Path(cfg['root']); python = os.path.abspath(sys.executable)  # retain venv symlink spelling
    probe = subprocess.run([python, '-c', 'import numpy,matplotlib,sys,json; print(json.dumps({"executable":sys.executable,"prefix":sys.prefix,"numpy":numpy.__version__,"matplotlib":matplotlib.__version__}))'],
                           env=environment(), capture_output=True, text=True, check=True)
    now = time.time(); deadline = now + 24*3600
    cmd = ['nice', '-n', '19', python, str(Path(__file__).absolute()), 'run', '--config', cfg['_config_path']]
    intent = {'utc': utc(), 'command': cmd, 'deadline_epoch': deadline, 'config': pin(cfg['_config_path']),
              'environment_probe': json.loads(probe.stdout), 'workers': cfg['workers'], 'threads': 4, 'once_only': True}
    exclusive(root/'LAUNCH_INTENT.json', intent)
    with (root/'queue.log').open('x') as log:
        proc = subprocess.Popen(cmd, env=environment(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    exclusive(root/'LAUNCH_RECEIPT.json', {'utc': utc(), 'pid': proc.pid, 'intent': pin(root/'LAUNCH_INTENT.json'), 'deadline_epoch': deadline})
    print(json.dumps({'status': 'STARTED', 'pid': proc.pid, 'deadline_epoch': deadline}))


def run_queue(cfg):
    root = Path(cfg['root']); intent = read(root/'LAUNCH_INTENT.json')
    if intent['config']['sha256'] != sha(cfg['_config_path']): raise ValueError('Launch/config identity changed')
    deadline = intent['deadline_epoch']; python = intent['command'][3]
    with (root/'.run_all.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        state = {'status': 'RUNNING', 'utc': utc(), 'pid': os.getpid(), 'deadline_epoch': deadline,
                 'workers': cfg['workers'], 'threads_per_worker': 4, 'results': {}}
        exclusive(root/'RUN_ALL.json', state)
        def worker(job):
            if time.time() >= deadline: return {'status': 'NOT_DISPATCHED_DEADLINE'}
            jr = root/'jobs'/job['job_id']; jr.mkdir(parents=True, exist_ok=False)
            command = ['nice', '-n', '19', python, str(Path(__file__).absolute()), 'train', '--config', cfg['_config_path'],
                       '--job', job['job_id'], '--deadline-epoch', str(deadline)]
            exclusive(jr/'ATTEMPT.json', {'utc': utc(), 'command': command, 'job': job, 'attempt': 1})
            start = time.monotonic()
            with (jr/'train_launcher.log').open('x') as log: code = subprocess.run(command, env=environment(), stdout=log, stderr=subprocess.STDOUT).returncode
            result = {'training_returncode': code, 'elapsed_training_seconds': time.monotonic()-start, 'finished_utc': utc()}
            if code:
                result['status'] = 'TRAINING_FAILED_OR_PARTIAL_NO_RETRY'; atomic(jr/'RESULT.json', result); return result
            ecmd = ['nice', '-n', '19', python, str(Path(__file__).absolute()), 'evaluate', '--config', cfg['_config_path'], '--job', job['job_id']]
            exclusive(jr/'EVALUATION_ATTEMPT.json', {'utc': utc(), 'command': ecmd, 'attempt': 1})
            start = time.monotonic()
            with (jr/'evaluation.log').open('x') as log: ecode = subprocess.run(ecmd, env=environment(), stdout=log, stderr=subprocess.STDOUT).returncode
            result.update(evaluation_returncode=ecode, elapsed_evaluation_seconds=time.monotonic()-start,
                          status='COMPLETE' if ecode == 0 else 'TRAINED_EVALUATION_FAILED_NO_RETRY')
            atomic(jr/'RESULT.json', result); return result
        with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
            pending = {pool.submit(worker, j): j['job_id'] for j in cfg['jobs']}
            for future in as_completed(pending):
                try: state['results'][pending[future]] = future.result()
                except BaseException as exc: state['results'][pending[future]] = {'status': 'FAILED_PRESERVED_NO_RETRY', 'error': str(exc)}
                atomic(root/'RUN_ALL.json', state)
        # Exactly one statistics callback after all queue tasks terminate; it
        # decides completeness separately per family and never retrains a job.
        cmd = ['nice', '-n', '19', python, cfg['sources']['aggregator']['path'], '--manifest', cfg['_config_path'], '--out', str(root/'aggregate_v1')]
        exclusive(root/'AGGREGATE_ATTEMPT.json', {'utc': utc(), 'command': cmd, 'source': cfg['sources']['aggregator']})
        with (root/'aggregate.log').open('x') as log: acode = subprocess.run(cmd, env=environment(1), stdout=log, stderr=subprocess.STDOUT).returncode
        all_ok = all(r.get('status') == 'COMPLETE' for r in state['results'].values())
        state.update(status='COMPLETE' if all_ok and acode == 0 else 'FINISHED_WITH_MISSING_OR_FAILURE',
                     aggregate_returncode=acode, finished_utc=utc(), deadline_drain_seconds=max(0,time.time()-deadline))
        atomic(root/'RUN_ALL.json', state)
        return state


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['configure','launch','run','train','resume-job','evaluate'])
    p.add_argument('--config', type=Path); p.add_argument('--dataset-root', type=Path); p.add_argument('--architecture-root', type=Path)
    p.add_argument('--out-root', type=Path); p.add_argument('--aggregate-script', type=Path); p.add_argument('--workers', type=int, choices=range(1,6), default=5)
    p.add_argument('--job'); p.add_argument('--deadline-epoch', type=float)
    args = p.parse_args()
    if args.mode == 'configure':
        if not all([args.dataset_root,args.architecture_root,args.out_root,args.aggregate_script]): p.error('configure needs explicit preserved roots/new root/aggregator')
        configure(args); return
    if not args.config: p.error('--config required')
    cfg = load_config(args.config)
    if args.mode == 'launch': launch(cfg); return
    if args.mode == 'run': print(json.dumps(run_queue(cfg))); return
    if not args.job: p.error('--job required')
    job = select_job(cfg,args.job)
    if args.mode == 'evaluate': print(json.dumps(evaluate(cfg,job))); return
    if args.deadline_epoch is None: p.error('train/resume-job needs explicit original or authorized continuation deadline')
    if args.deadline_epoch > time.time()+24*3600+2: raise ValueError('Training entry cannot allocate more than24h')
    print(json.dumps(train(cfg,job,args.deadline_epoch,resume=args.mode=='resume-job')))


if __name__ == '__main__': main()
