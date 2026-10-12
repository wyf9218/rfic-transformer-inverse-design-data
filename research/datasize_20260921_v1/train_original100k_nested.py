#!/usr/bin/env python3
"""Small explicit-index adapter around the unchanged deployed-100K NumPy trainer.

No data generation, EM invocation, remote access, or alternate neural framework.
The supplied qualification and index manifests must close before run/resume.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import types

import numpy as np

ROOT = Path('/Users/wyf/Documents/模拟变压器AI反向建模')
STUDY = Path(__file__).resolve().parent
TOOLKIT = ROOT / 'reports/frozen_mlp_synthesis_toolkit_20260804/toolkit/rfic_transformer_inverse_synthesis_toolkit_v1/model'
RUNTIME = Path(os.environ.get('EUCAP_DATASIZE_TRAINER',str(TOOLKIT / 'trainer_runtime.py')))
RUNTIME_SHA = 'fbfda2023f862af2f0b185e41c4bf899f97cb79b0d0d763d20501c03525e8a4f'
SPLITTER = Path(os.environ.get('EUCAP_DATASIZE_SPLITTER',str(ROOT / 'github_worktrees/eucap15-mlp-capacity-20260909/rfic_transformer_inverse_design/model_splitting.py')))
SPLITTER_SHA = '9db4b7f1ed9435f73e95e49c558003bdaaf153579c5c44f04fb779aa31e0ef8a'
SUMMARY = Path(os.environ.get('EUCAP_DATASIZE_SUMMARY',str(TOOLKIT / 'physical_feature_tandem_inverse_summary.json')))
SUMMARY_SHA = 'f05c4c0b3937bf9107f95d38d14679995aedc936071852ac7696f9838c5d43c1'
MOTHER_SHA = 'fe94d15b0d083330fe7a64d3e059d083a5aa32d73394c85fc4b881e9904e68a2'
COUNTS = (10000, 50000, 100000)
SPLIT_MODE = 'explicit_frozen_indices_external_holdout'
SCHEMA = 'original100k_nested_external_holdout.v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_pin(pin):
    p = Path(pin['path']).resolve(strict=True)
    if not p.is_file() or sha(p) != pin['sha256']:
        raise ValueError(f'Identity mismatch: {p}')
    return p


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_atomic(path, payload):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    temp.replace(path)


def index_sha(values):
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def validate_indices(indices, mother_rows, combined_rows, counts=COUNTS):
    expected = {f'train_{n}' for n in counts} | {'validation', 'test'}
    if set(indices) != expected:
        raise ValueError('Index keys must identify every arm and the one shared validation/test pair')
    for name, values in indices.items():
        if not isinstance(values, list) or not values or any(type(v) is not int for v in values):
            raise ValueError(f'Nonempty integer index list required: {name}')
        if len(set(values)) != len(values) or min(values) < 0 or max(values) >= combined_rows:
            raise ValueError(f'Duplicate or out-of-bounds index: {name}')
    for n in counts:
        values = indices[f'train_{n}']
        if len(values) != n or max(values) >= mother_rows:
            raise ValueError('Training rows must be the stated count from the original mother prefix only')
    largest = indices[f'train_{counts[-1]}']
    if set(largest) != set(range(mother_rows)):
        raise ValueError('Largest arm must use the complete original mother dataset exactly once')
    for n in counts[:-1]:
        if indices[f'train_{n}'] != largest[:n]:
            raise ValueError('Arms must preserve prefixes of one frozen nested training order')
    val, test = set(indices['validation']), set(indices['test'])
    if val & test or min(val | test) < mother_rows:
        raise ValueError('The identical validation/test cohorts must be external and disjoint')
    if set(range(mother_rows)) | val | test != set(range(combined_rows)):
        raise ValueError('Combined table must consist of mother plus exactly the two frozen holdouts')


def require_qualification(contract, qualification):
    if qualification.get('status') != 'PASS':
        raise ValueError('NO_GO: external holdout qualification is not PASS')
    for key, value in {
        'combined_csv_sha256': contract['combined_csv']['sha256'],
        'mother_csv_sha256': contract['mother']['sha256'],
        'indices_sha256': contract['indices']['sha256'],
    }.items():
        if qualification.get(key) != value:
            raise ValueError(f'NO_GO: qualification does not bind {key}')
    required = ['physical_compatibility', 'unique_geometry', 'mother_prefix_exact',
                'holdout_disjoint_from_mother', 'actual_15ghz_labels',
                'holdout_frozen_before_current_study_training',
                'holdout_not_used_for_current_study_gradient_or_tuning']
    checks = qualification.get('checks', {})
    missing = [k for k in required if checks.get(k) is not True]
    if missing:
        raise ValueError('NO_GO: missing actual qualification evidence: ' + ', '.join(missing))
    if qualification.get('independence_scope') != 'CURRENT_STUDY_ONLY':
        raise ValueError('Qualification must state CURRENT_STUDY_ONLY independence scope')
    if not qualification.get('historical_exposure_status') or not qualification.get('family_lineage_status'):
        raise ValueError('Historical exposure and family lineage need explicit status, including UNKNOWN where necessary')
    if qualification.get('known_family_overlap') is not False:
        raise ValueError('NO_GO: known-family overlap must be explicitly assessed and absent')
    assessment=qualification.get('source_method_family_assessment',{})
    if assessment.get('status') not in ('PASS_INDEPENDENT_GEOMETRY_PROPOSALS','PASS_FAMILY_GROUPED_HOLDOUT') or not assessment.get('evidence'):
        raise ValueError('NO_GO: source-method or family-group separation evidence is missing; UNKNOWN does not pass')


def load_contract(path):
    contract = read(path)
    if contract.get('schema') != SCHEMA or contract['mother']['rows'] != 100000:
        raise ValueError('Wrong study schema or original mother row count')
    if contract['mother']['sha256'] != MOTHER_SHA:
        raise ValueError('Only the pinned original100K mother is supported')
    # Fail before reading the large table if qualification is not yet closed.
    qpath = checked_pin(contract['qualification'])
    qualification = read(qpath)
    require_qualification(contract, qualification)
    ipath = checked_pin(contract['indices'])
    indices = read(ipath)
    validate_indices(indices, 100000, int(contract['combined_csv']['rows']))
    checked_pin(contract['combined_csv'])
    return contract, indices, qualification


def load_runtime():
    checked_pin({'path': str(RUNTIME), 'sha256': RUNTIME_SHA})
    checked_pin({'path': str(SPLITTER), 'sha256': SPLITTER_SHA})
    # Load the sole dependency without importing package __init__ (which imports
    # simulator/optimizer modules irrelevant to this offline training adapter).
    names = ('rfic_transformer_inverse_design', 'rfic_transformer_inverse_design.model_splitting')
    old = {n: sys.modules.get(n) for n in names}
    old_path = list(sys.path)
    try:
        package = types.ModuleType(names[0])
        package.__path__ = [str(SPLITTER.parent)]
        sys.modules[names[0]] = package
        spec = importlib.util.spec_from_file_location(names[1], SPLITTER)
        dep = importlib.util.module_from_spec(spec)
        sys.modules[names[1]] = dep
        spec.loader.exec_module(dep)
        spec = importlib.util.spec_from_file_location('original100k_frozen_runtime', RUNTIME)
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
    finally:
        sys.path[:] = old_path
        for n, value in old.items():
            if value is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = value
    required = {
        '_build_matrix': ['rows', 'input_columns', 'geometry_columns', 'split_reference_columns'],
        '_split_indices': ['matrix', 'args'], '_normalize': ['matrix', 'split', 'floor'],
        '_physical_cell_strict_pass_summary': ['target_physical', 'strict_pass', 'split_audit', 'partition'],
    }
    for name, args in required.items():
        if list(inspect.signature(getattr(runtime, name)).parameters) != args:
            raise ValueError(f'Historical function signature changed: {name}')
    return runtime


def select_matrix(full, indices, arm, expected_full_rows):
    if full['count'] != expected_full_rows or full['source_indices'] != list(range(expected_full_rows)):
        raise ValueError('Finite-row conversion changed the CSV position mapping; refuse implicit remapping')
    chosen = indices[f'train_{arm}'] + indices['validation'] + indices['test']
    identities = [full['source_geometry_identities'][i] for i in chosen]
    if len(set(identities)) != len(identities):
        raise ValueError('Geometry identities overlap inside the selected arm and holdout')
    out = dict(full)
    out['count'] = len(chosen)
    for key in ['x', 'y', 'split_x']:
        out[key] = np.asarray(full[key])[chosen].copy()
    for key in ['source_indices', 'source_evaluations', 'source_geometry_identities']:
        out[key] = [full[key][i] for i in chosen]
    out['global_combined_indices'] = chosen
    return out


def explicit_split(matrix, indices, arm, contract_sha):
    nval, ntest = len(indices['validation']), len(indices['test'])
    if matrix['count'] != arm + nval + ntest:
        raise ValueError('Arm matrix count mismatch')
    split = {'train': np.arange(arm), 'validation': np.arange(arm,arm+nval),
             'test': np.arange(arm+nval,arm+nval+ntest)}
    lower,upper = np.array([.5,.5,5,0]),np.array([3,3,25,.8])
    values=np.asarray(matrix['split_x'])
    norm=(values-lower)/(upper-lower)
    # Preserve the original checkpoint-selection domain; no new clipping/filtering.
    if np.any(~np.isfinite(norm)) or np.any(norm < -1e-12) or np.any(norm > 1+1e-12):
        raise ValueError('Data exceed original cell-tail checkpoint domain; do not clip or silently change selection')
    cells=np.clip(np.floor(np.clip(norm,0,1)*4).astype(int),0,3)
    all_ids=[':'.join(map(str,r)) for r in cells]
    cell_ids={k:sorted({all_ids[i] for i in v}) for k,v in split.items()}
    overlap=(set(cell_ids['train'])&set(cell_ids['validation'])) | (set(cell_ids['train'])&set(cell_ids['test'])) | (set(cell_ids['validation'])&set(cell_ids['test']))
    audit={'split_mode':SPLIT_MODE,'physical_cell_grouped':False,'physical_cell_range_source':'explicit',
           'physical_cell_bins_per_dimension':4,'physical_cell_lower':lower.tolist(),'physical_cell_upper':upper.tolist(),
           'cell_ids':cell_ids,'physical_cell_overlap_count':len(overlap),
           'row_counts':{k:len(v) for k,v in split.items()},
           'split_index_sha256':{k:index_sha(v) for k,v in split.items()},
           'global_split_index_sha256':{'train':index_sha(indices[f'train_{arm}']),
                                        'validation':index_sha(indices['validation']),'test':index_sha(indices['test'])},
           'all_rows_assigned_once':True,'explicit_external_holdout':True,'formal_cell_ood':False,
           'study_contract_sha256':contract_sha,'adapter_sha256':sha(__file__),
           'boundary':'Fixed disjoint external row holdouts; shared physical cells are reported, never claimed cell-OOD.'}
    return split,audit


def install_adapter(runtime, indices, arm, combined_rows, contract_sha):
    original_build=runtime._build_matrix
    original_parse=runtime._parse_args
    original_cells=runtime._physical_cell_strict_pass_summary
    def build(rows,input_columns,geometry_columns,split_reference_columns):
        return select_matrix(original_build(rows,input_columns,geometry_columns,split_reference_columns),indices,arm,combined_rows)
    def split(matrix,args):
        return explicit_split(matrix,indices,arm,contract_sha)
    def parse(argv):
        args=original_parse(argv)
        args.split_mode=SPLIT_MODE
        args.validation_fraction=len(indices['validation'])/(arm+len(indices['validation'])+len(indices['test']))
        args.test_fraction=len(indices['test'])/(arm+len(indices['validation'])+len(indices['test']))
        return args
    def cells(target_physical,strict_pass,split_audit,*,partition='validation'):
        # The old helper's name check conflates grouping a validation metric with
        # how partitions were chosen. Reuse its exact arithmetic on validation
        # cells only. The authoritative audit remains explicit/non-cell-OOD.
        local=dict(split_audit)
        local['split_mode']='physical_cell_grouped'
        return original_cells(target_physical,strict_pass,local,partition=partition)
    runtime._build_matrix=build
    runtime._split_indices=split
    runtime._parse_args=parse
    runtime._physical_cell_strict_pass_summary=cells


def trainer_argv(contract, out_dir):
    checked_pin({'path':str(SUMMARY),'sha256':SUMMARY_SHA})
    args=dict(read(SUMMARY)['arguments'])
    args.update(training_csv=contract['combined_csv']['path'],out_dir=str(out_dir),
                min_training_rows=1,evaluation_mode='validation_only',stage_checkpoint_mode='resume_exact')
    argv=[]
    for key,value in args.items():
        if value is None or value is False:
            continue
        argv.append('--'+key.replace('_','-'))
        if value is not True:
            argv.append(str(value))
    return argv


def protected_output(arm):
    path=STUDY/'runs'/f'train_{arm}'
    if path.is_symlink() or (path.parent.exists() and path.parent.resolve()!=STUDY/'runs'):
        raise ValueError('Output may not follow a symlink')
    return path


def launch(contract_path,contract,indices,qualification,arm,resume=False):
    out=protected_output(arm)
    contract_sha=sha(contract_path)
    identity={'schema':'original100k_nested_arm.v1','arm_train_rows':arm,
              'contract_sha256':contract_sha,'adapter_sha256':sha(__file__),
              'runtime_sha256':RUNTIME_SHA,'splitter_sha256':SPLITTER_SHA,
              'global_index_sha256':{k:index_sha(v) for k,v in indices.items() if k in (f'train_{arm}','validation','test')},
              'independence_scope':qualification['independence_scope'],
              'historical_exposure_status':qualification['historical_exposure_status'],
              'family_lineage_status':qualification['family_lineage_status'],
              'trained_old_holdout_mother_rows':True,
              'old_model_outputs_untouched':True,'fresh_emx':'NOT_RUN'}
    if out.exists() and not resume:
        raise FileExistsError(f'No-clobber: {out}. Use resume only for this same frozen arm.')
    if resume:
        if not out.is_dir() or read(out/'ENTRY_IDENTITY.json') != identity:
            raise ValueError('Resume requires the exact existing arm identity')
        terminal=out/'ENTRY_TERMINAL.json'
        if terminal.exists() and read(terminal).get('status')=='COMPLETE':
            print(f'Already complete; not repeated: {out}')
            return
    else:
        out.mkdir(parents=True,exist_ok=False)
        write_atomic(out/'ENTRY_IDENTITY.json',identity)
    with (out/'.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Arm already has a live execution owner; no duplicate launch')
        runtime=load_runtime()
        install_adapter(runtime,indices,arm,int(contract['combined_csv']['rows']),contract_sha)
        argv=trainer_argv(contract,out)
        write_atomic(out/'ENTRY_STARTED.json',{'utc':utc(),'pid':os.getpid(),'argv':argv,'resume_completed_stages_only':resume})
        try:
            with (out/'console.log').open('a') as log, contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                code=runtime.main(argv)
            summary=read(out/'physical_feature_tandem_inverse_summary.json')
            complete=summary.get('execution_status')=='PASS'
            write_atomic(out/'ENTRY_TERMINAL.json',{'utc':utc(),'status':'COMPLETE' if complete else 'FAILED',
                         'returncode':code,'summary_sha256':sha(out/'physical_feature_tandem_inverse_summary.json'),
                         'actual_train_rows':arm,'validation_rows':len(indices['validation']),
                         'test_rows_reserved':len(indices['test']),'test_evaluation':'NOT_RUN','fresh_emx':'NOT_RUN'})
            if not complete:
                raise RuntimeError('Historical trainer did not report execution_status PASS; inspect preserved output')
        except BaseException as exc:
            write_atomic(out/f'ENTRY_ERROR_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}.json',
                         {'utc':utc(),'type':type(exc).__name__,'error':str(exc),'resume_boundary':'completed stages only; no live Adam/RNG resume'})
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['check','run','resume'])
    parser.add_argument('--contract',required=True,type=Path)
    parser.add_argument('--arm',choices=['10000','50000','100000','all'],default='all')
    options=parser.parse_args()
    contract,indices,qualification=load_contract(options.contract)
    if options.mode=='check':
        print(json.dumps({'status':'PASS_INPUT_IDENTITIES_AND_EXPLICIT_SPLITS',
              'arms':list(COUNTS),'validation':len(indices['validation']),'test':len(indices['test']),
              'training_started':False,'independence_scope':qualification['independence_scope']}))
        return
    arms=COUNTS if options.arm=='all' else (int(options.arm),)
    for arm in arms:
        launch(options.contract,contract,indices,qualification,arm,resume=options.mode=='resume')


if __name__=='__main__':
    main()
