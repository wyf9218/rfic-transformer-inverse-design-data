"""Bind existing MARS inputs and start the bounded ordinary training queue once."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PYTHON = '/volumes/research-localdata/ywang3652/rfic-transformer-inverse-design-mars56-s4p-20260702/.venv/bin/python'
MOTHER = '/volumes/research-localdata/ywang3652/current_foundry_campaign_20260728/first100k_v1/real_emx_round_v2/model_checkpoint_1_n100000/training_table/physical_feature_inverse_training_table.csv'


def pin(p):
    p = Path(p)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}


def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    if (ROOT/'QUEUE_LAUNCH_INTENT.json').exists() or (ROOT/'RUN_ALL.json').exists():
        raise RuntimeError('Existing launch/queue evidence: inspect; do not submit again')
    env = dict(os.environ)
    env.update(EUCAP_DATASET_CORE=str(ROOT/'runtime_inputs/train_original100k_nested.py'),
               EUCAP_DATASIZE_TRAINER=str(ROOT/'runtime_inputs/trainer_runtime.py'),
               EUCAP_DATASIZE_SPLITTER=str(ROOT/'runtime_inputs/model_splitting.py'),
               EUCAP_DATASIZE_SUMMARY=str(ROOT/'runtime_inputs/physical_feature_tandem_inverse_summary.json'),
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               MKL_NUM_THREADS='4', VECLIB_MAXIMUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    original = ROOT/'data/DATASET_SIZE_CONTRACT.json'
    contract = json.loads(original.read_text())
    contract['mother']['path'] = MOTHER
    for key, name in [('indices','DATASET_SIZE_INDICES.json'), ('qualification','QUALIFICATION.json')]:
        contract[key]['path'] = str(ROOT/'data'/name)
    for key, name in [('original_summary','physical_feature_tandem_inverse_summary.json'), ('splitter','model_splitting.py')]:
        contract[key]['path'] = str(ROOT/'runtime_inputs'/name)
    remote = ROOT/'data/DATASET_SIZE_CONTRACT_MARS.json'
    write(remote, contract)
    check = subprocess.run([PYTHON, '-B', str(ROOT/'train_dataset_sizes.py'), 'check', '--contract', str(remote)],
                           env=env, text=True, capture_output=True)
    write(ROOT/'MARS_INPUT_CHECK.json', {'returncode': check.returncode, 'stdout': check.stdout, 'stderr': check.stderr})
    if check.returncode:
        raise RuntimeError('Input check failed; no training started')
    mem = {l.split(':')[0]: int(l.split(':')[1].split()[0])*1024
           for l in Path('/proc/meminfo').read_text().splitlines()}
    cpus, load, free = os.cpu_count(), os.getloadavg(), shutil.disk_usage(ROOT).free
    if cpus < 16 or load[0] > cpus-12 or mem['MemAvailable'] < 16*1024**3 or free < 20*1024**3:
        raise RuntimeError('Research resource admission insufficient; no training started')
    cmd = [PYTHON, '-B', str(ROOT/'run_all.py'), 'run', '--contract', str(remote),
           '--arms', 'D10k', 'D50k', '--workers', '2', '--threads', '4']
    sources = [ROOT/'train_dataset_sizes.py', ROOT/'run_all.py', ROOT/'launch_mars_once.py',
               ROOT/'runtime_inputs/train_original100k_nested.py', ROOT/'runtime_inputs/trainer_runtime.py',
               ROOT/'runtime_inputs/model_splitting.py', ROOT/'runtime_inputs/physical_feature_tandem_inverse_summary.json']
    identity = {'utc': datetime.now(timezone.utc).isoformat(), 'command': cmd,
                'source_contract': pin(original), 'runtime_contract': pin(remote),
                'sources': [pin(p) for p in sources],
                'resource_observation': {'logical_cpus': cpus, 'load': load,
                  'available_memory_bytes': mem['MemAvailable'], 'free_disk_bytes': free,
                  'maximum_worker_processes': 2, 'blas_threads_per_worker': 4, 'nice': 19},
                'native_simulation': False, 'historical_weights_overwritten': False}
    write(ROOT/'QUEUE_LAUNCH_INTENT.json', identity)
    with (ROOT/'QUEUE.stdout.log').open('xb') as log:
        child = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
    start_ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'pid': child.pid,
               'process_start_ticks': start_ticks, 'status': 'QUEUE_STARTED',
               'intent': pin(ROOT/'QUEUE_LAUNCH_INTENT.json'), 'arms': ['D10k','D50k'],
               'log': str(ROOT/'QUEUE.stdout.log'), 'run_root': str(ROOT/'runs')}
    write(ROOT/'QUEUE_START_RECEIPT.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
