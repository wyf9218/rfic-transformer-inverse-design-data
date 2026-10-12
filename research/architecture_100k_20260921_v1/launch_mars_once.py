"""One-time bounded MARS launch; leaves the already-running size queue alone."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
DATA = Path('/volumes/research-localdata/ywang3652/eucap_paper_dataset_size_811_20260921_v1')
PYTHON = '/volumes/research-localdata/ywang3652/rfic-transformer-inverse-design-mars56-s4p-20260702/.venv/bin/python'


def pin(p):
    p = Path(p)
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    if (ROOT/'QUEUE_LAUNCH_INTENT.json').exists() or (ROOT/'RUN_ALL.json').exists():
        raise RuntimeError('Prior queue/launch evidence exists: no duplicate submission')
    env = dict(os.environ)
    env.update(EUCAP_DATASET_ADAPTER=str(DATA/'train_dataset_sizes.py'),
               EUCAP_DATASET_CORE=str(DATA/'runtime_inputs/train_original100k_nested.py'),
               EUCAP_DATASIZE_TRAINER=str(DATA/'runtime_inputs/trainer_runtime.py'),
               EUCAP_DATASIZE_SPLITTER=str(DATA/'runtime_inputs/model_splitting.py'),
               EUCAP_DATASIZE_SUMMARY=str(DATA/'runtime_inputs/physical_feature_tandem_inverse_summary.json'),
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    contract = DATA/'data/DATASET_SIZE_CONTRACT_MARS.json'
    check = subprocess.run([PYTHON, '-B', str(ROOT/'train_architectures.py'), 'check',
                            '--contract', str(contract), '--arm', 'Small'],
                           env=env, text=True, capture_output=True)
    write(ROOT/'MARS_INPUT_CHECK.json', dict(returncode=check.returncode, stdout=check.stdout, stderr=check.stderr))
    if check.returncode:
        raise RuntimeError('Architecture input/environment check failed; no training launched')
    mem = {l.split(':')[0]: int(l.split(':')[1].split()[0])*1024
           for l in Path('/proc/meminfo').read_text().splitlines()}
    cpus, load, free = os.cpu_count(), os.getloadavg(), shutil.disk_usage(ROOT).free
    if cpus < 32 or load[0] > cpus-16 or mem['MemAvailable'] < 24*1024**3 or free < 20*1024**3:
        raise RuntimeError('Research resource admission insufficient; no training launched')
    cmd = [PYTHON, '-B', str(ROOT/'run_all.py'), 'run', '--contract', str(contract), '--workers', '3', '--threads', '4']
    write(ROOT/'QUEUE_LAUNCH_INTENT.json', dict(utc=datetime.now(timezone.utc).isoformat(), command=cmd,
          contract=pin(contract), sources=[pin(ROOT/n) for n in ('train_architectures.py','run_all.py','launch_mars_once.py')],
          resource_observation=dict(logical_cpus=cpus, load=load, available_memory_bytes=mem['MemAvailable'],
               free_disk_bytes=free, additional_blas_threads=12, existing_size_queue_threads=8, nice=19),
          no_existing_tasks_modified=True, native_simulation=False))
    with (ROOT/'QUEUE.stdout.log').open('xb') as log:
        proc = subprocess.Popen(cmd, env=env, cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ticks = Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    receipt = dict(utc=datetime.now(timezone.utc).isoformat(), pid=proc.pid, process_start_ticks=ticks,
                   arms=['Small','Medium','Large'], status='QUEUE_STARTED', intent=pin(ROOT/'QUEUE_LAUNCH_INTENT.json'),
                   log=str(ROOT/'QUEUE.stdout.log'), run_root=str(ROOT/'runs'))
    write(ROOT/'QUEUE_START_RECEIPT.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
