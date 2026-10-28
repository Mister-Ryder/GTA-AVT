"""Deferred, duplicate-guarded handoff to the authorized two-arm GDF revision.

Only consumes completion receipts. It never signals or modifies the old runs.
No SSH passwords are needed by this supervisor. Failed jobs stop, never retry.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

OLD_JOBS = {'tail_20261032_C0', 'tail_20261032_AR', 'tail_20261032_C0_GDF'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(str(temporary), str(path))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()


def alive(identity):
    try:
        fields = Path('/proc/%s/stat' % identity['pid']).read_text().split()
        return fields[21] == str(identity['start_ticks']) and fields[2] != 'Z'
    except FileNotFoundError:
        return False


def barrier(config):
    """True only after all four previous endpoints include native VAL8 results."""
    main_path = Path(config['main_completion_receipt'])
    queue_path = Path(config['old_queue_state'])
    if not main_path.exists() or not queue_path.exists():
        return False, 'waiting for old completion receipts'
    main = read(main_path)
    if (main.get('status') != 'complete' or main.get('exit_code') != 0 or
            main.get('successful_updates') != 122000 or main.get('native_cases') != 8 or
            main.get('test_opened') is not False):
        return False, 'main endpoint not certified complete'
    queue = read(queue_path)
    jobs = {j['name']: j for j in queue.get('jobs', []) if j['name'] in OLD_JOBS}
    if set(jobs) != OLD_JOBS:
        raise ValueError('Old queue does not contain the three exact authorized controls')
    for name, job in jobs.items():
        if job.get('status') in ('failed', 'error'):
            raise RuntimeError('Old run failed; no automatic handoff: ' + name)
        if job.get('status') != 'complete' or job.get('exit_code') != 0:
            return False, 'waiting for ' + name
        if alive(job['process']):
            return False, 'old training process still alive: ' + name
        pipeline = read(Path(job['run']) / 'pipeline.json')
        if pipeline.get('status') != 'complete' or pipeline.get('completed_successful_updates') != 42000:
            raise ValueError('Old pipeline is not a completed 42k endpoint: ' + name)
        native = Path(job['run']) / 'metrics' / ('native_VAL_' + job['arm']) / 'metrics.json'
        result = read(native)
        if result.get('split') != 'VAL' or len(result.get('per_case', [])) != 8:
            raise ValueError('Old native VAL8 evaluation missing: ' + name)
    return True, 'all four previous endpoints complete including native VAL8'


def environment(source, gpu):
    return dict(os.environ, PYTHONPATH=str(source), CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='4', PYTHONUNBUFFERED='1')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--poll-seconds', type=float, default=30)
    args = ap.parse_args()
    c = read(args.config)
    control = Path(c['control_dir'])
    control.mkdir(parents=True, exist_ok=True)
    lock = control / 'supervisor.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    state = dict(schema='GTA-AVT-gdf-joint-handoff-1', status='waiting_old_controls',
                 supervisor=dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21]),
                 started_unix=time.time(), stages=[], old_runs_modified=False)
    path = control / 'state.json'
    children = []
    def launch(name, argv, gpu):
        log_path = control / (name + '.log')
        with log_path.open('xb') as log:
            p = subprocess.Popen(argv, cwd=c['source_dir'], env=environment(c['source_dir'], gpu),
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        item = dict(name=name, pid=p.pid, start_ticks=Path('/proc/%d/stat' % p.pid).read_text().split()[21],
                    gpu=gpu, log=str(log_path), status='running', started_unix=time.time())
        state['stages'].append(item)
        write(path, state)
        return p, item
    def finish(p, item):
        code = p.wait()
        item.update(exit_code=code, status='complete' if code == 0 else 'failed', finished_unix=time.time())
        write(path, state)
        if code:
            raise RuntimeError('Revision job failed without retry: ' + item['name'])
    try:
        while True:
            ready, reason = barrier(c)
            state.update(wait_reason=reason, observed_unix=time.time())
            write(path, state)
            if ready:
                break
            time.sleep(args.poll_seconds)
        state.update(status='preparing_revision', barrier_passed_unix=time.time())
        if c['authorized_arms'] != ['AR', 'AR_GDF']:
            raise ValueError('Only the authorized matched pair may run')
        if sha(c['parent_checkpoint']) != c['parent_sha256']:
            raise ValueError('Clean condition checkpoint identity changed')
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True)
        if not {'0', '1'}.issubset(set(inventory.strip().splitlines())):
            raise RuntimeError('Two actual GPUs required for the paired revision')
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        if occupied.strip():
            raise RuntimeError('GPUs still occupied; will not interrupt existing work')
        source = Path(c['source_dir'])
        manifest = read(source / 'SOURCE_MANIFEST.json')
        for relative, expected in manifest['files'].items():
            expected = expected['sha256'] if isinstance(expected, dict) else expected
            if sha(source / relative) != expected:
                raise ValueError('Released source changed: ' + relative)
        py = c['python']
        # CPU preparation only writes a new Gaussian overlay, never old cache.
        p, item = launch('bounded_target_overlay', [py, '-B', str(source / 'scripts' / 'prepare_gdf_overlay.py'),
            '--source-manifest', c['legacy_manifest'], '--output', c['overlay_dir']], '')
        finish(p, item)
        # Actual128/B2 mixed-precision QA is deferred until the old GPU work finishes.
        p, item = launch('cuda_qa', [py, '-B', str(source / 'scripts' / 'smoke_gdf_joint_cuda.py'),
            '--output', str(control / 'cuda_qa.json')], '0')
        finish(p, item)
        p, item = launch('shared_head_and_calibration', [py, '-B', '-m', 'gta_avt.gdf_joint',
            'prepare', '--config', c['method_plan']], '0')
        finish(p, item)
        state['status'] = 'paired_training'
        for gpu, arm in enumerate(c['authorized_arms']):
            p, item = launch(arm, [py, '-B', '-m', 'gta_avt.gdf_joint', 'run',
                                  '--config', c['method_plan'], '--arm', arm], str(gpu))
            children.append((p, item))
        for p, item in children:
            finish(p, item)
        state.update(status='complete', finished_unix=time.time())
        write(path, state)
    except Exception as exc:
        state.update(status='failed', error_type=type(exc).__name__, error=str(exc), finished_unix=time.time())
        write(path, state)
        raise


if __name__ == '__main__':
    main()
