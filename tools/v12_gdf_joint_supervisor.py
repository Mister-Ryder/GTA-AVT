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


def _acquire_waiting_lock_unserialized(control, resume=False):
    """Resume only a dead, child-free waiting controller; archive its lock/state."""
    control = Path(control)
    lock, state_path = control / 'supervisor.lock', control / 'state.json'
    previous = None
    if lock.exists():
        if not resume:
            raise FileExistsError('A controller lock already exists; explicit waiting resume required')
        previous = read(state_path)
        if previous.get('status') not in ('waiting_old_controls', 'waiting_resource_change') or previous.get('stages') != []:
            raise ValueError('Only a waiting controller without any revision child may resume this way')
        if alive(previous['supervisor']):
            raise RuntimeError('Previous controller identity is still alive; refuse duplicate')
        if lock.read_text().strip() != str(previous['supervisor']['pid']):
            raise ValueError('Lock PID does not match the previous waiting state')
        stamp = str(time.time_ns())
        write(control / ('waiting_state_before_resource_change_' + stamp + '.json'), previous)
        os.replace(str(lock), str(control / ('supervisor.lock.before_resource_change_' + stamp)))
    elif resume or state_path.exists():
        raise RuntimeError('Missing audited waiting lock/state; no implicit fresh restart')
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    return previous


def acquire_waiting_lock(control, resume=False):
    if not resume:
        return _acquire_waiting_lock_unserialized(control, False)
    claim = Path(control) / 'waiting_resume.claim'
    with claim.open('x') as f:
        f.write(str(os.getpid()))
    try:
        return _acquire_waiting_lock_unserialized(control, True)
    finally:
        claim.unlink()


def resource_permission(config, inventory):
    """An operator release and exact requested GPU count are both necessary."""
    gate_path = config.get('resource_gate_file')
    if gate_path is None:
        return False, 'waiting for explicit two-GPU resource gate configuration'
    if not Path(gate_path).is_file():
        return False, 'waiting for explicit resource gate'
    gate = read(gate_path)
    if gate.get('schema') != 'GTA-AVT-resource-gate-1' or gate.get('requested_gpu_count') != 2:
        raise ValueError('This paired handoff requires the explicit two-GPU resource contract')
    if gate.get('allow_revision_start') is not True:
        return False, 'old controls complete; waiting for operator verification after two-GPU restart'
    if sorted(inventory) != ['0', '1']:
        return False, 'waiting for exactly two visible GPUs'
    return True, 'two-GPU resource configuration explicitly released'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--poll-seconds', type=float, default=30)
    ap.add_argument('--resume-waiting', action='store_true')
    args = ap.parse_args()
    c = read(args.config)
    control = Path(c['control_dir'])
    control.mkdir(parents=True, exist_ok=True)
    previous = acquire_waiting_lock(control, args.resume_waiting)
    state = dict(schema='GTA-AVT-gdf-joint-handoff-1', status='waiting_old_controls',
                 supervisor=dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21]),
                 started_unix=time.time(), stages=[], old_runs_modified=False)
    if previous is not None:
        state['previous_waiting_supervisor'] = previous['supervisor']
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
            state.update(status='waiting_old_controls', old_controls_complete=ready,
                         wait_reason=reason, observed_unix=time.time())
            if ready:
                inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True)
                allowed, resource_reason = resource_permission(c, [x.strip() for x in inventory.strip().splitlines()])
                state['resource_gate'] = read(c['resource_gate_file']) if c.get('resource_gate_file') and Path(c['resource_gate_file']).exists() else None
                if not allowed:
                    state.update(status='waiting_resource_change', wait_reason=resource_reason)
                    write(control / 'resource_change_ready.json', dict(
                        old_controls_complete=True, requested_gpu_count=2,
                        revision_children_started=False, observed_unix=time.time()))
                else:
                    state['resource_release_reason'] = resource_reason
                    write(path, state)
                    break
            write(path, state)
            time.sleep(args.poll_seconds)
        state.update(status='preparing_revision', barrier_passed_unix=time.time())
        write(path, state)
        if c['authorized_arms'] != ['AR', 'AR_GDF']:
            raise ValueError('Only the authorized matched pair may run')
        if sha(c['parent_checkpoint']) != c['parent_sha256']:
            raise ValueError('Clean condition checkpoint identity changed')
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True)
        allowed, reason = resource_permission(c, [x.strip() for x in inventory.strip().splitlines()])
        if not allowed:
            raise RuntimeError('Resource release changed before launch: ' + reason)
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
