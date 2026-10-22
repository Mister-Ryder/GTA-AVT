"""Finite internal GTA-AVT queue. A phase ends; it never starts another phase.

No SSH, credentials, external methods, final-test evaluation or hyperparameter
search. Invoke formal explicitly only after reviewing the pilot evidence.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time


SEED = 20261022
FROZEN_SPLIT_SHA256 = 'df9647da8b331260fb9f66b294eced01180c1eaf812de27b5148b6800a344a3b'
COHORT_PILOT_IDS = ['Dongyang_D2', 'KiTS_K13', 'Rider_R10']
EXPECTED = {
    ('main', 'pilot'): [('C_concat', 'faithful_concat'), ('A_soft', 'faithful_soft'), ('A_soft_constant', 'faithful_soft_constant')],
    ('ablations', 'pilot'): [('A_soft_platform_bridge', 'faithful_soft'), ('A_soft_scale', 'faithful_soft_scale')],
    ('main', 'formal'): [('C_concat', 'faithful_concat'), ('A_soft', 'faithful_soft'), ('A_soft_constant', 'faithful_soft_constant')],
    ('ablations', 'formal'): [('A_soft_platform_bridge', 'faithful_soft'), ('A_soft_scale', 'faithful_soft_scale'),
                              ('A_soft_structure', 'faithful_soft_structure'), ('A_hard_anchor', 'faithful_joint')],
}
ACTIVE_CHILD = None


def stop_owned_child():
    global ACTIVE_CHILD
    child = ACTIVE_CHILD
    ACTIVE_CHILD = None
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def interrupted(signum, frame):
    stop_owned_child()
    raise KeyboardInterrupt('Runner received signal%d; its owned training subprocess was stopped' % signum)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def event(journal, name, **data):
    row = dict(utc=utc(), event=name, **data)
    with journal.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)


def require_storage(path, gib=2.):
    free = shutil.disk_usage(path).free
    if free < gib * 2 ** 30:
        raise RuntimeError('Storage reserve below %.1fGiB at%s:%.2fGiB' % (gib, path, free / 2 ** 30))
    return free


def load_frozen_split(project):
    copied = project / 'configs' / 'bounded_20261022' / 'split_seed20261022.json'
    original = project / 'configs' / 'split_seed20261022.json'
    if digest(copied) != FROZEN_SPLIT_SHA256:
        raise ValueError('Frozen bounded split file byte identity changed')
    if original.exists() and digest(original) != FROZEN_SPLIT_SHA256:
        raise ValueError('Preprocessing split file differs from the frozen bounded split')
    fixed = read(copied)
    if fixed['seed'] != SEED:
        raise ValueError('Frozen split seed changed')
    prefix = {'D': 'Dongyang', 'K': 'KiTS', 'R': 'Rider'}
    return {name: [prefix[case_id[0]] + '_' + case_id for case_id in fixed[name]]
            for name in ('train', 'val', 'test')}


def load_plan(project, role, phase):
    directory = project / 'configs' / 'bounded_20261022'
    plan_path = directory / 'plan.json'
    plan = read(plan_path)
    if plan['seed'] != SEED or plan['external_network_comparisons'] or plan['automatic_final_test']:
        raise ValueError('Bounded plan policy changed unexpectedly')
    for (expected_role, expected_phase), expected_pairs in EXPECTED.items():
        actual = [(row['job_id'], row['method']) for row in plan['queues'][expected_role][expected_phase]]
        if actual != expected_pairs:
            raise ValueError('Another queue escaped the globally finite predeclared allowlist')
    specification = plan['queues'][role][phase]
    pairs = [(row['job_id'], row['method']) for row in specification]
    if pairs != EXPECTED[(role, phase)]:
        raise ValueError('Queue is outside the finite predeclared internal allowlist')
    jobs = []
    shared_recipe = None
    for row in specification:
        config_path = (directory / row['config']).resolve()
        if directory.resolve() not in config_path.parents:
            raise ValueError('Configuration escapes bounded directory')
        config = read(config_path)
        if config['method'] != row['method'] or config['seed'] != SEED or config['stage'] != phase:
            raise ValueError('Method/seed/stage does not match the bounded plan')
        if config['patch_size'] != 96 or config['batch_size'] != 2 or config['width'] != 32:
            raise ValueError('Unapproved patch/batch/backbone configuration')
        if phase == 'pilot':
            if config['max_steps'] != 200 or config['validate_every'] != 200 or config.get('validation_case_ids') != COHORT_PILOT_IDS:
                raise ValueError('Pilot must be200steps with the three fixed source-cohort validation cases')
        else:
            formal_steps = plan['formal_steps']
            if not isinstance(formal_steps, int) or formal_steps < 1000 or formal_steps % 1000 or config['max_steps'] != formal_steps:
                raise ValueError('Formal budget must be a common finite multiple of1000updates')
            if formal_steps != 5000 and not plan.get('budget_revision_reason'):
                raise ValueError('Changing the initial5000budget requires an explicit pilot-based reason in the new plan version')
            if config['validate_every'] != formal_steps // 2 or config['checkpoint_every'] != 500:
                raise ValueError('Formal validation/checkpoint cadence differs from the frozen plan')
            if config.get('validation_case_ids') or config.get('validation_case_limit', 0):
                raise ValueError('Formal validation must use all8cases')
        common = {key: value for key, value in config.items() if key != 'method'}
        if shared_recipe is None:
            shared_recipe = common
        elif common != shared_recipe:
            raise ValueError('Same-phase methods do not share the same training recipe')
        jobs.append(dict(row, config_path=str(config_path), config=config, config_sha256=digest(config_path)))
    # A platform bridge changes hardware/runtime only. It must not silently
    # change learning rate, augmentation, validation, or any other recipe field.
    for other_role in ('main', 'ablations'):
        for row in plan['queues'][other_role][phase]:
            other_path = (directory / row['config']).resolve()
            if directory.resolve() not in other_path.parents:
                raise ValueError('Cross-server configuration escapes bounded directory')
            other = read(other_path)
            if {key: value for key, value in other.items() if key != 'method'} != shared_recipe:
                raise ValueError('Cross-server platform bridge does not share the exact phase recipe')
    return plan, jobs, digest(plan_path)


def wait_ready(project, cache, smoke_path, current_source, required_methods, deadline, journal, status_path):
    last_reason = None
    manifest_path = cache / 'manifest.json'
    while time.monotonic() < deadline:
        reasons = []
        manifest, smoke = None, None
        if manifest_path.exists():
            try:
                manifest = read(manifest_path)
                if not manifest.get('complete') or len(manifest.get('cases', [])) != 56:
                    reasons.append('cache_preparation_%s_of56' % len(manifest.get('cases', [])))
            except (ValueError, OSError):
                reasons.append('cache_manifest_not_ready')
        else:
            reasons.append('cache_manifest_missing')
        if smoke_path.exists():
            try:
                smoke = read(smoke_path)
                if smoke.get('source_sha256') != current_source:
                    reasons.append('smoke_source_mismatch')
                elif smoke.get('state') == 'failed':
                    raise RuntimeError('Current-source faithful smoke reports failure')
                elif smoke.get('state') != 'passed':
                    reasons.append('current_source_smoke_incomplete')
                elif not set(required_methods).issubset(smoke.get('models', {})):
                    reasons.append('smoke_missing_queued_role_methods:' + ','.join(sorted(set(required_methods) - set(smoke.get('models', {})))))
            except (ValueError, OSError):
                reasons.append('smoke_report_not_ready')
        else:
            reasons.append('faithful_smoke_missing')
        if not reasons:
            return manifest_path, manifest, smoke
        reason = ';'.join(reasons)
        if reason != last_reason:
            event(journal, 'waiting', reason=reason)
            write(status_path, dict(state='waiting', reason=reason, utc=utc()))
            last_reason = reason
        time.sleep(15)
    raise TimeoutError('Bounded prerequisites were not ready within the explicit wait budget')


def verify_cache(cache, manifest, current_source, journal, frozen_split):
    started = time.time()
    cases = manifest['cases']
    ids = [case['id'] for case in cases]
    splits = manifest['splits']
    if manifest.get('split_sha256') != FROZEN_SPLIT_SHA256 or splits != frozen_split:
        raise ValueError('Cache patient IDs/order or split SHA differ from the exact predeclared patient split')
    if len(set(ids)) != 56 or [len(splits[name]) for name in ('train', 'val', 'test')] != [36, 8, 12]:
        raise ValueError('Expected the frozen56-case36/8/12 patient partition')
    flattened = sum([splits[name] for name in ('train', 'val', 'test')], [])
    if len(set(flattened)) != 56 or set(flattened) != set(ids):
        raise ValueError('Patient partition does not exactly cover the56cache records')
    if not math.isclose(float(manifest['spacing_mm']), 1., abs_tol=1e-9):
        raise ValueError('This bounded plan requires1.0mm cache')
    expected_preprocess = {name: value for name, value in current_source.items()
                           if name in ('prepare.py', 'geometry.py', 'common.py', 'intensity.py')}
    total_bytes, identities = 0, {}
    for index, case in enumerate(cases, 1):
        directory = Path(case['cache_dir']).resolve()
        if directory.parent != cache.resolve() or directory.name != case['id']:
            raise ValueError('Cache case escapes requested cache root: ' + case['id'])
        train = case['id'] in splits['train']
        if case['geometry_training_only'] != train or case['preprocessing_source_sha256'] != expected_preprocess:
            raise ValueError('Cache geometry scope or preprocessing source mismatch: ' + case['id'])
        hashes = case['cache_file_sha256']
        required = {'image.npy', 'mask.npy', 'geometry.npz'} if train else {'image.npy', 'mask.npy'}
        if set(hashes) != required:
            raise ValueError('Unexpected cached file set: ' + case['id'])
        if read(directory / 'record.json') != case:
            raise ValueError('Cache record and completed manifest disagree: ' + case['id'])
        for filename, expected in hashes.items():
            path = directory / filename
            if not path.is_file() or digest(path) != expected:
                raise ValueError('Actual cache byte hash mismatch: ' + str(path))
            total_bytes += path.stat().st_size
        identities[case['id']] = {name: case[name] for name in ('image_sha256', 'label_sha256', 'cache_file_sha256')}
        if index % 8 == 0 or index == 56:
            event(journal, 'cache_hash_progress', verified_cases=index, verified_bytes=total_bytes)
    return {'state': 'passed', 'verified_utc': utc(), 'cases': 56, 'all_actual_cache_bytes_hashed': True,
            'bytes': total_bytes, 'seconds': time.time() - started, 'canonical_data_sha256': canonical_digest(identities),
            'preprocessing_source_sha256': expected_preprocess}


def verify_run_signature(run, config, manifest_path, manifest, current_source):
    record = read(run / 'config.json')
    expected_val = config.get('validation_case_ids') or manifest['splits']['val']
    required = {'manifest_sha256': digest(manifest_path), 'source_sha256': current_source,
                'recipe': config, 'train_ids': manifest['splits']['train'], 'validation_ids': expected_val,
                'manifest_path': str(manifest_path.resolve()), 'test_used': False}
    for key, expected in required.items():
        if record.get(key) != expected:
            raise ValueError('Existing run/source/config mismatch:' + key + ':' + str(run))
    for relative, expected in current_source.items():
        if digest(run / 'source' / relative) != expected:
            raise ValueError('Frozen run source snapshot changed:' + relative)
    return record


def audit_checkpoint(project, python, path, record, expected_step=None):
    # Use the explicitly selected training environment, not whichever interpreter
    # happened to invoke this orchestration script.
    program = """import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from common import load_trusted_checkpoint,load_json
c=load_trusted_checkpoint(sys.argv[2],'cpu')
r=load_json(sys.argv[3])
assert c['signature']==r,'checkpoint signature differs from frozen run'
assert 'model' in c and 'optimizer' in c and 'scaler' in c,'resume state missing'
assert all(k in c for k in ('python_rng','numpy_rng','torch_rng','cuda_rng')),'RNG state missing'
print(json.dumps({'step':c['step'],'best_val_dice':c['best_val_dice'],'training_seconds':c['training_seconds']}))
"""
    result = subprocess.run([python, '-c', program, str(project / 'code'), str(path), str(path.parent / 'config.json')],
                            capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise ValueError('Checkpoint audit failed:%s\n%s' % (path, result.stderr[-3000:]))
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    if expected_step is not None and summary['step'] != expected_step:
        raise ValueError('Checkpoint step differs from expected completed budget')
    return summary


def verified_completion(project, python, run, config, manifest_path, manifest, source):
    record = verify_run_signature(run, config, manifest_path, manifest, source)
    completion = read(run / 'complete.json')
    if completion['state'] != 'complete' or completion['steps'] != config['max_steps'] or completion.get('test_used') is not False:
        raise ValueError('Completion marker is inconsistent')
    latest = audit_checkpoint(project, python, run / 'latest.pt', record, config['max_steps'])
    best = audit_checkpoint(project, python, run / 'best.pt', record)
    if not 0 < best['step'] <= config['max_steps']:
        raise ValueError('Best checkpoint has an invalid update index')
    expected_val = config.get('validation_case_ids') or manifest['splits']['val']
    for step in range(config['validate_every'], config['max_steps'] + 1, config['validate_every']):
        directory = run / 'validation' / ('step_%06d' % step)
        summary = read(directory / 'summary.json')
        if summary['cases'] != len(expected_val):
            raise ValueError('Completed validation case count differs from frozen plan')
        for case_id in expected_val:
            metric = read(directory / case_id / 'metrics.json')
            if metric.get('case_id') != case_id or metric.get('fixed_threshold') != .5 or metric.get('condition_mode') != 'predicted':
                raise ValueError('Validation case identity/threshold/condition differs from protocol')
            if metric.get('evaluation_grid') != 'original CT' or metric.get('postprocessing') != 'none':
                raise ValueError('Validation metric grid or processing differs from protocol')
    attestation = {'state': 'verified_complete', 'verified_utc': utc(), 'steps': latest['step'],
                   'latest_sha256': digest(run / 'latest.pt'), 'best_sha256': digest(run / 'best.pt'),
                   'config_sha256': digest(run / 'config.json'), 'complete_sha256': digest(run / 'complete.json'),
                   'source_sha256': source, 'test_used': False}
    write(run / 'bounded_completion_verified.json', attestation)
    return attestation


def timing_estimate(run, config, formal_steps=5000):
    values = []
    with (run / 'training.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                values.append(json.loads(line))
    # Resumes may repeat updates after the last checkpoint; retain the latest
    # observation of each step rather than calling those independent samples.
    latest = {row['step']: row for row in values}
    steady = [row['seconds'] for step, row in sorted(latest.items()) if step > 10]
    if not steady:
        steady = [row['seconds'] for row in latest.values()]
    median = statistics.median(steady)
    val_times = [read(path)['seconds'] for path in (run / 'validation').glob('step_*/*/metrics.json')]
    mean_case = statistics.mean(val_times) if val_times else None
    estimate = {'observed_updates': len(latest), 'median_training_step_seconds_excluding_first10': median,
                'observed_validation_cases': len(val_times), 'mean_native_validation_case_seconds': mean_case,
                'formal_planned_updates': formal_steps,
                'formal_update_training_hours_estimate': formal_steps * median / 3600.,
                'formal_all8_twice_validation_hours_estimate': 16 * mean_case / 3600. if mean_case is not None else None,
                'caveat': 'Pilot-based estimate; case sizes and checkpoint overhead vary; no new runs are started from this estimate'}
    estimate['formal_run_training_plus_validation_hours_estimate'] = (estimate['formal_update_training_hours_estimate'] +
        estimate['formal_all8_twice_validation_hours_estimate']) if mean_case is not None else None
    return estimate


def main():
    global ACTIVE_CHILD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--server-role', choices=('main', 'ablations'), required=True)
    parser.add_argument('--phase', choices=('pilot', 'formal'), required=True)
    parser.add_argument('--smoke-report', help='Defaults to PROJECT/results/faithful_smoke.json')
    parser.add_argument('--wait-hours', type=float, default=6.)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    project, cache = Path(args.project_root).resolve(), Path(args.cache_root).resolve()
    plan, jobs, plan_hash = load_plan(project, args.server_role, args.phase)
    frozen_split = load_frozen_split(project)
    run_root = project / 'runs' / 'bounded_20261022' / args.phase / args.server_role
    if args.plan_only:
        print(json.dumps({'phase': args.phase, 'server_role': args.server_role, 'seed': SEED,
              'jobs': [{key: value for key, value in job.items() if key != 'config'} for job in jobs],
              'run_root': str(run_root), 'formal_requires_separate_explicit_invocation': True,
              'automatic_test': False, 'automatic_next_phase': False, 'plan_sha256': plan_hash}, indent=2))
        return
    run_root.mkdir(parents=True, exist_ok=True)
    journal, status_path = run_root / 'journal.jsonl', run_root / 'status.json'
    lock_path = run_root / 'runner.lock'
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError('Runner lock exists; inspect its PID and journal before removing a stale owned lock:' + str(lock_path))
    with os.fdopen(descriptor, 'w') as lock:
        json.dump({'pid': os.getpid(), 'utc': utc(), 'phase': args.phase, 'role': args.server_role}, lock)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        sys.path.insert(0, str(project / 'code'))
        from common import source_hashes
        source = source_hashes()
        event(journal, 'phase_started', phase=args.phase, role=args.server_role, plan_sha256=plan_hash,
              jobs=[job['job_id'] for job in jobs], automatic_test=False, automatic_next_phase=False)
        manifest_path, manifest, smoke = wait_ready(project, cache,
            Path(args.smoke_report).resolve() if args.smoke_report else project / 'results' / 'faithful_smoke.json',
            source, [job['method'] for job in jobs], time.monotonic() + args.wait_hours * 3600., journal, status_path)
        require_storage(run_root)
        require_storage(cache)
        cache_check = verify_cache(cache, manifest, source, journal, frozen_split)
        manifest_hash = digest(manifest_path)
        write(run_root / 'cache_all_bytes_verified.json', dict(cache_check, manifest_sha256=manifest_hash))
        completed = []
        for job in jobs:
            if source_hashes() != source:
                raise ValueError('Frozen model source changed while this queue was active')
            if digest(manifest_path) != manifest_hash:
                raise ValueError('Frozen cache manifest changed after byte verification')
            if digest(project / 'configs' / 'bounded_20261022' / 'plan.json') != plan_hash:
                raise ValueError('Bounded plan changed while this phase was active')
            if digest(job['config_path']) != job['config_sha256']:
                raise ValueError('Bounded job configuration changed while queue active')
            require_storage(run_root)
            run = run_root / (job['job_id'] + '_seed%d' % SEED)
            command = [args.python, '-u', str(project / 'code' / 'train.py'), '--config', job['config_path'],
                       '--manifest', str(manifest_path), '--output', str(run)]
            if (run / 'complete.json').exists():
                verified_completion(project, args.python, run, job['config'], manifest_path, manifest, source)
                event(journal, 'verified_completed_run_skipped', job=job['job_id'], run=str(run))
            else:
                if run.exists() and any(run.iterdir()):
                    verify_run_signature(run, job['config'], manifest_path, manifest, source)
                    if not (run / 'latest.pt').exists():
                        raise ValueError('Interrupted run lacks a resumable checkpoint; refusing automatic overwrite:' + str(run))
                    checkpoint = audit_checkpoint(project, args.python, run / 'latest.pt', read(run / 'config.json'))
                    if checkpoint['step'] >= job['config']['max_steps']:
                        raise ValueError('Latest checkpoint reached budget without verified completion; inspect instead of rerunning')
                    command.append('--resume')
                    event(journal, 'resume_verified', job=job['job_id'], checkpoint_step=checkpoint['step'])
                run.mkdir(parents=True, exist_ok=True)
                started = time.time()
                log_path = run / 'subprocess.log'
                event(journal, 'job_started', job=job['job_id'], method=job['method'], run=str(run), log=str(log_path),
                      resume='--resume' in command)
                write(status_path, {'state': 'training', 'utc': utc(), 'job': job['job_id'], 'run': str(run), 'completed': completed})
                with log_path.open('ab') as log:
                    process = subprocess.Popen(command, cwd=project, stdout=log, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL, start_new_session=True)
                    ACTIVE_CHILD = process
                    write(run_root / 'active_process.json', {'state': 'running', 'pid': process.pid, 'job': job['job_id'], 'run': str(run), 'utc': utc()})
                    returncode = process.wait()
                    ACTIVE_CHILD = None
                write(run_root / 'active_process.json', {'state': 'finished', 'pid': process.pid, 'job': job['job_id'],
                      'run': str(run), 'utc': utc(), 'returncode': returncode})
                if returncode:
                    with log_path.open('rb') as error_log:
                        error_log.seek(max(0, log_path.stat().st_size - 4096))
                        tail = error_log.read().decode('utf-8', errors='replace')
                    event(journal, 'job_failed', job=job['job_id'], returncode=returncode, log=str(log_path),
                          seconds=time.time() - started, error_tail=tail)
                    raise RuntimeError('Training subprocess failed; queue stopped without retry: ' + job['job_id'])
                verified_completion(project, args.python, run, job['config'], manifest_path, manifest, source)
                event(journal, 'job_complete_verified', job=job['job_id'], seconds=time.time() - started)
            estimate = timing_estimate(run, job['config'], plan['formal_steps'])
            completed.append({'job': job['job_id'], 'method': job['method'], 'run': str(run), 'timing': estimate})
            write(run_root / 'summary.json', {'state': 'in_progress', 'phase': args.phase, 'role': args.server_role,
                  'seed': SEED, 'completed': completed, 'test_used': False, 'canonical_data_sha256': cache_check['canonical_data_sha256']})
            event(journal, 'timing_estimate', job=job['job_id'], **estimate)
        summary = {'state': 'phase_complete', 'phase': args.phase, 'role': args.server_role, 'utc': utc(),
                   'seed': SEED, 'completed': completed, 'test_used': False, 'automatic_next_phase': False,
                   'canonical_data_sha256': cache_check['canonical_data_sha256'], 'plan_sha256': plan_hash,
                   'source_sha256': source}
        write(run_root / 'summary.json', summary)
        write(status_path, summary)
        event(journal, 'phase_complete', phase=args.phase, jobs=len(completed), test_used=False, automatic_next_phase=False)
    except BaseException as exc:
        event(journal, 'runner_stopped', error_type=type(exc).__name__, reason=str(exc))
        write(status_path, {'state': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                           'utc': utc(), 'error_type': type(exc).__name__, 'reason': str(exc),
                           'test_used': False, 'automatic_retry': False})
        raise
    finally:
        try:
            # Also stop an owned child if orchestration fails after Popen but
            # before wait/logging completes; failure must never orphan training.
            stop_owned_child()
        finally:
            lock_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
