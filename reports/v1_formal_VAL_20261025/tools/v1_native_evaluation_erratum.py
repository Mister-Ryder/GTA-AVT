"""Bounded evaluation-only NIfTI header precision erratum for fixed formal v1.

This is NOT a training resume. Original source, checkpoints, pipeline status and
predictions are read-only. New inference uses the original frozen infer function.
Evaluation accepts only the original CT grid or its exact float32 serialization,
after all eight original CT/label grids have been certified before any scores.

Run on a terminal host with its existing training Python:
  python -B v1_native_evaluation_erratum.py --port 34153
An existing external output directory is always an error; there is no resume.
The complete evaluation.json, grid_certificate.json and metrics/ form a sidecar
to the collected original run. Do not edit/replace its failed pipeline.json.
The objective exporter must explicitly validate and consume this sidecar.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

SOURCE = Path('/root/GTA_AVT_v1_formal_20261024_r1')
RUN = Path('/root/GTA_AVT_v1_runs_20261024_r1')
CONTROL = Path('/root/GTA_AVT_v1_control_20261024_r1')
OUTPUT = Path('/root/GTA_AVT_v1_evaluation_erratum_20261024_r1')
SOURCE_COMMIT = 'e17e91465022fd38f8ca1fb82a73b3c504f050d0'
SOURCE_MANIFEST_SHA = 'c709eafcb43d8a1fce02d3678aeb9f0785d0c32d5e56d03672733e0cf211b722'
DATA_MANIFEST_SHA = 'cdf74b47c3cde32dda039ea1f2072e57c6b7362829380b1f97805106dd8c33fa'
CASES_SHA = '410d641a261efa0e568efa2298f7e113d646f24c92632fe41c74a6542a2e4e17'
SEEDS = {17552: 20261030, 34153: 20261031, 38773: 20261032}
ARMS = ('C0', 'A_only', 'AR', 'AR_GDF')
STAGES = ('attribute', 'warmup') + ARMS
CASE_IDS = ('Dongyang_D2', 'Dongyang_D5', 'KiTS_K13', 'KiTS_K15',
            'KiTS_K18', 'KiTS_K8', 'Rider_R10', 'Rider_R9')
FIELDS = ('spacing', 'origin', 'direction')
SCHEMA = 'gta-avt-v1-evaluation-erratum-1'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def header(image):
    """Accept either a SimpleITK image or a header-only ImageFileReader."""
    result = dict(size=list(image.GetSize()), spacing=list(image.GetSpacing()),
                  origin=list(image.GetOrigin()), direction=list(image.GetDirection()),
                  dimension=image.GetDimension(), components=image.GetNumberOfComponents())
    return validate_header(result)


def image_header(image):
    # SimpleITK Image uses a different scalar-components accessor than its reader.
    result = dict(size=list(image.GetSize()), spacing=list(image.GetSpacing()),
                  origin=list(image.GetOrigin()), direction=list(image.GetDirection()),
                  dimension=image.GetDimension(), components=image.GetNumberOfComponentsPerPixel())
    return validate_header(result)


def read_header(path):
    import SimpleITK as sitk
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return header(reader)


def validate_header(value):
    require(value['dimension'] == 3 and value['components'] == 1, 'Scalar 3D grid required')
    require(len(value['size']) == 3 and all(type(n) is int and n > 0 for n in value['size']),
            'Positive integer grid size required')
    for key, length in (('spacing', 3), ('origin', 3), ('direction', 9)):
        require(len(value[key]) == length and np.isfinite(value[key]).all(), 'Invalid ' + key)
    require((np.asarray(value['spacing']) > 0).all(), 'Positive physical spacing required')
    return value


def corners(value):
    indices = np.asarray([(x, y, z) for x in (0, value['size'][0] - 1)
                          for y in (0, value['size'][1] - 1)
                          for z in (0, value['size'][2] - 1)], dtype=np.float64)
    direction = np.asarray(value['direction']).reshape(3, 3)
    return indices * np.asarray(value['spacing']) @ direction.T + np.asarray(value['origin'])


def differences(first, second):
    distance = float(np.max(np.linalg.norm(corners(first) - corners(second), axis=1)))
    return dict(max_abs_header_difference={key: float(np.max(np.abs(
                    np.asarray(first[key]) - np.asarray(second[key])))) for key in FIELDS},
                max_corner_distance_mm=distance, corner_mm=distance)


def certify_source_pair(ct, label):
    validate_header(ct); validate_header(label)
    require(ct['size'] == label['size'], 'Original CT/label sizes differ')
    delta = differences(ct, label)
    require(all(value <= 1e-10 for value in delta['max_abs_header_difference'].values()),
            'Original CT/label physical grids differ beyond 1e-10')
    require(delta['corner_mm'] <= 1e-8, 'Original CT/label corner grids differ beyond 1e-8 mm')
    return dict(status='certified', size_equal=True, image=ct, label=label, **delta)


def certify_prediction(ct, prediction):
    validate_header(ct); validate_header(prediction)
    require(prediction['size'] == ct['size'], 'Prediction size differs from original CT')
    modes = {}
    quantized_equal = exact_equal = True
    for key in FIELDS:
        original = np.asarray(ct[key], dtype=np.float64)
        quantized = original.astype(np.float32).astype(np.float64)
        actual = np.asarray(prediction[key], dtype=np.float64)
        q_equal = np.array_equal(actual, quantized)
        x_equal = np.array_equal(actual, original)
        require(q_equal or x_equal, 'Prediction ' + key + ' is neither exact CT nor exact float32 CT')
        modes[key] = 'float32_CT' if q_equal else 'exact_CT'
        quantized_equal = quantized_equal and q_equal
        exact_equal = exact_equal and x_equal
    return dict(status='certified', size_equal=True, prediction_header=prediction,
                field_modes=modes, float32_ctgrid_equal=bool(quantized_equal),
                ct_grid_exact_equal=bool(exact_equal), **differences(ct, prediction))


def verify_inference_report(report, *, case_id, checkpoint_sha, arm, prediction_path):
    expected_arm = 'AR' if arm == 'AR_GDF' else arm
    expected = dict(case_id=case_id, checkpoint_sha256=checkpoint_sha,
                    checkpoint_successful_updates=5000, arm=expected_arm,
                    condition_mode='predicted', threshold=.5, patch_size=128, overlap=.5)
    require(all(report.get(key) == value for key, value in expected.items()),
            'Inference metadata identity or fixed configuration mismatch: ' + case_id)
    require(Path(report.get('prediction', '')).resolve() == Path(prediction_path).resolve(),
            'Inference metadata prediction path mismatch: ' + case_id)
    require(report.get('probability') is None, 'Unexpected probability output')
    require(type(report.get('tiles')) is int and report['tiles'] > 0, 'Missing inference tile count')


def verify_source():
    manifest_path = SOURCE / 'SOURCE_MANIFEST.json'
    require(sha(manifest_path) == SOURCE_MANIFEST_SHA, 'Frozen capsule manifest identity changed')
    manifest = read(manifest_path)
    for name, metadata in manifest['files'].items():
        path = (SOURCE / name).resolve()
        require(SOURCE.resolve() in path.parents, 'Unsafe capsule file path')
        require(sha(path) == metadata['sha256'] and path.stat().st_size == metadata['bytes'],
                'Frozen capsule file changed: ' + name)
    return dict(source_commit=SOURCE_COMMIT, source_manifest_sha256=SOURCE_MANIFEST_SHA,
                verified_file_count=len(manifest['files']))


def verify_terminal(port):
    process = read(CONTROL / 'process.json')
    plan = read(RUN / 'plan.json')
    pipeline = read(RUN / 'pipeline.json')
    require(process.get('status') in ('complete', 'failed') and type(process.get('exit_code')) is int,
            'Original formal process is not terminal')
    require(process.get('source_commit') == plan.get('source_commit') == SOURCE_COMMIT,
            'Original source commit mismatch')
    require(process.get('seed') == plan.get('seed') == SEEDS[port], 'Host/seed identity mismatch')
    require(pipeline.get('status') == 'failed' and pipeline.get('error_type') == 'ValueError'
            and str(pipeline.get('error', '')).startswith('Native prediction/label physical grids differ for ')
            and pipeline.get('current') == 'native_VAL_C0', 'Not the bounded original native header failure')
    require(process['status'] == 'failed' and process['exit_code'] != 0, 'Original failure wait status mismatch')
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    require(pipeline.get('plan_sha256') == signature, 'Original pipeline plan signature mismatch')
    require(plan.get('data_manifest_sha256') == process.get('data_manifest_sha256') == DATA_MANIFEST_SHA
            and sha(plan['manifest']) == DATA_MANIFEST_SHA, 'Prepared data manifest identity mismatch')
    require(plan.get('pretrain_steps') == 20000 and plan.get('warmup_steps') == 10000
            and plan.get('adapt_steps') == 5000 and plan.get('alpha_ramp_steps') == 1000
            and plan.get('adapt_lr') == .0001 and plan.get('lambda_gdf') == .1
            and plan.get('device') == 'cuda' and plan.get('num_workers') == 4,
            'Fixed formal training plan changed')
    require(not plan.get('common'), 'Unexpected formal common overrides')
    require(sha(plan['evaluation_cases']) == CASES_SHA, 'Original evaluation cases identity changed')
    cases_obj = read(plan['evaluation_cases'])
    cases = cases_obj['cases'] if isinstance(cases_obj, dict) else cases_obj
    require(tuple(case['id'] for case in cases) == CASE_IDS
            and all(case.get('split') == 'VAL' for case in cases), 'Fixed eight VAL cases changed')
    artifacts = {}
    for stage in STAGES:
        expected_steps = 20000 if stage == 'attribute' else 10000 if stage == 'warmup' else 5000
        summary = read(RUN / stage / 'summary.json')
        config = read(RUN / stage / 'config.json')
        require(stage in pipeline.get('completed', {})
                and summary == pipeline['completed'][stage]
                and summary.get('successful_updates') == expected_steps
                and config.get('max_steps') == expected_steps and config.get('seed') == SEEDS[port],
                'Missing, incomplete or inconsistent original training stage: ' + stage)
        for name in ('config.json', 'summary.json'):
            artifacts[stage + '/' + name] = sha(RUN / stage / name)
    metadata = dict(seed=SEEDS[port], original_plan_sha256=sha(RUN / 'plan.json'),
                    original_plan_signature=signature, original_pipeline_sha256=sha(RUN / 'pipeline.json'),
                    original_process_sha256=sha(CONTROL / 'process.json'),
                    original_pipeline_status=pipeline['status'], original_error_type=pipeline['error_type'],
                    original_error_stage=pipeline['current'], original_failure_reason='native_header_precision_check',
                    original_process_exit_code=process['exit_code'], original_stage_artifacts_sha256=artifacts)
    return plan, cases, metadata


def import_original():
    sys.path.insert(0, str(SOURCE))
    result = {}
    for name in ('inference', 'engine', 'metrics'):
        module = importlib.import_module('gta_avt.' + name)
        require(Path(module.__file__).resolve() == SOURCE / 'gta_avt' / (name + '.py'),
                'Imported module is not the certified frozen source')
        result[name] = module
    return result


def certify_checkpoint(path, arm, seed, load_checkpoint):
    checkpoint = load_checkpoint(path)
    expected_arm = 'AR' if arm == 'AR_GDF' else arm
    config = checkpoint.get('config', {})
    require(checkpoint.get('kind') == 'train' and checkpoint.get('step') == 5000
            and checkpoint.get('progress', {}).get('step') == 5000
            and config.get('stage') == 'adapt' and config.get('max_steps') == 5000
            and config.get('seed') == seed and config.get('arm') == expected_arm
            and config.get('patch_size') == 128 and config.get('batch_size') == 2
            and config.get('amp') is True and config.get('use_gdf') is True
            and config.get('lambda_gdf') == (.1 if arm == 'AR_GDF' else 0.)
            and checkpoint.get('model_config', {}).get('arm') == expected_arm
            and checkpoint.get('model_config', {}).get('use_gdf') is True,
            'Checkpoint is not the final fixed arm endpoint: ' + arm)
    require(Path(config.get('output_dir', '')).resolve() == RUN / arm,
            'Checkpoint source output directory mismatch')
    require(config == read(RUN / arm / 'config.json'), 'Checkpoint/config artifact mismatch')
    info = dict(checkpoint_sha256=sha(path), checkpoint_successful_updates=5000,
                checkpoint_arm=expected_arm)
    del checkpoint
    gc.collect()
    return info


def metric_row(case_id, prediction, label, binary_metrics):
    import SimpleITK as sitk
    label_values = sitk.GetArrayFromImage(label)
    value = sitk.GetArrayFromImage(prediction)
    require(np.isfinite(label_values).all(), 'Nonfinite native label: ' + case_id)
    require(np.isfinite(value).all() and np.isin(value, (0, 1)).all(),
            'Native prediction is not finite binary data: ' + case_id)
    target = label_values > 0
    pred = value >= .5
    row = {'case_id': case_id, 'source': None,
           **binary_metrics(pred, target, centerline=True, spacing_zyx=label.GetSpacing()[::-1])}
    voxel_ml = float(np.prod(label.GetSpacing()) / 1000.)
    row.update(predicted_volume_ml=float(pred.sum()) * voxel_ml,
               target_volume_ml=float(target.sum()) * voxel_ml,
               fp_volume_ml=row['fp_voxels'] * voxel_ml, fn_volume_ml=row['fn_voxels'] * voxel_ml)
    return row, hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_metrics(directory, rows, summarize_cases):
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(scope='native whole-case evaluation', split='unspecified', scores='fractions in [0,1]',
        definitions={
            'centerline': 'Lee skeleton voxel overlap on each native grid; not physical branch length',
            'fine_centerline': 'GT native Lee skeleton sites with native physical EDT radius <=2mm; report supported voxel count; no sites yields null recall',
            'dispersion': 'population SD across defined cases; worst20 means ceil(20% cases)',
            'empty': 'Both masks empty Dice/clDice=1; undefined precision or recall is null'},
        per_case=rows, summary=summarize_cases(rows))
    write(directory / 'metrics.json', report)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (directory / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    return report


def assert_original_unchanged(metadata):
    for path, expected in ((RUN / 'plan.json', metadata['original_plan_sha256']),
                           (RUN / 'pipeline.json', metadata['original_pipeline_sha256']),
                           (CONTROL / 'process.json', metadata['original_process_sha256'])):
        require(sha(path) == expected, 'Original formal identity changed during external evaluation')
    for relative, expected in metadata['original_stage_artifacts_sha256'].items():
        require(sha(RUN / relative) == expected, 'Original training artifact changed during evaluation')


def run(port):
    # Exclusive directory creation is the only dispatch lock. No implicit resume.
    OUTPUT.mkdir(parents=False, exist_ok=False)
    state = dict(schema=SCHEMA, status='running', current='source_certification', seed=SEEDS[port],
                 started_unix=time.time(), erratum_code_sha256=sha(__file__), case_ids=list(CASE_IDS),
                 evaluation_only=True, training_untouched=True, prediction_voxels_modified=False,
                 arms={})
    write(OUTPUT / 'evaluation_state.json', state)
    try:
        state.update(verify_source())
        plan, cases, metadata = verify_terminal(port)
        state.update(metadata)
        state['current'] = 'original_source_grid_certification'
        write(OUTPUT / 'evaluation_state.json', state)
        source_pairs = []
        for case in cases:
            source_pairs.append(dict(case_id=case['id'], **certify_source_pair(
                read_header(case['image']), read_header(case['label']))))
        # Complete all eight certifications BEFORE loading prediction arrays or scores.
        certificate = dict(schema='gta-avt-v1-grid-certificate-1', status='in_progress', seed=SEEDS[port],
                           source_commit=SOURCE_COMMIT, erratum_code_sha256=state['erratum_code_sha256'],
                           source_pairs=source_pairs, arms={})
        write(OUTPUT / 'grid_certificate.json', certificate)
        state['source_grids_certified'] = len(source_pairs)
        state['current'] = 'checkpoint_certification'
        write(OUTPUT / 'evaluation_state.json', state)
        modules = import_original()
        for arm in ARMS:
            state['arms'][arm] = certify_checkpoint(RUN / arm / 'last.pt', arm, SEEDS[port],
                                                    modules['engine'].load_checkpoint)
        write(OUTPUT / 'evaluation_state.json', state)
        import SimpleITK as sitk
        for arm in ARMS:
            state['current'] = 'inference_' + arm
            write(OUTPUT / 'evaluation_state.json', state)
            checkpoint = RUN / arm / 'last.pt'
            endpoint = state['arms'][arm]
            predictions = []
            for case, original in zip(cases, source_pairs):
                existing = RUN / 'native_VAL' / arm / case['id'] / 'inference.json'
                reused = arm == 'C0' and existing.is_file()
                if reused:
                    prediction_path = existing.with_name('prediction.nii.gz')
                    report = read(existing)
                    # Preserve original small metadata bytes in the sidecar, while
                    # prediction remains at its existing read-only original path.
                    info_path = OUTPUT / 'native_VAL' / arm / case['id'] / 'inference.json'
                    info_path.parent.mkdir(parents=True, exist_ok=False)
                    shutil.copyfile(str(existing), str(info_path))
                else:
                    report = modules['inference'].infer(dict(checkpoint=str(checkpoint), image=case['image'],
                        case_id=case['id'], output_dir=str(OUTPUT / 'native_VAL' / arm), device='cuda',
                        patch_size=128, batch_size=2, overlap=.5, reviewed_override=case.get('intensity_override'),
                        amp=True))
                    info_path = OUTPUT / 'native_VAL' / arm / case['id'] / 'inference.json'
                    prediction_path = info_path.with_name('prediction.nii.gz')
                verify_inference_report(report, case_id=case['id'], checkpoint_sha=endpoint['checkpoint_sha256'],
                                        arm=arm, prediction_path=prediction_path)
                proof = certify_prediction(original['image'], read_header(prediction_path))
                predictions.append((case, prediction_path, info_path, reused, proof))
            # Certify every prediction header within the arm before its first metric.
            state['current'] = 'metrics_' + arm
            write(OUTPUT / 'evaluation_state.json', state)
            rows = []; arm_proofs = []
            for case, path, info_path, reused, proof in predictions:
                prediction = sitk.ReadImage(str(path))
                label = sitk.ReadImage(case['label'])
                require(image_header(prediction) == proof['prediction_header'], 'Prediction grid changed after certification')
                row, voxel_sha = metric_row(case['id'], prediction, label, modules['metrics'].binary_metrics)
                rows.append(row)
                arm_proofs.append(dict(case_id=case['id'], prediction_file_sha256=sha(path),
                    prediction_voxel_sha256=voxel_sha, inference_metadata_sha256=sha(info_path),
                    reused_original_prediction=reused, prediction_voxels_modified=False, **proof))
                del prediction, label
                gc.collect()
            directory = OUTPUT / 'metrics' / ('native_VAL_' + arm)
            write_metrics(directory, rows, modules['metrics'].summarize_cases)
            certificate['arms'][arm] = arm_proofs
            endpoint['metrics_sha256'] = sha(directory / 'metrics.json')
            endpoint['metrics_relative_path'] = 'metrics/native_VAL_' + arm + '/metrics.json'
            write(OUTPUT / 'grid_certificate.json', certificate)
            write(OUTPUT / 'evaluation_state.json', state)
            require(sha(checkpoint) == endpoint['checkpoint_sha256'], 'Checkpoint changed during evaluation')
        assert_original_unchanged(metadata)
        verify_source()
        certificate['status'] = 'certified'
        write(OUTPUT / 'grid_certificate.json', certificate)
        state.update(status='complete', current=None, finished_unix=time.time(), original_pipeline_untouched=True,
                     grid_certificate_sha256=sha(OUTPUT / 'grid_certificate.json'))
        write(OUTPUT / 'evaluation_state.json', state)
        write(OUTPUT / 'evaluation.json', state)
        print(json.dumps(dict(status='complete', seed=SEEDS[port], arms=list(ARMS), cases=len(cases))))
        return state
    except BaseException as exc:
        state.update(status='failed', error_type=type(exc).__name__, error=str(exc), failed_unix=time.time())
        write(OUTPUT / 'evaluation_state.json', state)
        (OUTPUT / 'failure_private.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, choices=sorted(SEEDS), required=True)
    args = parser.parse_args()
    run(args.port)
