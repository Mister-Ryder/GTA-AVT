"""Package complete fixed-v1 objective outputs; no remote access or training."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = 'e17e91465022fd38f8ca1fb82a73b3c504f050d0'
ERRATUM_SHA = '93900d714dc3927aa0f416509ba7309d353cc0a86f2761e6b27e6e4d03cbf9ea'
SEEDS = (20261030, 20261031, 20261032)
ARMS = ('C0', 'A_only', 'AR', 'AR_GDF')
STAGES = ('attribute', 'warmup') + ARMS
BUDGETS = dict(attribute=20000, warmup=10000, C0=5000, A_only=5000, AR=5000, AR_GDF=5000)
EXPORT_FILES = ('native_per_case.csv', 'native_summary.csv', 'paired_deltas.csv', 'manifest.json')
TOOL_FILES = ('v1_native_evaluation_erratum.py', 'test_v1_native_evaluation_erratum.py',
              'v1_objective_readout.py', 'v1_objective_readout_erratum_checks.py',
              'v1_package_objective_release.py')
LOSS_KEYS = ('loss', 'availability_bce', 'radius_mse', 'dice_loss', 'segmentation_bce',
             'gdf_mse', 'auxiliary_bce')
HISTORY_KEYS = ('seed', 'stage', 'history_line', 'successful_update', 'attempt', 'epoch', 'alpha', 'lr') + tuple(
    'training_' + key for key in LOSS_KEYS) + ('sampled_val_selection_score',) + tuple(
    'sampled_val_' + key for key in LOSS_KEYS)
SAFE_ID = re.compile(r'^[A-Za-z0-9_.-]+$')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def numeric(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def nullable_float(value):
    if value == '' or value is None:
        return None
    result = float(value)
    require(math.isfinite(result), 'Nonfinite exported CSV value')
    return result


def write_csv(path, rows, keys):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def validate_export(export):
    value = read_json(export / 'manifest.json')
    require(value.get('status') == 'complete' and value.get('problems') == []
            and value.get('input_runs') == 3 and value.get('expected_seed_repeats') == 3
            and value.get('expected_native_val_cases') == 8, 'Complete three-seed native export required')
    require(set(value.get('expected_seeds', [])) == set(SEEDS), 'Unexpected seed set')
    require(len(value.get('case_ids', [])) == len(set(value.get('case_ids', []))) == 8
            and all(SAFE_ID.fullmatch(v) for v in value['case_ids']), 'Eight unique safe case IDs required')
    runs = value.get('runs', [])
    require(len(runs) == 3 and {run.get('seed') for run in runs} == set(SEEDS), 'Three unique runs required')
    for run in runs:
        require(run.get('status') == 'complete_with_evaluation_erratum'
                and run.get('pipeline_status') == 'failed' and run.get('problems') == []
                and run.get('external_evaluation_status') == 'complete', 'Original failure/sidecar status mismatch')
        correction = run.get('evaluation_erratum', {})
        require(correction.get('source_commit') == SOURCE_COMMIT
                and correction.get('erratum_code_sha256') == ERRATUM_SHA,
                'Fixed source or evaluation erratum identity mismatch')
        require(set(run.get('stages', {})) == set(STAGES), 'Six complete training stages required')
        for stage in STAGES:
            record = run['stages'][stage]
            require(record.get('status') == 'complete'
                    and record.get('successful_updates') == BUDGETS[stage]
                    and record.get('expected_successful_updates') == BUDGETS[stage], 'Training budget mismatch')
        ready = run.get('selected_attribute_ready') or {}
        attrs = (ready.get('validation') or {}).get('attributes') or {}
        require(attrs.get('readiness_passed') is True and len(attrs.get('readiness_tests', {})) == 5
                and all(v is True for v in attrs['readiness_tests'].values()), 'Selected attribute prerequisite missing')
        require(set(run.get('native', {})) == set(ARMS), 'Four native arms required')
        for arm in ARMS:
            native = run['native'][arm]
            require(native.get('status') == 'recorded' and native.get('external_evaluation_complete') is True
                    and native.get('pipeline_completed') is False
                    and set(native.get('per_case', {})) == set(value['case_ids']), 'Native completion/case mismatch')
    scopes = [run.get('platform', {}).get('platform_scope') for run in runs]
    require(scopes.count('RTX4080') == 2 and scopes.count('vGPU32GB') == 1, 'Expected hardware blocks missing')
    rows = csv_rows(export / 'native_per_case.csv')
    expected = {(str(seed), arm, case) for seed in SEEDS for arm in ARMS for case in value['case_ids']}
    actual = {(row['seed'], row['arm'], row['case_id']) for row in rows}
    require(len(rows) == 96 and len(actual) == 96 and actual == expected
            and all(row['status'] == 'recorded' for row in rows), 'Complete 3 x 4 x 8 native CSV required')
    for row in rows:
        for key in ('dice', 'cldice'):
            metric = nullable_float(row[key])
            require(metric is not None and 0 <= metric <= 1, 'Defined primary score required')
    return value


def across_seed_summary(export):
    groups = {}
    for row in csv_rows(export / 'native_summary.csv'):
        require(row['arm'] in ARMS and row['seed'] in {str(s) for s in SEEDS}
                and SAFE_ID.fullmatch(row['metric']), 'Unexpected summary identity')
        groups.setdefault((row['arm'], row['metric']), {})[int(row['seed'])] = row
    result = []
    for (arm, metric), data in sorted(groups.items(), key=lambda x: (ARMS.index(x[0][0]), x[0][1])):
        require(set(data) == set(SEEDS), 'Missing seed in metric summary')
        means = [nullable_float(data[seed]['mean']) for seed in SEEDS]
        defined = [v for v in means if v is not None]
        def average(key):
            values = [nullable_float(data[seed][key]) for seed in SEEDS]
            values = [v for v in values if v is not None]
            return statistics.fmean(values) if values else None
        result.append(dict(arm=arm, metric=metric, expected_seeds=3, defined_seeds=len(defined),
            mean_of_seed_means=statistics.fmean(defined) if defined else None,
            seed_mean_std_population=statistics.pstdev(defined) if defined else None,
            mean_within_case_std_population=average('std_population'), mean_worst20_mean=average('worst20_mean'),
            minimum_seed_mean=min(defined) if defined else None, maximum_seed_mean=max(defined) if defined else None,
            minimum_defined_cases=min(int(data[seed]['defined_cases']) for seed in SEEDS)))
    require({row['arm'] for row in result} == set(ARMS), 'Four summary arms required')
    return result


def history_row(item, seed, stage, line):
    step = item.get('successful_update')
    require(type(step) is int and 1 <= step <= BUDGETS[stage], 'History successful update outside fixed budget')
    result = dict(seed=seed, stage=stage, history_line=line, successful_update=step)
    for key in ('attempt', 'epoch', 'alpha', 'lr'):
        result[key] = numeric(item.get(key))
    for key in LOSS_KEYS:
        result['training_' + key] = numeric((item.get('losses') or {}).get(key))
    validation = item.get('validation') or {}
    result['sampled_val_selection_score'] = numeric(validation.get('selection_score'))
    for key in LOSS_KEYS:
        result['sampled_val_' + key] = numeric((validation.get('losses') or {}).get(key))
    return result


def validate_histories(inputs, manifest):
    by_seed = {}
    for root in inputs:
        seed = read_json(root / 'plan.json').get('seed')
        require(seed in SEEDS and seed not in by_seed, 'Original histories must bind each seed once')
        by_seed[seed] = root
    require(set(by_seed) == set(SEEDS), 'All three original history inputs required')
    runs = {run['seed']: run for run in manifest['runs']}
    records = []
    for seed in SEEDS:
        root = by_seed[seed]
        for stage in STAGES:
            summary_path = root / stage / 'summary.json'
            summary = read_json(summary_path)
            require(summary.get('successful_updates') == BUDGETS[stage], 'Original history summary incomplete')
            expected_sha = runs[seed]['provenance_sha256'].get(stage + '/summary.json')
            require(expected_sha == digest(summary_path), 'Original training summary identity mismatch')
            path = root / stage / 'history.jsonl'
            seen = set(); count = 0; last = None
            with path.open(encoding='utf-8-sig') as stream:
                for count, text in enumerate(stream, 1):
                    item = json.loads(text)
                    row = history_row(item, seed, stage, count)
                    last = row['successful_update']; seen.add(last)
            require(last == BUDGETS[stage] and seen == set(range(1, BUDGETS[stage] + 1)), 'Training history endpoint or coverage incomplete')
            records.append(dict(seed=seed, stage=stage, path=path, sha256=digest(path),
                                observations=count, replayed_updates=count - len(seen)))
    return records


def history_rows(records):
    for record in records:
        with record['path'].open(encoding='utf-8-sig') as stream:
            for line, text in enumerate(stream, 1):
                yield history_row(json.loads(text), record['seed'], record['stage'], line)
        require(digest(record['path']) == record['sha256'], 'Original history changed while packaging')


def readme(manifest, across):
    selected = {(row['arm'], row['metric']): row for row in across}
    def score(arm, metric):
        row = selected[(arm, metric)]
        if row['defined_seeds'] != 3:
            return 'undefined'
        return '{:.4f} ± {:.4f}'.format(100 * row['mean_of_seed_means'], 100 * row['seed_mean_std_population'])
    lines = ['# GTA-AVT v1 fixed formal VAL results', '',
        'Training source commit: `' + SOURCE_COMMIT + '`.', '',
        'Three independent initializations (seeds 20261030, 20261031, 20261032), four matched arms, '
        'eight native VAL cases per arm: 12 trained endpoints and 96 case measurements. '
        'No new TEST evaluation is included. Case identifiers do not establish patient independence.', '',
        'Each seed used 20,000 successful attribute-predictor updates, a frozen predictor, '
        '10,000 common segmentation warmup updates, and 5,000 updates per arm. '
        'Each arm independently loaded its own seed\'s final common warmup with an optimizer reset. '
        'All five fixed attribute prerequisites passed for each selected predictor. '
        'A finite update budget is not a convergence claim.', '',
        '| Arm | Consumed conditions | GDF loss weight |',
        '| --- | --- | --- |',
        '| C0 | Zero condition inputs; learned static biases allowed | 0 |',
        '| A_only | Predicted operational applicability A | 0 |',
        '| AR | Predicted applicability A and local log-radius R | 0 |',
        '| AR_GDF | Same predicted A and R | 0.1 |', '',
        'All arms share pretrained predictor features, backbone, guidance path and adapter capacity. '
        'These contrasts measure condition consumption and the Gaussian auxiliary within that shared design; '
        'there is no pure backbone-only comparator in this release.', '',
        '## Objective native VAL measurements', '',
        'Values below are percentages: equal-weight mean of the three within-seed eight-case means ± '
        'population SD of those three means. Two runs used RTX 4080 and one used vGPU-32GB; '
        'the between-run SD is descriptive across these hardware blocks, not three homogeneous-device repeats.', '',
        '| Arm | Dice (%) | clDice (%) | Fine centerline recall (%) |',
        '| --- | ---: | ---: | ---: |']
    for arm in ARMS:
        lines.append('| ' + arm + ' | ' + ' | '.join(score(arm, m) for m in ('dice', 'cldice', 'fine_centerline_recall')) + ' |')
    lines += ['',
        'The CSV files retain every positive, zero and negative observed difference. '
        '`native_summary.csv` records each seed\'s case population SD, extrema and worst-20% mean; '
        'for eight defined cases, the latter averages two cases. `paired_deltas.csv` uses to minus from '
        'on matched seed/case pairs; multiply score deltas by 100 for percentage points. '
        'Undefined metrics remain blank/null and are not replaced by zero.', '',
        'Centerline metrics use native-grid Lee skeleton voxel overlap, not physical branch length. '
        'Fine centerline recall uses reference skeleton sites with native physical EDT radius ≤2 mm. '
        'FP/FN volumes are in mL. `across_seed_summary.csv` reports the mean of available seed means, '
        'their population SD, the mean within-seed case SD and mean worst-20% value; '
        'defined seed and case counts accompany every metric.', '',
        '## Evaluation engineering erratum', '',
        'The original three pipelines completed all six training stages and then failed the native C0 '
        'prediction/label header check. The original failed statuses remain recorded in `manifest.json`. '
        'The check used an absolute header-component tolerance of 0.00001 (origin and spacing in mm); NIfTI float32 serialization of original '
        'CT geometry exceeded that tolerance for some fields. An external evaluator accepted only exact '
        'CT header values or their exact float32 conversions after verifying all eight original CT/label grids. '
        'It reused certified original C0 predictions and called the unchanged original inference function '
        'for the remaining arms. It applied the original metric functions to the unchanged prediction arrays '
        'and original labels with the original label spacing and 0.5 threshold. No resampling, prediction '
        'voxel editing, training/model changes or score-based setting selection was performed.', '',
        'External evaluator SHA256: `' + ERRATUM_SHA + '`. Its exact code and focused tests are in `tools/`; '
        'the immutable training capsule is identified by the source commit above. '
        'Each run explicitly has `pipeline_status=failed`, `status=complete_with_evaluation_erratum` '
        'and `external_evaluation_status=complete`.', '',
        '## Files', '',
        '- `native_per_case.csv`: all 96 native case measurements.',
        '- `native_summary.csv`: per-seed, per-arm case summaries for all exported metrics.',
        '- `paired_deltas.csv`: case and seed-mean paired contrasts.',
        '- `across_seed_summary.csv`: descriptive summaries across the three seed means.',
        '- `training_history.csv`: numeric update history and explicitly named sampled-VAL losses; these are not native metrics.',
        '- `manifest.json`: fixed recipes, selected attribute readiness, training lineage, original failures and external evaluation provenance.',
        '- `package_manifest.json`: packaged file hashes and source history hashes.',
        '- `tools/`: exact evaluation, export, packaging and validation code.', '']
    return '\n'.join(lines)


def check_public_data(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    for marker in ('/autodl-fs/', '/dev/shm/', '/root/', 'E:\\', 'C:\\', 'root@'):
        require(marker not in text, 'Private connection or filesystem text in report data')
    require(not re.search(r'ssh\s+-p\s+\d', text), 'SSH connection text in report data')


def package(export, inputs, output):
    export, output = Path(export), Path(output)
    require(not output.exists(), 'Fresh output directory required')
    for name in EXPORT_FILES:
        require((export / name).is_file(), 'Missing objective export file')
        check_public_data(export / name)
    manifest = validate_export(export)
    across = across_seed_summary(export)
    histories = validate_histories([Path(v) for v in inputs], manifest)
    require(digest(ROOT / 'ops' / TOOL_FILES[0]) == ERRATUM_SHA, 'Evaluation worker changed')
    for name in TOOL_FILES:
        require((ROOT / 'ops' / name).is_file(), 'Required reproducibility tool missing')
    destination = output / 'reports' / 'v1_formal_VAL_20261025'
    destination.mkdir(parents=True, exist_ok=False)
    for name in EXPORT_FILES:
        shutil.copyfile(export / name, destination / name)
    write_csv(destination / 'across_seed_summary.csv', across, list(across[0]))
    write_csv(destination / 'training_history.csv', history_rows(histories), HISTORY_KEYS)
    (destination / 'README.md').write_text(readme(manifest, across), encoding='utf-8')
    tools = destination / 'tools'; tools.mkdir()
    for name in TOOL_FILES:
        shutil.copyfile(ROOT / 'ops' / name, tools / name)
    for path in destination.iterdir():
        if path.is_file():
            check_public_data(path)
    files = {str(path.relative_to(destination)).replace('\\', '/'): dict(sha256=digest(path), bytes=path.stat().st_size)
             for path in sorted(destination.rglob('*')) if path.is_file()}
    receipt = dict(schema='gta-avt-v1-objective-package-1', status='complete',
        created_utc=datetime.now(timezone.utc).isoformat(), training_source_commit=SOURCE_COMMIT,
        evaluation_erratum_code_sha256=ERRATUM_SHA, source_export_manifest_sha256=digest(export / 'manifest.json'),
        seeds=list(SEEDS), native_cases=8, trained_arm_endpoints=12, native_case_measurements=96,
        original_pipeline_status='failed', external_evaluation_status='complete',
        hardware_blocks=dict(RTX4080=2, vGPU32GB=1),
        history=[{key: value for key, value in record.items() if key != 'path'} for record in histories],
        files=files)
    (destination / 'package_manifest.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    check_public_data(destination / 'package_manifest.json')
    return dict(status='complete', output=str(destination), files=len(files) + 1,
                training_history_observations=sum(v['observations'] for v in histories),
                native_case_measurements=96)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', required=True)
    parser.add_argument('--inputs', nargs=3, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.export, args.inputs, args.output)))
