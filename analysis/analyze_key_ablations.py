"""Case-unit, paired GTA-AVT analysis; reads completed reference evaluations only.

No prediction, checkpoint selection, model tuning, or held-out evaluation occurs
here. Run --self-test for temporary synthetic fixtures, or supply --index,
--stage val|test and a new --output directory. NumPy is the only dependency.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PureWindowsPath
import tempfile

import numpy as np


SEED = 20261022
DRAWS = 20000
INDEX_SCHEMA = 'GTA-AVT-key-ablation-index-v1'
REFERENCE_SCHEMA = 'GTA-AVT-reference-evaluation-v1'
SPLIT_SHA256 = 'df9647da8b331260fb9f66b294eced01180c1eaf812de27b5148b6800a344a3b'
FIXED_CASES = {
    'final_test12': ['Dongyang_D11', 'Dongyang_D13', 'Dongyang_D14', 'Dongyang_D7',
                     'KiTS_K1', 'KiTS_K10', 'KiTS_K5', 'KiTS_K7',
                     'Rider_R13', 'Rider_R14', 'Rider_R16', 'Rider_R2'],
    'full_frozen8': ['Dongyang_D2', 'Dongyang_D5', 'KiTS_K13', 'KiTS_K15',
                     'KiTS_K18', 'KiTS_K8', 'Rider_R10', 'Rider_R9'],
    'pilot_fixed3': ['Dongyang_D2', 'KiTS_K13', 'Rider_R10'],
}
METHODS = {
    'A_soft': ('main', 'faithful_soft'),
    'C_concat': ('main', 'faithful_concat'),
    'A_soft_constant': ('main', 'faithful_soft_constant'),
    'A_soft_platform_bridge': ('ablations', 'faithful_soft'),
    'A_soft_scale': ('ablations', 'faithful_soft_scale'),
    'A_soft_structure': ('ablations', 'faithful_soft_structure'),
    'A_hard_anchor': ('ablations', 'faithful_joint'),
}
COMPARISONS = [
    ('main_A_minus_C', 'primary', 'A_soft', 'C_concat'),
    ('main_A_minus_constant', 'secondary', 'A_soft', 'A_soft_constant'),
    ('ablations_full_minus_scale', 'exploratory', 'A_soft_platform_bridge', 'A_soft_scale'),
    ('ablations_full_minus_structure', 'exploratory', 'A_soft_platform_bridge', 'A_soft_structure'),
    ('ablations_soft_minus_hard', 'exploratory', 'A_soft_platform_bridge', 'A_hard_anchor'),
]
# name: display unit, multiplier, favorable direction of A-B.
METRICS = {
    'dice': ('pp', 100., 1), 'precision': ('pp', 100., 1), 'recall': ('pp', 100., 1),
    'fp_ml': ('ml', 1., -1), 'fn_ml': ('ml', 1., -1),
    'assd_mm': ('mm', 1., -1), 'hd95_mm': ('mm', 1., -1),
    'centerline_recall': ('pp', 100., 1), 'centerline_precision': ('pp', 100., 1),
    'cldice': ('pp', 100., 1),
}
GROUPS = ('radius_le_2mm', 'radius_2_to_4mm', 'radius_4_to_8mm', 'radius_gt_8mm',
          'nonjunction', 'junction_neighbourhood')
FIXED_INFERENCE = dict(threshold=.5, patch_size=128, overlap=.5, batch_size=2,
                       condition_mode='predicted', postprocessing='none', grid='original CT')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(x in '0123456789abcdef' for x in value)


def validated_training_recipe(protocol, expected_method, expected_stage, allow_legacy_pilot=False):
    """Formal comparisons require complete typed recipes, not provenance SHA alone.

    The SHA uses the same sorted ASCII JSON representation as the reference
    exporter. Unknown extra recipe fields remain identity-bearing and must be
    identical across methods after removing only ``method``.
    """
    recipe, recorded = protocol.get('training_recipe'), protocol.get('training_recipe_sha256')
    if recipe is None and recorded is None and expected_stage == 'pilot' and allow_legacy_pilot:
        return None  # Explicit development-only compatibility, never formal.
    if not isinstance(recipe, dict) or not recipe or not valid_sha(recorded):
        raise ValueError('Complete training_recipe and canonical training_recipe_sha256 are required')
    try:
        actual = canonical_sha(recipe)
    except (TypeError, ValueError) as error:
        raise ValueError('Training recipe must contain finite JSON values') from error
    if actual != recorded:
        raise ValueError('Training recipe SHA256 does not match its canonical contents')
    positive_ints = ('width', 'patch_size', 'batch_size', 'max_steps', 'log_every',
                     'checkpoint_every', 'validate_every', 'inference_patch_size',
                     'inference_batch_size', 'cpu_threads')
    required = set(positive_ints) | {'method', 'stage', 'seed', 'lr', 'weight_decay',
        'foreground_probability', 'auxiliary_weight', 'gradient_clip',
        'validation_case_limit', 'inference_overlap', 'save_validation_predictions'}
    if not required.issubset(recipe):
        raise ValueError('Training recipe is missing required original training fields')
    if (recipe['method'] != expected_method or recipe['method'] != protocol.get('method') or
            recipe['stage'] != expected_stage or recipe['stage'] != protocol.get('stage')):
        raise ValueError('Training recipe method/stage differs from the declared trained model')
    if type(recipe['seed']) is not int or recipe['seed'] != SEED:
        raise ValueError('Training recipe must record the fixed integer seed20261022')
    if any(type(recipe[name]) is not int or recipe[name] <= 0 for name in positive_ints):
        raise ValueError('Training recipe dimensions/budgets/intervals must be positive integers')
    if type(recipe['validation_case_limit']) is not int or recipe['validation_case_limit'] < 0:
        raise ValueError('Training recipe validation_case_limit must be a nonnegative integer')
    if not isinstance(recipe['save_validation_predictions'], bool):
        raise ValueError('Training recipe save_validation_predictions must be a boolean')
    for name in ('lr', 'weight_decay', 'foreground_probability', 'auxiliary_weight', 'gradient_clip', 'inference_overlap'):
        value = recipe[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError('Training recipe requires finite numeric ' + name)
    if (recipe['lr'] <= 0 or recipe['weight_decay'] < 0 or recipe['auxiliary_weight'] < 0 or
            recipe['gradient_clip'] <= 0 or not 0 <= recipe['foreground_probability'] <= 1):
        raise ValueError('Training recipe contains an invalid learning/sampling parameter')
    if (recipe['inference_patch_size'] != FIXED_INFERENCE['patch_size'] or
            recipe['inference_batch_size'] != FIXED_INFERENCE['batch_size'] or
            recipe['inference_overlap'] != FIXED_INFERENCE['overlap']):
        raise ValueError('Training recipe does not match the fixed reference inference settings')
    if expected_stage == 'formal' and (recipe['validation_case_limit'] != 0 or
            ('validation_case_ids' in recipe and recipe['validation_case_ids'] != FIXED_CASES['full_frozen8'])):
        raise ValueError('Formal training recipe must retain all frozen validation cases')
    step = protocol.get('selected_step')
    if type(step) is not int or not 0 < step <= recipe['max_steps']:
        raise ValueError('Selected checkpoint update exceeds the recorded training recipe budget')
    return recipe


def relative_path(base, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError('Index and metric paths must be nonempty relative paths')
    return (base / value).resolve()


def cohort(case_id):
    value = case_id.split('_', 1)[0]
    if value not in ('Dongyang', 'KiTS', 'Rider'):
        raise ValueError('Unrecognized frozen case ID')
    return value


def clean_environment(value):
    """Keep execution identity while excluding local executable paths/hostnames."""
    if not isinstance(value, dict) or not value.get('python'):
        raise ValueError('Reference/training environment receipt is missing Python identity')
    packages = value.get('packages', {})
    if not all(packages.get(name) for name in ('torch', 'numpy', 'scipy', 'scikit-image', 'SimpleITK')):
        raise ValueError('Environment receipt must identify torch and all four metric/data dependencies')
    selected = ('python', 'python_version', 'platform', 'packages', 'torch_cuda',
                'cudnn_version', 'cuda_available', 'devices', 'backend_settings')
    result = {key: value[key] for key in selected if key in value}
    if value.get('hostname'):
        result['host_identity_sha256'] = canonical_sha({'hostname': value['hostname']})
    return result


def number(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError('Metric must be a finite number or explicit null: ' + name)
    result = float(value)
    if result < 0. or (METRICS.get(name, (None, 1.))[1] == 100. and result > 1.):
        raise ValueError('Metric outside its declared range: ' + name)
    return result


def missing_reason(row, name):
    if row.get(name) is not None:
        return ''
    if name not in row:
        return 'not_recorded'
    if row.get('empty_truth') or row.get('empty_prediction'):
        return 'empty_mask_undefined_by_evaluator'
    if name.startswith('centerline') or name == 'cldice':
        return 'undefined_skeleton_or_other_evaluator_null'
    return 'evaluator_null'


def group_record(row, group):
    record = row.get('geometry_groups', {}).get(group)
    if record is None:
        return None, None, 'group_not_recorded'
    support = record.get('support_nodes')
    if isinstance(support, bool) or not isinstance(support, int) or support < 0:
        raise ValueError('Geometry group requires a nonnegative integer GT support')
    value = number(record.get('centerline_recall'), 'centerline_recall')
    if support == 0:
        if value is not None:
            raise ValueError('A geometry group without GT support must have null recall')
        return support, None, 'no_gt_group_support'
    return support, value, '' if value is not None else 'null_despite_gt_group_support'


def distribution(values):
    finite = [float(x) for x in values if x is not None]
    return dict(total_cases=len(values), defined_cases=len(finite), missing_cases=len(values) - len(finite),
                all_patient_mean=float(np.mean(finite)) if finite and len(finite) == len(values) else None,
                defined_patient_mean=float(np.mean(finite)) if finite else None,
                std_defined_patients=float(np.std(finite, ddof=1)) if len(finite) > 1 else None)


def paired(values_a, values_b, direction):
    valid = [(a, b) for a, b in zip(values_a, values_b) if a is not None and b is not None]
    delta = np.asarray([a - b for a, b in valid], dtype=np.float64)
    result = dict(total_cases=len(values_a), paired_cases=len(valid),
                  missing_a=sum(x is None for x in values_a), missing_b=sum(x is None for x in values_b),
                  excluded_cases=len(values_a) - len(valid), a_mean_paired=None, b_mean_paired=None, delta_a_minus_b=None,
                  ci95_lower=None, ci95_upper=None, wins=0, ties=0, losses=0,
                  estimand='unweighted patient mean of paired A minus B; complete pairs only',
                  interval='percentile paired patient bootstrap; 20000 draws; seed20261022',
                  missing_policy='no imputation; undefined metrics excluded pairwise with support reported')
    if not len(delta):
        return result
    # Each row is a resample of patients, never voxels, patches, nodes or models.
    draws = np.random.default_rng(SEED).integers(0, len(delta), size=(DRAWS, len(delta)))
    lower, upper = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5])
    favorable = delta * direction
    result.update(a_mean_paired=float(np.mean([x[0] for x in valid])), b_mean_paired=float(np.mean([x[1] for x in valid])),
                  delta_a_minus_b=float(delta.mean()), ci95_lower=float(lower), ci95_upper=float(upper),
                  wins=int((favorable > 1e-12).sum()), ties=int((np.abs(favorable) <= 1e-12).sum()),
                  losses=int((favorable < -1e-12).sum()))
    if len(delta) < 2:
        result['support_warning'] = 'Fewer than two paired patients; degenerate interval is not uncertainty evidence'
    return result


def load_inputs(index_path, stage, allow_partial=False):
    index_path = Path(index_path).resolve()
    index = read(index_path)
    if index.get('schema') != INDEX_SCHEMA or index.get('stage') != stage:
        raise ValueError('Index schema or explicit val/test stage does not match')
    scope = index.get('patient_scope')
    if scope not in FIXED_CASES or (scope == 'final_test12') != (stage == 'test'):
        raise ValueError('Explicit patient_scope must match val/test stage')
    ids = FIXED_CASES[scope]
    if index.get('patient_ids') != ids or index.get('split_sha256') != SPLIT_SHA256:
        raise ValueError('Index must list the exact frozen patient IDs/order and split SHA')
    entries = index.get('methods', [])
    if not entries or len({x['id'] for x in entries}) != len(entries):
        raise ValueError('Require at least one uniquely identified internal method')
    if not allow_partial and {x['id'] for x in entries} != set(METHODS):
        raise ValueError('Full analysis requires all seven methods; use --allow-partial explicitly for an incomplete analysis')
    loaded, common, role_training = {}, None, {}
    for entry in entries:
        method_id = entry['id']
        if method_id not in METHODS or entry.get('role') != METHODS[method_id][0]:
            raise ValueError('Method or training-role mapping is outside the predeclared internal ablations')
        directory = relative_path(index_path.parent, entry['evaluation_dir'])
        protocol_path = relative_path(directory, entry.get('protocol_file', 'evaluation_protocol.json'))
        protocol = read(protocol_path)
        if protocol.get('schema') != REFERENCE_SCHEMA or protocol.get('split') != stage:
            raise ValueError('Require completed common-reference protocol with the same explicit stage')
        if protocol.get('method') != METHODS[method_id][1] or protocol.get('case_ids') != ids:
            raise ValueError('Method/case identity differs between index and evaluation protocol')
        expected_training_stage = 'pilot' if scope == 'pilot_fixed3' else 'formal'
        if protocol.get('stage') != expected_training_stage:
            raise ValueError('Pilot and formal training evaluations cannot be mixed')
        if stage == 'test' and (protocol.get('final_test_eligible') is not True or protocol.get('explicitly_authorized_formal_test') is not True):
            raise ValueError('Final test analysis requires an eligible explicitly authorized formal evaluation')
        if protocol.get('intervention', {}).get('name') != 'none' or protocol.get('inference') != FIXED_INFERENCE:
            raise ValueError('Mechanism interventions or a changed inference recipe cannot enter trained-model comparisons')
        for name in ('canonical_data_sha256', 'original_checkpoint_sha256', 'weights_sha256', 'provenance_sha256'):
            if not valid_sha(protocol.get(name)):
                raise ValueError('Missing checkpoint/data/export provenance SHA256: ' + name)
        if not isinstance(protocol.get('selected_step'), int) or isinstance(protocol['selected_step'], bool) or protocol['selected_step'] <= 0:
            raise ValueError('Checkpoint provenance requires a positive selected training update')
        recipe = validated_training_recipe(protocol, METHODS[method_id][1], expected_training_stage,
                                           allow_legacy_pilot=scope == 'pilot_fixed3')
        execution = protocol.get('selected_execution_device')
        if not isinstance(execution, dict) or execution.get('type') not in ('cpu', 'cuda') or not isinstance(protocol.get('autocast_enabled'), bool):
            raise ValueError('Reference protocol must record selected_execution_device and actual autocast flag')
        if protocol['autocast_enabled'] != (execution['type'] == 'cuda'):
            raise ValueError('Execution device and frozen CUDA autocast policy disagree')
        runtime = clean_environment(protocol['reference_environment'])
        torch_version = runtime['packages']['torch']
        if runtime.get('python_version', [])[:2] != [3, 8] or not isinstance(torch_version, str) or torch_version.split('+')[0].split('.')[:2] != ['1', '11']:
            raise ValueError('Reference evaluation requires recorded Python3.8 and torch1.11')
        backend = runtime.get('backend_settings', {})
        if backend.get('matmul_tf32') is not False or backend.get('cudnn_tf32') is not False:
            raise ValueError('Reference evaluation requires both TF32 backends explicitly disabled')
        source = protocol.get('frozen_source_sha256')
        if not isinstance(source, dict) or not source or any(not valid_sha(v) for v in source.values()):
            raise ValueError('Missing frozen evaluation source byte identity')
        tools = protocol.get('evaluator_tool_sha256')
        if not isinstance(tools, dict) or not tools or any(not valid_sha(v) for v in tools.values()):
            raise ValueError('Reference evaluator tool SHA256 identities are required')
        identity = dict(reference_environment=runtime, selected_execution_device=execution,
                        autocast_enabled=protocol['autocast_enabled'],
                        frozen_source_sha256=source, canonical_data_sha256=protocol['canonical_data_sha256'],
                        inference=protocol['inference'], precision=protocol['precision'],
                        evaluator_tool_sha256=tools, stage=stage,
                        training_stage=protocol['stage'], case_ids=ids,
                        training_recipe_without_method=None if recipe is None else {k: v for k, v in recipe.items() if k != 'method'},
                        training_recipe_without_method_sha256=None if recipe is None else canonical_sha({k: v for k, v in recipe.items() if k != 'method'}),
                        training_recipe_status='legacy_pilot_development_only' if recipe is None else 'complete_recipe_and_SHA_verified')
        if common is None:
            common = identity
        elif identity != common:
            raise ValueError('Paired methods differ in reference runtime/source/data/inference/patient identity or common training recipe')
        training_receipt = protocol.get('training_environment_receipt', {})
        if not valid_sha(training_receipt.get('sha256')):
            raise ValueError('Training environment receipt SHA256 is required')
        training = clean_environment(training_receipt.get('value'))
        role = entry['role']
        if role in role_training and training != role_training[role]:
            raise ValueError('Within-role training runtime differs; resolve the confound before pairing')
        role_training[role] = training
        rows, metric_hashes = {}, {}
        supplied = entry.get('metric_files')
        if supplied is not None and set(supplied) != set(ids):
            raise ValueError('Explicit metric file map must cover exactly the stage patient set')
        for case_id in ids:
            name = supplied[case_id] if supplied is not None else case_id + '/metrics.json'
            metric_path = relative_path(directory, name)
            row = read(metric_path)
            if row.get('case_id') != case_id or row.get('cohort') != cohort(case_id):
                raise ValueError('Metric patient/cohort identity mismatch')
            expected = dict(fixed_threshold=.5, condition_mode='predicted', evaluation_grid='original CT', postprocessing='none')
            if any(row.get(k) != v for k, v in expected.items()):
                raise ValueError('Per-patient threshold/condition/grid/postprocessing mismatch')
            for metric in METRICS:
                number(row.get(metric), metric)
            if any(row.get(name) is None for name in ('dice', 'fp_ml', 'fn_ml')):
                raise ValueError('Dice, FP and FN volumes are required for every patient')
            if not row.get('assd_definition') or not row.get('centerline_definition'):
                raise ValueError('Per-patient metric definitions are required')
            for group in GROUPS:
                group_record(row, group)
            rows[case_id] = row
            metric_hashes[case_id] = sha(metric_path)
        loaded[method_id] = dict(id=method_id, role=role, rows=rows,
            provenance=dict(evaluation_dir=entry['evaluation_dir'], protocol_sha256=sha(protocol_path),
                metric_sha256=metric_hashes, original_checkpoint_sha256=protocol['original_checkpoint_sha256'],
                weights_sha256=protocol['weights_sha256'], provenance_sha256=protocol['provenance_sha256'], selected_step=protocol['selected_step'],
                training_recipe=recipe, training_recipe_sha256=protocol.get('training_recipe_sha256'),
                training_environment_sha256=canonical_sha(training),
                training_environment_receipt_sha256=training_receipt['sha256']))
    # Geometry comes from GT and must not vary with the predicted method.
    definitions = {'assd_definition': set(), 'centerline_definition': set()}
    for case_id in ids:
        truth_supports, group_supports = set(), {g: set() for g in GROUPS}
        for method in loaded.values():
            row = method['rows'][case_id]
            for name in definitions:
                definitions[name].add(row[name])
            if row.get('truth_skeleton_support') is not None:
                truth_supports.add(row['truth_skeleton_support'])
            for group in GROUPS:
                support, _, _ = group_record(row, group)
                if support is not None:
                    group_supports[group].add(support)
        if len(truth_supports) > 1 or any(len(x) > 1 for x in group_supports.values()):
            raise ValueError('GT skeleton/group support differs across methods for patient ' + case_id)
    if any(len(x) != 1 for x in definitions.values()):
        raise ValueError('Metric definitions differ across paired evaluations')
    return index, ids, loaded, common, role_training, {k: next(iter(v)) for k, v in definitions.items()}


def scopes(ids):
    return [('all', ids)] + [(name, [x for x in ids if cohort(x) == name]) for name in ('Dongyang', 'KiTS', 'Rider')]


def analyze(index_path, stage, allow_partial=False):
    index, ids, methods, identity, training, definitions = load_inputs(index_path, stage, allow_partial)
    method_rows, case_rows, geometry_rows, geometry_method_rows = [], [], [], []
    for method_id in METHODS:
        if method_id not in methods:
            continue
        method = methods[method_id]
        for case_id in ids:
            row = method['rows'][case_id]
            record = dict(method=method_id, role=method['role'], stage=stage, case_id=case_id, cohort=cohort(case_id),
                          fraction_metric_value_unit='percent')
            for name, (_, factor, _) in METRICS.items():
                record[name] = None if row.get(name) is None else row[name] * factor
                record[name + '_missing_reason'] = missing_reason(row, name)
            record.update(truth_skeleton_support=row.get('truth_skeleton_support'),
                          prediction_skeleton_support=row.get('prediction_skeleton_support'),
                          empty_prediction=row.get('empty_prediction'), empty_truth=row.get('empty_truth'))
            case_rows.append(record)
            for group in GROUPS:
                support, value, reason = group_record(row, group)
                geometry_rows.append(dict(method=method_id, role=method['role'], stage=stage, case_id=case_id,
                    cohort=cohort(case_id), group=group, gt_support_nodes=support,
                    centerline_recall_percent=None if value is None else value * 100., missing_reason=reason))
        for scope_name, selected in scopes(ids):
            for name, (unit, factor, _) in METRICS.items():
                values = [method['rows'][c].get(name) for c in selected]
                values = [None if v is None else v * factor for v in values]
                method_rows.append(dict(method=method_id, role=method['role'], stage=stage, cohort=scope_name,
                    metric=name, value_unit='percent' if factor == 100. else unit, **distribution(values)))
            for group in GROUPS:
                records = [group_record(method['rows'][c], group) for c in selected]
                values = [None if v is None else v * 100. for _, v, _ in records]
                geometry_method_rows.append(dict(method=method_id, role=method['role'], stage=stage, cohort=scope_name,
                    group=group, metric='GT_group_centerline_recall', value_unit='percent',
                    no_gt_support_cases=sum(s == 0 for s, _, _ in records),
                    unrecorded_group_cases=sum(s is None for s, _, _ in records),
                    aggregation='equal weight per supported patient, never per node', **distribution(values)))
    paired_rows, paired_cases, geometry_pairs, worst, unavailable = [], [], [], [], []
    for comparison_id, priority, a_id, b_id in COMPARISONS:
        if a_id not in methods or b_id not in methods:
            unavailable.append(comparison_id)
            continue
        a, b = methods[a_id], methods[b_id]
        if a['role'] != b['role']:
            raise AssertionError('Cross-training-role comparison is prohibited')
        common = dict(comparison=comparison_id, priority=priority, role=a['role'], method_a=a_id, method_b=b_id, stage=stage)
        for case_id in ids:
            ra, rb = a['rows'][case_id], b['rows'][case_id]
            for name, (unit, factor, direction) in METRICS.items():
                av, bv = ra.get(name), rb.get(name)
                av, bv = None if av is None else av * factor, None if bv is None else bv * factor
                paired_cases.append(dict(common, case_id=case_id, cohort=cohort(case_id), metric=name,
                    difference_unit=unit, a=av, b=bv, delta_a_minus_b=av - bv if av is not None and bv is not None else None,
                    favorable_direction='higher' if direction == 1 else 'lower',
                    missing_a=missing_reason(ra, name), missing_b=missing_reason(rb, name)))
        for scope_name, selected in scopes(ids):
            for name, (unit, factor, direction) in METRICS.items():
                av = [a['rows'][c].get(name) for c in selected]
                bv = [b['rows'][c].get(name) for c in selected]
                av = [None if x is None else x * factor for x in av]
                bv = [None if x is None else x * factor for x in bv]
                paired_rows.append(dict(common, cohort=scope_name, metric=name, difference_unit=unit,
                    favorable_direction='higher' if direction == 1 else 'lower', **paired(av, bv, direction)))
            for group in GROUPS:
                ga = [group_record(a['rows'][c], group) for c in selected]
                gb = [group_record(b['rows'][c], group) for c in selected]
                av = [None if v is None else v * 100. for _, v, _ in ga]
                bv = [None if v is None else v * 100. for _, v, _ in gb]
                geometry_pairs.append(dict(common, cohort=scope_name, group=group, difference_unit='pp',
                    metric='GT_group_centerline_recall',
                    no_gt_support_cases=sum(x[0] == 0 and y[0] == 0 for x, y in zip(ga, gb)),
                    unrecorded_group_cases=sum(x[0] is None or y[0] is None for x, y in zip(ga, gb)),
                    aggregation='unweighted patient mean among defined supported paired patients; nodes are never pooled',
                    precision_scope='No GT-group precision is defined; FP/FN tradeoffs are global', **paired(av, bv, 1)))
        ranked = sorted(ids, key=lambda c: (a['rows'][c]['dice'] - b['rows'][c]['dice'], c))
        for rank, case_id in enumerate(ranked[:3], 1):
            row = dict(common, worst_dice_rank=rank, case_id=case_id, cohort=cohort(case_id))
            for name in ('dice', 'fp_ml', 'fn_ml', 'assd_mm', 'hd95_mm'):
                factor = METRICS[name][1]
                av, bv = a['rows'][case_id].get(name), b['rows'][case_id].get(name)
                row[name + '_a'] = None if av is None else av * factor
                row[name + '_b'] = None if bv is None else bv * factor
                row[name + '_delta'] = (av - bv) * factor if av is not None and bv is not None else None
            worst.append(row)
    return dict(schema='GTA-AVT-key-ablation-analysis-v1', stage=stage, patient_scope=index['patient_scope'],
        patient_ids=ids, split_sha256=SPLIT_SHA256, index_sha256=sha(index_path), seed=SEED, bootstrap_draws=DRAWS,
        complete_method_set=set(methods) == set(METHODS), unavailable_comparisons=unavailable,
        reference_identity=identity, training_environments_by_role=training,
        metric_definitions=definitions, method_provenance={k: v['provenance'] for k, v in methods.items()},
        method_summaries=method_rows, per_case_metrics=case_rows, paired_summaries=paired_rows,
        paired_case_differences=paired_cases, geometry_case_recall=geometry_rows, geometry_method_summaries=geometry_method_rows,
        geometry_paired_summaries=geometry_pairs, worst_dice_pairs=worst,
        limitations=[
            'One training seed per role: intervals quantify patient resampling, not seed-to-seed stability.',
            'Principal comparisons use main-trained models; ablations use their own same-runtime full-A bridge.',
            'Common reference inference/metrics remove evaluation-runtime differences, not training-runtime or hardware confounding.',
            'Distances and skeleton metrics can be undefined; no imputation; complete-pair estimates can be selectively observed.',
            'Geometry recall averages supported patients equally; GT-defined groups do not provide group precision.',
            'Cohort summaries are exploratory (four test patients per cohort; fewer in validation).',
            'Multiple exploratory outcomes and percentile intervals are descriptive; no significance or multiplicity-adjusted claims.',
            'Different published cross-validation/cohort/label protocols do not support a SOTA ranking here.',
        ])


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_report(result):
    lines = ['GTA-AVT finite internal ablation analysis',
             'Stage: %s; patient scope: %s; patients: %d; complete method set: %s' %
             (result['stage'], result['patient_scope'], len(result['patient_ids']), result['complete_method_set']),
             'Paired differences are A minus B. Positive Dice/centerline/clDice differences favor A; negative FP/FN/distance differences favor A.',
             'Dice and centerline values are percentages; their differences are percentage points (pp).',
             'Intervals: patient-unit paired percentile bootstrap, 20000 draws, fixed seed20261022.',
             'Undefined values are never zero-filled. Distance and skeleton estimates use complete pairs with explicit support.', '']
    for row in result['paired_summaries']:
        if row['cohort'] != 'all':
            continue
        if row['delta_a_minus_b'] is None:
            line = '%s | %s: undefined; paired0/%d' % (row['comparison'], row['metric'], row['total_cases'])
        else:
            line = '%s | %s: delta %.3f %s, 95%% CI [%.3f, %.3f]; paired%d/%d; W/T/L %d/%d/%d' % (
                row['comparison'], row['metric'], row['delta_a_minus_b'], row['difference_unit'],
                row['ci95_lower'], row['ci95_upper'], row['paired_cases'], row['total_cases'], row['wins'], row['ties'], row['losses'])
        lines.append(line)
    if result['unavailable_comparisons']:
        lines += ['', 'INCOMPLETE: unavailable comparisons: ' + ', '.join(result['unavailable_comparisons'])]
    lines += ['', 'Geometry-group recall and cohort summaries are in the corresponding CSV/JSON tables; supported patient means are used, never pooled nodes.',
              'Worst pairs are the three smallest Dice A-B differences per comparison; this descriptive selection is fixed in advance.', '']
    lines.extend(result['limitations'])
    return '\n'.join(lines) + '\n'


def save_result(result, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Analysis output is immutable; choose a new empty directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    for key in ('method_summaries', 'per_case_metrics', 'paired_summaries', 'paired_case_differences',
                'geometry_case_recall', 'geometry_method_summaries', 'geometry_paired_summaries', 'worst_dice_pairs'):
        write_csv(output / (key + '.csv'), result[key])
    (output / 'report.txt').write_text(render_report(result), encoding='utf-8')


def _synthetic_recipe(method, stage='formal'):
    """Self-test inputs only, never an inferred recipe for a real evaluation."""
    return dict(stage=stage, method=method, seed=SEED, width=32, patch_size=96,
        batch_size=2, max_steps=5000, lr=.0003, weight_decay=.0001,
        foreground_probability=.5, auxiliary_weight=.1, gradient_clip=12.,
        log_every=25, checkpoint_every=500, validate_every=2500, validation_case_limit=0,
        inference_patch_size=128, inference_batch_size=2, inference_overlap=.5,
        save_validation_predictions=False, cpu_threads=4)


def self_test():
    """Temporary synthetic inputs are removed; fixture values never enter real outputs."""
    with tempfile.TemporaryDirectory(prefix='gta_avt_analysis_fixture_') as temp:
        root = Path(temp)
        ids = FIXED_CASES['full_frozen8']
        env = dict(python='fixture-python', packages={k: 'fixture-version' for k in
            ('torch', 'numpy', 'scipy', 'scikit-image', 'SimpleITK')}, devices=[], python_version=[3, 8, 0],
            backend_settings={'matmul_tf32': False, 'cudnn_tf32': False})
        env['packages']['torch'] = '1.11.0+fixture'
        entries = []
        for method_id, (role, method) in METHODS.items():
            directory = root / method_id
            directory.mkdir()
            protocol = dict(schema=REFERENCE_SCHEMA, split='val', case_ids=ids, method=method, stage='formal',
                selected_step=5000, original_checkpoint_sha256='a' * 64, weights_sha256='b' * 64, provenance_sha256='0' * 64,
                frozen_source_sha256={'fixture.py': 'c' * 64}, canonical_data_sha256='d' * 64,
                final_test_eligible=True, explicitly_authorized_formal_test=False, inference=FIXED_INFERENCE,
                intervention={'name': 'none'}, precision='fixture-fp32', reference_environment=env,
                selected_execution_device={'type': 'cpu', 'index': None, 'name': 'CPU'}, autocast_enabled=False,
                evaluator_tool_sha256={'fixture.py': 'e' * 64}, training_environment_receipt={'sha256': 'f' * 64, 'value': env})
            protocol['training_recipe'] = _synthetic_recipe(method)
            protocol['training_recipe_sha256'] = canonical_sha(protocol['training_recipe'])
            (directory / 'evaluation_protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
            for i, case_id in enumerate(ids):
                value = .6 + i * .01
                if method_id == 'C_concat':
                    value -= .02
                row = dict(case_id=case_id, cohort=cohort(case_id), fixed_threshold=.5, condition_mode='predicted',
                    evaluation_grid='original CT', postprocessing='none', dice=value, precision=.8, recall=.7,
                    fp_ml=1. + i, fn_ml=2. + i, assd_mm=None if i == 0 and method_id == 'C_concat' else 1.,
                    hd95_mm=2., centerline_recall=.7, centerline_precision=.8, cldice=.7466666666666666,
                    empty_prediction=False, empty_truth=False, truth_skeleton_support=10, prediction_skeleton_support=11,
                    assd_definition='fixture-pooled-surface', centerline_definition='fixture-Lee-voxel',
                    geometry_groups={g: {'support_nodes': 0 if g == 'junction_neighbourhood' else 10,
                        'centerline_recall': None if g == 'junction_neighbourhood' else .7} for g in GROUPS})
                row['geometry_groups']['radius_le_2mm'] = {'support_nodes': i + 1, 'centerline_recall': i / (len(ids) - 1.)}
                if method_id == 'A_soft_scale':
                    # A nonconstant within-role contrast exercises asymmetric
                    # interval plotting as well as patient-level estimates.
                    row['dice'] -= .01 * i
                    row['fp_ml'] -= .1 * i
                    for group in GROUPS:
                        if row['geometry_groups'][group]['centerline_recall'] is not None:
                            row['geometry_groups'][group]['centerline_recall'] -= .01 * i
                patient = directory / case_id
                patient.mkdir()
                (patient / 'metrics.json').write_text(json.dumps(row), encoding='utf-8')
            entries.append(dict(id=method_id, role=role, evaluation_dir=method_id))
        index = dict(schema=INDEX_SCHEMA, stage='val', patient_scope='full_frozen8', patient_ids=ids,
                     split_sha256=SPLIT_SHA256, methods=entries)
        index_path = root / 'index.json'
        index_path.write_text(json.dumps(index), encoding='utf-8')
        result = analyze(index_path, 'val')
        def pick(comparison, metric):
            return next(x for x in result['paired_summaries'] if x['comparison'] == comparison and x['metric'] == metric and x['cohort'] == 'all')
        identical = pick('main_A_minus_constant', 'dice')
        assert identical['delta_a_minus_b'] == identical['ci95_lower'] == identical['ci95_upper'] == 0.
        assert identical['ties'] == len(ids)
        offset = pick('main_A_minus_C', 'dice')
        assert all(abs(offset[k] - 2.) < 1e-10 for k in ('delta_a_minus_b', 'ci95_lower', 'ci95_upper'))
        assert offset['wins'] == len(ids)
        distance = pick('main_A_minus_C', 'assd_mm')
        assert distance['paired_cases'] == 7 and distance['excluded_cases'] == 1
        c_distance = next(x for x in result['method_summaries'] if x['method'] == 'C_concat' and x['metric'] == 'assd_mm' and x['cohort'] == 'all')
        assert c_distance['all_patient_mean'] is None and c_distance['defined_cases'] == 7
        empty_group = next(x for x in result['geometry_paired_summaries'] if x['comparison'] == 'main_A_minus_C' and x['group'] == 'junction_neighbourhood' and x['cohort'] == 'all')
        assert empty_group['paired_cases'] == 0 and empty_group['no_gt_support_cases'] == 8 and empty_group['delta_a_minus_b'] is None
        radius_group = next(x for x in result['geometry_method_summaries'] if x['method'] == 'A_soft' and x['group'] == 'radius_le_2mm' and x['cohort'] == 'all')
        assert abs(radius_group['defined_patient_mean'] - 50.) < 1e-10
        assert abs(radius_group['defined_patient_mean'] - np.average(np.linspace(0., 100., 8), weights=np.arange(1, 9))) > 1.
        # Nonconstant differences verify deterministic resampling, beyond a trivial constant-offset CI.
        check = paired([0., 2., 4.], [0., 0., 0.], 1)
        assert check == paired([0., 2., 4.], [0., 0., 0.], 1)
        assert check['ci95_lower'] <= 2. <= check['ci95_upper'] and check['ci95_upper'] > check['ci95_lower']
        save_result(result, root / 'temporary_output')
        assert 'fixture' not in render_report(result)
        # These negative cases recompute the hash where appropriate: a valid
        # per-method SHA must not hide a different budget/patch/extra field.
        changed = root / 'C_concat/evaluation_protocol.json'
        saved_recipe_protocol = read(changed)
        recipe_failures = ('mixed_v1_v2_budget', 'mixed_v1_v2_patch', 'recipe_SHA_mismatch',
            'recipe_method_mismatch', 'recipe_stage_mismatch', 'recipe_seed_mismatch',
            'formal_recipe_missing', 'malformed_recipe', 'selected_step_exceeds_budget',
            'extra_recipe_field_mismatch', 'recipe_JSON_numeric_type_mismatch')
        for mutation in recipe_failures:
            bad = json.loads(json.dumps(saved_recipe_protocol))
            if mutation == 'mixed_v1_v2_budget': bad['training_recipe']['max_steps'] = 10000
            if mutation == 'mixed_v1_v2_patch': bad['training_recipe']['patch_size'] = 128
            if mutation == 'recipe_method_mismatch': bad['training_recipe']['method'] = 'faithful_soft'
            if mutation == 'recipe_stage_mismatch': bad['training_recipe']['stage'] = 'pilot'
            if mutation == 'recipe_seed_mismatch': bad['training_recipe']['seed'] += 1
            if mutation == 'extra_recipe_field_mismatch': bad['training_recipe']['recipe_version'] = 'synthetic-v2'
            if mutation == 'recipe_JSON_numeric_type_mismatch': bad['training_recipe']['gradient_clip'] = 12
            if mutation == 'malformed_recipe': bad['training_recipe'] = ['not_a_recipe_dictionary']
            if mutation == 'selected_step_exceeds_budget': bad['selected_step'] = 5001
            bad['training_recipe_sha256'] = canonical_sha(bad['training_recipe'])
            if mutation == 'recipe_SHA_mismatch': bad['training_recipe_sha256'] = '1' * 64
            if mutation == 'formal_recipe_missing':
                bad.pop('training_recipe'); bad.pop('training_recipe_sha256')
            changed.write_text(json.dumps(bad), encoding='utf-8')
            try:
                load_inputs(index_path, 'val')
            except ValueError as error:
                assert 'recipe' in str(error) or 'budget' in str(error)
            else:
                raise AssertionError('Invalid formal training recipe accepted: ' + mutation)
        changed.write_text(json.dumps(saved_recipe_protocol), encoding='utf-8')
        saved_protocols = {method_id: read(root / method_id / 'evaluation_protocol.json') for method_id in METHODS}
        for method_id, original in saved_protocols.items():
            updated = json.loads(json.dumps(original))
            updated['training_recipe'].update(max_steps=10000, patch_size=128)
            updated['training_recipe_sha256'] = canonical_sha(updated['training_recipe'])
            (root / method_id / 'evaluation_protocol.json').write_text(json.dumps(updated), encoding='utf-8')
        newer_common = load_inputs(index_path, 'val')[3]['training_recipe_without_method']
        assert newer_common['max_steps'] == 10000 and newer_common['patch_size'] == 128
        for method_id, original in saved_protocols.items():
            (root / method_id / 'evaluation_protocol.json').write_text(json.dumps(original), encoding='utf-8')
        assert validated_training_recipe(dict(method='faithful_soft', stage='pilot'),
            'faithful_soft', 'pilot', allow_legacy_pilot=True) is None
        # Reject an execution/runtime mismatch, not just a changed threshold.
        changed = root / 'C_concat/evaluation_protocol.json'
        protocol = read(changed)
        protocol['selected_execution_device'] = {'type': 'cuda', 'index': 0, 'name': 'fixture-GPU'}
        protocol['autocast_enabled'] = True
        changed.write_text(json.dumps(protocol), encoding='utf-8')
        try:
            analyze(index_path, 'val')
        except ValueError as error:
            assert 'reference runtime' in str(error)
        else:
            raise AssertionError('Mixed execution devices were accepted')
        first = root / 'A_soft/evaluation_protocol.json'
        saved_first = read(first)
        for changed_field in ('runtime', 'TF32'):
            bad = json.loads(json.dumps(saved_first))
            if changed_field == 'runtime':
                bad['reference_environment']['python_version'] = [3, 11, 0]
            else:
                bad['reference_environment']['backend_settings']['matmul_tf32'] = True
            first.write_text(json.dumps(bad), encoding='utf-8')
            try:
                analyze(index_path, 'val')
            except ValueError as error:
                assert 'Python3.8' in str(error) or 'TF32' in str(error)
            else:
                raise AssertionError('Wrong reference runtime/backend was accepted')
        first.write_text(json.dumps(saved_first), encoding='utf-8')
        # Gate failures occur before any test metric file is read. No synthetic
        # test predictions or extra test metric artifacts are needed.
        test_index = dict(index, stage='test', patient_scope='final_test12', patient_ids=FIXED_CASES['final_test12'])
        index_path.write_text(json.dumps(test_index), encoding='utf-8')
        test_protocol = dict(read(first), split='test', case_ids=FIXED_CASES['final_test12'])
        for eligibility, authorization in ((False, True), (True, False), ('true', True)):
            test_protocol.update(final_test_eligible=eligibility, explicitly_authorized_formal_test=authorization)
            first.write_text(json.dumps(test_protocol), encoding='utf-8')
            try:
                analyze(index_path, 'test')
            except ValueError as error:
                assert 'eligible explicitly authorized' in str(error)
            else:
                raise AssertionError('Ineligible or unauthorized test receipt was accepted')
        test_protocol.update(final_test_eligible=True, explicitly_authorized_formal_test=True, intervention={'name': 'availability_only'})
        first.write_text(json.dumps(test_protocol), encoding='utf-8')
        try:
            analyze(index_path, 'test')
        except ValueError as error:
            assert 'interventions' in str(error)
        else:
            raise AssertionError('Runtime test intervention was accepted')
    return dict(state='passed', fixtures=['identical_methods', 'known_paired_Dice_offset',
        'missing_distance_complete_pairs', 'no_GT_group_support', 'geometry_patient_not_node_weighting', 'deterministic_nonconstant_bootstrap',
        'CSV_JSON_report_roundtrip', 'mixed_execution_rejected', 'exact_reference_runtime_TF32_gates', 'test_eligibility_and_authorization_gates',
        'test_runtime_intervention_rejected', 'valid_complete_formal_recipes',
        'all_methods_new_recipe_accepted', 'legacy_pilot_development_compatibility'] + list(recipe_failures),
        synthetic_values_saved=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index')
    parser.add_argument('--stage', choices=('val', 'test'))
    parser.add_argument('--output')
    parser.add_argument('--allow-partial', action='store_true', help='Mark incomplete method sets explicitly; never fill missing comparisons')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        if args.index or args.output or args.stage or args.allow_partial:
            parser.error('--self-test cannot be combined with real analysis options')
        print(json.dumps(self_test()))
        return
    if not args.index or not args.stage or not args.output:
        parser.error('Real analysis requires --index, --stage and --output')
    result = analyze(args.index, args.stage, args.allow_partial)
    save_result(result, args.output)
    print(json.dumps(dict(state='complete', stage=args.stage, cases=len(result['patient_ids']),
                         complete_method_set=result['complete_method_set'], automatic_evaluation=False)))


if __name__ == '__main__':
    main()
