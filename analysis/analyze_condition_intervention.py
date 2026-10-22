"""Analyze SAME-weight full-composition versus availability-only FORMAL VAL8.

Reads completed reference-evaluation directories only. No model inference,
test-split option, checkpoint selection, p-values or tuning is performed.
Delta is NORMAL minus AVAILABILITY_ONLY. Run --self-test for temporary fixtures.
"""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

import analyze_key_ablations as base


IDS = base.FIXED_CASES['full_frozen8']
NORMAL = 'normal_full_composition'
AVAILABILITY = 'availability_only_same_weights'
FORMULA = {
    'scale': '[q_scale, (1-q_scale)/4, (1-q_scale)/4, (1-q_scale)/4, (1-q_scale)/4]',
    'structure': '[q_structure, (1-q_structure)/2, (1-q_structure)/2]',
    'q_definition': "each original image-only head's predicted p_empty",
}


def reviewed_intervention_source():
    """Private ops/ and public tools/ carry the same reviewed implementation."""
    project = Path(__file__).resolve().parents[1]
    candidates = [project / folder / 'reference_interventions.py' for folder in ('ops', 'tools')]
    present = [path for path in candidates if path.is_file()]
    if not present:
        raise ValueError('Reviewed reference_interventions.py is required in ops/ or tools/; SHA verification cannot be skipped')
    if len({base.sha(path) for path in present}) != 1:
        raise ValueError('Ambiguous ops/tools intervention implementations have different SHA256 identities')
    return present[0]


def check_runtime(protocol):
    runtime = base.clean_environment(protocol['reference_environment'])
    version = runtime['packages']['torch']
    if runtime.get('python_version', [])[:2] != [3, 8] or not isinstance(version, str) or version.split('+')[0].split('.')[:2] != ['1', '11']:
        raise ValueError('Require the common Python3.8/torch1.11 reference runtime')
    flags = runtime.get('backend_settings', {})
    if flags.get('matmul_tf32') is not False or flags.get('cudnn_tf32') is not False:
        raise ValueError('Both reference TF32 backends must be explicitly disabled')
    device = protocol.get('selected_execution_device', {})
    if device.get('type') not in ('cpu', 'cuda') or protocol.get('autocast_enabled') is not (device.get('type') == 'cuda'):
        raise ValueError('Selected execution device and actual autocast policy are required')
    receipt = protocol.get('training_environment_receipt', {})
    if not base.valid_sha(receipt.get('sha256')):
        raise ValueError('Actual main training-runtime receipt is missing')
    training = base.clean_environment(receipt.get('value'))
    # This checks software compatibility, not training-role identity. After
    # runtime harmonization the owner must still supply the genuine main A
    # export; matching software alone cannot distinguish the platform bridge.
    if training['python'] != runtime['python'] or training['packages'] != runtime['packages']:
        raise ValueError('Main A training software differs from the common reference software')
    return runtime, training


def check_protocol(protocol):
    if (protocol.get('schema') != base.REFERENCE_SCHEMA or protocol.get('split') != 'val' or
            protocol.get('stage') != 'formal' or protocol.get('method') != 'faithful_soft' or
            protocol.get('case_ids') != IDS):
        raise ValueError('Only the completed formal main A_soft exact VAL8 reference evaluation is eligible')
    if protocol.get('final_test_eligible') is not True or protocol.get('explicitly_authorized_formal_test') is not False:
        raise ValueError('Require eligible frozen formal weights and a validation-only reference invocation')
    if protocol.get('inference') != base.FIXED_INFERENCE:
        raise ValueError('Normal and intervention must use the fixed reference inference recipe')
    step = protocol.get('selected_step')
    if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
        raise ValueError('Selected genuine training update is missing')
    base.validated_training_recipe(protocol, 'faithful_soft', 'formal')
    for key in ('original_checkpoint_sha256', 'weights_sha256', 'provenance_sha256', 'canonical_data_sha256'):
        if not base.valid_sha(protocol.get(key)):
            raise ValueError('Missing same-weight checkpoint/data/export identity: ' + key)
    for key in ('frozen_source_sha256', 'evaluator_tool_sha256'):
        value = protocol.get(key)
        if not isinstance(value, dict) or not value or not all(base.valid_sha(x) for x in value.values()):
            raise ValueError('Missing frozen source/evaluator byte identity')
    if not protocol.get('precision'):
        raise ValueError('Reference precision policy is missing')
    return check_runtime(protocol)


def check_interventions(normal, availability):
    ordinary = normal.get('intervention', {})
    expected_normal = dict(name='none', separately_trained_model=False, changes_weights=False)
    if ordinary != expected_normal or ordinary.get('separately_trained_model') is not False or ordinary.get('changes_weights') is not False:
        raise ValueError('Normal reference must contain exactly the unmodified no-intervention declaration')
    intervention = availability.get('intervention', {})
    expected = dict(name='availability_only', stage='runtime-validation-diagnostic',
                    separately_trained_model=False, changes_weights=False, uses_GT=False, formula=FORMULA)
    for key, value in expected.items():
        if key not in intervention or intervention[key] != value or (isinstance(value, bool) and intervention[key] is not value):
            raise ValueError('Intervention must preserve p_empty and uniformly redistribute nonempty mass: ' + key)
    tool = reviewed_intervention_source()
    if (not base.valid_sha(intervention.get('intervention_source_sha256')) or
            not tool.is_file() or base.sha(tool) != intervention['intervention_source_sha256']):
        raise ValueError('Recorded availability-only implementation differs from the reviewed intervention source')
    # Exact equality is stricter than a selected-field comparison: identical
    # provenance SHA also binds the full original training recipe/export.
    a = {k: v for k, v in normal.items() if k != 'intervention'}
    b = {k: v for k, v in availability.items() if k != 'intervention'}
    if a != b:
        raise ValueError('Reference protocols differ beyond intervention; weights/checkpoint/source/runtime/device/data/patients/recipe must be common')
    return a


def load_metrics(directory):
    directory = Path(directory)
    rows, hashes = {}, {}
    definitions = {'assd_definition': set(), 'centerline_definition': set()}
    for case_id in IDS:
        path = directory / case_id / 'metrics.json'
        row = base.read(path)
        required = dict(case_id=case_id, cohort=base.cohort(case_id), fixed_threshold=.5,
                        condition_mode='predicted', evaluation_grid='original CT', postprocessing='none')
        if any(row.get(k) != v for k, v in required.items()):
            raise ValueError('Per-patient identity/inference differs from the common VAL8 protocol')
        for name in base.METRICS:
            base.number(row.get(name), name)
        if any(row.get(name) is None for name in ('dice', 'fp_ml', 'fn_ml')):
            raise ValueError('Dice/FP/FN must be defined for all8validation patients')
        for name in definitions:
            if not row.get(name):
                raise ValueError('Native metric definitions must be explicit')
            definitions[name].add(row[name])
        for group in base.GROUPS:
            base.group_record(row, group)
        rows[case_id], hashes[case_id] = row, base.sha(path)
    if any(len(v) != 1 for v in definitions.values()):
        raise ValueError('Metric definitions differ between validation patients')
    summary_path = directory / 'summary.json'
    summary = base.read(summary_path)
    if summary.get('cases') != 8 or summary.get('statistical_unit') != 'case':
        raise ValueError('Reference evaluation is incomplete or uses a different statistical unit')
    for name in base.METRICS:
        values = [rows[c].get(name) for c in IDS]
        finite = [v for v in values if v is not None]
        statistic = summary.get('metrics', {}).get(name, {})
        expected_mean = float(np.mean(finite)) if len(finite) == 8 else None
        if statistic.get('defined_case_count') != len(finite):
            raise ValueError('Completed summary support differs from the patient metric files')
        observed = statistic.get('mean')
        if (expected_mean is None and observed is not None) or (expected_mean is not None and
                (not isinstance(observed, (int, float)) or isinstance(observed, bool) or
                 not np.isfinite(observed) or not np.isclose(expected_mean, observed, rtol=1e-10, atol=1e-12))):
            raise ValueError('Completed summary mean differs from the patient metric files')
    return rows, hashes, {k: next(iter(v)) for k, v in definitions.items()}, base.sha(summary_path)


def analyze(normal_directory, availability_directory):
    normal_directory, availability_directory = Path(normal_directory), Path(availability_directory)
    normal_protocol_path = normal_directory / 'evaluation_protocol.json'
    availability_protocol_path = availability_directory / 'evaluation_protocol.json'
    normal, availability = base.read(normal_protocol_path), base.read(availability_protocol_path)
    runtime, training = check_protocol(normal)
    check_protocol(availability)
    common = check_interventions(normal, availability)
    normal_rows, normal_hashes, normal_definitions, normal_summary_sha = load_metrics(normal_directory)
    availability_rows, availability_hashes, other_definitions, availability_summary_sha = load_metrics(availability_directory)
    if normal_definitions != other_definitions:
        raise ValueError('Native metric definitions differ between the normal and intervened evaluations')
    per_case, summaries, paired_cases, paired_summaries, geometry_cases, geometry_summary = [], [], [], [], [], []
    for label, rows in ((NORMAL, normal_rows), (AVAILABILITY, availability_rows)):
        for case_id in IDS:
            source = rows[case_id]
            record = dict(condition=label, case_id=case_id, cohort=base.cohort(case_id), stage='val',
                          fraction_metric_value_unit='percent', empty_prediction=source.get('empty_prediction'),
                          truth_skeleton_support=source.get('truth_skeleton_support'),
                          prediction_skeleton_support=source.get('prediction_skeleton_support'))
            for name, (_, multiplier, _) in base.METRICS.items():
                record[name] = None if source.get(name) is None else source[name] * multiplier
                record[name + '_missing_reason'] = base.missing_reason(source, name)
            per_case.append(record)
            for group in base.GROUPS:
                support, value, reason = base.group_record(source, group)
                geometry_cases.append(dict(condition=label, case_id=case_id, cohort=base.cohort(case_id), group=group,
                    gt_support_nodes=support, centerline_recall_percent=None if value is None else value * 100.,
                    missing_reason=reason))
        for name, (unit, multiplier, _) in base.METRICS.items():
            values = [None if rows[c].get(name) is None else rows[c][name] * multiplier for c in IDS]
            summaries.append(dict(condition=label, metric=name, value_unit='percent' if multiplier == 100. else unit,
                                  **base.distribution(values)))
    for case_id in IDS:
        n, a = normal_rows[case_id], availability_rows[case_id]
        supports = [r.get('truth_skeleton_support') for r in (n, a) if r.get('truth_skeleton_support') is not None]
        if len(set(supports)) > 1:
            raise ValueError('GT skeleton support differs between same-data evaluations')
        for name, (unit, multiplier, direction) in base.METRICS.items():
            nv, av = n.get(name), a.get(name)
            nv, av = None if nv is None else nv * multiplier, None if av is None else av * multiplier
            paired_cases.append(dict(case_id=case_id, cohort=base.cohort(case_id), metric=name, difference_unit=unit,
                normal=nv, availability_only=av, delta_normal_minus_availability_only=nv - av if nv is not None and av is not None else None,
                favorable_direction='higher' if direction == 1 else 'lower',
                missing_normal=base.missing_reason(n, name), missing_availability_only=base.missing_reason(a, name)))
        for group in base.GROUPS:
            ns, _, _ = base.group_record(n, group)
            ass, _, _ = base.group_record(a, group)
            if ns is not None and ass is not None and ns != ass:
                raise ValueError('GT group support differs between same-data evaluations')
    for name, (unit, multiplier, direction) in base.METRICS.items():
        nv = [None if normal_rows[c].get(name) is None else normal_rows[c][name] * multiplier for c in IDS]
        av = [None if availability_rows[c].get(name) is None else availability_rows[c][name] * multiplier for c in IDS]
        paired_summaries.append(dict(metric=name, difference_unit=unit, delta_definition='normal minus availability_only',
            favorable_direction='higher' if direction == 1 else 'lower', **base.paired(nv, av, direction)))
    for group in base.GROUPS:
        nr = [base.group_record(normal_rows[c], group) for c in IDS]
        ar = [base.group_record(availability_rows[c], group) for c in IDS]
        nv = [None if v is None else v * 100. for _, v, _ in nr]
        av = [None if v is None else v * 100. for _, v, _ in ar]
        geometry_summary.append(dict(group=group, metric='GT_group_centerline_recall', difference_unit='pp',
            no_gt_support_cases=sum(n[0] == 0 and a[0] == 0 for n, a in zip(nr, ar)),
            unrecorded_group_cases=sum(n[0] is None or a[0] is None for n, a in zip(nr, ar)),
            aggregation='equal supported patient weight; nodes are never pooled',
            precision_scope='GT-defined recall only; FP/FN and centerline precision reported globally',
            **base.paired(nv, av, 1)))
    worst = sorted([r for r in paired_cases if r['metric'] == 'dice'],
                   key=lambda r: (r['delta_normal_minus_availability_only'], r['case_id']))[:3]
    # Publish a sanitized identity, not raw runtime receipts with local paths.
    public_common = {k: common[k] for k in ('schema', 'split', 'case_ids', 'method', 'stage', 'selected_step',
        'original_checkpoint_sha256', 'weights_sha256', 'provenance_sha256', 'frozen_source_sha256',
        'canonical_data_sha256', 'inference', 'selected_execution_device', 'autocast_enabled',
        'precision', 'evaluator_tool_sha256', 'training_recipe', 'training_recipe_sha256')}
    public_common.update(reference_environment=runtime, training_environment=training,
                         complete_raw_protocol_identity_sha256=base.canonical_sha(common))
    return dict(schema='GTA-AVT-condition-intervention-analysis-v1', split='val', training_stage='formal',
        patient_ids=IDS, case_count=8, comparison='normal_full_composition_minus_availability_only_same_weights',
        separately_trained_model=False, weights_changed=False, seed=base.SEED, bootstrap_draws=base.DRAWS,
        common_identity=public_common, intervention=availability['intervention'], metric_definitions=normal_definitions,
        input_provenance={NORMAL: dict(evaluation_directory_label=normal_directory.name, protocol_sha256=base.sha(normal_protocol_path),
                            summary_sha256=normal_summary_sha, metric_sha256=normal_hashes),
                          AVAILABILITY: dict(evaluation_directory_label=availability_directory.name, protocol_sha256=base.sha(availability_protocol_path),
                            summary_sha256=availability_summary_sha, metric_sha256=availability_hashes)},
        per_case_metrics=per_case, condition_metric_summaries=summaries, paired_case_differences=paired_cases,
        paired_summaries=paired_summaries, geometry_case_recall=geometry_cases,
        geometry_paired_summaries=geometry_summary, worst_dice_pairs=worst,
        interpretation=[
            'This is a validation-only intervention on the SAME trained main A_soft weights, not a newly trained comparator.',
            'Each head retains its image-predicted p_empty=q; the remaining mass is uniform across four scale or two structure classes.',
            'The intervention removes nonempty composition variation beyond q while retaining the q-dependent availability term; features, weights and operator forms remain fixed.',
            'The current soft operators consume fine/junction components and p_empty; this is not a separate test of every radius-class component.',
            'Retained p_empty may still encode visibility or anatomical correlates. It is an operational availability proxy, not calibrated uncertainty.',
            'A favorable normal-minus-intervention difference is conditional evidence about this runtime composition intervention, not proof of anatomical causality.',
            'Replacing compositions also changes their joint distribution. Read this sensitivity diagnostic with matched trained controls and the separate head-faithfulness audit.',
            'Zero and unfavorable differences are reported. No threshold, architecture, method choice or checkpoint is changed after this diagnostic.',
            'VAL8 already served model selection; these results are development diagnostics, not independent held-out confirmation.',
            'One seed; paired patient bootstrap intervals quantify patient resampling only. No p-values or multiple-outcome significance claims.',
            'Patient macro means and ddof1 SD; complete pairs for undefined metrics; no imputation. GT group recall uses supported patient means, not pooled nodes.',
        ])


def save(result, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Analysis outputs are immutable; use a new empty directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    for key in ('per_case_metrics', 'condition_metric_summaries', 'paired_case_differences', 'paired_summaries',
                'geometry_case_recall', 'geometry_paired_summaries', 'worst_dice_pairs'):
        base.write_csv(output / (key + '.csv'), result[key])
    lines = ['Same-weight condition intervention: formal main A_soft, VAL8 only',
             'Delta = normal full composition minus availability_only. Positive Dice/centerline favors normal; negative FP/FN/distances favors normal.',
             'Eight paired patients; percentile95% patient bootstrap CI, 20000 draws, seed20261022. Undefined metrics use complete pairs with support.', '']
    for row in result['paired_summaries']:
        if row['delta_a_minus_b'] is None:
            lines.append('%s: undefined; paired0/%d' % (row['metric'], row['total_cases']))
        else:
            lines.append('%s: delta %.3f %s, CI [%.3f, %.3f]; paired%d/%d; W/T/L %d/%d/%d' %
                (row['metric'], row['delta_a_minus_b'], row['difference_unit'], row['ci95_lower'], row['ci95_upper'],
                 row['paired_cases'], row['total_cases'], row['wins'], row['ties'], row['losses']))
    lines.extend(['', 'Geometry group results and all patient differences remain in CSV/JSON, including zero/unfavorable cases.', ''])
    lines.extend(result['interpretation'])
    (output / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def self_test():
    with tempfile.TemporaryDirectory(prefix='gta_avt_intervention_fixture_') as folder:
        root = Path(folder)
        env = dict(python='fixture Python3.8', python_version=[3, 8, 0],
            packages={k: 'fixture-version' for k in ('torch', 'numpy', 'scipy', 'scikit-image', 'SimpleITK')},
            backend_settings={'matmul_tf32': False, 'cudnn_tf32': False})
        env['packages']['torch'] = '1.11.0+fixture'
        tool = reviewed_intervention_source()
        protocol = dict(schema=base.REFERENCE_SCHEMA, split='val', stage='formal', case_ids=IDS,
            method='faithful_soft', selected_step=5000, final_test_eligible=True, explicitly_authorized_formal_test=False,
            weights_sha256='a' * 64, original_checkpoint_sha256='b' * 64, provenance_sha256='c' * 64,
            canonical_data_sha256='d' * 64, frozen_source_sha256={'fixture.py': 'e' * 64},
            evaluator_tool_sha256={'fixture.py': 'f' * 64}, precision='fixtureFP32',
            inference=base.FIXED_INFERENCE, selected_execution_device={'type': 'cpu', 'index': None, 'name': 'CPU'},
            autocast_enabled=False, reference_environment=env,
            training_environment_receipt={'sha256': '0' * 64, 'value': env})
        protocol['training_recipe'] = base._synthetic_recipe('faithful_soft')
        protocol['training_recipe_sha256'] = base.canonical_sha(protocol['training_recipe'])
        normal, availability = root / 'normal', root / 'availability'
        raw = {}
        for directory, intervened in ((normal, False), (availability, True)):
            directory.mkdir()
            p = dict(protocol, intervention=(dict(name='availability_only', stage='runtime-validation-diagnostic',
                separately_trained_model=False, changes_weights=False, uses_GT=False, formula=FORMULA,
                intervention_source_sha256=base.sha(tool)) if intervened else
                dict(name='none', separately_trained_model=False, changes_weights=False)))
            (directory / 'evaluation_protocol.json').write_text(json.dumps(p), encoding='utf-8')
            raw[directory.name] = []
            for i, case_id in enumerate(IDS):
                row = dict(case_id=case_id, cohort=base.cohort(case_id), fixed_threshold=.5, condition_mode='predicted',
                    evaluation_grid='original CT', postprocessing='none', dice=.7 + i * .01 - (.01 if intervened else 0.),
                    precision=.8, recall=.7, fp_ml=2. + i, fn_ml=3. + i,
                    assd_mm=None if intervened and i == 0 else 1., hd95_mm=2.,
                    centerline_recall=.7, centerline_precision=.8, cldice=.74,
                    empty_prediction=False, empty_truth=False, truth_skeleton_support=10, prediction_skeleton_support=11,
                    assd_definition='fixture pooled boundary', centerline_definition='fixture Lee voxel coverage',
                    geometry_groups={g: {'support_nodes': 0 if g == 'junction_neighbourhood' else i + 1,
                        'centerline_recall': None if g == 'junction_neighbourhood' else i / 10.} for g in base.GROUPS})
                raw[directory.name].append(row)

        def write_rows():
            for directory in (normal, availability):
                rows = raw[directory.name]
                for row in rows:
                    patient = directory / row['case_id']
                    patient.mkdir(exist_ok=True)
                    (patient / 'metrics.json').write_text(json.dumps(row), encoding='utf-8')
                summary = dict(cases=8, statistical_unit='case', metrics={})
                for name in base.METRICS:
                    values = [r.get(name) for r in rows]
                    finite = [v for v in values if v is not None]
                    summary['metrics'][name] = {'mean': float(np.mean(finite)) if len(finite) == 8 else None,
                                               'defined_case_count': len(finite)}
                (directory / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')

        write_rows()
        result = analyze(normal, availability)
        dice = next(r for r in result['paired_summaries'] if r['metric'] == 'dice')
        assert all(abs(dice[k] - 1.) < 1e-10 for k in ('delta_a_minus_b', 'ci95_lower', 'ci95_upper'))
        assert next(r for r in result['paired_summaries'] if r['metric'] == 'assd_mm')['paired_cases'] == 7
        missing = next(r for r in result['geometry_paired_summaries'] if r['group'] == 'junction_neighbourhood')
        assert missing['paired_cases'] == 0 and missing['no_gt_support_cases'] == 8 and missing['delta_a_minus_b'] is None
        save(result, root / 'temporary_output')
        for original, changed in zip(raw['normal'], raw['availability']):
            changed['dice'] = original['dice']
        write_rows()
        zero = next(r for r in analyze(normal, availability)['paired_summaries'] if r['metric'] == 'dice')
        assert zero['delta_a_minus_b'] == zero['ci95_lower'] == zero['ci95_upper'] == 0. and zero['ties'] == 8
        for row in raw['availability']:
            row['dice'] += .01
        write_rows()
        negative = next(r for r in analyze(normal, availability)['paired_summaries'] if r['metric'] == 'dice')
        assert abs(negative['delta_a_minus_b'] + 1.) < 1e-10 and negative['losses'] == 8
        original = base.read(availability / 'evaluation_protocol.json')
        recipe_failures = ('mixed_recipe_budget', 'mixed_recipe_patch', 'recipe_SHA',
                           'recipe_method', 'recipe_stage', 'recipe_seed', 'missing_formal_recipe')
        for mutation in ('weight', 'formula', 'test', 'pilot', 'runtime') + recipe_failures:
            bad = json.loads(json.dumps(original))
            if mutation == 'weight': bad['weights_sha256'] = '1' * 64
            if mutation == 'formula': bad['intervention']['formula']['scale'] = 'does_not_preserve_q'
            if mutation == 'test': bad['split'] = 'test'
            if mutation == 'pilot': bad['stage'] = 'pilot'
            if mutation == 'runtime': bad['reference_environment']['python_version'] = [3, 11, 0]
            if mutation == 'mixed_recipe_budget': bad['training_recipe']['max_steps'] = 10000
            if mutation == 'mixed_recipe_patch': bad['training_recipe']['patch_size'] = 128
            if mutation == 'recipe_method': bad['training_recipe']['method'] = 'faithful_concat'
            if mutation == 'recipe_stage': bad['training_recipe']['stage'] = 'pilot'
            if mutation == 'recipe_seed': bad['training_recipe']['seed'] += 1
            bad['training_recipe_sha256'] = base.canonical_sha(bad['training_recipe'])
            if mutation == 'recipe_SHA': bad['training_recipe_sha256'] = '1' * 64
            if mutation == 'missing_formal_recipe':
                bad.pop('training_recipe'); bad.pop('training_recipe_sha256')
            (availability / 'evaluation_protocol.json').write_text(json.dumps(bad), encoding='utf-8')
            try: analyze(normal, availability)
            except ValueError: pass
            else: raise AssertionError('Invalid same-weight intervention gate accepted: ' + mutation)
        (availability / 'evaluation_protocol.json').write_text(json.dumps(original), encoding='utf-8')
        for q in (0., .2, .7, 1.):
            for nonempty in (4, 2):
                vector = np.r_[q, np.full(nonempty, (1. - q) / nonempty)]
                assert vector[0] == q and np.isclose(vector.sum(), 1.) and (vector >= 0).all()
        summary = base.read(availability / 'summary.json')
        summary['cases'] = 7
        (availability / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        try: analyze(normal, availability)
        except ValueError as error: assert 'incomplete' in str(error)
        else: raise AssertionError('Partial evaluation was accepted')
    return dict(state='passed', fixtures=['known_Dice_pp_offset', 'identical_zero_effect', 'unfavorable_effect_visible',
        'missing_distance_support', 'unsupported_GT_group', 'same_weight_gate', 'q_preserving_formula_gate',
        'test_pilot_wrongruntime_rejected', 'q_mass_arithmetic', 'incomplete_evaluation_rejected', 'temporary_output_roundtrip'],
        recipe_fixtures=['valid_new_recipe_protocol'] + list(recipe_failures), synthetic_results_retained=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--normal', help='Completed formal main A_soft normal VAL8 reference directory')
    parser.add_argument('--availability-only', help='Same-weight availability_only VAL8 reference directory')
    parser.add_argument('--output')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        if args.normal or args.availability_only or args.output:
            parser.error('--self-test cannot be combined with actual inputs/output')
        print(json.dumps(self_test()))
        return
    if not args.normal or not args.availability_only or not args.output:
        parser.error('Specify --normal, --availability-only and a new --output directory')
    result = analyze(args.normal, args.availability_only)
    save(result, args.output)
    print(json.dumps(dict(state='complete', cases=8, split='val', same_trained_weights=True,
                         inference_or_tuning_performed=False)))


if __name__ == '__main__':
    main()
