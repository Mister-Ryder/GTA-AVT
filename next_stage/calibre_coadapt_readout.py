"""Fixed TRAIN-only oracle-calibre co-adaptation readout.

Default is inert. A short capacity screen does not replace native validation,
prove GTA efficacy or authorize any successor. Seeds are retained separately.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ARMS = ('C0', 'A_only', 'AR')
INTERVENTIONS = ('own_A_half_ring_radius',)
WORKER_SHA = '231b99fefb5b17e9e71d7b24bf9a4ba74f8cb032fd05b6ab62b8fa11ecee63f8'
MODEL_SHA = '7741821580cf6fded92b2d5033aa4649bdbab51a8f218378525c28533763c942'
ORACLE_BANK_SHA = '98c639be40fea8df7d72e9fc97835e8147c87c47fe1ce57e317f232794c2cde5'
ORACLE_SHA = '00dbec1b9230f2613a3fa90236eed9842228bcd6f6fb96f6c40f9d29eb4f86ed'
SOURCE_SHA = 'f311c9fbe6ea55be70e72ed220ab0562da08b3ee267863fc0d78509a30c3654e'
WEIGHT_SHA = '7406a3ba7e5bac1a7d8ee24898cb954ff8576f37fcf5fba0e26ef9a2cb5c43e6'
SEEDS = (20261030, 20261031, 20261032)
BANK_SHA = '9be42890017341e4cc32fb0f17688d859a13e84643f95cd21e347462043419d6'
METRICS = ('dice', 'cldice', 'centerline_recall', 'centerline_precision', 'fine_recall', 'BCE', 'FP', 'FN', 'TP')
METHOD = dict(primary='Known-nonempty GT-jitter slots0..7; patch means within case, then equal PROBE12 cases',
    fine='Sum fine_covered/sum fine_nodes within each case primary patches; then equal supported cases; patch means retained separately',
    counts='FP/FN/TP means per sampled primary patch then per case, not native volumes; overlapping predictions are not independent cases',
    utility='AR mean Dice strictly greater than BOTH C0/A_only; clDice/CLrecall/fine and own-worst2 nonworse',
    training='2000 successful updates/arm; same seed/plan/initialization, common4226 projection parameters plus192 gates; original main Dice+BCE only; no attribute/DS/GDF loss',
    mechanism='OwnA unchanged; same patch A1 flattened R values half-ring roll floor(N/2); exact support and complete R multiset retained',
    mechanism_gate='Mean correct-minus-permuted Dice>=.1pp, at least8 strictly positive cases, all12 supported',
    mechanism_support='At least2 primary known-nonempty slots, positive supported log-radius variance, actual consumed FP32 zR changes, finite measurements',
    missing='Unsupported values remain null/inconclusive, never imputed0 or used as scientific negative evidence',
    oracle='GT sparse native-EDT proxy support/radius only; geometry unknowns consume neutral0 but are not negative radius labels',
    stability='Cross-case dispersion and tails only; not test-time variance or independent cohorts',
    seeds='Three fixed seeds reported separately; PRIMARY cannot be rescued; no best seed or pooling',
    scope='Oracle-conditioned TRAIN capability candidate, not CT deployment, not a mathematical upper bound, not native validation or GTA efficacy',
    cross_round='New radius definition plus co-adaptation; differences from old rounds cannot isolate either change')



def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def lines(path):
    return [json.loads(row) for row in Path(path).read_bytes().splitlines()]


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def variance(values):
    values = list(values)
    m = mean(values)
    return mean((v - m) ** 2 for v in values) if values else None


def validate_metrics(metrics):
    expected = {'dice', 'cldice', 'centerline_recall', 'centerline_precision', 'fine_recall', 'fine_nodes',
        'fine_covered', 'GT_nodes', 'GT_covered', 'FP', 'FN', 'TP', 'BCE', 'real_CT_voxels', 'truth_foreground'}
    require(expected <= set(metrics), 'Missing pilot metrics')
    for key in expected:
        value = metrics[key]
        require(value is None or finite(value), 'Nonfinite pilot metric: ' + key)
    for key in ('dice', 'cldice', 'centerline_recall', 'centerline_precision', 'fine_recall'):
        require(metrics[key] is None or 0. <= metrics[key] <= 1., 'Invalid metric fraction: ' + key)
    for key in ('fine_nodes', 'fine_covered', 'GT_nodes', 'GT_covered', 'FP', 'FN', 'TP', 'real_CT_voxels', 'truth_foreground'):
        require(type(metrics[key]) is int and metrics[key] >= 0, 'Invalid count: ' + key)
    require(metrics['fine_covered'] <= metrics['fine_nodes'] and metrics['GT_covered'] <= metrics['GT_nodes'], 'Coverage exceeds support')


def aggregate(rows):
    result = dict(patches=len(rows))
    for name in METRICS:
        values = [r['metrics'][name] for r in rows if r['metrics'][name] is not None]
        result[name] = mean(values)
        result[name + '_defined_patches'] = len(values)
    result['fine_patch_mean_recall'] = result['fine_recall']
    result['fine_nodes'] = sum(r['metrics']['fine_nodes'] for r in rows)
    result['fine_covered'] = sum(r['metrics']['fine_covered'] for r in rows)
    result['fine_recall'] = result['fine_covered'] / result['fine_nodes'] if result['fine_nodes'] else None
    result['GT_nodes'] = sum(r['metrics']['GT_nodes'] for r in rows)
    result['GT_covered'] = sum(r['metrics']['GT_covered'] for r in rows)
    result['pooled_patch_GT_node_recall_descriptive'] = result['GT_covered'] / result['GT_nodes'] if result['GT_nodes'] else None
    return result


def case_arms(rows, case_ids):
    output = {}
    for arm in ARMS:
        per_case = {}
        for cid in case_ids:
            group = [r for r in rows[arm] if r['case_id'] == cid]
            primary = [r for r in group if r['primary']]
            per_case[cid] = dict(source=group[0]['source'], primary=aggregate(primary),
                all16=aggregate(group), GT_jitter_all8=aggregate([r for r in group if r['slot'] < 8]),
                CT_uniform_all8=aggregate([r for r in group if r['slot'] >= 8]),
                uniform_known_nonempty=aggregate([r for r in group if r['slot'] >= 8 and r['known_nonempty']]),
                uniform_empty_patches=sum(r['slot'] >= 8 and r['valid_empty'] for r in group),
                uniform_empty_FP_voxels=sum(r['metrics']['FP'] for r in group if r['slot'] >= 8 and r['valid_empty']),
                historical_patch_attribute_unknown_slots=[r['slot'] for r in group if not r['attribute_valid']],
                oracle_support=dict(all16_mean_available_sites=mean(r['oracle']['available_radius_sites'] for r in group),
                    primary_mean_available_sites=mean(r['oracle']['available_radius_sites'] for r in primary),
                    all16_mean_unknown_sites=mean(r['oracle']['unknown_sites'] for r in group)))
        summary = dict(per_case=per_case, expected_cases=len(case_ids))
        for name in METRICS:
            values = [v['primary'][name] for v in per_case.values() if v['primary'][name] is not None]
            summary[name] = mean(values); summary[name + '_supported_cases'] = len(values)
        dice = [per_case[c]['primary']['dice'] for c in case_ids if per_case[c]['primary']['dice'] is not None]
        summary['Dice_population_SD'] = math.sqrt(variance(dice)) if dice else None
        defined = sorted((c for c in case_ids if per_case[c]['primary']['dice'] is not None),
            key=lambda c: (per_case[c]['primary']['dice'], c))
        worst = defined[:2]
        summary['own_worst2'] = dict(case_ids=worst, Dice_mean=mean(per_case[c]['primary']['dice'] for c in worst) if len(worst) == 2 else None)
        summary['source_groups'] = {}
        summary['strata_case_equal'] = {}
        for stratum in ('primary', 'all16', 'GT_jitter_all8', 'CT_uniform_all8', 'uniform_known_nonempty'):
            summary['strata_case_equal'][stratum] = dict(
                metrics={key: dict(mean=mean(per_case[c][stratum][key] for c in case_ids if per_case[c][stratum][key] is not None),
                    supported_cases=sum(per_case[c][stratum][key] is not None for c in case_ids)) for key in METRICS},
                source_groups={s: {key: dict(mean=mean(per_case[c][stratum][key] for c in case_ids
                    if per_case[c]['source'] == s and per_case[c][stratum][key] is not None),
                    supported_cases=sum(per_case[c]['source'] == s and per_case[c][stratum][key] is not None for c in case_ids))
                    for key in METRICS} for s in ('Dongyang', 'KiTS', 'Rider')})
        for source in ('Dongyang', 'KiTS', 'Rider'):
            members = [c for c in case_ids if per_case[c]['source'] == source]
            summary['source_groups'][source] = dict(expected_cases=len(members),
                metrics={key: dict(mean=mean(per_case[c]['primary'][key] for c in members if per_case[c]['primary'][key] is not None),
                    supported_cases=sum(per_case[c]['primary'][key] is not None for c in members)) for key in METRICS})
        output[arm] = summary
    for reference in ARMS:
        ids = output[reference]['own_worst2']['case_ids']
        output[reference]['matched_worst2'] = dict(reference_arm=reference, case_ids=ids,
            Dice_means={arm: mean(output[arm]['per_case'][c]['primary']['dice'] for c in ids)
                if len(ids) == 2 and all(output[arm]['per_case'][c]['primary']['dice'] is not None for c in ids) else None for arm in ARMS})
    return output


def compare(arms, case_ids, control):
    result = dict(control=control, per_case={}, metrics={})
    for cid in case_ids:
        a, c = arms['AR']['per_case'][cid]['primary'], arms[control]['per_case'][cid]['primary']
        result['per_case'][cid] = {k: None if a[k] is None or c[k] is None else a[k] - c[k] for k in METRICS}
    for key in METRICS:
        ac = [cid for cid in case_ids if arms['AR']['per_case'][cid]['primary'][key] is not None]
        cc = [cid for cid in case_ids if arms[control]['per_case'][cid]['primary'][key] is not None]
        common = [cid for cid in ac if cid in cc]; delta = [result['per_case'][cid][key] for cid in common]
        result['metrics'][key] = dict(case_ids=common, AR_supported_cases=ac, control_supported_cases=cc,
            supports_identical=ac == cc, mean_AR_minus_control=mean(delta),
            mean_AR_minus_control_pp=None if not delta or key in ('FP', 'FN', 'TP', 'BCE') else 100. * mean(delta),
            positive_cases=sum(v > 0 for v in delta), negative_cases=sum(v < 0 for v in delta), ties=sum(v == 0 for v in delta))
    dice = result['metrics']['dice']
    result['mean_Dice_positive'] = (dice['mean_AR_minus_control'] > 0 if len(dice['case_ids']) == len(case_ids) else None)
    safety = {}
    for key in ('cldice', 'centerline_recall', 'fine_recall'):
        m = result['metrics'][key]
        safety[key + '_nonworse'] = None if not m['supports_identical'] or not m['case_ids'] else m['mean_AR_minus_control'] >= 0.
    at, ct = arms['AR']['own_worst2']['Dice_mean'], arms[control]['own_worst2']['Dice_mean']
    safety['own_worst2_Dice_nonworse'] = None if at is None or ct is None else at >= ct
    result['no_vascular_or_tail_harm'] = safety
    result['source_groups'] = {source: {key: dict(
        mean_AR_minus_control=mean(result['per_case'][c][key] for c in case_ids
            if c.startswith(source + '_') and result['per_case'][c][key] is not None),
        supported_cases=sum(c.startswith(source + '_') and result['per_case'][c][key] is not None for c in case_ids),
        positive_cases=sum(c.startswith(source + '_') and result['per_case'][c][key] is not None and result['per_case'][c][key] > 0. for c in case_ids))
        for key in METRICS} for source in ('Dongyang', 'KiTS', 'Rider')}
    return result


def mechanism(rows, normal_rows, case_ids):
    lookup = {(r['case_id'], r['slot']): r for r in normal_rows}
    per_case = {}
    for cid in case_ids:
        selected = [r for r in rows if r['case_id'] == cid and lookup[(cid, r['slot'])]['primary']]
        count = sum(r['radius_moments']['count'] for r in selected)
        total = math.fsum(r['radius_moments']['sum'] for r in selected)
        squares = math.fsum(r['radius_moments']['sum_squares'] for r in selected)
        radius_variance = max(0., squares / count - (total / count) ** 2) if count else None
        changed = sum(r['zR_changed'] for r in selected)
        reasons = []
        if len(selected) < 2:
            reasons.append('fewer_than_two_primary_known_nonempty_slots')
        if radius_variance is None or radius_variance <= 0.:
            reasons.append('supported_log_radius_variance_not_positive')
        if changed < 1:
            reasons.append('no_actual_consumed_primary_zR_change')
        valid = all(finite(r['normal']['dice']) and finite(r['permuted']['dice']) and
            finite(r['normal']['BCE']) and finite(r['permuted']['BCE']) and finite(r['mean_abs_logit_response']) for r in selected)
        if not valid:
            reasons.append('required_metric_undefined')
        effect = mean(100. * (r['normal']['dice'] - r['permuted']['dice']) for r in selected) if selected and valid else None
        per_case[cid] = dict(source=cid.split('_')[0], supported=not reasons, null_reasons=reasons,
            primary_patches=len(selected), radius_sites=count, raw_log_radius_variance=radius_variance,
            actual_zR_changed_patches=changed,
            correct_minus_permuted_Dice_pp=effect if not reasons else None, descriptive_Dice_pp=effect,
            permuted_minus_correct_BCE=mean(r['permuted']['BCE'] - r['normal']['BCE'] for r in selected) if selected and valid else None,
            correct_minus_permuted_FP=mean(r['normal']['FP'] - r['permuted']['FP'] for r in selected),
            correct_minus_permuted_FN=mean(r['normal']['FN'] - r['permuted']['FN'] for r in selected),
            mean_abs_logit_response=mean(r['mean_abs_logit_response'] for r in selected),
            binary_changed_voxels=sum(r['binary_changed_voxels'] for r in selected))
    supported = [c for c in case_ids if per_case[c]['supported']]
    effects = [per_case[c]['correct_minus_permuted_Dice_pp'] for c in supported]
    value = mean(effects)
    parts = dict(all12_supported=len(supported) == 12,
        mean_effect_at_least_point1pp=None if value is None else value >= .1,
        at_least8_positive=sum(v > 0. for v in effects) >= 8)
    return dict(intervention='own_A_half_ring_radius', per_case=per_case,
        supported_case_ids=supported, mean_correct_minus_permuted_Dice_pp=value,
        positive_cases=sum(v > 0. for v in effects), parts=parts,
        passed=None if len(supported) != 12 else all(parts.values()),
        source_groups={s: dict(expected_cases=sum(c.startswith(s + '_') for c in case_ids),
            supported_cases=sum(c.startswith(s + '_') for c in supported),
            mean_Dice_effect_pp=mean(per_case[c]['correct_minus_permuted_Dice_pp'] for c in supported if c.startswith(s + '_')))
            for s in ('Dongyang', 'KiTS', 'Rider')},
        own_A_and_radius_multiset_preserved=True)


def metadata(root, summary, bank):
    import importlib.util
    require(summary.get('schema') == 'GTA-AVT-calibre-coadapt-pilot-v1' and summary.get('status') == 'verified_complete' and
        summary.get('mode') == 'run' and summary.get('seed') in SEEDS and summary.get('worker_sha256') == WORKER_SHA and
        summary.get('model_sha256') == MODEL_SHA and summary.get('oracle_bank_sha256') == ORACLE_BANK_SHA and
        summary.get('fit_cases') == bank['fit_cases'] and summary.get('probe_cases') == bank['probe_cases'],
        'Completed actual2000-update oracle candidate required, not QA')
    required = ['intent.json', 'inputs.json', 'normalization.json', 'training_plan.json']
    required += [arm + '/' + name for arm in ARMS for name in ('training.jsonl', 'patches.jsonl', 'analysis.json')]
    required += ['AR/mechanism.jsonl', 'AR/mechanism_analysis.json']
    for name in required:
        item = summary['files'][name]; path = root / name
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Collected metadata bytes differ: ' + name)
    inputs, intent = read(root / 'inputs.json'), read(root / 'intent.json')
    binding = dict(source_manifest_sha256=SOURCE_SHA, bank_sha256=BANK_SHA, common_weights_sha256=WEIGHT_SHA,
        model_sha256=MODEL_SHA, oracle_bank_sha256=ORACLE_BANK_SHA, oracle_helper_sha256=ORACLE_SHA)
    require(all(inputs.get(k) == v for k, v in binding.items()) and inputs['cache_before'] == inputs['cache_after'] and
        inputs['no_VAL_or_TEST_arrays'] is True and set(inputs['oracle_files']) == set(bank['case_ids']) and
        type(inputs['initial_all_arm_state_sha256']) is str and len(inputs['initial_all_arm_state_sha256']) == 64,
        'Source/oracle/data/initialization identity differs')
    active = {'structure_guidance.morph_fusion.weight', 'structure_guidance.morph_fusion.bias',
        'anatomical_output_fusion.3.weight', 'anatomical_output_fusion.3.bias',
        'calibre_bias', 'calibre_availability_weight', 'calibre_radius_weight'}
    require(set(inputs['trainable_parameter_names']) == active, 'Co-adaptation inventory differs')
    worker_path = Path(__file__).with_name('calibre_coadapt_pilot.py')
    require(sha(worker_path) == WORKER_SHA, 'Executed worker recipe source changed')
    spec = importlib.util.spec_from_file_location('_calibre_fixed_recipe', worker_path)
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    require(intent['recipe'] == worker.RECIPE and intent['worker_sha256'] == WORKER_SHA and
        intent['model_sha256'] == MODEL_SHA and intent['oracle_bank_sha256'] == ORACLE_BANK_SHA and
        intent['oracle_helper_sha256'] == ORACLE_SHA and intent['mode'] == 'run' and intent['seed'] == summary['seed'] and
        intent['engineering_only'] is False, 'Formal recipe or oracle identity differs')
    normalization = read(root / 'normalization.json')
    require(normalization['case_equal'] is True and normalization['fit_cases'] == bank['fit_cases'] and
        finite(normalization['mean']) and finite(normalization['population_std']) and normalization['population_std'] > 0.,
        'Fixed case-equal FIT24 normalization required')
    import numpy as np
    plan = worker.make_plan(np, summary['seed'], bank['fit_cases'])
    plan_sha = hashlib.sha256(worker.encoded(plan)).hexdigest()
    actual_plan = read(root / 'training_plan.json')
    require(actual_plan['sha256'] == inputs['training_plan_sha256'] == plan_sha and
        actual_plan['common_to_all_three_arms'] is True and worker.encoded(actual_plan['plan']) == worker.encoded(plan),
        'Same fixed2000-step sampling plan required')
    for arm in ARMS:
        updates = lines(root / arm / 'training.jsonl'); analysis = read(root / arm / 'analysis.json')
        require([r['step'] for r in updates] == list(range(1, 2001)) and
            all(finite(r['loss']) and r['alpha'] == min(1., (i + 1) / 250.) for i, r in enumerate(updates)) and
            analysis['arm'] == arm and analysis['successful_updates'] == 2000 and analysis['frozen_state_unchanged'] is True,
            'Successful training budget, alpha schedule or frozen-state receipt differs: ' + arm)
    return {name: sha(root / name) for name in required}


def load_seed(root, case_ids, bank=None):
    require(bank is not None, 'Original bank required for provenance validation')
    root = Path(root); summary = read(root / 'summary.json')
    hashes = metadata(root, summary, bank); hashes['summary.json'] = sha(root / 'summary.json')
    rows = {}; expected = [(cid, slot) for cid in case_ids for slot in range(16)]
    for arm in ARMS:
        data = lines(root / arm / 'patches.jsonl')
        require([(r['case_id'], r['slot']) for r in data] == expected, 'Complete original192 patch order required')
        for row in data:
            require(row['source'] == row['case_id'].split('_')[0] and row['stratum'] ==
                ('GT_centerline_plus_jitter' if row['slot'] < 8 else 'CT_uniform') and
                all(type(row[k]) is bool for k in ('known_nonempty', 'attribute_valid', 'valid_empty', 'primary')) and
                row['primary'] == (row['slot'] < 8 and row['known_nonempty']), 'Patch population identity differs')
            validate_metrics(row['metrics'])
            coverage = row['oracle']
            require(type(coverage['available_radius_sites']) is int and type(coverage['unknown_sites']) is int and
                0 <= coverage['available_radius_sites'] <= 64 ** 3 and
                coverage['available_radius_sites'] + coverage['unknown_sites'] == 64 ** 3,
                'Oracle support/unknown population differs')
        rows[arm] = data
    for arm in ('A_only', 'AR'):
        for before, after in zip(rows['C0'], rows[arm]):
            require(all(before[k] == after[k] for k in ('case_id', 'slot', 'stratum', 'known_nonempty', 'attribute_valid', 'valid_empty', 'primary', 'oracle')) and
                all(before['metrics'][k] == after['metrics'][k] for k in ('fine_nodes', 'GT_nodes', 'truth_foreground', 'real_CT_voxels')),
                'Arms do not use identical geometry/foreground/oracle supports')
    interventions = lines(root / 'AR' / 'mechanism.jsonl')
    require([(r['case_id'], r['slot']) for r in interventions] == expected, 'Complete192 radius-correspondence rows required')
    for row, normal in zip(interventions, rows['AR']):
        require(row['normal'] == normal['metrics'] and all(row[k] == normal[k] for k in ('case_id', 'source', 'slot', 'stratum', 'primary')),
            'Mechanism normal prediction differs')
        validate_metrics(row['permuted'])
        require(all(row['permuted'][k] == normal['metrics'][k] for k in ('fine_nodes', 'GT_nodes', 'truth_foreground', 'real_CT_voxels')),
            'Permuted endpoint uses different truth support')
        permutation, moments = row['permutation'], row['radius_moments']
        count = permutation['eligible_sites']
        require(type(count) is int and count == moments['count'] == normal['oracle']['available_radius_sites'] and
            permutation['shift'] == count // 2 and permutation['own_A_and_support_preserved'] is True and
            permutation['correct_radius_multiset_sha256'] == permutation['permuted_radius_multiset_sha256'] and
            type(permutation['changed_radius_sites']) is int and 0 <= permutation['changed_radius_sites'] <= count and
            type(row['zR_changed']) is bool and type(row['binary_changed_voxels']) is int and
            0 <= row['binary_changed_voxels'] <= 128 ** 3 and finite(row['mean_abs_logit_response']) and
            row['mean_abs_logit_response'] >= 0. and finite(moments['sum']) and finite(moments['sum_squares']) and moments['sum_squares'] >= 0.,
            'Half-ring intervention/support/multiset or actual-response identity differs')
    return summary, rows, interventions, hashes


def analyze_seed(summary, rows, interventions, case_ids):
    arms = case_arms(rows, case_ids)
    comparisons = {control: compare(arms, case_ids, control) for control in ('C0', 'A_only')}
    mechanisms = {'own_A_half_ring_radius': mechanism(interventions, rows['AR'], case_ids)}
    tests = {name + '_AR_mean_Dice_positive': value['mean_Dice_positive'] for name, value in comparisons.items()}
    for name, value in comparisons.items():
        tests.update({name + '_' + k: v for k, v in value['no_vascular_or_tail_harm'].items()})
    tests.update({name + '_sameweight_gate': value['passed'] for name, value in mechanisms.items()})
    passed = None if any(v is None for v in tests.values()) else all(tests.values())
    return dict(seed=summary['seed'], arms=arms, AR_vs_controls=comparisons, mechanisms=mechanisms,
        screen=dict(parts=tests, passed=passed, status='inconclusive_support' if passed is None else 'positive_capacity_screen' if passed else 'screen_not_passed'),
        native_validation=False, GTA_goal_completed=False, training_authorized=False)


def analyze(roots, bank_path):
    require(len(roots) == 3 and len({str(Path(p).resolve()) for p in roots}) == 3, 'Three predeclared distinct seed outputs required')
    require(sha(bank_path) == BANK_SHA, 'Original frozen TRAIN partition required')
    bank = read(bank_path); ids = bank['probe_cases']
    require(len(ids) == 12 and len(set(ids)) == 12 and len(bank['fit_cases']) == 24 and set(ids).isdisjoint(bank['fit_cases']), 'Original24/12 partition required')
    seeds, provenance = [], []
    for root in roots:
        summary, rows, interventions, hashes = load_seed(root, ids, bank)
        seeds.append(analyze_seed(summary, rows, interventions, ids)); provenance.append(hashes)
    require(tuple(r['seed'] for r in seeds) == SEEDS, 'Fixed PRIMARY and replica seed order required')
    require(len({p['normalization.json'] for p in provenance}) == 1, 'Three seeds must use identical normalization bytes')
    primary = seeds[0]['screen']['passed']; flags = [r['screen']['passed'] for r in seeds]
    whole = None if any(v is None for v in flags) else all(flags)
    return dict(schema='GTA-AVT-calibre-coadapt-final-readout-v1', method=METHOD, bank_sha256=BANK_SHA,
        readout_sha256=sha(__file__), probe_cases=ids, per_seed=seeds, input_hashes=provenance,
        primary_seed=seeds[0]['seed'], primary_screen_passed=primary, all3_screens_passed=whole,
        primary_failure_cannot_be_rescued=primary is not True, seeds_pooled=False,
        oracle_conditioned_not_CT_only=True, mathematical_or_true_upper_bound=False,
        no_attribute_auxiliary_loss=True, all_three_seeds_retained=True,
        next_experiment_authorized=False, native_confirmation_required=True, GTA_goal_completed=False)


def render(result):
    lines = ['局部半径GT oracle与有限共适配能力筛查；不是CT部署、真正上界或native确认。']
    for seed in result['per_seed']:
        lines.append('Seed %s：screen=%s。' % (seed['seed'], seed['screen']['status']))
        for control, value in seed['AR_vs_controls'].items():
            dice = value['metrics']['dice']
            lines.append(' AR−%s Dice=%s pp；正病例%s/%s。' % (control, dice['mean_AR_minus_control_pp'], dice['positive_cases'], len(dice['case_ids'])))
        for name, value in seed['mechanisms'].items():
            lines.append(' %s：correct−donor=%s pp；positive=%s，supported=%s/12，gate=%s。' % (name,
                value['mean_correct_minus_permuted_Dice_pp'], value['positive_cases'], len(value['supported_case_ids']), value['passed']))
    lines.append('三个seed单独保留；不互救、不自动启动后续。GT支持与半径已直接给定，且共享投影有限共适配；不能写成CT条件化GTA有效验证或真正性能上界。')
    return '\n'.join(lines) + '\n'


def execute(args):
    output = Path(args.output); require(not output.exists(), 'Fresh output required')
    result = analyze(args.inputs, args.bank); payload = encoded(result)
    output.mkdir(parents=True); (output / 'analysis.json').write_bytes(payload)
    (output / 'readout.txt').write_bytes(render(result).encode('utf-8'))
    case_rows = []
    for seed in result['per_seed']:
        for arm, value in seed['arms'].items():
            for cid, case in value['per_case'].items():
                case_rows.append(dict(seed=seed['seed'], arm=arm, case_id=cid, source=case['source'], **case['primary']))
    with (output / 'casewise.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0])); writer.writeheader(); writer.writerows(case_rows)
    mechanisms = [dict(seed=seed['seed'], case_id=cid, **value) for seed in result['per_seed']
        for cid, value in seed['mechanisms']['own_A_half_ring_radius']['per_case'].items()]
    with (output / 'mechanism_cases.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mechanisms[0])); writer.writeheader(); writer.writerows(mechanisms)
    return dict(status='complete', analysis_sha256=sha(output / 'analysis.json'), next_experiment_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--inputs', nargs=3)
    parser.add_argument('--bank'); parser.add_argument('--output'); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='inert', method=METHOD, files_read=False, writes=False))); return
    require(args.inputs and args.bank and args.output, 'Explicit completed seed outputs, original bank and fresh output required')
    print(json.dumps(execute(args)))


if __name__ == '__main__':
    main()
