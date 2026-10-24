"""Completed block-mean TRAIN oracle readout; default inert, CPU only.

The new AR has 2000 updates. C0/A_only are the explicitly reused, unchanged
same-seed 2000-update controls, not new independent repeats. Pure segmentation
aggregation and utility/uniform guards reuse the original fixed basis math.
Only the correspondence estimator changes to same-case block-scalar pairing.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import struct


BASE_SHA = '1f7fe250df11b96f831850c9eccafa5a4ba3e51d0b3dd224cea9953d95542d44'
WORKER_SHA = '622decef7578eac151f9b64e0ed71bf7fb4622d24c7520a09832ecf0dfabc7e1'
REUSED_WORKER_SHA = 'a9320edc72b1dcde9fe4486355566bf9c491b78d109bf5315f578e08db32c9ef'
MODEL_SHA = '9acd1c9d8b04fdef3e5a7e4cefe9d8f0d234978093eaf09d7d2a03a64288f58e'
BANK_SHA = '9be42890017341e4cc32fb0f17688d859a13e84643f95cd21e347462043419d6'
SEEDS = (20261030, 20261031, 20261032)
ARMS = ('C0', 'A_only', 'AR')
SOURCES = ('Dongyang', 'KiTS', 'Rider')
STRATA = ('primary', 'all16', 'GT_jitter_all8', 'CT_uniform_all8', 'uniform_known_nonempty')
COEFFICIENTS = ('calibre_bias', 'calibre_availability_weight', 'calibre_radius_weight')
INTERVENTION = 'own_A_same_case_primary_blockmean_shift4'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def lines(path):
    return [json.loads(row) for row in Path(path).read_bytes().splitlines()]


def module(path, digest, name):
    require(sha(path) == digest, 'Pinned source differs: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def combine(values):
    values = list(values)
    return None if any(v is None for v in values) else all(values)


def active_names():
    return {'structure_guidance.morph_fusion.weight', 'structure_guidance.morph_fusion.bias',
        'anatomical_output_fusion.3.weight', 'anatomical_output_fusion.3.bias'} | set(COEFFICIENTS) | {
        'structure_guidance.morphology_convs.%s.%s' % (i, key)
        for i in range(4) for key in ('weight', 'bias')}


def arm_home(root, arm):
    return root / arm if arm == 'AR' else root / 'reused_controls' / arm


def metadata(root, summary, bank, base):
    """Read completed compact artifacts, never weights or old AR endpoints."""
    require(summary.get('schema') == 'GTA-AVT-calibre-blockmean-coadapt-pilot-v1' and
        summary.get('status') == 'verified_complete' and summary.get('mode') == 'run' and
        summary.get('seed') in SEEDS and summary.get('worker_sha256') == WORKER_SHA and
        summary.get('model_sha256') == MODEL_SHA and summary.get('oracle_bank_sha256') == base.ORACLE_BANK_SHA and
        summary.get('fit_cases') == bank['fit_cases'] and summary.get('probe_cases') == bank['probe_cases'] and
        summary.get('controls_reused_not_retrained') is True and summary.get('original_control_updates') == 2000 and
        summary.get('new_control_updates') == 0, 'Completed formal block-mean run required, not QA or fresh control training')
    receipt = read(root.parent / 'collection.json')
    child = receipt.get('child', {})
    require(receipt.get('terminal', {}).get('exit_code') == 0 and
        (not child.get('exists') or child.get('state') in ('Z', 'X')), 'Natural completion and retired child required')
    required = ['intent.json', 'inputs.json', 'normalization.json', 'training_plan.json', 'reused_controls.json',
        'analysis.json'] + [str(arm_home(Path('.'), a) / n).replace('\\', '/') for a in ARMS
        for n in ('training.jsonl', 'patches.jsonl', 'analysis.json')] + ['AR/mechanism.jsonl', 'AR/mechanism_analysis.json']
    for name in ['summary.json'] + required:
        item = receipt['files'][name]
        require((root / name).stat().st_size == item['bytes'] and sha(root / name) == item['sha256'],
            'Collected artifact bytes differ: ' + name)
        if name != 'summary.json':
            require(summary['files'][name] == item, 'Completed summary binding differs: ' + name)
    inputs, intent = read(root / 'inputs.json'), read(root / 'intent.json')
    bindings = dict(source_manifest_sha256=base.SOURCE_SHA, bank_sha256=BANK_SHA,
        common_weights_sha256=base.WEIGHT_SHA, model_sha256=MODEL_SHA,
        oracle_bank_sha256=base.ORACLE_BANK_SHA, oracle_helper_sha256=base.ORACLE_SHA)
    require(all(inputs.get(k) == v for k, v in bindings.items()) and inputs['cache_before'] == inputs['cache_after'] and
        inputs['no_VAL_or_TEST_arrays'] is True and set(inputs['oracle_files']) == set(bank['case_ids']) and
        isinstance(inputs['initial_all_arm_state_sha256'], str) and len(inputs['initial_all_arm_state_sha256']) == 64 and
        set(inputs['trainable_parameter_names']) == active_names(), 'Fixed source/data/common-init/15-tensor identity differs')
    worker = module(Path(__file__).with_name('calibre_blockmean_pilot_reused.py'), WORKER_SHA, '_blockmean_executed_recipe')
    require(intent['schema'] == 'GTA-AVT-calibre-blockmean-coadapt-pilot-intent-v1' and intent['mode'] == 'run' and
        intent['seed'] == summary['seed'] and intent['recipe'] == worker.RECIPE and intent['worker_sha256'] == WORKER_SHA and
        intent['model_sha256'] == MODEL_SHA and intent['oracle_bank_sha256'] == base.ORACLE_BANK_SHA and
        intent['oracle_helper_sha256'] == base.ORACLE_SHA and intent['engineering_only'] is False and
        worker.RECIPE['trainable_parameters'] == 115458 and worker.RECIPE['shared_parameters'] == 114882 and
        worker.RECIPE['gate_parameters'] == 576 and worker.RECIPE['formal_new_arms'] == ['AR'] and
        worker.RECIPE['mechanism_gates'] == dict(minimum_mean_Dice_pp=.1, minimum_positive_cases=8, all12_supported=True),
        'Executed recipe, estimator or fixed gates differ')
    require(sha(Path(__file__).with_name('calibre_routing_stage_model.py')) == MODEL_SHA, 'Original model source differs')
    norm = read(root / 'normalization.json')
    require(norm['case_equal'] is True and norm['fit_cases'] == bank['fit_cases'] and norm['mean'] == 2.068768924624306 and
        norm['population_std'] == .5812830275631903, 'Original point-level FIT24 moments required')
    import numpy as np
    plan = worker.make_plan(np, summary['seed'], bank['fit_cases'])
    actual = read(root / 'training_plan.json')
    require(actual['sha256'] == inputs['training_plan_sha256'] == hashlib.sha256(worker.encoded(plan)).hexdigest() and
        actual['common_to_all_three_arms'] is True and worker.encoded(actual['plan']) == worker.encoded(plan),
        'Original common2000-step plan differs')
    reuse = read(root / 'reused_controls.json')
    original_names = {'intent.json', 'inputs.json', 'normalization.json', 'training_plan.json', 'summary.json'} | {
        a + '/' + n for a in ('C0', 'A_only') for n in ('analysis.json', 'patches.jsonl', 'training.jsonl')}
    copied_names = {a + '/' + n for a in ('C0', 'A_only') for n in ('analysis.json', 'patches.jsonl', 'training.jsonl')}
    require(reuse['reused_not_retrained'] is True and reuse['independent_new_replicas'] is False and
        reuse['old_worker_sha256'] == REUSED_WORKER_SHA and reuse['unchanged_model_sha256'] == MODEL_SHA and
        reuse['seed'] == summary['seed'] and reuse['original_successful_updates_per_control'] == 2000 and
        reuse['new_updates_per_control'] == 0 and reuse['no_old_AR_weights_or_optimizer_read'] is True and
        set(reuse['proof_sha256']) == original_names and set(reuse['copied_control_file_sha256']) == copied_names,
        'Fixed reused-control provenance/budget differs')
    for name in copied_names:
        require(sha(root / 'reused_controls' / name) == reuse['copied_control_file_sha256'][name] == reuse['proof_sha256'][name],
            'Reused control content differs: ' + name)
    require(reuse['proof_sha256']['normalization.json'] == sha(root / 'normalization.json') and
        reuse['proof_sha256']['training_plan.json'] == sha(root / 'training_plan.json'), 'Reused moments or plan bytes differ')
    for arm in ARMS:
        home = arm_home(root, arm)
        updates, result = lines(home / 'training.jsonl'), read(home / 'analysis.json')
        schema = 'GTA-AVT-calibre-' + ('blockmean' if arm == 'AR' else 'routing') + '-coadapt-arm-v1'
        require(result['schema'] == schema and result['arm'] == arm and result['successful_updates'] == 2000 and
            result['frozen_state_unchanged'] is True and [u['step'] for u in updates] == list(range(1, 2001)) and
            all(base.finite(u['loss']) and u['alpha'] == min(1., (i + 1) / 250.) for i, u in enumerate(updates)),
            'Formal control/newAR budget, alpha or frozen state differs')
        for update in updates:
            require(set(update['gradient_maxabs']) == active_names() and
                all(v is None or base.finite(v) and v >= 0. for v in update['gradient_maxabs'].values()) and
                set(update['coefficient_row_gradient_maxabs']) == set(COEFFICIENTS) and
                all(len(update['coefficient_row_gradient_maxabs'][k]) == 3 and
                    all(v is None or base.finite(v) and v >= 0. for v in update['coefficient_row_gradient_maxabs'][k])
                    for k in COEFFICIENTS) and [a['axis'] for a in update['fields']['routing_axes']] == [1, 2, 3],
                'Runtime15-tensor/three-axis records differ')
    return {name: sha(root / name) for name in ['summary.json'] + required}


def validate_rows(rows, ids, arm, base):
    require([(r['case_id'], r['slot']) for r in rows] == [(c, s) for c in ids for s in range(16)],
        'Original complete192 patch order required')
    for row in rows:
        require(row['source'] == row['case_id'].split('_')[0] and row['stratum'] ==
            ('GT_centerline_plus_jitter' if row['slot'] < 8 else 'CT_uniform') and
            all(type(row[k]) is bool for k in ('known_nonempty', 'attribute_valid', 'valid_empty', 'primary')) and
            row['primary'] == (row['slot'] < 8 and row['known_nonempty']), 'Original patch population differs')
        base.validate_metrics(row['metrics'])
        coverage = row['oracle']; count, unknown = coverage['available_radius_sites'], coverage['unknown_sites']
        require(type(count) is int and type(unknown) is int and 0 <= count <= 64 ** 3 and count + unknown == 64 ** 3,
            'Original oracle support/unknown population differs')
        if arm == 'AR':
            m64, m32 = coverage['mean_logR_FP64'], coverage['mean_logR_FP32']
            require(coverage['supplied_logR_is_block_mean'] is True and
                ((count == 0 and m64 is None and m32 is None) or
                 (count > 0 and base.finite(m64) and base.finite(m32) and
                  struct.unpack('f', struct.pack('f', m64))[0] == m32)), 'Supported F64-to-F32 block mean differs')


def validate_interventions(interventions, normals, ids, base):
    require([(r['case_id'], r['slot']) for r in interventions] == [(c, s) for c in ids for s in range(8)],
        'Only original primary96 mechanism rows required; no invented uniform mechanism')
    lookup = {(r['case_id'], r['slot']): r for r in normals}
    for row in interventions:
        own = lookup[(row['case_id'], row['slot'])]
        donor = lookup[(row['case_id'], (row['slot'] + 4) % 8)]
        require(row['normal'] == own['metrics'] and
            all(row[k] == own[k] for k in ('case_id', 'source', 'slot', 'stratum', 'primary')), 'Mechanism normal endpoint differs')
        base.validate_metrics(row['permuted'])
        require(all(row['permuted'][k] == own['metrics'][k] for k in
            ('fine_nodes', 'GT_nodes', 'truth_foreground', 'real_CT_voxels')), 'Donor endpoint changed truth support')
        p, oc, dc = row['permutation'], own['oracle'], donor['oracle']
        require(p['intervention'] == 'same_case_primary_blockmean_shift4' and p['recipient_slot'] == row['slot'] and
            p['donor_slot'] == (row['slot'] + 4) % 8 and p['recipient_supported_sites'] == oc['available_radius_sites'] and
            p['donor_supported_sites'] == dc['available_radius_sites'] and
            p['recipient_mean_logR_FP64'] == oc['mean_logR_FP64'] and p['recipient_mean_logR_FP32'] == oc['mean_logR_FP32'] and
            p['donor_mean_logR_FP64'] == dc['mean_logR_FP64'] and p['donor_mean_logR_FP32'] == dc['mean_logR_FP32'] and
            type(p['scalar_changed']) is bool and p['scalar_changed'] == (oc['mean_logR_FP32'] != dc['mean_logR_FP32']) and
            p['changed_radius_sites'] == (oc['available_radius_sites'] if p['scalar_changed'] else 0) and
            isinstance(p['own_A_sha256'], str) and len(p['own_A_sha256']) == 64 and
            p['own_A_and_support_preserved'] is True and p['block_scalar_multiset_preserved'] is True and
            p['site_weighted_radius_multiset_preserved'] is False and type(row['zR_changed']) is bool and
            type(row['binary_changed_voxels']) is int and 0 <= row['binary_changed_voxels'] <= 128 ** 3 and
            base.finite(row['mean_abs_logit_response']) and row['mean_abs_logit_response'] >= 0.,
            'Block donor/support/scalar/actual-response identity differs')


def load_seed(root, ids, bank, base):
    root = Path(root); summary = read(root / 'summary.json')
    hashes = metadata(root, summary, bank, base)
    rows = {arm: lines(arm_home(root, arm) / 'patches.jsonl') for arm in ARMS}
    for arm in ARMS:
        validate_rows(rows[arm], ids, arm, base)
    for arm in ('A_only', 'AR'):
        for before, after in zip(rows['C0'], rows[arm]):
            require(all(before[k] == after[k] for k in
                ('case_id', 'slot', 'stratum', 'known_nonempty', 'attribute_valid', 'valid_empty', 'primary')) and
                all(before['oracle'][k] == after['oracle'][k] for k in ('available_radius_sites', 'unknown_sites')) and
                all(before['metrics'][k] == after['metrics'][k] for k in
                    ('fine_nodes', 'GT_nodes', 'truth_foreground', 'real_CT_voxels')), 'Arms changed original geometry/metric support')
    interventions = lines(root / 'AR' / 'mechanism.jsonl')
    validate_interventions(interventions, rows['AR'], ids, base)
    return summary, rows, interventions, hashes


def mechanism(rows, normal_rows, ids, base):
    lookup = {(r['case_id'], r['slot']): r for r in normal_rows}
    cases = {}
    for cid in ids:
        all8 = [r for r in rows if r['case_id'] == cid]
        selected = [r for r in all8 if lookup[(cid, r['slot'])]['primary']]
        joint = [r for r in selected if r['permutation']['scalar_changed'] and r['zR_changed'] and
            r['mean_abs_logit_response'] > 0.]
        reasons = []
        if len(all8) != 8 or any(r['permutation']['recipient_supported_sites'] <= 0 or
                r['permutation']['donor_supported_sites'] <= 0 for r in all8):
            reasons.append('not_all_original_eight_blocks_and_donors_supported')
        if len(selected) < 2:
            reasons.append('fewer_than_two_primary_known_nonempty_slots')
        if not joint:
            reasons.append('no_joint_scalar_actual_FP32_zR_and_nonzero_logit_change')
        valid = bool(selected) and all(base.finite(r['normal']['dice']) and base.finite(r['permuted']['dice']) and
            base.finite(r['normal']['BCE']) and base.finite(r['permuted']['BCE']) and
            base.finite(r['mean_abs_logit_response']) for r in selected)
        if not valid:
            reasons.append('required_metric_undefined')
        correct = base.aggregate([dict(metrics=r['normal']) for r in selected])
        donor = base.aggregate([dict(metrics=r['permuted']) for r in selected])
        effects = {key: None if correct[key] is None or donor[key] is None else correct[key] - donor[key]
            for key in base.METRICS}
        effect = base.mean(100. * (r['normal']['dice'] - r['permuted']['dice']) for r in selected) if valid else None
        cases[cid] = dict(source=cid.split('_')[0], supported=not reasons, null_reasons=reasons,
            primary_patches=len(selected), original_eight_supported_blocks=len(all8) == 8 and all(
                r['permutation']['recipient_supported_sites'] > 0 and r['permutation']['donor_supported_sites'] > 0 for r in all8),
            different_primary_block_scalars=sum(r['permutation']['scalar_changed'] for r in selected),
            actual_zR_changed_primary_patches=sum(r['zR_changed'] for r in selected),
            jointly_informative_primary_patches=len(joint), joint_slots=[r['slot'] for r in joint],
            correct_minus_permuted_Dice_pp=effect if not reasons else None, descriptive_Dice_pp=effect,
            correct=correct, donor=donor, correct_minus_donor=effects,
            own_block_scalar_FP32=[r['permutation']['recipient_mean_logR_FP32'] for r in all8],
            donor_block_scalar_FP32=[r['permutation']['donor_mean_logR_FP32'] for r in all8],
            mean_abs_logit_response=base.mean(r['mean_abs_logit_response'] for r in selected),
            binary_changed_voxels=sum(r['binary_changed_voxels'] for r in selected))
    supported = [c for c in ids if cases[c]['supported']]
    values = [cases[c]['correct_minus_permuted_Dice_pp'] for c in supported]
    value = base.mean(values) if len(supported) == 12 else None
    parts = dict(all12_supported=len(supported) == 12,
        mean_effect_at_least_point1pp=None if value is None else value >= .1,
        at_least8_positive=sum(v > 0. for v in values) >= 8)
    source_groups = {}
    for source in SOURCES:
        members = [c for c in ids if c.startswith(source + '_')]
        source_groups[source] = dict(expected_case_ids=members, supported_case_ids=[c for c in members if c in supported],
            metrics={key: dict(mean_correct_minus_donor=base.mean(cases[c]['correct_minus_donor'][key] for c in members
                if cases[c]['correct_minus_donor'][key] is not None),
                supported_cases=sum(cases[c]['correct_minus_donor'][key] is not None for c in members), descriptive_only=True)
                for key in base.METRICS})
        for key, value_record in source_groups[source]['metrics'].items():
            value_record['correspondence_supported_cases'] = sum(c in supported and
                cases[c]['correct_minus_donor'][key] is not None for c in members)
    stability = {}
    for name in ('correct', 'donor'):
        defined = [c for c in ids if cases[c][name]['dice'] is not None]
        dice = [cases[c][name]['dice'] for c in defined]
        worst = sorted(defined, key=lambda c: (cases[c][name]['dice'], c))[:2]
        stability[name] = dict(
            metrics={key: dict(mean=base.mean(cases[c][name][key] for c in ids if cases[c][name][key] is not None),
                defined_cases=sum(cases[c][name][key] is not None for c in ids)) for key in base.METRICS},
            Dice_population_SD=base.variance(dice) ** .5 if dice else None,
            own_worst2=dict(case_ids=worst, Dice_mean=base.mean(cases[c][name]['dice'] for c in worst) if len(worst) == 2 else None))
    for name in ('correct', 'donor'):
        tail = stability[name]['own_worst2']['case_ids']
        stability[name]['matched_worst2'] = dict(reference_variant=name, case_ids=tail,
            Dice_means={other: base.mean(cases[c][other]['dice'] for c in tail)
                if len(tail) == 2 and all(cases[c][other]['dice'] is not None for c in tail) else None
                for other in ('correct', 'donor')})
    return dict(intervention=INTERVENTION, per_case=cases, supported_case_ids=supported,
        mean_correct_minus_permuted_Dice_pp=value, descriptive_mean_defined_Dice_pp=base.mean(values),
        positive_cases=sum(v > 0. for v in values), parts=parts,
        passed=None if len(supported) != 12 else all(parts.values()), source_groups=source_groups,
        own_A_and_support_preserved=True, block_scalar_multiset_preserved=True,
        site_weighted_radius_multiset_preserved=False, within_block_radius_variance_used_for_support=False,
        uniform_mechanism='not_evaluated', primary_variant_stability=stability,
        stability_descriptive_not_additional_gate=True)


def SOURCE_safety(record, ids, base):
    groups = {}
    for source in SOURCES:
        expected = [c for c in ids if c.startswith(source + '_')]
        supported = [c for c in expected if record['per_case'][c]['supported'] and
            record['per_case'][c]['correct_minus_permuted_Dice_pp'] is not None]
        complete = len(expected) == 4 and supported == expected
        effect = base.mean(record['per_case'][c]['correct_minus_permuted_Dice_pp'] for c in expected) if complete else None
        groups[source] = dict(expected_case_ids=expected, supported_case_ids=supported,
            complete_unchanged_support=complete, case_equal_correct_minus_donor_blockmean_Dice_pp=effect,
            passed=complete and effect >= 0., descriptive_metrics=record['source_groups'][source]['metrics'])
    return dict(scope='Sameweight primary own block mean minus fixed samecase donor block mean; original4 cases/source',
        source_groups=groups, passed=all(g['passed'] for g in groups.values()),
        no_source_reclassification_or_removal=True, source_vascular_metrics_descriptive_only=True)


def analyze_seed(summary, rows, interventions, ids, base):
    arms = base.case_arms(rows, ids)
    comparisons = {control: base.compare(arms, ids, control) for control in ('C0', 'A_only')}
    record = mechanism(interventions, rows['AR'], ids, base)
    tests = {name + '_AR_mean_Dice_positive': value['mean_Dice_positive'] for name, value in comparisons.items()}
    for name, value in comparisons.items():
        tests.update({name + '_' + key: v for key, v in value['no_vascular_or_tail_harm'].items()})
    tests[INTERVENTION + '_sameweight_gate'] = record['passed']
    original = dict(parts=dict(tests), passed=combine(tests.values()))
    uniform = base.CT_uniform_safety(arms, ids)
    for control, comparison in uniform['comparisons'].items():
        tests.update({'CT_uniform_' + control + '_' + key + '_nonworse': value['passed']
            for key, value in comparison['metrics'].items()})
    primary_uniform = dict(parts=dict(tests), passed=combine(tests.values()))
    source = SOURCE_safety(record, ids, base)
    tests.update({'SOURCE_' + name + '_sameweight_nonnegative': v['passed']
        for name, v in source['source_groups'].items()})
    passed = combine(tests.values())
    return dict(seed=summary['seed'], arms=arms, AR_vs_controls=comparisons, mechanisms={INTERVENTION: record},
        original_primary_screen=original, original_primary_screen_passed=original['passed'],
        CT_uniform_transfer_safety=uniform, primary_and_uniform_screen=primary_uniform,
        SOURCE_safety=source, capacity_continue=passed,
        screen=dict(parts=tests, passed=passed, status='inconclusive_support' if passed is None else
            'positive_capacity_uniform_source_screen' if passed else 'screen_not_passed'),
        controls_reused_not_retrained=True, original_successful_control_updates=2000,
        new_control_updates=0, new_AR_successful_updates=2000, shared_controls_not_new_independent_replicas=True,
        native_validation=False, GTA_goal_completed=False, training_authorized=False)


def analyze(roots, bank_path):
    require(len(roots) == 3 and len({str(Path(r).resolve()) for r in roots}) == 3 and sha(bank_path) == BANK_SHA,
        'Three distinct completed seeds and original bank required')
    base = module(Path(__file__).with_name('calibre_basis_readout.py'), BASE_SHA, '_blockmean_fixed_math')
    bank = read(bank_path); ids = bank['probe_cases']
    require(len(ids) == len(set(ids)) == 12 and len(bank['fit_cases']) == 24 and set(ids).isdisjoint(bank['fit_cases']),
        'Original FIT24/PROBE12 partition required')
    seeds, proofs = [], []
    for root in roots:
        summary, rows, interventions, hashes = load_seed(root, ids, bank, base)
        seeds.append(analyze_seed(summary, rows, interventions, ids, base)); proofs.append(hashes)
    require(tuple(s['seed'] for s in seeds) == SEEDS and len({p['normalization.json'] for p in proofs}) == 1,
        'Prespecified seed order/common normalization required')
    flags = [s['screen']['passed'] for s in seeds]
    method = dict(base.METHOD,
        training='Only AR newly trained2000 updates from common7406; C0/A_only reused original same-seed2000 update controls; no new independent control repeats. All use115458 parameters/15 tensors and unchanged recipe.',
        mechanism='Samecase primary0..7 donor=(slot+4)%8; donor FP64 ownA1 logR mean roundedFP32, broadcast on recipient ownA1. Eight block scalars preserved; site-weighted radius multiset NOT preserved.',
        mechanism_support='Every original8 block and donor supported; at least2 known-nonempty primary patches and at least1 joint scalar difference, actualFP32 zR change and positive mean absolute logit response. No within-block radius variance criterion.',
        cross_round='Only supplied R granularity changes; original9acd model/point-level FITmoments/seeds/plans/recipe/budget/utility/uniform/source thresholds fixed. New block-pairing estimator differs from old pointwise half-ring.',
        source_guard='Each fixed Dongyang/KiTS/Rider4case primary sameweight Dice effect>=0 on complete original support. Source vascular effects are descriptive.',
        uniform_mechanism='Not evaluated; only normal segmentation retains all original192 patches.',
        scope='Repeated TRAIN oracle capability screen using shared controls; not CT predictability, CT deployment, native confirmation or independent confirmation.')
    original = [s['original_primary_screen_passed'] for s in seeds]
    uniform = [s['CT_uniform_transfer_safety']['passed'] for s in seeds]
    source = [s['SOURCE_safety']['passed'] for s in seeds]
    return dict(schema='GTA-AVT-calibre-blockmean-coadapt-final-readout-v1', readout_sha256=sha(__file__),
        method=method, bank_sha256=BANK_SHA, math_source_sha256=BASE_SHA, worker_sha256=WORKER_SHA,
        reused_control_worker_sha256=REUSED_WORKER_SHA, model_sha256=MODEL_SHA, coefficient_shape=[3, 64],
        trainable_parameters=115458, probe_cases=ids, per_seed=seeds, input_hashes=proofs, primary_seed=SEEDS[0],
        original_primary_screen_passed=original[0], all3_original_primary_screens_passed=combine(original),
        primary_CT_uniform_transfer_safety_passed=uniform[0], all3_CT_uniform_transfer_safety_passed=combine(uniform),
        primary_SOURCE_safety_passed=source[0], all3_SOURCE_safety_passed=combine(source),
        primary_screen_passed=flags[0], all3_screens_passed=combine(flags), capacity_continue=combine(flags),
        primary_failure_cannot_be_rescued=flags[0] is not True, seeds_pooled=False,
        controls_reused_not_retrained=True, controls_new_updates=0, new_AR_updates_per_seed=2000,
        shared_controls_not_new_independent_replicas=True, oracle_conditioned_not_CT_only=True,
        mathematical_or_true_upper_bound=False, no_attribute_auxiliary_loss=True, all_three_seeds_retained=True,
        next_experiment_authorized=False, independent_prelocked_confirmation_required=True,
        native_confirmation_required=True, GTA_goal_completed=False)


def render(result):
    text = ['块摘要R：完成后的TRAIN oracle能力筛查；C0/A_only复用原各2000步，AR新各2000步。']
    for seed in result['per_seed']:
        text.append('seed%s: original=%s; CT_uniform=%s; SOURCE=%s; combined=%s。' % (
            seed['seed'], seed['original_primary_screen_passed'], seed['CT_uniform_transfer_safety']['passed'],
            seed['SOURCE_safety']['passed'], seed['capacity_continue']))
        for control, value in seed['AR_vs_controls'].items():
            text.append(' AR−%s Dice=%s pp；cl=%s，CL=%s，fine=%s；own-worst2保护=%s。' % (control,
                value['metrics']['dice']['mean_AR_minus_control_pp'], value['metrics']['cldice']['mean_AR_minus_control_pp'],
                value['metrics']['centerline_recall']['mean_AR_minus_control_pp'], value['metrics']['fine_recall']['mean_AR_minus_control_pp'],
                value['no_vascular_or_tail_harm']['own_worst2_Dice_nonworse']))
        mechanism_record = seed['mechanisms'][INTERVENTION]
        text.append(' own−donor-blockmean Dice=%s pp；positive=%s，supported=%s/12，gate=%s。' % (
            mechanism_record['mean_correct_minus_permuted_Dice_pp'], mechanism_record['positive_cases'],
            len(mechanism_record['supported_case_ids']), mechanism_record['passed']))
        for source, value in seed['SOURCE_safety']['source_groups'].items():
            text.append(' %s own−donor Dice=%s pp；supported=%s/4；gate=%s。' % (source,
                value['case_equal_correct_minus_donor_blockmean_Dice_pp'], len(value['supported_case_ids']), value['passed']))
    text.append('全部预定种子/病例/来源保留；不改门槛，不以来源血管/SD/matched-tail替换原保护。')
    text.append('新机制只保留8块标量集合，不保留站点加权R多重集；uniform机制未测。')
    text.append('复用控制不是新独立重复；当前不是CT条件GTA或独立native确认，不自动启动后继。')
    return '\n'.join(text) + '\n'


def objective_tables(result):
    summary, cases, mechanisms, sources = [], [], [], []
    for seed in result['per_seed']:
        for arm, record in seed['arms'].items():
            for st in STRATA:
                row = dict(seed=seed['seed'], arm=arm, stratum=st,
                    reused_not_retrained=arm != 'AR', original_successful_updates=2000,
                    new_successful_updates=2000 if arm == 'AR' else 0)
                for key, value in record['strata_case_equal'][st]['metrics'].items():
                    row[key] = value['mean']; row[key + '_supported_cases'] = value['supported_cases']
                row.update(Dice_population_SD=record['Dice_population_SD'] if st == 'primary' else None,
                    own_worst2=record['own_worst2'] if st == 'primary' else None,
                    matched_worst2=record['matched_worst2'] if st == 'primary' else None,
                    original_primary_screen_passed=seed['original_primary_screen_passed'],
                    CT_uniform_safety_passed=seed['CT_uniform_transfer_safety']['passed'],
                    SOURCE_safety_passed=seed['SOURCE_safety']['passed'], combined=seed['capacity_continue'],
                    sameweight_stability=seed['mechanisms'][INTERVENTION]['primary_variant_stability']
                        if arm == 'AR' and st == 'primary' else None)
                summary.append(row)
            for cid, case in record['per_case'].items():
                for st in STRATA:
                    cases.append(dict(seed=seed['seed'], arm=arm, case_id=cid, source=case['source'], stratum=st,
                        reused_not_retrained=arm != 'AR', **case[st]))
        mechanisms.extend(dict(seed=seed['seed'], case_id=cid, **value)
            for cid, value in seed['mechanisms'][INTERVENTION]['per_case'].items())
        for source, value in seed['SOURCE_safety']['source_groups'].items():
            row = dict(seed=seed['seed'], source=source, expected_case_ids=value['expected_case_ids'],
                supported_case_ids=value['supported_case_ids'], complete_unchanged_support=value['complete_unchanged_support'],
                sameweight_Dice_pp=value['case_equal_correct_minus_donor_blockmean_Dice_pp'], source_Dice_gate=value['passed'])
            for key, metric in value['descriptive_metrics'].items():
                row[key + '_correct_minus_donor'] = metric['mean_correct_minus_donor']
                row[key + '_supported_cases'] = metric['supported_cases']
            row['source_vascular_descriptive_only'] = True
            sources.append(row)
    return [('objective_summary.csv', summary), ('objective_cases.csv', cases),
        ('objective_mechanism.csv', mechanisms), ('objective_sources.csv', sources)]


def execute(args):
    out = Path(args.output)
    require(not out.exists(), 'Fresh output required')
    result = analyze(args.inputs, args.bank)
    out.mkdir(parents=True)
    (out / 'analysis.json').write_bytes((json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
    (out / 'readout.txt').write_bytes(render(result).encode('utf-8'))
    for name, rows in objective_tables(result):
        with (out / name).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list)) else v
                for k, v in row.items()} for row in rows)
    return dict(status='complete', analysis_sha256=sha(out / 'analysis.json'),
        all3_screens_passed=result['all3_screens_passed'], next_experiment_authorized=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs=3)
    parser.add_argument('--bank')
    parser.add_argument('--output')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        require(args.inputs and args.bank and args.output, 'Three completed seeds, original bank and fresh output required')
        print(json.dumps(execute(args)))
    else:
        print(json.dumps(dict(status='inert', files_read=False, writes=False, GPU=False, new_training_authorized=False)))
