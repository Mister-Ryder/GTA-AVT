"""CPU descriptive readout of completed frozen routing interventions; default inert.

Four original fields retain all192 patches. Cross-patch block means were
evaluated only on slots0..7; absent uniform evaluations remain explicit nulls.
No fitting, optimizer, efficacy gate, STOP rescue or CT-predictor authorization.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

WORKER_SHA = 'c08cb97ea9728689217aefe303fb62130e29c3dacddd6a1eafe968dd673fd9ad'
ROUTING_SHA = 'a9320edc72b1dcde9fe4486355566bf9c491b78d109bf5315f578e08db32c9ef'
MODEL_SHA = '9acd1c9d8b04fdef3e5a7e4cefe9d8f0d234978093eaf09d7d2a03a64288f58e'
FROZEN_READER_SHA = '9807c001dfe91ffd44054cd637ce329d5338db1f96f91ca2d327246cf3e413bb'
BANK_SHA = '9be42890017341e4cc32fb0f17688d859a13e84643f95cd21e347462043419d6'
SEEDS = (20261030, 20261031, 20261032)
ORIGINAL = ('correct', 'half_ring', 'patch_mean_FP32', 'FIT_mean_FP64')
FIFTH = 'cross_patch_mean_FP32'
VARIANTS = ORIGINAL + (FIFTH,)
STRATA = ('primary', 'all16', 'GT_jitter_all8', 'CT_uniform_all8', 'uniform_known_nonempty')
SOURCES = ('Dongyang', 'KiTS', 'Rider')
REGIONS = ('0_to15', '16_to31', 'at_least32')
METRICS = ('dice', 'cldice', 'centerline_recall', 'centerline_precision', 'fine_recall', 'BCE', 'FP', 'FN', 'TP')
FRACTIONS = METRICS[:5]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def prior_math():
    path = Path(__file__).with_name('calibre_basis_frozen_readout.py')
    require(sha(path) == FROZEN_READER_SHA, 'Original frozen aggregation source changed')
    spec = importlib.util.spec_from_file_location('_routing_frozen_pure_math', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.metric_math()


def validate_variant(value, correct, mathlib):
    mathlib.validate_metrics(value['metrics'])
    require(type(value['actual_zR_changed_from_correct']) is bool and set(value['regions']) == set(REGIONS),
        'Actual condition/region records missing')
    metrics, regions = value['metrics'], value['regions']
    require(all(metrics[k] == correct['metrics'][k] for k in
        ('real_CT_voxels', 'truth_foreground', 'fine_nodes', 'GT_nodes')), 'Variant GT/real support changed')
    require(metrics['real_CT_voxels'] == sum(z['real_CT_voxels'] for z in regions.values()) and
        all(metrics[k] == sum(z[k] for z in regions.values()) for k in ('TP', 'FP', 'FN', 'truth_foreground')),
        'Regional counts do not partition whole real voxels')
    require(all(regions[z][k] == correct['regions'][z][k] for z in REGIONS
        for k in ('real_CT_voxels', 'truth_foreground')), 'Regional GT/support changed')
    bce = metrics['BCE']
    regional_bce = math.fsum(z['BCE_sum'] for z in regions.values()) / metrics['real_CT_voxels']
    require(bce is not None and math.isfinite(bce) and bce >= 0 and
        ((bce == regional_bce) if bce == 0 or regional_bce == 0 else
         math.isclose(bce, regional_bce, rel_tol=1e-12, abs_tol=1e-12)), 'Regional BCE partition differs')
    for z in regions.values():
        require(all(type(z[k]) is int and z[k] >= 0 for k in ('real_CT_voxels', 'truth_foreground', 'TP', 'FP', 'FN')) and
            math.isfinite(z['BCE_sum']) and z['BCE_sum'] >= 0 and z['TP'] + z['FN'] == z['truth_foreground'] and
            (z['BCE'] == z['BCE_sum'] / z['real_CT_voxels'] if z['real_CT_voxels'] else z['BCE'] is None),
            'Invalid regional support/count/BCE')


def validate_scalars(rows, ids):
    """Check the exact eight-block FP32 scalar permutation, never a site multiset."""
    for cid in ids:
        primary = [r for r in rows if r['case_id'] == cid and r['slot'] < 8]
        own = {r['slot']: r['variants'][FIFTH]['donor'] for r in primary}
        require(len(own) == 8, 'All eight prespecified donor blocks required')
        for slot, donor in own.items():
            paired = (slot + 4) % 8
            require(donor['case_id'] == cid and donor['recipient_slot'] == slot and donor['donor_slot'] == paired and
                donor['recipient_supported_sites'] > 0 and donor['donor_supported_sites'] > 0 and
                donor['recipient_supported_sites'] == primary[slot]['radius_support'] and
                donor['donor_supported_sites'] == primary[paired]['radius_support'] and
                donor['site_weighted_radius_multiset_preserved'] is False, 'Fixed donor/support identity differs')
            for prefix in ('recipient', 'donor'):
                x64, x32 = donor[prefix + '_mean_logR_FP64'], donor[prefix + '_mean_logR_FP32']
                require(math.isfinite(x64) and math.isfinite(x32) and
                    struct.unpack('f', struct.pack('f', x64))[0] == x32, 'Mean must be finite and rounded to FP32')
            require(donor['donor_mean_logR_FP32'] == own[paired]['recipient_mean_logR_FP32'] and
                donor['donor_mean_logR_FP64'] == own[paired]['recipient_mean_logR_FP64'], 'Donor mean differs from its own block')
        require(sorted(v['recipient_mean_logR_FP32'] for v in own.values()) ==
            sorted(v['donor_mean_logR_FP32'] for v in own.values()), 'Eight block scalar multiset differs')


def load_seed(root, seed, ids, mathlib):
    root = Path(root)
    collection = read(root.parent / 'collection.json')
    require(collection.get('terminal', {}).get('exit_code') == 0, 'Natural completed collection required')
    child = collection.get('child', {})
    require(not child.get('exists') or child.get('state') in ('Z', 'X'), 'Diagnostic child still active')
    for name in ('summary.json', 'intent.json', 'diagnostics.jsonl'):
        proof, path = collection['files'][name], root / name
        require(path.stat().st_size == proof['bytes'] and sha(path) == proof['sha256'], 'Collected diagnostic bytes differ')
    summary, intent = read(root / 'summary.json'), read(root / 'intent.json')
    require(summary['schema'] == 'GTA-AVT-calibre-routing-frozen-diagnostic-v1' and
        summary['status'] == 'verified_complete' and summary['mode'] == 'run' and summary['seed'] == seed and
        summary['patches'] == 192 and tuple(summary['variants']) == VARIANTS and
        tuple(summary['original_variants']) == ORIGINAL and summary['primary_only_variant'] == FIFTH and
        summary['primary_only_evaluated_patches'] == 96 and summary['primary_only_not_evaluated_patches'] == 96 and
        summary['block_scalar_multiset_preserved_per_case'] is True and summary['site_weighted_radius_multiset_preserved'] is False and
        summary['whole_metric_reference_parity_passed'] is True and 0 <= summary['maximum_BCE_ULP'] <= 2 and
        summary['model_state_unchanged'] is True and summary['optimizer_created'] is False and
        summary['weights_updated'] is False and summary['candidate_STOP_unchanged'] is True and
        summary['no_new_success_gate'] is True and summary['CT_predictor_training_authorized'] is False,
        'Complete frozen routing/no-update summary required')
    require(intent['schema'] == 'GTA-AVT-calibre-routing-frozen-diagnostic-intent-v1' and
        intent['seed'] == seed and intent['mode'] == 'run' and intent['worker_sha256'] == WORKER_SHA and
        intent['routing_worker_sha256'] == ROUTING_SHA and intent['model_sha256'] == MODEL_SHA and
        tuple(intent['variants']) == VARIANTS and tuple(intent['original_variants']) == ORIGINAL and
        intent['primary_only_variant'] == FIFTH and intent['expected_patches'] == 192 and
        intent['primary_only_expected_patches'] == 96 and intent['trainable_parameter_tensors'] == 15 and
        intent['trainable_parameter_count'] == 115458 and intent['coefficient_shape'] == [3, 64] and
        intent['optimizer_created'] is False and intent['model_eval'] is True and intent['no_grad'] is True and
        intent['candidate_STOP_unchanged'] is True and intent['new_success_gate'] is False,
        'Exact routing worker/model/intervention identity differs')
    rows = [json.loads(line) for line in (root / 'diagnostics.jsonl').read_bytes().splitlines()]
    require([(r['case_id'], r['slot']) for r in rows] == [(c, slot) for c in ids for slot in range(16)],
        'Full original12x16 row order required')
    for row in rows:
        require(row['source'] == row['case_id'].split('_')[0] and type(row['known_nonempty']) is bool and
            row['stratum'] == ('GT_centerline_plus_jitter' if row['slot'] < 8 else 'CT_uniform') and
            row['primary'] == (row['slot'] < 8 and row['known_nonempty']) and set(row['variants']) == set(VARIANTS),
            'Original patch/variant population differs')
        for variant in ORIGINAL:
            validate_variant(row['variants'][variant], row['variants']['correct'], mathlib)
        fifth = row['variants'][FIFTH]
        if row['slot'] < 8:
            require(row['primary'] is True and fifth['status'] == 'evaluated' and
                type(fifth['actual_zR_changed_from_own_patch_mean']) is bool, 'All96 original primary fifth fields required')
            validate_variant(fifth, row['variants']['correct'], mathlib)
        else:
            require(fifth['status'] == 'not_evaluated' and set(fifth) == {'status', 'reason'},
                'Uniform fifth-field evaluations must remain explicit not_evaluated, without fake metrics')
        for variant in ('correct', 'half_ring'):
            require(row['parity'][variant]['all_count_and_derived_metrics_exact'] is True and
                0 <= row['parity'][variant]['BCE_binary64_ULP'] <= 2, 'Original reference metric parity missing')
    validate_scalars(rows, ids)
    return rows, dict(summary_sha256=sha(root / 'summary.json'), intent_sha256=sha(root / 'intent.json'),
        diagnostics_sha256=sha(root / 'diagnostics.jsonl'), trained_state_sha256=intent['trained_state_sha256'],
        reference_patches_sha256=intent['reference_patches_sha256'], reference_mechanism_sha256=intent['reference_mechanism_sha256'])


def support_summary(rows, ids):
    cases = {}
    for cid in ids:
        primary = [r for r in rows if r['case_id'] == cid and r['slot'] < 8]
        blocks = []
        for row in primary:
            v = row['variants'][FIFTH]
            donor = v['donor']
            blocks.append(dict(slot=row['slot'], **donor,
                own_minus_donor_logR_FP32=donor['recipient_mean_logR_FP32'] - donor['donor_mean_logR_FP32'],
                block_scalar_FP32_different=donor['recipient_mean_logR_FP32'] != donor['donor_mean_logR_FP32'],
                actual_zR_changed_from_own_patch_mean=v['actual_zR_changed_from_own_patch_mean'],
                actual_zR_changed_from_correct=v['actual_zR_changed_from_correct']))
        cases[cid] = dict(blocks=blocks, primary_patches=len(blocks),
            recipient_supported_sites=sum(b['recipient_supported_sites'] for b in blocks),
            distinct_own_FP32_block_scalars=len({b['recipient_mean_logR_FP32'] for b in blocks}),
            scalar_FP32_different_patches=sum(b['block_scalar_FP32_different'] for b in blocks),
            actual_zR_changed_from_own_patch_mean_patches=sum(b['actual_zR_changed_from_own_patch_mean'] for b in blocks),
            mean_abs_scalar_FP32_difference=mean(abs(b['own_minus_donor_logR_FP32']) for b in blocks),
            original_fields_actual_zR_changed_from_correct={v: sum(r['variants'][v]['actual_zR_changed_from_correct'] for r in primary) for v in ORIGINAL},
            uniform_fifth=dict(status='not_evaluated', patches=8, actual_zR_changed_from_own_patch_mean_patches=None))
    def group(members):
        return dict(case_ids=members, cases=len(members), primary_patches=sum(cases[c]['primary_patches'] for c in members),
            scalar_FP32_different_patches=sum(cases[c]['scalar_FP32_different_patches'] for c in members),
            actual_zR_changed_from_own_patch_mean_patches=sum(cases[c]['actual_zR_changed_from_own_patch_mean_patches'] for c in members),
            case_equal_mean_abs_scalar_FP32_difference=mean(cases[c]['mean_abs_scalar_FP32_difference'] for c in members),
            all8_supported_blocks_case_ids=[c for c in members if cases[c]['primary_patches'] == 8 and
                all(b['recipient_supported_sites'] > 0 and b['donor_supported_sites'] > 0 for b in cases[c]['blocks'])],
            actual_FP32_changed_case_ids=[c for c in members if cases[c]['actual_zR_changed_from_own_patch_mean_patches'] > 0])
    return dict(per_case=cases, all12=group(ids), source_groups={s: group([c for c in ids if c.startswith(s + '_')]) for s in SOURCES},
        uniform_fifth=dict(status='not_evaluated', patches=96, condition_change_count=None, segmentation_metrics=None))


def generic_contrast(value):
    if isinstance(value, dict):
        return {k.replace('mean_correct_minus_variant', 'mean_reference_minus_comparator'): generic_contrast(v) for k, v in value.items()}
    if isinstance(value, list):
        return [generic_contrast(v) for v in value]
    return value


def analyze(inputs, bank_path):
    require(len(inputs) == 3 and len({str(Path(p).resolve()) for p in inputs}) == 3, 'Three distinct completed seed outputs required')
    require(sha(bank_path) == BANK_SHA, 'Original bank bytes differ')
    bank = read(bank_path)
    ids = bank['probe_cases']
    require(len(ids) == 12 and len(set(ids)) == 12 and all(sum(c.startswith(s + '_') for c in ids) == 4 for s in SOURCES),
        'Original12 cases/three sources required')
    prior, mathlib = prior_math()
    result, proofs = [], []
    for root, seed in zip(inputs, SEEDS):
        rows, proof = load_seed(root, seed, ids, mathlib)
        variants, contrasts = {}, {}
        for variant in VARIANTS:
            strata = {}
            for stratum in STRATA:
                if variant == FIFTH and stratum not in ('primary', 'GT_jitter_all8'):
                    strata[stratum] = dict(status='not_evaluated', per_case=None, case_equal=None, source_groups=None,
                        reason='Fifth field is primary-only; no complete all16 or uniform population was evaluated')
                    continue
                cases = {c: prior.case_record([r for r in rows if r['case_id'] == c and prior.selected(r, stratum)], variant, mathlib) for c in ids}
                strata[stratum] = dict(status='evaluated', per_case=cases, case_equal=prior.levels(cases, ids),
                    source_groups={s: prior.levels(cases, [c for c in ids if c.startswith(s + '_')]) for s in SOURCES})
            defined = [(c, strata['primary']['per_case'][c]['whole']['dice']) for c in ids
                if strata['primary']['per_case'][c]['whole']['dice'] is not None]
            worst = [c for c, v in sorted(defined, key=lambda x: (x[1], x[0]))[:2]]
            variants[variant] = dict(strata=strata,
                Dice_population_SD=math.sqrt(mathlib.variance(v for c, v in defined)) if defined else None,
                own_worst2=dict(case_ids=worst, Dice_mean=mean(strata['primary']['per_case'][c]['whole']['dice'] for c in worst) if len(worst) == 2 else None))
        for reference in VARIANTS:
            members = variants[reference]['own_worst2']['case_ids']
            variants[reference]['matched_worst2'] = dict(case_ids=members,
                Dice_means={v: mean(variants[v]['strata']['primary']['per_case'][c]['whole']['dice'] for c in members)
                    if len(members) == 2 else None for v in VARIANTS})
        for variant in ORIGINAL[1:]:
            contrasts[variant] = {st: prior.comparison(variants['correct']['strata'][st]['per_case'],
                variants[variant]['strata'][st]['per_case'], ids) for st in STRATA}
        own, cross = variants['patch_mean_FP32'], variants[FIFTH]
        block_contrast = generic_contrast(prior.comparison(own['strata']['primary']['per_case'], cross['strata']['primary']['per_case'], ids))
        block_contrast.update(reference_variant='patch_mean_FP32', comparator_variant=FIFTH,
            scope='Original primary96 patches; case equal across12 cases; no uniform counterpart',
            mean_difference_direction='own_patchmean_minus_cross_patchmean',
            Dice_population_SD_reference_minus_comparator=own['Dice_population_SD'] - cross['Dice_population_SD'],
            own_worst2_Dice_reference_minus_comparator=own['own_worst2']['Dice_mean'] - cross['own_worst2']['Dice_mean'])
        result.append(dict(seed=seed, variants=variants, correct_minus_variants=contrasts,
            own_patchmean_minus_cross_patchmean=block_contrast, block_scalar_support=support_summary(rows, ids),
            Rider_R15_patchwise=[r for r in rows if r['case_id'] == 'Rider_R15']))
        proofs.append(proof)
    return dict(schema='GTA-AVT-calibre-routing-frozen-readout-v1', source_sha256=sha(__file__), worker_sha256=WORKER_SHA,
        routing_worker_sha256=ROUTING_SHA, model_sha256=MODEL_SHA, bank_sha256=BANK_SHA,
        frozen_math_source_sha256=FROZEN_READER_SHA, original_metric_source_sha256=prior.READER_SHA,
        case_ids=ids, per_seed=result, input_hashes=proofs,
        method=dict(scope='Descriptive same final routing AR weights; eval/no_grad, no optimizer/update, fitting or gate',
            original_four='All192 original patches/five strata/regions retained with original pure aggregation and comparison math',
            fifth='For each case slots0..7 donor=(slot+4)%8; donor FP64 ownA1 mean rounded to FP32 and broadcast on recipient ownA1',
            estimand='Own block scalar mean correspondence versus a fixed between-block scalar reassignment. Eight block scalar means preserved per case; site-weighted radius multiset NOT preserved. This differs from pointwise own-A half-ring spatial association.',
            support='All eight donor and recipient blocks require positive measured support; record FP32 scalar differences AND actual consumed zR differences. Support/variation is not a new efficacy gate.',
            aggregation='Original patch equal within case, case equal across cases; fine sum covered/sum nodes within each case. Sources each retain four cases.',
            uniform='Fifth-field uniform96 patches NOT EVALUATED; no invented zeros, full-all16 fifth aggregate, or support extrapolation',
            regional='TP/FP/FN patch mean per case, then case equal; BCE sum/real voxels within case then case equal. Real observed six-face distance bins; no regional skeleton/Dice/anatomical wall interpretation.',
            limitations='Oracle A is geometry-dependent support, not anatomical background; mean controls are not retrained A_only; nonlinear effects are not additive information decomposition. Same12 TRAIN cases/overlapping patches, no patient/zero-shot independence.',
            precision='Original count/derived metrics parity exact; nonnegative BCE allows2 binary64 ULP. No old voxelwise mask parity claim; regional BCE closure1e-12 FP64 tolerance.'),
        candidate_STOP_unchanged=True, scientific_verdict_modified=False, new_success_gate=False,
        CT_predictor_training_authorized=False, oracle_not_CT_only=True, GTA_goal_completed=False)


def render(result):
    out = ['冻结 routing AR 条件诊断：描述性分析，不改STOP、不建立新gate。',
        '原四字段保留全部192patch/五层/区域；第五字段只测primary96，uniform96未测，不能以0代替。',
        '第五对照只保留每例8个区块FP32均值集合，不保留站点加权R多重集；与点内半环是不同估计对象。',
        'Dice/cl/CL/fine差以pp显示；BCE为loss单位；FP/FN先patch平均再病例等权。']
    for seed in result['per_seed']:
        out.append('seed %s' % seed['seed'])
        for variant in VARIANTS:
            v = seed['variants'][variant]
            m = v['strata']['primary']['case_equal']
            out.append(' %s primary：%s；SD=%spp；ownworst2=%s%%。' % (variant,
                '；'.join('%s=%s%%' % (k, None if m[k]['mean'] is None else 100 * m[k]['mean']) for k in FRACTIONS),
                None if v['Dice_population_SD'] is None else 100 * v['Dice_population_SD'],
                None if v['own_worst2']['Dice_mean'] is None else 100 * v['own_worst2']['Dice_mean']))
        for variant in ORIGINAL[1:]:
            for st in ('primary', 'uniform_known_nonempty'):
                c = seed['correct_minus_variants'][variant][st]['case_equal']
                out.append(' correct−%s %s：%s。' % (variant, st, '；'.join('%s=%s' % (k,
                    c[k]['mean_correct_minus_variant_pp'] if k in FRACTIONS else c[k]['mean_correct_minus_variant'])
                    for k in ('dice', 'cldice', 'centerline_recall', 'fine_recall', 'BCE', 'FP', 'FN'))))
        contrast = seed['own_patchmean_minus_cross_patchmean']
        for label, group in [('all12', contrast['case_equal'])] + list(contrast['source_groups'].items()):
            out.append(' own_patchmean−cross_patchmean %s：%s。' % (label, '；'.join('%s=%s' % (k,
                group[k]['mean_reference_minus_comparator_pp'] if k in FRACTIONS else group[k]['mean_reference_minus_comparator'])
                for k in ('dice', 'cldice', 'centerline_recall', 'fine_recall', 'BCE', 'FP', 'FN'))))
        out.append(' own−cross SD=%spp，ownworst2=%spp；同病例尾部比较保留于analysis。' %
            (100 * contrast['Dice_population_SD_reference_minus_comparator'], 100 * contrast['own_worst2_Dice_reference_minus_comparator']))
        out.append(' FP32/实际消费支持：%s。' % json.dumps(seed['block_scalar_support']['all12'], ensure_ascii=False))
        for row in seed['Rider_R15_patchwise']:
            if row['slot'] in (3, 5):
                out.append(' R15 slot%s 五字段：%s。' % (row['slot'], json.dumps({v: row['variants'][v]['metrics'] for v in VARIANTS}, ensure_ascii=False)))
    out += ['全部12例/3来源/SD/自身及匹配尾部/区域/R15逐patch/FP32区块支持均保留于analysis。',
        '这是GT代理属性的冻结消费诊断；任何描述性改善均不能挽救当前STOP或授权CT预测器训练。']
    return '\n'.join(out) + '\n'


def execute(args):
    output = Path(args.output)
    require(not output.exists(), 'Fresh output required')
    result = analyze(args.inputs, args.bank)
    cases = []
    for seed in result['per_seed']:
        for variant, value in seed['variants'].items():
            for st, group in value['strata'].items():
                for cid in result['case_ids']:
                    evaluated = group['status'] == 'evaluated'
                    base = seed['variants']['correct']['strata'][st]['per_case'][cid]['whole']
                    whole = group['per_case'][cid]['whole'] if evaluated else {k: None for k in base}
                    cases.append(dict(seed=seed['seed'], variant=variant, stratum=st, case_id=cid, source=cid.split('_')[0],
                        status=group['status'], **whole,
                        **{'correct_minus_variant_' + k: None if base[k] is None or whole[k] is None else base[k] - whole[k] for k in METRICS}))
    output.mkdir(parents=True)
    (output / 'analysis.json').write_bytes((json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
    (output / 'readout.txt').write_bytes(render(result).encode('utf-8'))
    with (output / 'casewise.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(cases)
    return dict(status='complete', analysis_sha256=sha(output / 'analysis.json'), case_rows=len(cases), scientific_verdict_modified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs=3)
    parser.add_argument('--bank')
    parser.add_argument('--output')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        require(args.inputs and args.bank and args.output, 'Three completed collected roots, original bank and fresh output required')
        print(json.dumps(execute(args)))
    else:
        print(json.dumps(dict(status='inert', files_read=False, writes=False, new_success_gate=False)))
