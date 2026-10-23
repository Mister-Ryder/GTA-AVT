"""Pure arithmetic copied from the frozen statistical readers; no execution attestation."""
import math
import statistics
import statistics as st
import struct

def require(value, message):
    if not value:
        raise ValueError(message)

def dist(values):
    require(all(v is None or isinstance(v, (int, float)) and math.isfinite(v)
                for v in values), 'Finite numeric values or explicit null required')
    defined = [v for v in values if v is not None]
    return dict(cases=len(values), defined_cases=len(defined),
                mean=st.mean(defined) if len(defined) == len(values) and defined else None,
                defined_case_mean=st.mean(defined) if defined else None,
                sample_SD=st.stdev(defined) if len(defined) > 1 else None)

def paired(a, b):
    changes = [None if x is None or y is None else x-y for x, y in zip(a, b)]
    valid = [v for v in changes if v is not None]
    return dict(after=dist(a), before=dist(b), difference=dist(changes),
                positive=sum(v > 1e-10 for v in valid),
                ties=sum(abs(v) <= 1e-10 for v in valid),
                negative=sum(v < -1e-10 for v in valid))

def relation(rows, predicted='raw_fine'):
    pairs = [(float(row['GT_fine']), float(row[predicted])) for row in rows
             if row['GT_fine'] is not None and row[predicted] is not None]
    if not pairs:
        return dict(support=0, MSE=None, constant_point2_MSE=None,
                    in_bank_GT_mean_MSE=None, pearson=None)
    x, y = zip(*pairs)
    mx, my = statistics.mean(x), statistics.mean(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    return dict(support=len(x), GT_mean=mx, predicted_mean=my,
                GT_population_std=math.sqrt(vx / len(x)),
                predicted_population_std=math.sqrt(vy / len(y)),
                predicted_min=min(y), predicted_max=max(y),
                MSE=sum((a - b) ** 2 for a, b in pairs) / len(pairs),
                constant_point2_MSE=sum((a - .2) ** 2 for a in x) / len(x),
                in_bank_GT_mean_MSE=vx / len(x),
                pearson=(sum((a - mx) * (b - my) for a, b in pairs) /
                         math.sqrt(vx * vy) if vx > 0. and vy > 0. else None),
                warning='The in-bank mean constant is descriptive and optimistic; not a fitted deployable baseline')

def semantic_summary(rows, bank, gate):
    supported = [r for r in rows if r['known_nonempty']]
    per_case = {}
    informative = []
    for case in bank['case_ids']:
        selected = [r for r in supported if r['case_id'] == case]
        c = [r['raw_fine'] for r in selected]
        gt = [r['GT_fine'] for r in selected]
        std = statistics.pstdev(c) if c else None
        variance = statistics.pvariance(gt) if gt else None
        active = bool(len(c) >= gate['informative_case_minimum_known_nonempty_patches']
                      and std > gate['informative_case_raw_c_population_std_strictly_greater']
                      and variance >= gate['informative_case_GT_fine_population_variance_at_least'])
        slots = [r['slot'] for r in selected]
        shift = max(1, len(slots) // 2) if len(slots) > 1 else 0
        mapping = dict(zip(map(str, slots), slots[-shift:] + slots[:-shift])) if shift else {}
        per_case[case] = dict(known_nonempty_patches=len(c), raw_c_population_std=std,
                              GT_fine_population_variance=variance, informative=active,
                              fixed_r_repair_slot_mapping=mapping,
                              raw_fine=relation(selected),
                              conditional_fine=relation(selected, 'conditional_fine'))
        if active:
            informative.append(case)
    c = [r['raw_fine'] for r in supported]
    std = statistics.pstdev(c) if c else None
    complete = len(rows) == 96 and len(per_case) == 12
    passed = bool(complete and std is not None and
                  std > gate['global_nonempty_raw_c_population_std_strictly_greater'] and
                  len(informative) >= gate['minimum_informative_cases'])
    groups = {}
    for source in ('Dongyang', 'KiTS', 'Rider'):
        chosen = [r for r in supported if r['source'] == source]
        groups[source] = dict(raw_fine=relation(chosen),
                              conditional_fine=relation(chosen, 'conditional_fine'))
    return dict(passed=passed, complete=complete,
                global_known_nonempty_patches=len(c),
                global_nonempty_raw_c_population_std=std,
                informative_cases=informative, informative_case_count=len(informative),
                per_case=per_case, source_groups=groups,
                all_attribute_valid_raw_fine=relation([r for r in rows if r['attribute_valid']]),
                known_nonempty_raw_fine=relation(supported),
                known_nonempty_conditional_fine=relation(supported, 'conditional_fine'),
                primary_GT_jitter_raw_fine=relation([r for r in supported if r['slot'] < 4]),
                CT_uniform_nonempty_raw_fine=relation([r for r in supported if r['slot'] >= 4]),
                semantic_scores_are_hard_gate=False,
                interpretation='Support and nondegeneracy only; neither information nor segmentation efficacy proven')

def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None

def summarize_metrics(rows, case_ids):
    result = {}
    mechanism_keys = ('normal_minus_repaired_Dice_pp', 'repaired_minus_normal_BCE',
                      'normal_minus_repaired_FP_voxels', 'normal_minus_repaired_FN_voxels',
                      'mean_absolute_logit_response', 'normal_minus_repaired_real_CT_Dice_pp')
    for case in case_ids:
        group = [row for row in rows if row['case_id'] == case]
        primary = [row for row in group if row['slot'] < 4]
        require(len(primary) == 4 and all(r['normal']['full']['truth_foreground'] > 0 for r in primary),
                'Original four primary patches must remain nonempty; no empty-Dice pooling')
        nonempty = [row for row in group if row['slot'] >= 4 and row['normal']['full']['truth_foreground'] > 0]
        empty = [row for row in group if row['slot'] >= 4 and row['normal']['full']['truth_foreground'] == 0]

        def aggregate(selected):
            if not selected:
                return dict(patches=0, normal_Dice=None, repaired_Dice=None,
                            normal_minus_repaired_Dice_pp=None, repaired_minus_normal_BCE=None,
                            mean_absolute_logit_response=None)
            return dict(patches=len(selected),
                normal_Dice=mean([r['normal']['full']['dice'] for r in selected]),
                repaired_Dice=mean([r['repaired']['full']['dice'] for r in selected]),
                normal_minus_repaired_Dice_pp=100. * mean([r['normal']['full']['dice'] - r['repaired']['full']['dice'] for r in selected]),
                repaired_minus_normal_BCE=mean([r['repaired']['full']['BCE_mean'] - r['normal']['full']['BCE_mean'] for r in selected]),
                normal_minus_repaired_FP_voxels=mean([r['normal']['full']['FP'] - r['repaired']['full']['FP'] for r in selected]),
                normal_minus_repaired_FN_voxels=mean([r['normal']['full']['FN'] - r['repaired']['full']['FN'] for r in selected]),
                mean_absolute_logit_response=mean([r['mean_abs_logit_difference'] for r in selected]),
                binary_changed_voxels=sum(r['binary_changed_voxels'] for r in selected),
                normal_minus_repaired_real_CT_Dice_pp=100. * mean([r['normal']['real_CT']['dice'] - r['repaired']['real_CT']['dice'] for r in selected]))
        result[case] = dict(primary_four_GT_jitter=aggregate(primary), CT_uniform_nonempty=aggregate(nonempty),
            CT_uniform_empty=dict(patches=len(empty),
                normal_FP_voxels=sum(r['normal']['full']['FP'] for r in empty),
                repaired_FP_voxels=sum(r['repaired']['full']['FP'] for r in empty),
                normal_mean_BCE=mean([r['normal']['full']['BCE_mean'] for r in empty]),
                repaired_mean_BCE=mean([r['repaired']['full']['BCE_mean'] for r in empty]),
                Dice_not_pooled=True))
        # Conditions were consumed in FP32. Distinguish a zero response to an
        # actual intervention from no intervention; no fitted tolerance/filter.
        supported = [r for r in group if r['known_nonempty']]
        eligible_primary = [r for r in primary if r['known_nonempty']]

        def consumed(value):
            require(math.isfinite(value) and 0. <= value <= 1., 'Invalid recorded condition')
            return struct.unpack('f', struct.pack('f', value))[0]

        normal_c = [consumed(r['normal_c']) for r in supported]
        variance = statistics.pvariance(normal_c) if normal_c else None
        changed = [r for r in supported if consumed(r['normal_c']) != consumed(r['repaired_c'])]
        primary_changed = [r for r in eligible_primary if consumed(r['normal_c']) != consumed(r['repaired_c'])]
        reasons = []
        if len(supported) < 2:
            reasons.append('fewer_than_two_known_nonempty_patches')
        if variance is None or variance == 0.:
            reasons.append('normal_consumed_FP32_condition_has_no_variation')
        if not primary_changed:
            reasons.append('no_eligible_primary_consumed_FP32_condition_changed')
        active = not reasons
        result[case]['mechanism_support'] = dict(
            known_nonempty_patch_count=len(supported),
            eligible_primary_patch_count=len(eligible_primary),
            known_nonempty_normal_consumed_FP32_population_variance=variance,
            known_nonempty_effective_c_changed_patch_count=len(changed),
            primary_effective_c_changed_patch_count=len(primary_changed),
            intervention_supported=active, null_reasons=reasons,
            criterion='At least2 known-nonempty patches, nonzero exact FP32 normal-c variance, and at least1 eligible primary patch with changed FP32 c; no tuned threshold')
        result[case]['primary_mechanism_delta'] = {
            key: result[case]['primary_four_GT_jitter'][key] if active else None
            for key in mechanism_keys}
    primary = [result[case]['primary_four_GT_jitter'] for case in case_ids]
    active_cases = [case for case in case_ids if result[case]['mechanism_support']['intervention_supported']]
    return dict(per_case=result, primary_case_equal={key: mean([r[key] for r in primary])
        for key in ('normal_Dice', 'repaired_Dice', 'normal_minus_repaired_Dice_pp',
                    'repaired_minus_normal_BCE', 'normal_minus_repaired_FP_voxels',
                    'normal_minus_repaired_FN_voxels', 'mean_absolute_logit_response',
                    'normal_minus_repaired_real_CT_Dice_pp')},
        positive_primary_cases=sum(r['normal_minus_repaired_Dice_pp'] > 0. for r in primary),
        negative_primary_cases=sum(r['normal_minus_repaired_Dice_pp'] < 0. for r in primary),
        case_count=len(case_ids), primary_patch_count=4 * len(case_ids),
        primary_definition='Original four GT-jitter full128 Dice/BCE means, then equal mean over12 cases',
        primary_case_equal_scope='Fixed-bank effect over ALL original cases/primary inputs, including unperturbed cases; not an intervention-supported mechanism estimate',
        primary_mechanism_case_equal={key: mean([result[case]['primary_mechanism_delta'][key] for case in active_cases])
                                     for key in mechanism_keys},
        mechanism_supported_case_ids=active_cases, mechanism_supported_case_count=len(active_cases),
        mechanism_null_case_ids=[case for case in case_ids if case not in active_cases],
        mechanism_null_case_count=len(case_ids)-len(active_cases),
        eligible_primary_patch_count=sum(result[case]['mechanism_support']['eligible_primary_patch_count'] for case in case_ids),
        primary_effective_c_changed_patch_count=sum(result[case]['mechanism_support']['primary_effective_c_changed_patch_count'] for case in case_ids),
        positive_mechanism_primary_cases=sum(result[case]['primary_mechanism_delta']['normal_minus_repaired_Dice_pp'] > 0. for case in active_cases),
        negative_mechanism_primary_cases=sum(result[case]['primary_mechanism_delta']['normal_minus_repaired_Dice_pp'] < 0. for case in active_cases),
        mechanism_supported_mean_scope='Original four-primary-patch case effects, equal mean over intervention-supported cases; null cases retained explicitly, not replaced by zeros or used to choose an experiment',
        interpretation='TRAIN correspondence diagnostic only; no new efficacy decision threshold or automatic experiment')
