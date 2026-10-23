"""Self-contained standard-library recomputation of released derived numbers.

No image, mask, prediction volume, checkpoint, model or server is accessed.
Public data preserve the same reused cases, fixed rows, weights and nulls.
Default prints an inert plan. --package DIR --check verifies the manifest,
arithmetic identity and expected summary by recomputing every block.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics as st
import struct
from types import SimpleNamespace

PROTOCOL_SHA='478276000b84a941ec5c59083eba53a852d1a97f8f227b93def1cdcc852ffb47'
NULL_SHA='73f394afb2513dd8043993edb820b7832377dcb5ddaadd9aa3950c630dca250b'
BLOCKS=('A_PRIMARY','B_REPLICATION','C_LOCKED_CONFIRMATION')
SEEDS=(20261024,20261025,20261026)

NAMES = ('raw_consumed_score_permutation', 'own_presence_composition_permutation')

VAL = ['Dongyang_D2', 'Dongyang_D5', 'KiTS_K13', 'KiTS_K15',
       'KiTS_K18', 'KiTS_K8', 'Rider_R10', 'Rider_R9']

GROUPS = ('radius_le_2mm', 'radius_2_to_4mm', 'radius_4_to_8mm',
          'radius_gt_8mm', 'nonjunction', 'junction_neighbourhood')

METRICS = ('dice', 'cldice', 'centerline_recall', 'centerline_precision',
           'precision', 'recall', 'fp_ml', 'fn_ml', 'assd_mm', 'hd95_mm')

PROPORTIONS = set(METRICS[:6])

def require(condition, message):
    if not condition:
        raise ValueError(message)

def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]

def finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True

def permutations(rows, case_ids):
    """No GT values enter overrides: GT only defines frozen eligibility pools."""
    values = {name: [f32(row['raw_ordinal']) for row in rows] for name in NAMES}
    mappings, invalid = [], {name: set() for name in NAMES}
    for case in case_ids:
        for stratum in ('GT_centerline_plus_jitter', 'CT_uniform'):
            indices = sorted([i for i, row in enumerate(rows) if row['case_id'] == case and
                row['stratum'] == stratum and row['attribute_valid'] and row['known_nonempty']],
                key=lambda i: rows[i]['slot'])
            shift = max(1, len(indices) // 2) if len(indices) >= 2 else 0
            donors = indices[-shift:] + indices[:-shift] if shift else indices[:]
            details = []
            before_r, after_r = [], []
            for index, donor_index in zip(indices, donors):
                own, donor = rows[index], rows[donor_index]
                values[NAMES[0]][index] = f32(donor['raw_ordinal'])
                own_u, donor_u = own['conditional_ordinal'], donor['conditional_ordinal']
                presence = f32(1. - f32(own['predicted_empty']))
                self_product, donor_product = None, None
                reason = None
                if own_u is None or donor_u is None:
                    invalid[NAMES[1]].add(index)
                    reason = 'own_or_donor_zero_predicted_nonempty_mass'
                    repaired = f32(own['raw_ordinal'])
                    self_error = None
                    equal_u = None
                    equal_products = None
                    rounding_only_change = None
                    numerical_null_reason = None
                    rounding_only_change_prevented = None
                else:
                    own_u, donor_u = f32(own_u), f32(donor_u)
                    self_product, donor_product = f32(presence * own_u), f32(presence * donor_u)
                    self_error = abs(self_product - f32(own['raw_ordinal']))
                    equal_u = own_u == donor_u
                    equal_products = self_product == donor_product
                    repaired = (f32(own['raw_ordinal']) if equal_u or equal_products else donor_product)
                    rounding_only_change = bool(not equal_u and equal_products and repaired != f32(own['raw_ordinal']))
                    rounding_only_change_prevented = bool(not equal_u and equal_products and
                        donor_product != f32(own['raw_ordinal']))
                    numerical_null_reason = ('equal_FP32_conditional_ordinal' if equal_u else
                        'equal_FP32_own_presence_weighted_products' if equal_products else None)
                require(math.isfinite(repaired) and 0 <= repaired <= 1, 'Invalid FP32 repaired score')
                values[NAMES[1]][index] = repaired
                before_r.append(own['conditional_r']); after_r.append(donor['conditional_r'])
                details.append(dict(slot=own['slot'], donor_slot=donor['slot'],
                    normal_c=f32(own['raw_ordinal']), raw_permuted_c=values[NAMES[0]][index],
                    own_presence_permuted_c=repaired, conditional_ordinal_exactly_equal=equal_u,
                    own_presence_FP32=presence, own_conditional_ordinal_FP32=own_u,
                    donor_conditional_ordinal_FP32=donor_u,
                    self_product_FP32=self_product, donor_product_FP32=donor_product,
                    self_and_donor_FP32_products_equal=equal_products,
                    different_ordinal_but_equal_FP32_products=bool(equal_u is False and equal_products),
                    consumed_c_change_only_self_reconstruction_rounding=rounding_only_change,
                    reconstruction_rounding_only_c_change_prevented=rounding_only_change_prevented,
                    numerical_null_reason=numerical_null_reason,
                    self_reconstruction_rounding_abs_error=self_error, unsupported_reason=reason))
            require(sorted(values[NAMES[0]][i] for i in indices) ==
                sorted(f32(rows[i]['raw_ordinal']) for i in indices), 'Raw score multiset differs')
            # The complete r vectors are re-paired, including unused mixture detail.
            require(sorted(json.dumps(r) for r in before_r) == sorted(json.dumps(r) for r in after_r),
                    'Conditional four-component composition multiset differs')
            mappings.append(dict(case_id=case, stratum=stratum, eligible_slots=[rows[i]['slot'] for i in indices],
                half_ring_shift=shift, slot_mapping=details, conditional_r4_multiset_preserved=True,
                raw_c_multiset_preserved=True, composition_raw_c_multiset_preserved=False,
                different_ordinal_equal_product_slots=sum(r['different_ordinal_but_equal_FP32_products'] for r in details),
                reconstruction_rounding_only_changed_slots=sum(bool(r['consumed_c_change_only_self_reconstruction_rounding']) for r in details),
                reconstruction_rounding_only_c_change_prevented_slots=sum(bool(r['reconstruction_rounding_only_c_change_prevented']) for r in details)))
    return values, mappings, invalid

def average(values):
    values = [value for value in values if value is not None]
    return st.mean(values) if values else None

def aggregate(group, name):
    if not group:
        return dict(patches=0, correct_minus_permuted_Dice_pp=None, permuted_minus_correct_BCE=None,
                    correct_minus_permuted_FP_voxels=None, correct_minus_permuted_FN_voxels=None,
                    mean_absolute_logit_response=None, correct_minus_permuted_real_CT_Dice_pp=None)
    return dict(patches=len(group), normal_Dice=average([r['normal']['full']['dice'] for r in group]),
        permuted_Dice=average([r['interventions'][name]['score']['full']['dice'] for r in group]),
        correct_minus_permuted_Dice_pp=100. * average([r['normal']['full']['dice'] -
            r['interventions'][name]['score']['full']['dice'] for r in group]),
        permuted_minus_correct_BCE=average([r['interventions'][name]['score']['full']['BCE_mean'] -
            r['normal']['full']['BCE_mean'] for r in group]),
        correct_minus_permuted_FP_voxels=average([r['normal']['full']['FP'] -
            r['interventions'][name]['score']['full']['FP'] for r in group]),
        correct_minus_permuted_FN_voxels=average([r['normal']['full']['FN'] -
            r['interventions'][name]['score']['full']['FN'] for r in group]),
        mean_absolute_logit_response=average([r['interventions'][name]['mean_abs_logit_response'] for r in group]),
        binary_changed_voxels=sum(r['interventions'][name]['binary_changed_voxels'] for r in group),
        correct_minus_permuted_real_CT_Dice_pp=100. * average([r['normal']['real_CT']['dice'] -
            r['interventions'][name]['score']['real_CT']['dice'] for r in group]))

def summarize(rows, case_ids, probe_ids):
    output = {}
    for name in NAMES:
        per_case = {}
        for case in case_ids:
            group = [r for r in rows if r['case_id'] == case]
            primary = [r for r in group if r['slot'] < 8 and r['known_nonempty'] and r['attribute_valid']]
            variance = st.pvariance([f32(r['normal_c']) for r in primary]) if primary else None
            changed = sum(f32(r['normal_c']) != f32(r['interventions'][name]['consumed_c']) for r in primary)
            reasons = []
            if len(primary) < 2:
                reasons.append('fewer_than_two_valid_known_nonempty_primary_slots')
            if variance is None or variance <= 0:
                reasons.append('normal_primary_exact_FP32_c_population_variance_not_positive')
            if changed < 1:
                reasons.append('no_primary_actual_FP32_consumed_score_change')
            if any(r['interventions'][name].get('unsupported_reason') for r in primary):
                reasons.append('primary_permutation_has_undefined_predicted_composition')
            effects = aggregate(primary, name)
            supported = not reasons
            per_case[case] = dict(source=case.split('_')[0], primary_valid_known_nonempty=effects,
                mechanism_support=dict(supported=supported, null_reasons=reasons,
                    valid_known_nonempty_primary_slots=len(primary), normal_primary_FP32_c_variance=variance,
                    actually_changed_primary_FP32_c_slots=changed,
                    GT_ordinal_variance=st.pvariance([r['GT_ordinal'] for r in primary]) if primary else None),
                primary_mechanism_effect={key: value if supported else None for key, value in effects.items()},
                all_primary_descriptive=aggregate([r for r in group if r['slot'] < 8 and
                    r['normal']['full']['truth_foreground'] > 0], name),
                uniform_known_nonempty_descriptive=aggregate([r for r in group if r['slot'] >= 8 and r['known_nonempty']], name),
                uniform_empty_descriptive=dict(patches=sum(r['slot'] >= 8 and r['valid_empty'] for r in group),
                    normal_FP_voxels=sum(r['normal']['full']['FP'] for r in group if r['slot'] >= 8 and r['valid_empty']),
                    permuted_FP_voxels=sum(r['interventions'][name]['score']['full']['FP'] for r in group if r['slot'] >= 8 and r['valid_empty']),
                    Dice_not_pooled=True),
                unknown_slots=[r['slot'] for r in group if not r['attribute_valid']])
        probe = [per_case[case] for case in probe_ids]
        effects = [r['primary_mechanism_effect']['correct_minus_permuted_Dice_pp'] for r in probe]
        supported = [case for case in probe_ids if per_case[case]['mechanism_support']['supported']]
        value = average(effects)
        parts = dict(all12_probe_cases_supported=len(supported) == 12,
            mean_correct_minus_permuted_Dice_at_least_point1pp=value is not None and value >= .1,
            at_least8_positive_probe_cases=sum(x is not None and x > 0 for x in effects) >= 8,
            all_required_metrics_finite=finite_tree(per_case))
        source = {source: dict(case_count=sum(case.startswith(source + '_') for case in probe_ids),
            supported_case_count=sum(case.startswith(source + '_') for case in supported),
            supported_mean_Dice_effect_pp=average([per_case[case]['primary_mechanism_effect']['correct_minus_permuted_Dice_pp']
                for case in probe_ids if case.startswith(source + '_')])) for source in ('Dongyang', 'KiTS', 'Rider')}
        descriptive_keys = ('correct_minus_permuted_Dice_pp', 'permuted_minus_correct_BCE',
            'correct_minus_permuted_FP_voxels', 'correct_minus_permuted_FN_voxels',
            'mean_absolute_logit_response', 'correct_minus_permuted_real_CT_Dice_pp')
        output[name] = dict(per_case=per_case, probe12_supported_case_ids=supported,
            probe12_null_case_ids=[case for case in probe_ids if case not in supported],
            probe12_case_equal_supported_mean_Dice_effect_pp=value,
            probe12_case_equal_descriptive= {key: average([r['primary_valid_known_nonempty'].get(key) for r in probe])
                for key in descriptive_keys},
            all36_case_equal_primary_descriptive={key: average([
                per_case[case]['primary_valid_known_nonempty'].get(key) for case in case_ids]) for key in descriptive_keys},
            all36_source_case_equal_primary_descriptive={source: {key: average([
                per_case[case]['primary_valid_known_nonempty'].get(key) for case in case_ids
                if case.startswith(source + '_')]) for key in descriptive_keys} for source in ('Dongyang', 'KiTS', 'Rider')},
            source_groups=source, gate=dict(parts=parts, passed=all(parts.values())))
    return dict(interventions=output, both_mechanism_gates_passed=all(value['gate']['passed'] for value in output.values()),
        unsupported_effects_are_null_not_zero=True, averaging='Valid nonempty primary patches within case, then equal probe12 cases',
        training_authorized=False)

def dist(values):
    require(all(v is None or isinstance(v, (int, float)) and math.isfinite(v)
                for v in values), 'Finite numeric values or explicit null required')
    defined = [v for v in values if v is not None]
    return dict(cases=len(values), defined_cases=len(defined),
                mean=st.mean(defined) if len(defined) == len(values) and defined else None,
                defined_case_mean=st.mean(defined) if defined else None,
                sample_SD=st.stdev(defined) if len(defined) > 1 else None)

def scaled(row, metric):
    value = row[metric]
    require(value is None or type(value) in (int, float) and math.isfinite(value),
            'Metric must be finite numeric or null: '+metric)
    require(value is None or (0 <= value <= 1 if metric in PROPORTIONS else value >= 0),
            'Metric outside its valid range: '+metric)
    return None if value is None else value * (100 if metric in PROPORTIONS else 1)

def paired(a, b):
    changes = [None if x is None or y is None else x-y for x, y in zip(a, b)]
    valid = [v for v in changes if v is not None]
    return dict(after=dist(a), before=dist(b), difference=dist(changes),
                positive=sum(v > 1e-10 for v in valid),
                ties=sum(abs(v) <= 1e-10 for v in valid),
                negative=sum(v < -1e-10 for v in valid))

def count_summary(a, b):
    result = {}
    for key in ('fp_voxels', 'fn_voxels'):
        av, bv = [a[4][c][key] for c in core.VAL], [b[4][c][key] for c in core.VAL]
        result[key] = dict(paired=core.paired(av, bv),
            after_total=sum(av), before_total=sum(bv), difference_total=sum(av)-sum(bv),
            cases=[dict(case_id=c, after=x, before=y, difference=x-y)
                   for c, x, y in zip(core.VAL, av, bv)],
            scope='Native voxel counts; totals reflect different native voxel volumes, not pooled physical volume')
    return result

def utility(result):
    """Literal six conjunctive utility criteria; undefined support fails."""
    required=('dice','cldice','centerline_recall')
    means={name:result['metrics'][name]['difference']['mean'] for name in required}
    small=result['geometry']['radius_le_2mm']['paired']['difference']['mean']
    defined=all(v is not None and math.isfinite(v) for v in list(means.values())+[small])
    parts=dict(all_required_case_equal_means_defined=defined,
        mean_clDice_at_least_point5pp=means['cldice'] is not None and means['cldice']>=.5,
        mean_Dice_nonnegative=means['dice'] is not None and means['dice']>=0,
        mean_centerline_recall_nonnegative=means['centerline_recall'] is not None and means['centerline_recall']>=0,
        mean_radius_le2mm_recall_nonnegative=small is not None and small>=0,
        own_worst_two_Dice_nonnegative=result['worst_two_mean_Dice_difference_pp']>=0,
        no_case_Dice_decline_gt3pp=result['worst_paired_Dice_difference_pp']>=-3)
    return dict(parts=parts,passed=all(parts.values()),protocol_sha256=PROTOCOL_SHA,
        evaluated_on_all_original_VAL8=True,mechanism_established=False,
        policy='All criteria required per block; SD cannot substitute; not significance or a clinical margin')

def metric(record):
    for grid in ('full','real_CT'):
        r = record[grid]
        require(all(type(r[k]) is int and r[k]>=0 for k in
            ('TP','FP','FN','truth_foreground','predicted_foreground','scored_voxels')) and
            r['scored_voxels']>0 and r['truth_foreground']==r['TP']+r['FN'] and
            r['predicted_foreground']==r['TP']+r['FP'] and
            r['TP']+r['FP']+r['FN']<=r['scored_voxels'],'Incomplete/invalid actual confusion counts')
        expected = (2.*r['TP']+1e-6)/(2.*r['TP']+r['FP']+r['FN']+1e-6)
        require(type(r['dice']) in (int,float) and math.isclose(r['dice'],expected,rel_tol=1e-12,abs_tol=1e-12) and
            type(r['BCE_mean']) in (int,float) and math.isfinite(r['BCE_mean']) and r['BCE_mean']>=0,
            'Incomplete/invalid measured Dice or BCE')


core=SimpleNamespace(VAL=VAL,METRICS=METRICS,GROUPS=GROUPS,paired=paired,scaled=scaled)


def canonical_ast(node):
    if isinstance(node,ast.AST):
        return [type(node).__name__,{k:canonical_ast(v) for k,v in ast.iter_fields(node)
            if k not in ('type_comment','type_params')}]
    if isinstance(node,list):return [canonical_ast(v) for v in node]
    return node


def ast_digest(node):
    raw=json.dumps(canonical_ast(node),sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def numeric(value,unit_interval=False,nullable=False):
    require(value is None and nullable or type(value) in (int,float) and math.isfinite(value) and
        (0<=value<=1 if unit_interval else True),'Invalid derived numeric field')
    return value


def validate_native(rows,case_ids):
    require([r['case_id'] for r in rows]==case_ids,'All original ordered native cases are required')
    for row in rows:
        require(row['evaluation_grid']=='original CT' and row['fixed_threshold']==.5 and
            row['condition_mode']=='predicted' and row['postprocessing']=='none','Native metric protocol differs')
        require(row['dice'] is not None,'No native case may be dropped')
        for name in METRICS:scaled(row,name)
        for name in ('fp_voxels','fn_voxels'):
            require(type(row[name]) is int and row[name]>=0,'Invalid native count')
        for name in GROUPS:
            value=row['geometry_groups'].get(name)
            require(value is not None or row['empty_prediction'] or row['empty_truth'],'Missing nonempty geometry group')
            if value is not None:
                require(type(value['support_nodes']) is int and value['support_nodes']>=0,'Invalid geometry support')
                numeric(value['centerline_recall'],True,True)
                require(value['support_nodes']>0 or value['centerline_recall'] is None,'Zero support must stay null')


def native_report(native,case_ids):
    before,after=[{r['case_id']:r for r in native[role]} for role in ('C_ord','D_ord')]
    for role in ('C_ord','D_ord'):validate_native(native[role],case_ids)
    metrics={m:paired([scaled(after[c],m) for c in case_ids],[scaled(before[c],m) for c in case_ids]) for m in METRICS}
    cases=[dict(case_id=c,before={m:scaled(before[c],m) for m in METRICS},
        after={m:scaled(after[c],m) for m in METRICS},delta={m:None if before[c][m] is None or after[c][m] is None else
        scaled(after[c],m)-scaled(before[c],m) for m in METRICS}) for c in case_ids]
    tails={}
    for label,rows in (('before',before),('after',after)):
        ids=sorted(case_ids,key=lambda c:(rows[c]['dice'],case_ids.index(c)))[:2]
        tails[label]=dict(case_ids=ids,mean_Dice_pp=st.mean(100*rows[c]['dice'] for c in ids))
    geometry={}
    for name in GROUPS:
        av,bv,records=[],[],[]
        for c in case_ids:
            x,y=after[c]['geometry_groups'].get(name),before[c]['geometry_groups'].get(name)
            if x is not None and y is not None:require(x['support_nodes']==y['support_nodes'],'GT support differs between arms')
            av.append(None if x is None or x['centerline_recall'] is None else 100*x['centerline_recall'])
            bv.append(None if y is None or y['centerline_recall'] is None else 100*y['centerline_recall'])
            records.append(dict(case_id=c,after_support_nodes=None if x is None else x['support_nodes'],
                before_support_nodes=None if y is None else y['support_nodes'],after_recall_pp=av[-1],before_recall_pp=bv[-1]))
        geometry[name]=dict(paired=paired(av,bv),case_support=records)
    result=dict(metrics=metrics,cases=cases,geometry=geometry,worst_two=tails,
        worst_two_mean_Dice_difference_pp=tails['after']['mean_Dice_pp']-tails['before']['mean_Dice_pp'],
        worst_paired_Dice_difference_pp=min(r['delta']['dice'] for r in cases),
        Dice_SD_reduction_pp=metrics['dice']['before']['sample_SD']-metrics['dice']['after']['sample_SD'])
    result['continue_gate']=utility(result)
    result['FP_FN_voxel_counts']=count_summary((None,None,None,None,after),(None,None,None,None,before))
    return result


def validate_mechanism(predictions,patches,case_ids,probe_ids):
    order=[(c,s) for c in case_ids for s in range(16)]
    require([(r['case_id'],r['slot']) for r in predictions]==order and
        [(r['case_id'],r['slot']) for r in patches]==order,'Exactly ordered36x16 derived rows are required')
    values,mapping,invalid=permutations(predictions,case_ids)
    for i,(pred,row) in enumerate(zip(predictions,patches)):
        require(pred['source']==row['source']==pred['case_id'].split('_')[0] and
            pred['stratum']==row['stratum']==('GT_centerline_plus_jitter' if pred['slot']<8 else 'CT_uniform'),
            'Fixed case/stratum identity differs')
        require(all(row[k]==pred[k] for k in ('known_nonempty','valid_empty','attribute_valid','GT_ordinal')) and
            row['normal_c']==f32(pred['raw_ordinal']),'Condition/target identity differs')
        numeric(pred['raw_ordinal'],True);numeric(pred['predicted_empty'],True)
        numeric(pred['conditional_ordinal'],True,True)
        require(len(pred['scale_probability'])==5,'Five class probabilities required')
        for x in pred['scale_probability']:numeric(x,True)
        require(pred['conditional_r'] is None or len(pred['conditional_r'])==4,'Full conditional composition required')
        if pred['conditional_r'] is not None:
            for x in pred['conditional_r']:numeric(x,True)
        metric(row['normal'])
        require(len(row['normal_gain'])==128 and all(type(x) in (int,float) and math.isfinite(x) and -1<=x<=1
            for x in row['normal_gain']),'Complete finite channel gains required')
        require(set(row['interventions'])==set(NAMES),'Both frozen interventions required')
        for name in NAMES:
            v=row['interventions'][name]
            require(v['consumed_c']==values[name][i] and v['condition_changed']==(v['consumed_c']!=row['normal_c']) and
                bool(v['unsupported_reason'])==(i in invalid[name]),'Actual donor/consumption/support differs')
            metric(v['score'])
            require(len(v['gain'])==128 and all(type(x) in (int,float) and math.isfinite(x) and -1<=x<=1 for x in v['gain']),
                'Complete finite intervention gains required')
            require(all(type(v[k]) in (int,float) and math.isfinite(v[k]) and v[k]>=0 for k in
                ('mean_abs_logit_response','max_abs_logit_response')) and
                type(v['binary_changed_voxels']) is int and v['binary_changed_voxels']>=0,'Invalid measured response')
            if not v['condition_changed']:
                require(v['score']==row['normal'] and v['mean_abs_logit_response']==0 and
                    v['max_abs_logit_response']==0 and v['binary_changed_voxels']==0,'Numerical null produced a response')
    for group in mapping:
        indices=[i for i,r in enumerate(patches) if r['case_id']==group['case_id'] and r['stratum']==group['stratum'] and
            r['slot'] in group['eligible_slots']]
        require(sorted(tuple(patches[i]['normal_gain']) for i in indices)==
            sorted(tuple(patches[i]['interventions'][NAMES[0]]['gain']) for i in indices),'Raw gain multiset differs')
    result=summarize(patches,case_ids,probe_ids)
    require(finite_tree(result),'Nonfinite mechanism summary')
    return result


def recompute(data,protocol):
    require(data['schema']=='GTA-AVT-ordinal-public-derived-data-v1' and set(data['blocks'])==set(BLOCKS),
        'Exactly three named blocks required')
    require(protocol['schema']=='GTA-AVT-ordinal-consumption-prospective-protocol-v1','Wrong original protocol')
    partition=protocol['case_partitions'];case_ids=partition['TRAIN36'];probe_ids=partition['probe12_for_prerequisite_and_primary_mechanism']
    require(partition['VAL8']==VAL and len(case_ids)==36 and len(probe_ids)==12,'Original case partitions required')
    blocks={}
    for label,seed in zip(BLOCKS,SEEDS):
        block=data['blocks'][label];require(block['seed']==seed,'Frozen seed differs')
        native=native_report(block['native_case_metrics'],VAL)
        mechanism=validate_mechanism(block['prediction_rows'],block['patch_rows'],case_ids,probe_ids)
        blocks[label]=dict(seed=seed,utility=native,mechanism=mechanism,
            all_scientific_gates_passed=native['continue_gate']['passed'] and mechanism['both_mechanism_gates_passed'])
    eligible=all(blocks[k]['all_scientific_gates_passed'] for k in BLOCKS[:2])
    decision='confirmation_authorized' if eligible else 'terminal_failure_unsealing_only'
    require(data['certificate_decision']==decision and data['candidate_claim_eligible']==eligible,
        'Release decision must match independently recomputed A/B gates')
    success=eligible and blocks[BLOCKS[2]]['all_scientific_gates_passed']
    return dict(schema='GTA-AVT-ordinal-public-numeric-readout-v1',scientific_success=success,
        candidate_claim_eligible=eligible,certificate_decision=decision,terminal_failure_cannot_be_rescued=True,
        blocks=blocks,blocks_pooled=False,native_case_prediction_records=48,
        distinct_native_validation_case_ids=8,mechanism_rows_per_block=576,
        scope='Derived-number recomputation; not a rerun of segmentation or an independent observation of private execution',
        limitations=['The three blocks share the same previously reused VAL8 and common warmstart;48 records are not48 independent patients.',
            'Case SD is not training variance; no pristine-patient, unseen-source, zero-shot, clinical or significance claim.',
            'Both arms retain attribute supervision. Utility alone cannot establish dynamic correspondence.',
            'Only released derived counts/scores are checked. No image, mask, volume, checkpoint or model is opened.'])


def safe_file(root,name):
    rel=PurePosixPath(name)
    require(rel.parts and not rel.is_absolute() and '..' not in rel.parts and '\\' not in name and ':' not in name,
        'Unsafe public relative filename')
    p=root.joinpath(*rel.parts);require(p.resolve()==p and p.is_file(),'Missing/symlink public file')
    return p


NUMERIC_COMPARISON_POLICY = dict(
    schema='GTA-AVT-ordinal-description-variance-binary64-policy-v2',
    allowed_path='interventions.{frozen_intervention}.per_case.{case_id}.mechanism_support.{allowed_field}',
    allowed_interventions=['raw_consumed_score_permutation','own_presence_composition_permutation'],
    allowed_fields=['GT_ordinal_variance','normal_primary_FP32_c_variance'],
    maximum_binary64_ULP_distance=2,finite_nonnegative_variance_only=True,
    zero_requires_identical_binary64=True,positivity_must_match=True,
    dictionary_key_order_exact=True,list_order_exact=True,all_types_exact=True,
    all_other_values_exact=True,scientific_gates_use_recomputed_values=True,
    effects_thresholds_consumption_donors_counts_and_gate_booleans_exact=True)
NUMERIC_COMPARISON_POLICY_SHA256 = hashlib.sha256(json.dumps(NUMERIC_COMPARISON_POLICY,
    sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('utf-8')).hexdigest()

# The two allowed variance names and all comparison rules are copied from the
# frozen cross-platform metadata policy. Only the public-summary prefix differs.
NUMERIC_COMPARISON_POLICY_SOURCE_SHA256='17effbff68757705659afb687c3640c4ea4aab7ee38fc098207be03d1af157f6'
PUBLIC_SUMMARY_VARIANCE_SCOPE='blocks.{original_block}.mechanism.'+NUMERIC_COMPARISON_POLICY['allowed_path']


def variance_path(path):
    return (len(path)==9 and path[0]=='blocks' and path[1] in BLOCKS and path[2]=='mechanism' and
        path[3]=='interventions' and path[4] in NUMERIC_COMPARISON_POLICY['allowed_interventions'] and
        path[5]=='per_case' and type(path[6]) is str and path[7]=='mechanism_support' and
        path[8] in NUMERIC_COMPARISON_POLICY['allowed_fields'])


def binary64_bits(value):
    return struct.unpack('>Q',struct.pack('>d',value))[0]

def compare_metric_readouts(recomputed,recorded):
    """Strict structure comparison, with just the two descriptive variance ULP exceptions."""
    differences=[];accepted=[]
    def fail(path,reason,left,right):
        differences.append(dict(path=list(path),reason=reason,recomputed=repr(left),recorded=repr(right)))
    def walk(left,right,path):
        if type(left) is not type(right):
            fail(path,'type_differs',left,right);return
        if isinstance(left,dict):
            if list(left)!=list(right) or any(type(a) is not type(b) for a,b in zip(left,right)):
                fail(path,'dictionary_keys_or_order_differs',list(left),list(right));return
            for key in left:walk(left[key],right[key],path+(key,))
        elif isinstance(left,(list,tuple)):
            if len(left)!=len(right):fail(path,'sequence_length_differs',len(left),len(right));return
            for index,(a,b) in enumerate(zip(left,right)):walk(a,b,path+(index,))
        elif variance_path(path) and left is not None:
            if type(left) is not float or not math.isfinite(left) or not math.isfinite(right) or left<0 or right<0:
                fail(path,'variance_must_be_finite_nonnegative_binary64_float',left,right);return
            lb,rb=binary64_bits(left),binary64_bits(right)
            if lb==rb:return
            if left==0 or right==0 or (left>0)!=(right>0):
                fail(path,'variance_zero_or_positivity_differs',left,right);return
            distance=abs(lb-rb)
            if distance>NUMERIC_COMPARISON_POLICY['maximum_binary64_ULP_distance']:
                fail(path,'variance_exceeds_two_binary64_ULPs',left,right);return
            accepted.append(dict(path=list(path),binary64_ULP_distance=distance,
                recomputed=left,recorded=right))
        elif type(left) is float:
            if not math.isfinite(left) or not math.isfinite(right) or binary64_bits(left)!=binary64_bits(right):
                fail(path,'exact_binary64_value_differs',left,right)
        elif left!=right:
            fail(path,'exact_value_differs',left,right)
    walk(recomputed,recorded,())
    return dict(equal=not differences,policy_sha256=NUMERIC_COMPARISON_POLICY_SHA256,
        accepted_descriptive_variance_rounding=accepted,exact_comparison_failures=differences)

def metric_readouts_equal(recomputed,recorded):
    return compare_metric_readouts(recomputed,recorded)['equal']


def check_package(package):
    root=Path(package).resolve();manifest=json.loads((root/'manifest.json').read_bytes())
    require(manifest['schema']=='GTA-AVT-ordinal-public-numeric-manifest-v1','Wrong numerical package')
    expected={'recompute.py','protocol.json','numerical_null_addendum.json','arithmetic_identity.json','expected_summary.json','README.txt'}
    expected.update('blocks/'+label+'/'+name for label in BLOCKS for name in
        ('native_case_metrics.json','prediction_rows.json','patch_rows.json'))
    require(set(manifest['files'])==expected,'Every exact public input must be manifest-bound')
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==expected|{'manifest.json'},
        'Unexpected or unlisted public file')
    for name,ref in manifest['files'].items():
        p=safe_file(root,name);raw=p.read_bytes()
        require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'Public file bytes differ: '+name)
    protocol=json.loads(safe_file(root,'protocol.json').read_bytes())
    require(hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()==PROTOCOL_SHA and
        hashlib.sha256((root/'numerical_null_addendum.json').read_bytes()).hexdigest()==NULL_SHA,'Frozen protocol/addendum differs')
    identity=json.loads((root/'arithmetic_identity.json').read_bytes())
    require(identity['schema']=='GTA-AVT-ordinal-public-pure-arithmetic-identity-v1' and
        set(identity['functions'])=={'require','f32','finite_tree','permutations','average','aggregate','summarize',
            'dist','scaled','paired','count_summary','utility','metric'} and identity['scientific_arithmetic_changed'] is False,
        'Exact frozen arithmetic inventory required')
    tree=ast.parse(Path(__file__).read_text(encoding='utf-8'))
    funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    for name,ref in identity['functions'].items():require(ast_digest(funcs[name])==ref['canonical_AST_sha256'],'Pure arithmetic identity differs')
    data=dict(schema='GTA-AVT-ordinal-public-derived-data-v1',certificate_decision=manifest['certificate_decision'],
        candidate_claim_eligible=manifest['candidate_claim_eligible'],blocks={})
    for label,seed in zip(BLOCKS,SEEDS):
        home=root/'blocks'/label
        data['blocks'][label]=dict(seed=seed,native_case_metrics=json.loads((home/'native_case_metrics.json').read_bytes()),
            prediction_rows=json.loads((home/'prediction_rows.json').read_bytes()),patch_rows=json.loads((home/'patch_rows.json').read_bytes()))
    result=recompute(data,protocol)
    require(metric_readouts_equal(result,json.loads((root/'expected_summary.json').read_bytes())),'Expected summary differs from independent recomputation')
    require(result['scientific_success']==manifest['scientific_success'],'Manifest scientific decision differs')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package');p.add_argument('--check',action='store_true');a=p.parse_args()
    if not a.check:
        print(json.dumps(dict(check=False,derived_numbers_read=False,models_executed=False,standard_library_only=True)));return
    require(a.package,'Explicit --package required');result=check_package(a.package)
    print(json.dumps(dict(check='passed',scientific_success=result['scientific_success'],
        candidate_claim_eligible=result['candidate_claim_eligible'],certificate_decision=result['certificate_decision'],
        blocks_pooled=False,models_executed=False)))


if __name__=='__main__':main()
