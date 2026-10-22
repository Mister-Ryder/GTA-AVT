"""Public four-slot statistics: five byte-identical functions from frozen427.

Only the released31f mathematics module is imported; no private provenance,
server workflow or model is accessed. Source authentication belongs to the
separate portable reader. This module alone does not authenticate input data.
"""
from pathlib import Path
import statistics
import sys
import analyze_key_ablations as math4

SCHEMA='GTA-AVT-minimal4-paired-case-analysis-v1'
NUMERIC_INPUT_SCHEMA='GTA-AVT-minimal4-numeric-input-v1'
SCOPE='bounded_20261022_v2_minimal4'
SLOTS=('main_C_concat','main_A_soft','ablations_A_soft_platform_bridge','ablations_A_soft_constant')
RULES_SHA='347d0850f9e54150f63c3cdee7a69216c22344eadc2b023a25e8fb92287ab565'
MATH_SHA='31f355f735f6744589e464bbfb50cef276bc4db650f5edc70c5f6938d834f3c4'
COMPARISONS=(('MAIN_A_minus_C','main_A_soft','main_C_concat'),
             ('SECOND_A_minus_constant','ablations_A_soft_platform_bridge','ablations_A_soft_constant'))
ORIGINAL_FIVE_FUNCTION_SOURCE_SHA256='427c4feeff804c13ce3e5033d02e56fe17aa837f47d69cde4f281bef322b2c45'
ORIGINAL_FUNCTION_SHA256={'distribution': '1f012ccf9c6cf6d1cc7243aad5c06ff33efb585ae369bdc4ab3af0ac1b64297b', 'scaled': 'beb37388ff42c2e4694bb116609acca94b68bff5758cc1b5554cf5205bdb4508', 'figure_cases': 'eb56731942c9a3303030c3d602dc290f46816316c39e11e1d9d0a61280599fe9', 'analyze_case_rows': 'd09439fc535ef38c6a7496bd2a9c0d879b59787096705aba55fcc0f0c4ca298f', 'text': 'ba61526b5ac7785571525de6014514de7ddd8dcf30c25dae0d631567460b90b4'}


def distribution(values):
    result=math4.distribution(values)
    return dict(cases=result['total_cases'],defined_cases=result['defined_cases'],missing_cases=result['missing_cases'],
        all_case_mean=result['all_patient_mean'],defined_case_mean=result['defined_patient_mean'],sample_SD=result['std_defined_patients'])


def scaled(row,key):
    value=math4.number(row[key],key)
    return None if value is None else value*math4.METRICS[key][1]


def figure_cases(ids,rows_a,rows_b):
    """One candidate per ordered rule; duplicates do not cause replacement/filling."""
    delta=lambda c,k:None if rows_a[c].get(k) is None or rows_b[c].get(k) is None else rows_a[c][k]-rows_b[c][k]
    candidates=[('worst_Dice_difference',min(ids,key=lambda c:(delta(c,'dice'),ids.index(c))))]
    mixed=[c for c in ids if delta(c,'dice')>0 and delta(c,'cldice') is not None and delta(c,'cldice')<0]
    if mixed:candidates.append(('Dice_improves_clDice_declines_most',min(mixed,key=lambda c:(delta(c,'cldice'),ids.index(c)))))
    false_positive=[c for c in ids if delta(c,'fp_ml')>0]
    if false_positive:candidates.append(('largest_positive_FP_increase',max(false_positive,key=lambda c:(delta(c,'fp_ml'),-ids.index(c)))))
    selected=[]
    for reason,cid in candidates:
        existing=next((r for r in selected if r['case_id']==cid),None)
        if existing:existing['reasons'].append(reason)
        else:selected.append(dict(case_id=cid,reasons=[reason]))
    return selected


def analyze_case_rows(index,rows):
    """Pure math with an explicit provenance level; never authenticates a source."""
    ids=index['case_ids'];stage=index['stage'];expected=math4.FIXED_CASES['full_frozen8' if stage=='val' else 'final_test12']
    if stage not in ('val','test') or index.get('schema')!=NUMERIC_INPUT_SCHEMA or ids!=expected or set(rows)!=set(SLOTS) or set(index['slots'])!=set(SLOTS):raise ValueError('Complete fixed four-slot/stage input required')
    if math4.sha(Path(math4.__file__))!=MATH_SHA:raise ValueError('Frozen pure analysis math changed')
    summaries={};casewise=[]
    for slot in SLOTS:
        identity=index['slots'][slot]['bundle_identity']
        role,job=slot.split('_',1);method='faithful_concat' if slot=='main_C_concat' else 'faithful_soft_constant' if slot=='ablations_A_soft_constant' else 'faithful_soft'
        if any(identity.get(k)!=v for k,v in dict(role=role,job=job,method=method).items()):raise ValueError('Fixed slot declaration differs')
        if set(rows[slot])!=set(ids):raise ValueError('Every exact frozen case must be present')
        for cid,row in rows[slot].items():
            if row.get('case_id')!=cid or (row.get('cohort') is not None and row['cohort']!=math4.cohort(cid)):raise ValueError('Wrong case/cohort')
            for key in math4.METRICS:math4.number(row[key],key)
            for group in math4.GROUPS:math4.group_record(row,group)
            if any(row[k] is None for k in ('dice','precision','recall','fp_ml','fn_ml')):raise ValueError('Defined overlap/FP/FN required')
            casewise.append(dict(slot=slot,case_id=cid,cohort=math4.cohort(cid),**{k:scaled(row,k) for k in math4.METRICS},
                **{k+'_missing_reason':math4.missing_reason(row,k) for k in math4.METRICS},FP_plus_FN_ml=row['fp_ml']+row['fn_ml']))
        summaries[slot]={scope:{key:distribution([scaled(rows[slot][c],key) for c in selected]) for key in math4.METRICS}
            for scope,selected in math4.scopes(ids)}
    # GT-only supports must agree across all four slots, beyond the two contrasts.
    # Unrecorded groups remain legal; they never supply an imputed zero support.
    for cid in ids:
        truth=set();groups={g:set() for g in math4.GROUPS}
        for slot in SLOTS:
            value=rows[slot][cid].get('truth_skeleton_support')
            if value is not None:
                if type(value) is not int or value<0:raise ValueError('GT skeleton support requires a nonnegative integer')
                truth.add(value)
            for group in math4.GROUPS:
                support,_,_=math4.group_record(rows[slot][cid],group)
                if support is not None:groups[group].add(support)
        if len(truth)>1 or any(len(values)>1 for values in groups.values()):raise ValueError('Explicit GT skeleton/group support differs across four slots')
    comparisons={};figure_selection={}
    for name,a,b in COMPARISONS:
        ia,ib=index['slots'][a]['bundle_identity'],index['slots'][b]['bundle_identity']
        if ia['role']!=ib['role']:raise ValueError('Cross-training-role contrast forbidden')
        ra,rb=rows[a],rows[b];cases=[]
        for cid in ids:
            cases.append(dict(case_id=cid,cohort=math4.cohort(cid),after={k:scaled(ra[cid],k) for k in math4.METRICS},
                before={k:scaled(rb[cid],k) for k in math4.METRICS},
                delta_after_minus_before={k:None if ra[cid][k] is None or rb[cid][k] is None else scaled(ra[cid],k)-scaled(rb[cid],k) for k in math4.METRICS},
                FP_plus_FN_after_ml=ra[cid]['fp_ml']+ra[cid]['fn_ml'],FP_plus_FN_before_ml=rb[cid]['fp_ml']+rb[cid]['fn_ml']))
        scopes={}
        for scope,selected in math4.scopes(ids):
            metrics={key:dict(difference_unit=unit,**math4.paired([scaled(ra[c],key) for c in selected],[scaled(rb[c],key) for c in selected],direction))
                for key,(unit,_,direction) in math4.METRICS.items()}
            geometry={}
            for group in math4.GROUPS:
                ga=[math4.group_record(ra[c],group) for c in selected];gb=[math4.group_record(rb[c],group) for c in selected]
                for x,y in zip(ga,gb):
                    if x[0] is not None and y[0] is not None and x[0]!=y[0]:raise ValueError('Explicit GT group support differs between methods')
                geometry[group]=dict(**math4.paired([None if x[1] is None else 100*x[1] for x in ga],[None if x[1] is None else 100*x[1] for x in gb],1),
                    no_GT_support_cases=sum(x[0]==0 and y[0]==0 for x,y in zip(ga,gb)),
                    unrecorded_cases=sum(x[0] is None or y[0] is None for x,y in zip(ga,gb)),
                    case_support_and_missing=[dict(case_id=c,after_GT_nodes=x[0],before_GT_nodes=y[0],after_recall=x[1],before_recall=y[1],
                        missing_after=x[2],missing_before=y[2]) for c,x,y in zip(selected,ga,gb)],
                    precision_scope='GT-group recall only; no group precision; combine global FP/CLprecision',aggregation='equal supported case pairs, never pooled nodes')
            scopes[scope]=dict(metrics=metrics,geometry=geometry,cohort_scope_is_descriptive_only=scope!='all')
        dice=[c['delta_after_minus_before']['dice'] for c in cases]
        leaveout={cid:statistics.mean(v for c,v in zip(ids,dice) if c!=cid) for cid in ids}
        cost=[c['FP_plus_FN_after_ml']-c['FP_plus_FN_before_ml'] for c in cases]
        comparisons[name]=dict(role=ia['role'],after_slot=a,before_slot=b,cases=cases,scopes=scopes,
            worst_Dice_difference_pp=min(dice),worst_Dice_cases=[c for c,v in zip(ids,dice) if v==min(dice)],
            leave_one_case_out_Dice_mean_difference_pp=leaveout,leave_one_out_range_pp=[min(leaveout.values()),max(leaveout.values())],
            leave_one_out_is_CI=False,FP_plus_FN_difference_ml=distribution(cost),FP_plus_FN_is_derived_descriptive_cost=True)
        figure_selection[name]=dict(case_selection=figure_cases(ids,ra,rb),native_FOV=True,GT_volume_axial_quantiles=[.25,.5,.75],
            shared_CT_window_and_colors=True,maximum_distinct_cases=3,no_replacement_or_filling=True,predictions_rendered_by_this_analysis=False)
    return dict(schema=SCHEMA,stage=stage,scope=SCOPE,case_ids=ids,case_unit_metadata=math4.CASE_UNIT_METADATA,
        source_verification_level=index.get('source_verification_level','not_supplied'),
        source_index_sha256=None,source_marker_FILE_sha256=index['marker_file_sha256'],selection_payload_sha256=index['selection_payload_sha256'],
        common_reference_runtime=index['common_reference_runtime'],analysis_runtime=dict(python=sys.version,numpy=math4.np.__version__),
        final_readout_rules_sha256=RULES_SHA,statistical_math=dict(source_file_sha256=math4.sha(Path(math4.__file__)),
            seed=math4.SEED,bootstrap_draws=math4.DRAWS,interval='exploratory95%percentile paired-case; no seed/selection/multiplicity correction'),
        slot_summaries=summaries,casewise=casewise,comparisons=comparisons,figure_selection=figure_selection,
        availability_only_VAL_evidence=index.get('availability_only_VAL'),
        limitations=['Two distinct within-role contrasts, never pooled or subtracted; no cross-role GTA effect.',
            'MAIN A-C changes operations and predicted conditioning jointly; missing MAIN constant prevents decomposition.',
            'SECOND A-constant changes empty mass, nonempty composition and gradient paths jointly; not isolated anatomy.',
            'VAL8 and TEST12 remain separate; all case files required while null/unsupported metrics retain complete-pair support.',
            'GT recall and Lee voxel coverage do not establish group precision, branch length/topology or local causation.',
            'Availability-only and classifier evidence remain separate diagnostics; never subtract from trained-constant effect.',
            'Training component-loss interpretation requires the separately collected actual logs; reference metrics do not prove optimization causes.'])


def text(result):
    f=lambda x:'NA' if x is None else '%.4f'%x
    lines=['四模型两项平台内配对：'+result['stage']+'；released case独立性未核实。',
        'VAL8与TEST12分开；概率变化pp，体积ml，距离mm。区间是20000次case配对bootstrap探索性percentile95%，不是确认性显著性。']
    for name,record in result['comparisons'].items():
        lines+=['','比较 '+name+'：'+record['after_slot']+' 减 '+record['before_slot'], 'metric | mean delta [exploratory CI] | pairs | W/T/L']
        for key,value in record['scopes']['all']['metrics'].items():lines.append('%s | %s [%s,%s] | %d | %d/%d/%d'%(key,f(value['delta_a_minus_b']),f(value['ci95_lower']),f(value['ci95_upper']),value['paired_cases'],value['wins'],value['ties'],value['losses']))
        lines+=['逐case Dice/FP/FN/clDice差：']
        for row in record['cases']:lines.append(row['case_id']+' | '+' / '.join(f(row['delta_after_minus_before'][k]) for k in ('dice','fp_ml','fn_ml','cldice')))
        lines+=['worst Dice difference '+f(record['worst_Dice_difference_pp'])+'pp；leave-one-out range '+str(record['leave_one_out_range_pp'])+'（非CI）。',
            'FP+FN派生代价平均变化 '+f(record['FP_plus_FN_difference_ml']['all_case_mean'])+'ml。', 'GT组recall差：']
        for group,value in record['scopes']['all']['geometry'].items():lines.append(group+' | '+f(value['delta_a_minus_b'])+'pp, pairs='+str(value['paired_cases'])+', unrecorded='+str(value['unrecorded_cases'])+', noGT='+str(value['no_GT_support_cases']))
        lines.append('固定规则选图case：'+str(result['figure_selection'][name]['case_selection'])+'；本工具未渲染/查看native预测。')
    lines+=['','完整原值/差值、全指标缺失与来源组、几何支持节点数见JSON/CSV；节点不是独立N。',*result['limitations']]
    return '\n'.join(lines)+'\n'

