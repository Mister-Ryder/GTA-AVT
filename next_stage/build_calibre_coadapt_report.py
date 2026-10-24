"""Scalar-only public report from the completed fixed calibre readout.

Default inert. --execute reads one explicitly hashed final CPU analysis only;
it never opens seed endpoints, arrays, CT images, checkpoints or live jobs.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re

READOUT_SHA = '478a41fd87b5d4def485c2a010f0c76e69ee29f089ab7185b9a5a94cd573871d'
ARMS = ('C0', 'A_only', 'AR')
SEEDS = (20261030, 20261031, 20261032)
STRATA = ('primary', 'all16', 'GT_jitter_all8', 'CT_uniform_all8', 'uniform_known_nonempty')
METRICS = ('dice', 'cldice', 'centerline_recall', 'centerline_precision', 'fine_recall', 'BCE', 'FP', 'FN', 'TP')
SOURCES = ('Dongyang', 'KiTS', 'Rider')
CASE_FIELDS = ('source',) + STRATA + ('uniform_empty_patches', 'uniform_empty_FP_voxels',
    'historical_patch_attribute_unknown_slots', 'oracle_support')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def csv_bytes(rows):
    names = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=names, lineterminator='\n')
    writer.writeheader()
    for source in rows:
        row = {k:json.dumps(v, ensure_ascii=False, separators=(',', ':')) if isinstance(v,(list,dict)) else v
               for k,v in source.items()}
        writer.writerow(row)
    return stream.getvalue().encode('utf-8')


def project(value, names, label):
    require(isinstance(value,dict) and set(value)==set(names), 'Unexpected report schema: '+label)
    return copy.deepcopy(value)


def finite_tree(value):
    if isinstance(value,dict):
        require(all(isinstance(k,str) for k in value), 'Non-string report key')
        for item in value.values(): finite_tree(item)
    elif isinstance(value,list):
        for item in value: finite_tree(item)
    elif isinstance(value,float):
        require(math.isfinite(value), 'Undefined measurements must be null, not NaN/Inf')
    else:
        require(value is None or type(value) in (bool,int,str), 'Non-scalar report payload')


def private_string_check(value):
    """Reject path/machine payloads after strict fixed-schema projection."""
    if isinstance(value,dict):
        for key,item in value.items():
            require(not any(t in key.lower() for t in ('credential','password','gpu_uuid','start_ticks','checkpoint_path','cache_dir')),
                    'Private metadata key in report')
            private_string_check(item)
    elif isinstance(value,list):
        for item in value: private_string_check(item)
    elif isinstance(value,str):
        lower=value.lower()
        require(not any(t in lower for t in ('/root/','/autodl','\\users\\','connect.westc','gpu-','ssh -p')) and
                re.search(r'[A-Za-z]:[\\/]',value) is None,
                'Private path/machine string in report')


def public_result(actual, readout, analysis_sha):
    require(actual.get('schema')=='GTA-AVT-calibre-coadapt-final-readout-v1' and
        actual.get('readout_sha256')==READOUT_SHA and actual.get('method')==readout.METHOD and
        actual.get('bank_sha256')==readout.BANK_SHA and tuple(s['seed'] for s in actual['per_seed'])==SEEDS,
        'Final fixed CPU readout required')
    ids=actual['probe_cases']
    require(len(ids)==len(set(ids))==12 and all(re.fullmatch(r'(Dongyang|KiTS|Rider)_[A-Za-z0-9]+',c) for c in ids),
        'Original complete12 case IDs required')
    require(actual['all_three_seeds_retained'] is True and actual['seeds_pooled'] is False and
        actual['oracle_conditioned_not_CT_only'] is True and actual['mathematical_or_true_upper_bound'] is False and
        actual['GTA_goal_completed'] is False and actual['next_experiment_authorized'] is False and
        actual['native_confirmation_required'] is True, 'Oracle/TRAIN scope differs')
    seeds=[]
    for index,seed in enumerate(actual['per_seed']):
        require(seed['native_validation'] is False and seed['GTA_goal_completed'] is False and seed['training_authorized'] is False,
                'Seed scope differs')
        arms={}
        for arm in ARMS:
            item=seed['arms'][arm]
            fields=('per_case','expected_cases','Dice_population_SD','own_worst2','source_groups','strata_case_equal','matched_worst2')
            fields+=METRICS+tuple(k+'_supported_cases' for k in METRICS)
            clean=project(item,fields,'arm')
            require(clean['expected_cases']==12 and set(clean['per_case'])==set(ids) and
                set(clean['source_groups'])==set(SOURCES) and set(clean['strata_case_equal'])==set(STRATA),
                'Complete arm population required')
            for cid,value in clean['per_case'].items():
                project(value,CASE_FIELDS,'case')
                require(value['source']==cid.split('_')[0], 'Case/source mismatch')
                for stratum in STRATA:
                    metric_fields=('patches','fine_patch_mean_recall','fine_nodes','fine_covered','GT_nodes','GT_covered',
                        'pooled_patch_GT_node_recall_descriptive')+METRICS+tuple(k+'_defined_patches' for k in METRICS)
                    project(value[stratum],metric_fields,'case/stratum')
                project(value['oracle_support'],('all16_mean_available_sites','primary_mean_available_sites','all16_mean_unknown_sites'),'oracle coverage')
            project(clean['own_worst2'],('case_ids','Dice_mean'),'own tail')
            project(clean['matched_worst2'],('reference_arm','case_ids','Dice_means'),'matched tail')
            require(set(clean['matched_worst2']['Dice_means'])==set(ARMS), 'Matched tail arms incomplete')
            arms[arm]=clean
        controls={control:project(seed['AR_vs_controls'][control],
            ('control','per_case','metrics','mean_Dice_positive','no_vascular_or_tail_harm','source_groups'),'comparison')
            for control in ('C0','A_only')}
        for control,value in controls.items():
            require(value['control']==control and set(value['per_case'])==set(ids) and set(value['metrics'])==set(METRICS),
                    'Complete paired comparison required')
            for case in value['per_case'].values():project(case,METRICS,'paired case metrics')
        mechanism=project(seed['mechanisms']['own_A_half_ring_radius'],('intervention','per_case','supported_case_ids',
            'mean_correct_minus_permuted_Dice_pp','positive_cases','parts','passed','source_groups',
            'own_A_and_radius_multiset_preserved'),'mechanism')
        require(set(seed['mechanisms'])=={'own_A_half_ring_radius'} and set(mechanism['per_case'])==set(ids) and
            mechanism['own_A_and_radius_multiset_preserved'] is True, 'Complete same-weight intervention required')
        for case in mechanism['per_case'].values():
            project(case,('source','supported','null_reasons','primary_patches','radius_sites','raw_log_radius_variance',
                'actual_zR_changed_patches','correct_minus_permuted_Dice_pp','descriptive_Dice_pp',
                'permuted_minus_correct_BCE','correct_minus_permuted_FP','correct_minus_permuted_FN',
                'mean_abs_logit_response','binary_changed_voxels'),'mechanism case')
        screen=project(seed['screen'],('parts','passed','status'),'screen')
        gate_names={control+'_AR_mean_Dice_positive' for control in ('C0','A_only')}
        gate_names|={control+'_'+name for control in ('C0','A_only') for name in
            ('cldice_nonworse','centerline_recall_nonworse','fine_recall_nonworse','own_worst2_Dice_nonworse')}
        gate_names.add('own_A_half_ring_radius_sameweight_gate')
        require(set(screen['parts'])==gate_names and set(mechanism['parts'])==
            {'all12_supported','mean_effect_at_least_point1pp','at_least8_positive'}, 'Locked gate inventory differs')
        require(all(v is None or type(v) is bool for v in screen['parts'].values()) and
            screen['passed'] is (None if any(v is None for v in screen['parts'].values()) else all(screen['parts'].values())),
            'Locked screen flags inconsistent')
        seeds.append(dict(seed=seed['seed'],role='PRIMARY' if index==0 else 'predeclared_replica_'+str(index),
            arms=arms,AR_vs_controls=controls,mechanisms={'own_A_half_ring_radius':mechanism},screen=screen))
    flags=[s['screen']['passed'] for s in seeds]
    require(actual['primary_seed']==SEEDS[0] and actual['primary_screen_passed'] is flags[0] and
        actual['all3_screens_passed'] is (None if any(v is None for v in flags) else all(flags)),
        'Whole/primary decision differs from fixed reader')
    result=dict(schema='GTA-AVT-calibre-coadapt-public-report-v1',method=copy.deepcopy(readout.METHOD),probe_cases=ids,
        recipe=dict(patch_size=128,batch_size=2,successful_updates_per_arm=2000,augmentation=False,
            optimizer='AdamW',shared_projection_learning_rate=.0001,gate_learning_rate=.001,
            weight_decay=.00001,gradient_clipping=False,alpha='min(1, successful_step/250)',
            sampler='FIT24 case-uniform and original16-slot-uniform; same plan per seed across all arms',
            trainable_shared_parameters=4226,trainable_gate_parameters=192,trainable_total_parameters=4418),
        per_seed=seeds,primary_seed=SEEDS[0],primary_screen_passed=actual['primary_screen_passed'],
        all3_screens_passed=actual['all3_screens_passed'],primary_cannot_be_rescued=True,seeds_pooled=False,
        scope=dict(oracle_capability_not_CT_GTA=True,not_native_validation=True,not_independent_validation=True,
            not_a_true_or_mathematical_upper_bound=True,no_attribute_auxiliary_loss=True,
            native_confirmation_required=True,next_experiment_authorized=False,GTA_goal_completed=False),
        units=dict(metric_fractions='Dice/clDice/recall stored0..1; displayed percentages are fraction×100',
            effects='Dice/recall differences are percentage points, not relative percent',
            FP_FN_TP='Mean voxel counts per sampled patch, then case equal; not native whole-case volumes',
            SD='Population SD across defined PROBE12 case Dice fractions; displayed×100pp',
            fine='Per case sum fine_covered / sum fine_nodes, then mean of supported cases; patch means separately retained',
            null='Undefined/unsupported remains null; CSV empty cell is null, never zero',
            tail='Own worst2 can contain different cases per arm; matched-worst2 holds reference case IDs fixed',
            dependence='Repeated/overlapping patches, same12 TRAIN cases and three seeds are not independent validation cohorts'),
        source_bindings=dict(final_analysis_sha256=analysis_sha,readout_sha256=READOUT_SHA,model_sha256=readout.MODEL_SHA,
            worker_sha256=readout.WORKER_SHA,source_manifest_sha256=readout.SOURCE_SHA,
            initial_common_weights_sha256=readout.WEIGHT_SHA,TRAIN_patch_bank_sha256=readout.BANK_SHA,
            oracle_bank_sha256=readout.ORACLE_BANK_SHA,oracle_helper_sha256=readout.ORACLE_SHA))
    finite_tree(result);private_string_check(result)
    return result


def tables(result):
    cases,mechanisms,groups=[],[],[]
    for seed in result['per_seed']:
        for arm in ARMS:
            value=seed['arms'][arm]
            for cid,case in value['per_case'].items():
                for stratum in STRATA:
                    cases.append(dict(seed=seed['seed'],arm=arm,case_id=cid,source=case['source'],stratum=stratum,
                        **case[stratum],uniform_empty_patches=case['uniform_empty_patches'],
                        uniform_empty_FP_voxels=case['uniform_empty_FP_voxels'],
                        historical_patch_attribute_unknown_slots=case['historical_patch_attribute_unknown_slots'],
                        **case['oracle_support']))
            for stratum in STRATA:
                aggregate=value['strata_case_equal'][stratum]
                for label,metrics in [('all12',aggregate['metrics'])]+list(aggregate['source_groups'].items()):
                    row=dict(seed=seed['seed'],arm=arm,stratum=stratum,group=label,
                        expected_cases=12 if label=='all12' else sum(c.startswith(label+'_') for c in result['probe_cases']))
                    for metric in METRICS:
                        row[metric]=metrics[metric]['mean'];row[metric+'_supported_cases']=metrics[metric]['supported_cases']
                    groups.append(row)
        for cid,case in seed['mechanisms']['own_A_half_ring_radius']['per_case'].items():
            mechanisms.append(dict(seed=seed['seed'],case_id=cid,intervention='own_A_half_ring_radius',**case))
    require(len(cases)==540 and len(mechanisms)==36 and len(groups)==180,'Complete report row counts differ')
    return cases,mechanisms,groups


def number(value,scale=1.,signed=False):
    if value is None:return '未定义'
    return ('%+.6f' if signed else '%.6f')%(scale*value)


def render(result):
    flags=[s['screen']['passed'] for s in result['per_seed']]
    if result['all3_screens_passed'] is True:
        outcome='三个预定种子均满足本轮TRAIN能力筛查。该结果仍需前瞻native确认，不能写成CT条件化GTA已经有效。'
    elif result['all3_screens_passed'] is None:
        outcome='至少一组存在支持不足或未定义判定，本轮不能作完整阳性结论；技术/支持不确定性不能填0当科学阴性。'
    else:
        outcome='本轮未满足全部预定TRAIN能力筛查。完整保留每个种子的结果；其他种子不能救回PRIMARY失败。'
    lines=['局部半径oracle与有限共适应：固定TRAIN能力筛查','',outcome,'',
        '范围与对照',
        'GT稀疏native-EDT代理支持A与连续log半径R被直接给定；这不是CT预测属性或可部署GTA，也不是性能的数学上界。FIT24训练、PROBE12读出均来自既有TRAIN36；旧骨干已训练过这些病例，反复使用的区块/病例不是独立或盲验证。没有VAL/TEST结果。',
        '三臂均从共同7406状态出发，各2000个成功更新，128³输入/B2、无增强；仅原SAAM融合投影4160参数、最终线性投影66参数与三组零初始化系数192参数可学，共4418；其余参数与全部BN/Dropout保持冻结/eval。损失仅原Dice+BCE，未加属性、深监督或GDF损失。AdamW共享投影学习率1e−4、门控1e−3、weight_decay1e−5，无梯度裁剪，alpha=min(1,step/250)。不将新增属性定义与共适应的跨轮变化解释成单因素因果。',
        'C0两个条件轴为0；A_only只用同一支持A；AR再用A×(logR−固定FIT24均值)/SD。A0处仅中性化，不称未知为背景或半径0。F0 + alpha×tanh(b+v×zA+w×zR)×Dθ；Dθ使用既有不同感受野分支的固定符号对比和可学习原融合W，不把分支命名成解剖专家。',
        '固定判定',
        '训练间AR相对C0与A_only的病例等权平均Dice必须分别严格>0；clDice、中心线召回、≤2mm细支召回在相同病例支持上不降，且各臂自身最差2例平均Dice不降。这里没有挪用机制的0.1pp/8例条件给训练间比较。',
        '同权机制固定ownA及完整R多重集，在每区块A1位置展平序列半环移动floor(N/2)。正确−错配Dice均值≥0.1pp、至少8例严格为正且12/12支持，才通过机制检查。支持还要求每例至少2个primary区块、正logR方差、实际FP32 zR改变及有限测量。未知/缺支持保留null，不补0。','']
    for seed in result['per_seed']:
        status={True:'通过',False:'未通过',None:'支持不足/未定义'}[seed['screen']['passed']]
        lines.append('种子%s（%s）：%s'%(seed['seed'],seed['role'],status))
        lines.append('  臂：Dice% / clDice% / 中心线召回% / 细支召回% / Dice病例SD(pp) / 自身最差2例Dice%')
        for arm in ARMS:
            a=seed['arms'][arm]
            lines.append('  %s：%s / %s / %s / %s / %s / %s'%(arm,
                number(a['dice'],100),number(a['cldice'],100),number(a['centerline_recall'],100),
                number(a['fine_recall'],100),number(a['Dice_population_SD'],100),number(a['own_worst2']['Dice_mean'],100)))
        for control,comparison in seed['AR_vs_controls'].items():
            metrics=comparison['metrics']
            lines.append('  AR−%s：Dice %s pp（正%d/支持%d例），clDice %s pp，中心线召回 %s pp，细支召回 %s pp；FP %s、FN %s 个/采样区块。'%(control,
                number(metrics['dice']['mean_AR_minus_control_pp'],signed=True),metrics['dice']['positive_cases'],len(metrics['dice']['case_ids']),
                number(metrics['cldice']['mean_AR_minus_control_pp'],signed=True),number(metrics['centerline_recall']['mean_AR_minus_control_pp'],signed=True),
                number(metrics['fine_recall']['mean_AR_minus_control_pp'],signed=True),number(metrics['FP']['mean_AR_minus_control'],signed=True),
                number(metrics['FN']['mean_AR_minus_control'],signed=True)))
        m=seed['mechanisms']['own_A_half_ring_radius']
        lines.append('  同权正确−半环错配：Dice %s pp，正%d/支持%d例，机制检查%s。'%(number(m['mean_correct_minus_permuted_Dice_pp'],signed=True),
            m['positive_cases'],len(m['supported_case_ids']),{True:'通过',False:'未通过',None:'不可判'}[m['passed']]))
        failed=[k for k,v in seed['screen']['parts'].items() if v is False]
        undefined=[k for k,v in seed['screen']['parts'].items() if v is None]
        lines.append('  未满足检查：'+('；'.join(failed) if failed else '无')+'。未定义检查：'+('；'.join(undefined) if undefined else '无')+'。')
        for reference in ARMS:
            tail=seed['arms'][reference]['matched_worst2']
            lines.append('  固定%s最差2例[%s]，C0/A_only/AR的匹配Dice%%：%s。'%(reference,','.join(tail['case_ids']),
                '/'.join(number(tail['Dice_means'][arm],100) for arm in ARMS)))
        lines.append('')
    lines += ['解释边界与完整数据',
        '非零参数、条件变化、logit响应与对应关系有益是不同证据。平均Dice或正病例个数不能单独替代全部锁定判定；跨病例SD降低也不能自动称测试时低方差或稳定性提高。自身最差2例可能不是同一病例，固定病例的matched-worst2同时保留。',
        '所有种子单独呈现，不合池、不选最佳种子。病例内重叠区块不是独立样本；细支指标使用每例累计fine_covered/fine_nodes后病例等权，另保留patch均值及支持数。FP/FN是采样区块平均计数，不是native整例体积。',
        'casewise.csv保留3种子×3臂×12病例×5采样层共540行；mechanism_cases.csv保留36例次及全部null理由/响应；source_strata_metrics.csv保留180个来源×采样层分组。summary.json完整保留mean、支持数、SD、自身与匹配tail、每项锁定检查。CSV空单元格表示null，不能作0。',
        '本包只报告已完成CPU读出；不授权下一训练、不证明真实CT属性泛化或native分割收益，不宣称达到GTA目标。下一决策由完整结果另行制定。','']
    return '\n'.join(lines)


def execute(args):
    require(re.fullmatch('[0-9a-f]{64}',args.analysis_sha256 or '') is not None,'Explicit analysis SHA256 required')
    require(sha(args.analysis)==args.analysis_sha256 and sha(args.readout)==READOUT_SHA,'Pinned analysis/readout differs')
    spec=importlib.util.spec_from_file_location('_calibre_report_fixed_readout',args.readout)
    readout=importlib.util.module_from_spec(spec);spec.loader.exec_module(readout)
    actual=json.loads(Path(args.analysis).read_bytes())
    result=public_result(actual,readout,args.analysis_sha256)
    cases,mechanisms,groups=tables(result)
    output=Path(args.output);require(not output.exists(),'Fresh report output required')
    payload={'readout_zh.txt':render(result).encode('utf-8'),'summary.json':encoded(result),
        'casewise.csv':csv_bytes(cases),'mechanism_cases.csv':csv_bytes(mechanisms),
        'source_strata_metrics.csv':csv_bytes(groups)}
    manifest=dict(schema='GTA-AVT-calibre-coadapt-report-manifest-v1',
        source_bindings=dict(result['source_bindings'],report_builder_sha256=sha(__file__)),
        files={name:dict(sha256=hashlib.sha256(body).hexdigest(),bytes=len(body)) for name,body in payload.items()},
        rows=dict(casewise=540,mechanism_cases=36,source_strata_metrics=180),
        all_three_seeds_retained=True,scalar_only=True,no_private_paths_or_machine_identity=True,
        no_data_arrays_or_weights=True,scope='Oracle TRAIN capability; not CT-only GTA or native/independent validation')
    private_string_check(manifest)
    output.mkdir(parents=True)
    for name,body in payload.items():
        with (output/name).open('xb') as stream:stream.write(body)
    with (output/'manifest.json').open('xb') as stream:stream.write(encoded(manifest))
    return dict(status='complete',files=6,summary_sha256=sha(output/'summary.json'),manifest_sha256=sha(output/'manifest.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis');parser.add_argument('--analysis-sha256');parser.add_argument('--output')
    parser.add_argument('--readout',default=str(Path(__file__).with_name('calibre_coadapt_readout.py')))
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='inert',files_read=False,outputs_written=False)));return
    require(args.analysis and args.analysis_sha256 and args.output,'Explicit completed readout/SHA and fresh output required')
    print(json.dumps(execute(args)))


if __name__=='__main__':
    main()
