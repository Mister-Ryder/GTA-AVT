"""CPU-only descriptive readout of completed four-variant frozen AR runs.

Default inert; no optimizer, fitting, new efficacy gate, candidate rescue or
CT-predictor authorization. Region metrics are voxel counts/BCE, not skeletons.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

WORKER_SHA = 'bcd6f63a40f4419d62b259b4e203f653b9fb4e91d92f3727812739127e1c87de'
BASIS_SHA = '68840bce80d3576c353fe6c43c49c5ea7bfc7db44e772e9bd687283af1198f7c'
MODEL_SHA = '6f5e6fcf84600c737fc8844c0fdce6839f980d19d734512686f374cce1a0c294'
READER_SHA = '1f7fe250df11b96f831850c9eccafa5a4ba3e51d0b3dd224cea9953d95542d44'
SEEDS = (20261030, 20261031, 20261032)
VARIANTS = ('correct', 'half_ring', 'patch_mean_FP32', 'FIT_mean_FP64')
STRATA = ('primary', 'all16', 'GT_jitter_all8', 'CT_uniform_all8', 'uniform_known_nonempty')
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


def selected(row, stratum):
    return row['primary'] if stratum == 'primary' else True if stratum == 'all16' else (
        row['slot'] < 8 if stratum == 'GT_jitter_all8' else row['slot'] >= 8 if stratum == 'CT_uniform_all8'
        else row['slot'] >= 8 and row['known_nonempty'])


def metric_math():
    path = Path(__file__).with_name('calibre_basis_readout.py')
    require(sha(path) == READER_SHA, 'Original aggregation source changed')
    spec = importlib.util.spec_from_file_location('_frozen_basis_metrics', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def regional(items, variant, region):
    values = [r['variants'][variant]['regions'][region] for r in items]
    voxels = sum(v['real_CT_voxels'] for v in values)
    return dict(patches=len(items), real_voxels_across_patches=voxels,
        mean_real_voxels_per_patch=mean(v['real_CT_voxels'] for v in values),
        truth_foreground=mean(v['truth_foreground'] for v in values),
        TP=mean(v['TP'] for v in values), FP=mean(v['FP'] for v in values), FN=mean(v['FN'] for v in values),
        BCE=math.fsum(v['BCE_sum'] for v in values) / voxels if voxels else None)


def case_record(items, variant, mathlib):
    return dict(whole=mathlib.aggregate([dict(metrics=r['variants'][variant]['metrics']) for r in items]),
        regions={zone: regional(items, variant, zone) for zone in REGIONS})


def levels(per_case, ids):
    return {key: dict(mean=mean(per_case[c]['whole'][key] for c in ids if per_case[c]['whole'][key] is not None),
        supported_case_ids=[c for c in ids if per_case[c]['whole'][key] is not None]) for key in METRICS}


def comparison(correct, other, ids):
    cases = {}; regions = {}
    for cid in ids:
        a, b = correct[cid]['whole'], other[cid]['whole']
        cases[cid] = {key: None if a[key] is None or b[key] is None else a[key] - b[key] for key in METRICS}
        regions[cid] = {zone: {key: None if correct[cid]['regions'][zone][key] is None or other[cid]['regions'][zone][key] is None else
            correct[cid]['regions'][zone][key] - other[cid]['regions'][zone][key] for key in ('TP', 'FP', 'FN', 'BCE')} for zone in REGIONS}
    def group(members):
        return {key: dict(mean_correct_minus_variant=mean(cases[c][key] for c in members if cases[c][key] is not None),
            mean_correct_minus_variant_pp=None if key not in FRACTIONS else
                mean(100. * cases[c][key] for c in members if cases[c][key] is not None),
            supported_case_ids=[c for c in members if cases[c][key] is not None],
            positive_cases=sum(cases[c][key] is not None and cases[c][key] > 0 for c in members),
            negative_cases=sum(cases[c][key] is not None and cases[c][key] < 0 for c in members)) for key in METRICS}
    return dict(per_case=cases, case_equal=group(ids), source_groups={source: group([c for c in ids if c.startswith(source + '_')])
        for source in ('Dongyang', 'KiTS', 'Rider')}, regional_per_case=regions)


def load_seed(root, seed, mathlib):
    root = Path(root); collection = read(root.parent / 'collection.json')
    require(collection.get('terminal', {}).get('exit_code') == 0, 'Naturally completed collection required')
    child = collection.get('child', {})
    require(not child.get('exists') or child.get('state') in ('Z', 'X'), 'Diagnostic child still active')
    for name in ('summary.json', 'intent.json', 'diagnostics.jsonl'):
        proof = collection['files'][name]; path = root / name
        require(path.stat().st_size == proof['bytes'] and sha(path) == proof['sha256'], 'Collected diagnostic bytes differ')
    summary, intent = read(root / 'summary.json'), read(root / 'intent.json')
    require(summary['schema'] == 'GTA-AVT-calibre-basis-frozen-diagnostic-v1' and summary['status'] == 'verified_complete' and
        summary['seed'] == seed and summary['patches'] == 192 and tuple(summary['variants']) == VARIANTS and
        summary['whole_metric_reference_parity_passed'] is True and summary['maximum_BCE_ULP'] <= 2 and
        summary['model_state_unchanged'] is True and summary['optimizer_created'] is False and summary['weights_updated'] is False and
        summary['candidate_STOP_unchanged'] is True and summary['no_new_success_gate'] is True and
        summary['CT_predictor_training_authorized'] is False, 'Four-variant frozen/no-update summary required')
    require(intent['schema'] == 'GTA-AVT-calibre-basis-frozen-diagnostic-intent-v1' and intent['seed'] == seed and
        intent['worker_sha256'] == WORKER_SHA and intent['basis_worker_sha256'] == BASIS_SHA and intent['model_sha256'] == MODEL_SHA and
        tuple(intent['variants']) == VARIANTS and intent['optimizer_created'] is False and intent['model_eval'] is True and
        intent['no_grad'] is True and intent['candidate_STOP_unchanged'] is True and intent['new_success_gate'] is False,
        'Exact frozen worker/model/intervention identity differs')
    rows = [json.loads(r) for r in (root / 'diagnostics.jsonl').read_bytes().splitlines()]
    ids = list(dict.fromkeys(r['case_id'] for r in rows))
    require(len(ids) == 12 and all(sum(c.startswith(s + '_') for c in ids) == 4 for s in ('Dongyang', 'KiTS', 'Rider')) and
        [(r['case_id'], r['slot']) for r in rows] == [(c, slot) for c in ids for slot in range(16)], 'Full original12x16 row order required')
    for row in rows:
        require(row['source'] == row['case_id'].split('_')[0] and type(row['known_nonempty']) is bool and
            row['stratum'] == ('GT_centerline_plus_jitter' if row['slot'] < 8 else 'CT_uniform') and
            row['primary'] == (row['slot'] < 8 and row['known_nonempty']) and set(row['variants']) == set(VARIANTS), 'Patch/variant population differs')
        for variant in VARIANTS:
            value = row['variants'][variant]; mathlib.validate_metrics(value['metrics'])
            require(all(value['metrics'][key] == row['variants']['correct']['metrics'][key] for key in
                ('real_CT_voxels','truth_foreground','fine_nodes','GT_nodes')), 'Variant GT/real support changed')
            require(type(value['actual_zR_changed_from_correct']) is bool and set(value['regions']) == set(REGIONS), 'Region/consumed-condition records missing')
            require(value['metrics']['real_CT_voxels'] == sum(z['real_CT_voxels'] for z in value['regions'].values()) and
                all(value['metrics'][k] == sum(z[k] for z in value['regions'].values()) for k in ('TP', 'FP', 'FN', 'truth_foreground')),
                'Observed-region counts do not partition whole real voxels')
            require(all(value['regions'][zone][key] == row['variants']['correct']['regions'][zone][key]
                for zone in REGIONS for key in ('real_CT_voxels','truth_foreground')), 'Regional GT/support changed')
            whole_bce = value['metrics']['BCE']
            regional_bce = math.fsum(z['BCE_sum'] for z in value['regions'].values()) / value['metrics']['real_CT_voxels']
            require(whole_bce is not None and math.isfinite(whole_bce) and whole_bce >= 0 and
                ((whole_bce == regional_bce) if whole_bce == 0 or regional_bce == 0 else
                    math.isclose(whole_bce, regional_bce, rel_tol=1e-12, abs_tol=1e-12)),
                'Regional BCE sums do not reproduce whole BCE within FP64 reduction tolerance')
            for zone in value['regions'].values():
                require(all(type(zone[k]) is int and zone[k] >= 0 for k in ('real_CT_voxels','truth_foreground','TP','FP','FN')) and
                    math.isfinite(zone['BCE_sum']) and zone['BCE_sum'] >= 0 and zone['TP'] + zone['FN'] == zone['truth_foreground'] and
                    (zone['BCE'] == zone['BCE_sum'] / zone['real_CT_voxels'] if zone['real_CT_voxels'] else zone['BCE'] is None),
                    'Invalid regional support/count/BCE measurement')
        for variant in ('correct', 'half_ring'):
            require(row['parity'][variant]['all_count_and_derived_metrics_exact'] is True and
                0 <= row['parity'][variant]['BCE_binary64_ULP'] <= 2, 'Reference metric parity missing')
    return ids, rows, dict(summary_sha256=sha(root / 'summary.json'), intent_sha256=sha(root / 'intent.json'),
        diagnostics_sha256=sha(root / 'diagnostics.jsonl'), trained_state_sha256=intent['trained_state_sha256'],
        reference_patches_sha256=intent['reference_patches_sha256'], reference_mechanism_sha256=intent['reference_mechanism_sha256'])


def analyze(inputs):
    require(len(inputs) == 3 and len({str(Path(p).resolve()) for p in inputs}) == 3, 'Three distinct completed seed outputs required')
    mathlib = metric_math(); result = []; inputs_proof = []; first_ids = None
    for root, seed in zip(inputs, SEEDS):
        ids, rows, proof = load_seed(root, seed, mathlib)
        require(first_ids is None or ids == first_ids, 'Original cases/order differ between seeds'); first_ids = ids
        variants = {}; contrasts = {}
        for variant in VARIANTS:
            strata = {}
            for stratum in STRATA:
                cases = {c: case_record([r for r in rows if r['case_id'] == c and selected(r, stratum)], variant, mathlib) for c in ids}
                strata[stratum] = dict(per_case=cases, case_equal=levels(cases, ids),
                    source_groups={s: levels(cases, [c for c in ids if c.startswith(s + '_')]) for s in ('Dongyang','KiTS','Rider')})
            dice = [(c, strata['primary']['per_case'][c]['whole']['dice']) for c in ids]
            defined = [(c,v) for c,v in dice if v is not None]; worst = [c for c,v in sorted(defined, key=lambda x:(x[1],x[0]))[:2]]
            variants[variant] = dict(strata=strata, Dice_population_SD=math.sqrt(mathlib.variance(v for c,v in defined)) if defined else None,
                own_worst2=dict(case_ids=worst, Dice_mean=mean(strata['primary']['per_case'][c]['whole']['dice'] for c in worst) if len(worst)==2 else None))
        for reference in VARIANTS:
            members = variants[reference]['own_worst2']['case_ids']
            variants[reference]['matched_worst2'] = dict(case_ids=members, Dice_means={v: mean(variants[v]['strata']['primary']['per_case'][c]['whole']['dice'] for c in members)
                if len(members)==2 and all(variants[v]['strata']['primary']['per_case'][c]['whole']['dice'] is not None for c in members) else None for v in VARIANTS})
        for variant in VARIANTS[1:]:
            contrasts[variant] = {st: comparison(variants['correct']['strata'][st]['per_case'], variants[variant]['strata'][st]['per_case'], ids) for st in STRATA}
        r15 = [r for r in rows if r['case_id'] == 'Rider_R15']
        result.append(dict(seed=seed, variants=variants, correct_minus_variants=contrasts, Rider_R15_patchwise=r15))
        inputs_proof.append(proof)
    return dict(schema='GTA-AVT-calibre-basis-frozen-readout-v1', source_sha256=sha(__file__), worker_sha256=WORKER_SHA,
        metric_source_sha256=READER_SHA, model_sha256=MODEL_SHA, case_ids=first_ids, per_seed=result, input_hashes=inputs_proof,
        method=dict(scope='Posthoc descriptive same-trained-AR-weight interventions; no optimizer/update, fitting or efficacy gate',
            variants='half_ring preserves ownA/R multiset but changes spatial association; patch_mean_FP32 collapses within-patch R variation and retains FP32-rounded patch mean; FIT_mean_FP64 makes consumed R axis exactly0',
            limitations='FIT_mean is NOT a retrained A_only control; A itself contains radius-dependent geometry. Nonlinear intervention effects are not an additive variance decomposition.',
            whole='Original metric aggregation: patch-equal within case then case equal; fine sum covered/sum nodes within case, not mean patch fine ratios',
            regional_counts='Regional TP/FP/FN are patch means within each case, then case-equal means, including0-count regions of selected patches',
            regional_BCE='Sum BCE_sum/sum real_CT_voxels within each case, then mean of defined cases; never pooled across cases or mean of patch BCE',
            region='Minimum grid-index distance to six actual observed patch/cache-intersection faces:0..15,16..31,>=32; padding excluded; no regional skeleton, Dice or anatomical wall claims',
            dependence='Same12 TRAIN cases/overlapping patches; R15 primary fine-node count27 may repeat anatomical nodes; no patient/zero-shot independence',
            precision='Whole count/derived metrics match prior endpoints; only nonnegative BCE parity permits up to2 binary64 ULP; no old voxelwise mask parity claim; regional BCE partition closure uses1e-12 relative/absolute FP64 reduction tolerance with exact-zero agreement'),
        candidate_STOP_unchanged=True, scientific_verdict_modified=False, new_success_gate=False,
        CT_predictor_training_authorized=False, oracle_not_CT_only=True, GTA_goal_completed=False)


def render(result):
    out=['冻结AR四条件诊断：事后描述，不改STOP、不建立新成功门槛。',
        'whole Dice/cl/recall的差为pp；BCE为原loss单位（正确−变体负值更好）；FP/FN为每patch均数后病例等权。',
        '区域BCE先按真实体素在每病例内加权，再病例等权；区域不计算骨架或Dice。']
    for seed in result['per_seed']:
        out.append('seed %s' % seed['seed'])
        for variant in VARIANTS:
            v=seed['variants'][variant]; m=v['strata']['primary']['case_equal']
            out.append(' %s primary Dice=%s%%，SD=%spp，ownworst2=%s%%。' % (variant,
                None if m['dice']['mean'] is None else 100*m['dice']['mean'],
                None if v['Dice_population_SD'] is None else 100*v['Dice_population_SD'],
                None if v['own_worst2']['Dice_mean'] is None else 100*v['own_worst2']['Dice_mean']))
        for variant in VARIANTS[1:]:
            for st in ('primary','uniform_known_nonempty'):
                c=seed['correct_minus_variants'][variant][st]['case_equal']
                out.append(' correct−%s %s：%s。' % (variant,st,'；'.join('%s=%s' % (k,
                    c[k]['mean_correct_minus_variant_pp'] if k in FRACTIONS else c[k]['mean_correct_minus_variant'])
                    for k in ('dice','cldice','centerline_recall','fine_recall','BCE','FP','FN'))))
            r=seed['correct_minus_variants'][variant]['primary']['per_case']['Rider_R15']
            display = {(key + '_pp' if key in FRACTIONS else key):
                100. * value if key in FRACTIONS and value is not None else value for key, value in r.items()}
            out.append(' R15 correct−%s：%s。' % (variant,json.dumps(display,ensure_ascii=False)))
    out += ['patchmean保留区块均值但改变完整半径分布；FITmean使R轴为0但不是重新训练的A_only。不能把它们的非线性差值当成独立相加的信息贡献。',
        'A是GT可测支持且本身含几何信息；当前无CT预测器结论或后继授权。全部病例/来源/五层/尾部/区域及R15逐patch保留。']
    return '\n'.join(out)+'\n'


def execute(args):
    output=Path(args.output);require(not output.exists(),'Fresh output required')
    result=analyze(args.inputs); cases=[];regions=[];r15=[]
    for seed in result['per_seed']:
        for variant,v in seed['variants'].items():
            for st,group in v['strata'].items():
                for cid,c in group['per_case'].items():
                    baseline=seed['variants']['correct']['strata'][st]['per_case'][cid]['whole']
                    cases.append(dict(seed=seed['seed'],variant=variant,stratum=st,case_id=cid,source=cid.split('_')[0],**c['whole'],
                        **{'correct_minus_variant_'+k:None if baseline[k] is None or c['whole'][k] is None else baseline[k]-c['whole'][k] for k in METRICS}))
                    for zone,z in c['regions'].items():
                        regions.append(dict(seed=seed['seed'],variant=variant,stratum=st,scope='case',case_id=cid,source=cid.split('_')[0],region=zone,**z,
                            **{k+'_supported_cases':int(z[k] is not None) for k in ('TP','FP','FN','BCE')},
                            **{'correct_minus_variant_'+k:None if seed['variants']['correct']['strata'][st]['per_case'][cid]['regions'][zone][k] is None or z[k] is None else
                                seed['variants']['correct']['strata'][st]['per_case'][cid]['regions'][zone][k]-z[k] for k in ('TP','FP','FN','BCE')}))
                for label,ids in [('all12',result['case_ids'])]+[(s,[c for c in result['case_ids'] if c.startswith(s+'_')]) for s in ('Dongyang','KiTS','Rider')]:
                    for zone in REGIONS:
                        values={k:mean(group['per_case'][c]['regions'][zone][k] for c in ids if group['per_case'][c]['regions'][zone][k] is not None) for k in ('TP','FP','FN','BCE')}
                        correct=seed['variants']['correct']['strata'][st]['per_case']
                        regions.append(dict(seed=seed['seed'],variant=variant,stratum=st,scope='case_equal_group',case_id='',source=label,region=zone,
                            patches=None,real_voxels_across_patches=None,mean_real_voxels_per_patch=None,truth_foreground=None,**values,
                            **{k+'_supported_cases':sum(group['per_case'][c]['regions'][zone][k] is not None for c in ids) for k in ('TP','FP','FN','BCE')},
                            **{'correct_minus_variant_'+k:mean(correct[c]['regions'][zone][k]-group['per_case'][c]['regions'][zone][k] for c in ids
                                if correct[c]['regions'][zone][k] is not None and group['per_case'][c]['regions'][zone][k] is not None) for k in ('TP','FP','FN','BCE')}))
        for row in seed['Rider_R15_patchwise']:
            for variant,v in row['variants'].items():
                base=row['variants']['correct']['metrics'];r15.append(dict(seed=seed['seed'],slot=row['slot'],stratum=row['stratum'],primary=row['primary'],variant=variant,
                    radius_support=row['radius_support'],actual_zR_changed=v['actual_zR_changed_from_correct'],**v['metrics'],
                    **{'correct_minus_variant_'+k:None if base[k] is None or v['metrics'][k] is None else base[k]-v['metrics'][k] for k in METRICS}))
    output.mkdir(parents=True)
    (output/'analysis.json').write_bytes((json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))
    (output/'readout.txt').write_bytes(render(result).encode('utf-8'))
    for name,rows in [('casewise.csv',cases),('regions.csv',regions),('R15_patchwise.csv',r15)]:
        with (output/name).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    return dict(status='complete',analysis_sha256=sha(output/'analysis.json'),case_rows=len(cases),region_rows=len(regions),R15_rows=len(r15),scientific_verdict_modified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--inputs',nargs=3);parser.add_argument('--output');parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.execute:
        require(args.inputs and args.output,'Three completed collected roots and fresh output required');print(json.dumps(execute(args)))
    else:
        print(json.dumps(dict(status='inert',files_read=False,writes=False,new_success_gate=False)))
