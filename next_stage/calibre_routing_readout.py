"""Completed three-axis routing readout; default inert and no live jobs.

Reuse the fixed basis metrics/primary/uniform math. Add only the prospectively
fixed per-source correspondence protection; no rescue, fitting or new job.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

BASE_SHA = '1f7fe250df11b96f831850c9eccafa5a4ba3e51d0b3dd224cea9953d95542d44'
WORKER_SHA = 'a9320edc72b1dcde9fe4486355566bf9c491b78d109bf5315f578e08db32c9ef'
MODEL_SHA = '9acd1c9d8b04fdef3e5a7e4cefe9d8f0d234978093eaf09d7d2a03a64288f58e'
BANK_SHA = '9be42890017341e4cc32fb0f17688d859a13e84643f95cd21e347462043419d6'
SEEDS = (20261030,20261031,20261032)
ARMS = ('C0','A_only','AR')
COEFFICIENTS = ('calibre_bias','calibre_availability_weight','calibre_radius_weight')


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def lines(path):
    return [json.loads(r) for r in Path(path).read_bytes().splitlines()]


def module(path, digest, name):
    require(sha(path)==digest,'Pinned source differs: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def metadata(root, summary, bank, base):
    require(summary.get('schema')=='GTA-AVT-calibre-routing-coadapt-pilot-v1' and summary.get('status')=='verified_complete' and
        summary.get('mode')=='run' and summary.get('seed') in SEEDS and summary.get('worker_sha256')==WORKER_SHA and
        summary.get('model_sha256')==MODEL_SHA and summary.get('oracle_bank_sha256')==base.ORACLE_BANK_SHA and
        summary.get('fit_cases')==bank['fit_cases'] and summary.get('probe_cases')==bank['probe_cases'], 'Completed formal routing run required, not QA')
    receipt=read(root.parent/'collection.json'); child=receipt.get('child',{})
    require(receipt.get('terminal',{}).get('exit_code')==0 and (not child.get('exists') or child.get('state') in ('Z','X')),
        'Natural completion and retired child required')
    required=['intent.json','inputs.json','normalization.json','training_plan.json']+[a+'/'+n for a in ARMS for n in ('training.jsonl','patches.jsonl','analysis.json')]+['AR/mechanism.jsonl','AR/mechanism_analysis.json']
    for name in ['summary.json']+required:
        item=receipt['files'][name]; path=root/name
        require(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'Collection bytes differ: '+name)
        if name!='summary.json': require(summary['files'][name]==item,'Completed summary/file binding differs')
    inputs,intent=read(root/'inputs.json'),read(root/'intent.json')
    bindings=dict(source_manifest_sha256=base.SOURCE_SHA,bank_sha256=BANK_SHA,common_weights_sha256=base.WEIGHT_SHA,
        model_sha256=MODEL_SHA,oracle_bank_sha256=base.ORACLE_BANK_SHA,oracle_helper_sha256=base.ORACLE_SHA)
    active={'structure_guidance.morph_fusion.weight','structure_guidance.morph_fusion.bias',
        'anatomical_output_fusion.3.weight','anatomical_output_fusion.3.bias'}|set(COEFFICIENTS)|{
        'structure_guidance.morphology_convs.%s.%s'%(i,k) for i in range(4) for k in ('weight','bias')}
    require(all(inputs.get(k)==v for k,v in bindings.items()) and inputs['cache_before']==inputs['cache_after'] and
        inputs['no_VAL_or_TEST_arrays'] is True and set(inputs['oracle_files'])==set(bank['case_ids']) and
        isinstance(inputs['initial_all_arm_state_sha256'],str) and len(inputs['initial_all_arm_state_sha256'])==64 and
        set(inputs['trainable_parameter_names'])==active,'Source/data/common-init/exact15-tensor identity differs')
    worker=module(Path(__file__).with_name('calibre_routing_pilot.py'),WORKER_SHA,'_routing_executed_recipe')
    require(intent['schema']=='GTA-AVT-calibre-routing-coadapt-pilot-intent-v1' and intent['recipe']==worker.RECIPE and
        worker.RECIPE['trainable_parameters']==115458 and worker.RECIPE['shared_parameters']==114882 and worker.RECIPE['gate_parameters']==576 and
        intent['worker_sha256']==WORKER_SHA and intent['model_sha256']==MODEL_SHA and intent['oracle_bank_sha256']==base.ORACLE_BANK_SHA and
        intent['oracle_helper_sha256']==base.ORACLE_SHA and intent['mode']=='run' and intent['seed']==summary['seed'] and
        intent['engineering_only'] is False,'Original routing recipe/shape-source binding differs')
    # The pinned executed model constructor asserts every coefficient shape(3,64).
    require(sha(Path(__file__).with_name('calibre_routing_stage_model.py'))==MODEL_SHA,'Executed3x64 model source differs')
    norm=read(root/'normalization.json');require(norm['case_equal'] is True and norm['fit_cases']==bank['fit_cases'] and
        norm['mean']==2.068768924624306 and norm['population_std']==0.5812830275631903,'Original fixed FIT24 moments required')
    import numpy as np
    plan=worker.make_plan(np,summary['seed'],bank['fit_cases']); plan_sha=hashlib.sha256(worker.encoded(plan)).hexdigest(); actual=read(root/'training_plan.json')
    require(actual['sha256']==inputs['training_plan_sha256']==plan_sha and actual['common_to_all_three_arms'] is True and
        worker.encoded(actual['plan'])==worker.encoded(plan),'Original common2000-step plan differs')
    for arm in ARMS:
        updates=lines(root/arm/'training.jsonl'); result=read(root/arm/'analysis.json')
        require([u['step'] for u in updates]==list(range(1,2001)) and all(base.finite(u['loss']) and u['alpha']==min(1.,(i+1)/250.)
            for i,u in enumerate(updates)) and result['schema']=='GTA-AVT-calibre-routing-coadapt-arm-v1' and
            result['arm']==arm and result['successful_updates']==2000 and result['frozen_state_unchanged'] is True,'Formal updates/alpha/frozen state differ')
        for u in updates:
            require(set(u['gradient_maxabs'])==active and set(u['coefficient_row_gradient_maxabs'])==set(COEFFICIENTS) and
                all(len(u['coefficient_row_gradient_maxabs'][k])==3 and all(v is None or base.finite(v) and v>=0 for v in u['coefficient_row_gradient_maxabs'][k])
                    for k in COEFFICIENTS) and [a['axis'] for a in u['fields']['routing_axes']]==[1,2,3], 'Runtime3-axis coefficient records differ')
    return {name:sha(root/name) for name in required}


def load_seed(root, case_ids, bank, base):
    require(bank is not None, 'Original bank required for provenance validation')
    root = Path(root); summary = read(root / 'summary.json')
    hashes = metadata(root, summary, bank, base); hashes['summary.json'] = sha(root / 'summary.json')
    rows = {}; expected = [(cid, slot) for cid in case_ids for slot in range(16)]
    for arm in ARMS:
        data = lines(root / arm / 'patches.jsonl')
        require([(r['case_id'], r['slot']) for r in data] == expected, 'Complete original192 patch order required')
        for row in data:
            require(row['source'] == row['case_id'].split('_')[0] and row['stratum'] ==
                ('GT_centerline_plus_jitter' if row['slot'] < 8 else 'CT_uniform') and
                all(type(row[k]) is bool for k in ('known_nonempty', 'attribute_valid', 'valid_empty', 'primary')) and
                row['primary'] == (row['slot'] < 8 and row['known_nonempty']), 'Patch population identity differs')
            base.validate_metrics(row['metrics'])
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
        base.validate_metrics(row['permuted'])
        require(all(row['permuted'][k] == normal['metrics'][k] for k in ('fine_nodes', 'GT_nodes', 'truth_foreground', 'real_CT_voxels')),
            'Permuted endpoint uses different truth support')
        permutation, moments = row['permutation'], row['radius_moments']
        count = permutation['eligible_sites']
        require(type(count) is int and count == moments['count'] == normal['oracle']['available_radius_sites'] and
            permutation['shift'] == count // 2 and permutation['own_A_and_support_preserved'] is True and
            permutation['correct_radius_multiset_sha256'] == permutation['permuted_radius_multiset_sha256'] and
            type(permutation['changed_radius_sites']) is int and 0 <= permutation['changed_radius_sites'] <= count and
            type(row['zR_changed']) is bool and type(row['binary_changed_voxels']) is int and
            0 <= row['binary_changed_voxels'] <= 128 ** 3 and base.finite(row['mean_abs_logit_response']) and
            row['mean_abs_logit_response'] >= 0. and base.finite(moments['sum']) and base.finite(moments['sum_squares']) and moments['sum_squares'] >= 0.,
            'Half-ring intervention/support/multiset or actual-response identity differs')
    return summary, rows, interventions, hashes


def SOURCE_safety(mechanism, ids, base):
    groups={}
    for source in ('Dongyang','KiTS','Rider'):
        expected=[c for c in ids if c.startswith(source+'_')]
        supported=[c for c in expected if mechanism['per_case'][c]['supported'] and mechanism['per_case'][c]['correct_minus_permuted_Dice_pp'] is not None]
        complete=len(expected)==4 and supported==expected
        effect=base.mean(mechanism['per_case'][c]['correct_minus_permuted_Dice_pp'] for c in expected) if complete else None
        groups[source]=dict(expected_case_ids=expected,supported_case_ids=supported,complete_unchanged_support=complete,
            case_equal_correct_minus_half_ring_Dice_pp=effect,passed=complete and effect>=0.)
    return dict(scope='Prospective SOURCE safety: sameweight original primary patches/cases',source_groups=groups,
        passed=all(g['passed'] for g in groups.values()),no_source_reclassification_or_removal=True,stricter_only=True,cannot_rescue_original_primary=True)


def analyze_seed(summary, rows, interventions, ids, base):
    result=base.analyze_seed(summary,rows,interventions,ids)
    previous=result['screen']; source=SOURCE_safety(result['mechanisms']['own_A_half_ring_radius'],ids,base)
    parts=dict(previous['parts']);parts.update({'SOURCE_'+s+'_sameweight_nonnegative':v['passed'] for s,v in source['source_groups'].items()})
    passed=None if any(v is None for v in parts.values()) else all(parts.values())
    result.update(primary_and_uniform_screen=previous,SOURCE_safety=source,capacity_continue=passed,
        screen=dict(parts=parts,passed=passed,status='inconclusive_support' if passed is None else 'positive_capacity_uniform_source_screen' if passed else 'screen_not_passed'))
    return result


def analyze(roots, bank_path):
    require(len(roots)==3 and len({str(Path(r).resolve()) for r in roots})==3 and sha(bank_path)==BANK_SHA,'Three distinct completed seeds and original bank required')
    base=module(Path(__file__).with_name('calibre_basis_readout.py'),BASE_SHA,'_routing_fixed_math')
    bank=read(bank_path);ids=bank['probe_cases'];require(len(ids)==len(set(ids))==12 and len(bank['fit_cases'])==24 and set(ids).isdisjoint(bank['fit_cases']),'Original FIT24/PROBE12 partition required')
    seeds=[];proofs=[]
    for root in roots:
        summary,rows,interventions,hashes=load_seed(root,ids,bank,base)
        seeds.append(analyze_seed(summary,rows,interventions,ids,base));proofs.append(hashes)
    require(tuple(s['seed'] for s in seeds)==SEEDS and len({p['normalization.json'] for p in proofs})==1,'Prespecified seed order/common normalization required')
    combine=lambda values:None if any(v is None for v in values) else all(values)
    flags=[s['screen']['passed'] for s in seeds]; original=[s['original_primary_screen_passed'] for s in seeds]
    uniform=[s['CT_uniform_transfer_safety']['passed'] for s in seeds];source=[s['SOURCE_safety']['passed'] for s in seeds]
    method=dict(base.METHOD,training='2000 successful updates/arm, shared114882 original branches/projections plus576 gates;3 tensors(3,64),15 active tensors115458parameters; no auxiliary loss',
        cross_round='Only conditional routing freedom1to3 axes changes; same original learning scope/labels/bank/seeds/plan/budget; source guard is separately prospectively stricter',
        operator='g_j=tanh(b_j+v_j*zA+w_j*zR); F=F0+alpha*(g1*D1+(1-abs(g1))*(g2*D2+g3*D3)); three (3,64) coefficient tensors,576 coefficients',
        source_guard='Each fixed Dongyang/KiTS/Rider4case primary sameweight Dice effect>=0 on complete original support; undefined effect stays null and cannot pass')
    return dict(schema='GTA-AVT-calibre-routing-coadapt-final-readout-v1',readout_sha256=sha(__file__),method=method,bank_sha256=BANK_SHA,
        math_source_sha256=BASE_SHA,worker_sha256=WORKER_SHA,model_sha256=MODEL_SHA,coefficient_shape=[3,64],trainable_parameters=115458,
        probe_cases=ids,per_seed=seeds,input_hashes=proofs,primary_seed=SEEDS[0],original_primary_screen_passed=original[0],all3_original_primary_screens_passed=combine(original),
        primary_CT_uniform_transfer_safety_passed=uniform[0],all3_CT_uniform_transfer_safety_passed=combine(uniform),
        primary_SOURCE_safety_passed=source[0],all3_SOURCE_safety_passed=combine(source),primary_screen_passed=flags[0],all3_screens_passed=combine(flags),
        capacity_continue=combine(flags),primary_failure_cannot_be_rescued=flags[0] is not True,seeds_pooled=False,
        oracle_conditioned_not_CT_only=True,mathematical_or_true_upper_bound=False,no_attribute_auxiliary_loss=True,all_three_seeds_retained=True,
        next_experiment_authorized=False,native_confirmation_required=True,GTA_goal_completed=False)


def execute(args):
    out=Path(args.output);require(not out.exists(),'Fresh output required');result=analyze(args.inputs,args.bank)
    base=module(Path(__file__).with_name('calibre_basis_readout.py'),BASE_SHA,'_routing_render_math')
    text=['Three-axis original-branch routing: completed TRAIN oracle capability screen, not CT deployment.']
    cases=[];mechanisms=[];sources=[]
    for seed in result['per_seed']:
        text.append('Seed%s: original=%s; CT_uniform=%s; SOURCE=%s; combined=%s.'%(seed['seed'],seed['original_primary_screen_passed'],
            seed['CT_uniform_transfer_safety']['passed'],seed['SOURCE_safety']['passed'],seed['capacity_continue']))
        for source,value in seed['SOURCE_safety']['source_groups'].items():
            text.append(' %s sameweight Dice=%s pp; supported%s/4; guard=%s.'%(source,value['case_equal_correct_minus_half_ring_Dice_pp'],len(value['supported_case_ids']),value['passed']))
            sources.append(dict(seed=seed['seed'],source=source,**value))
        for arm,record in seed['arms'].items():
            for cid,case in record['per_case'].items():
                for st in ('primary','all16','GT_jitter_all8','CT_uniform_all8','uniform_known_nonempty'):
                    cases.append(dict(seed=seed['seed'],arm=arm,case_id=cid,source=case['source'],stratum=st,**case[st]))
        mechanisms += [dict(seed=seed['seed'],case_id=cid,**value) for cid,value in seed['mechanisms']['own_A_half_ring_radius']['per_case'].items()]
    text.append(base.render(result).replace('两者合取继续','全部检查合取继续'));text.append('All fixed seeds/sources remain. No threshold change, source removal, candidate rescue or automatic successor.')
    out.mkdir(parents=True);(out/'analysis.json').write_bytes((json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))
    (out/'readout.txt').write_bytes(('\n'.join(text)+'\n').encode('utf-8'))
    for name,rows in [('casewise.csv',cases),('mechanism_cases.csv',mechanisms),('source_safety.csv',sources)]:
        with (out/name).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)
    return dict(status='complete',analysis_sha256=sha(out/'analysis.json'),all3_screens_passed=result['all3_screens_passed'],next_experiment_authorized=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--inputs',nargs=3);parser.add_argument('--bank');parser.add_argument('--output');parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.execute:
        require(args.inputs and args.bank and args.output,'Three completed seeds, original bank and fresh output required');print(json.dumps(execute(args)))
    else:print(json.dumps(dict(status='inert',files_read=False,writes=False,new_training_authorized=False)))
