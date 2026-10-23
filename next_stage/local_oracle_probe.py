"""One fixed, TRAIN-only, frozen local-scale oracle correspondence probe.

Default is metadata-only planning. Explicit --execute runs exactly three modes
on the twelve check cases/eight patches already fixed in the readout plan.
This private tool never changes r3qa, fits a head, or starts another experiment.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np

PLAN_SHA='8aacf68f45e706e9b1236a1d76c505c638851ae0fd508ae721d004ff02abeaac'
WEIGHT_SHA='7406a3ba7e5bac1a7d8ee24898cb954ff8576f37fcf5fba0e26ef9a2cb5c43e6'
HEAD_SHA='a635c3dfd7f50f69881c018c61c47559e6b8818143ec8639767f765490ecea4b'
MODES=('constant','correct','permuted')
Q0=S0=.2
CHECK_CASES=('Dongyang_D15','Dongyang_D3','Dongyang_D18','Dongyang_D12',
             'KiTS_K12','KiTS_K16','KiTS_K9','KiTS_K14',
             'Rider_R5','Rider_R15','Rider_R3','Rider_R11')
SOURCES={
 'code/faithful_model.py':'c02a222dc92b231cbaf5a3768217028ee74567d166e09d2df979c9ebdff72337',
 'code/data.py':'9ad4c237e2b0e7e1fdde00391ebba8dc320b3001cc4ee68c04ca23e64e5f62cf',
 'code/historical/__init__.py':'275f71602236a43c1b6288874186ef8ec11077124051b2733d3b95b9dc315ecb',
 'code/historical/func/__init__.py':'8b611c2073b7bec1691a687085deb8707402d6063f6625dc737f3b62599f24b3',
 'code/historical/func/AnatomicalClassifier.py':'f5288c74ce71db97eca63f9665e29dfb0099aa010326ce7d6748377bc0156222',
 'code/historical/func/AnatomicalFusionModules.py':'7309281bb296eb7d861050f5ead6b563823b8393addf6ff7c18a594225030b01',
 'code/historical/func/model_arch_model10.py':'55d51ea850efd2b31aa07acd68ad5e24f64fafe6a714d26dc357919f639f3003',
 'next_stage/stage_model.py':'bba378a8d384ddbe6694491bbabd3ef384cf3796f2e3eb6db647ef02ac965d7b',
 'next_stage/local_model.py':'53e70826e64f708e080e3aca17c1ce8c9666858f0fe126affcc1474b3c90fac3',
 'next_stage/local_learning_model.py':'4de40e538fbbcabe628aca230d2ad5ec3645cbe95874d42839bb344977475c58',
 'next_stage/frozen_readout.py':'81d3f26176095306ae7a1f724ffc36fb861da28992b750d3d58b06150dfe480e',
 'next_stage/frozen_readout_fit.py':'b65422401a096451977bcb92088e99831e3a61b687180e04126e6ca3cb6b376c',
}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path,value):
    path=Path(path)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def array_hash(value):
    value=np.ascontiguousarray(value)
    digest=hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape,dtype='<i8').tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def row_multiset_hash(rows):
    rows=np.asarray(rows,dtype=np.float32).reshape(-1,4)
    if len(rows):
        order=np.lexsort(tuple(rows[:,index] for index in (3,2,1,0)))
        rows=rows[order]
    return array_hash(rows)


def oracle_fields(local):
    """External oracle fields; no GT-derived mask enters normal inference.

    Both fields are uniform outside known nonempty complete windows. The entire
    original four-component row is permuted, with no renormalization/rounding.
    """
    target=np.asarray(local['target'],np.float32)
    if target.shape!=(5,32,32,32):raise ValueError('Expected the fixed five-class32 grid')
    eligible=(local['ct_support'] & local['loss_valid'] & local['foreground_window'])
    if eligible.shape!=(32,32,32) or eligible.dtype!=bool:raise ValueError('Invalid oracle eligibility')
    indices=np.flatnonzero(eligible)
    values=target[1:,eligible].T.copy()
    if len(values) and (not np.isfinite(values).all() or (values<0).any() or
                       not np.allclose(values.sum(1),1.,rtol=1e-6,atol=1e-6)):
        raise ValueError('Invalid original nonempty GT compositions')
    correct=np.full(target.shape,S0,np.float32)
    correct[:,eligible]=target[:,eligible]
    if len(indices) and not np.equal(correct[0,eligible],0.).all():
        raise ValueError('GT nonempty targets must have exactlyzero empty mass')
    permuted=correct.copy()
    source=np.roll(indices,max(1,len(indices)//2)) if len(indices)>1 else indices.copy()
    if len(indices):permuted.reshape(5,-1)[:,indices]=correct.reshape(5,-1)[:,source]
    outside=~eligible
    before=correct[1:,eligible].T
    after=permuted[1:,eligible].T
    fine=correct[1,eligible]
    moved=permuted[1,eligible]
    joint_before=row_multiset_hash(before)
    joint_after=row_multiset_hash(after)
    fine_before=array_hash(np.sort(fine))
    fine_after=array_hash(np.sort(moved))
    other_before=array_hash(correct[:,outside])
    other_after=array_hash(permuted[:,outside])
    if joint_before!=joint_after or fine_before!=fine_after or other_before!=other_after:
        raise AssertionError('Oracle permutation changed a required multiset or outside field')
    if not np.equal(correct[:,outside],np.float32(S0)).all():
        raise AssertionError('Noneligible oracle windows are not exactlyuniform')
    psum=correct.sum(0)
    permuted_psum=permuted.sum(0)
    if not np.allclose(psum,1.,atol=1e-6,rtol=1e-6) or not np.allclose(permuted_psum,1.,atol=1e-6,rtol=1e-6):
        raise AssertionError('External oracle simplex is invalid')
    mapping=np.full((32,)*3,-1,np.int64)
    mapping.ravel()[indices]=source
    audit=dict(eligible_windows=int(len(indices)),eligible_mask_sha256=array_hash(eligible),
        correct_probability_sha256=array_hash(correct),permuted_probability_sha256=array_hash(permuted),
        correct_joint_r_multiset_sha256=joint_before,permuted_joint_r_multiset_sha256=joint_after,
        correct_fine_multiset_sha256=fine_before,permuted_fine_multiset_sha256=fine_after,
        noneligible_correct_sha256=other_before,noneligible_permuted_sha256=other_after,
        half_ring_mapping_sha256=array_hash(mapping),half_ring_shift=max(1,len(indices)//2) if len(indices)>1 else 0,
        full_probability_sum_min=float(psum.min()),full_probability_sum_max=float(psum.max()),
        eligible_probability_sum_min=float(psum[eligible].min()) if len(indices) else None,
        eligible_probability_sum_max=float(psum[eligible].max()) if len(indices) else None,
        correct_eligible_probability_sums_multiset_sha256=array_hash(np.sort(psum[eligible])),
        permuted_eligible_probability_sums_multiset_sha256=array_hash(np.sort(permuted_psum[eligible])),
        eligible_GT_empty_mass_exact_zero=True,operator_q0=Q0,noneligible_score=S0,
        fine_min=float(fine.min()) if len(indices) else None,
        fine_max=float(fine.max()) if len(indices) else None,
        fine_mean=float(fine.mean(dtype=np.float64)) if len(indices) else None,
        nonconstant_fine=bool(len(indices)>=2 and fine.min()!=fine.max()),
        fine_positive_windows=int(np.count_nonzero(fine>0)),
        fine_value_changed_windows=int(np.count_nonzero(fine!=moved)),
        composition_row_changed_windows=int(np.count_nonzero(np.any(before!=after,axis=1))),
        eligible_fine_mean_abs_change=float(np.abs(fine-moved).mean(dtype=np.float64)) if len(indices) else 0.,
        GT_unknown_windows=int(np.count_nonzero(local['ct_support'] & local['foreground_window'] & ~local['loss_valid'])),
        trueempty_windows=int(np.count_nonzero(local['ct_support'] & ~local['foreground_window'])),
        ct_supported_windows=int(np.count_nonzero(local['ct_support'])),
        invariant_joint_r_multiset=True,invariant_raw_fine_multiset=True,
        invariant_outside_uniform=True,invariant_eligibility=True,
        scope='explicit oracle construction only; trueempty andunknown receive neutral.2, not GT presence')
    return correct,permuted,eligible,audit


def validate_bank(plan):
    if tuple(plan['check_cases'])!=CHECK_CASES or plan['weights_sha256']!=WEIGHT_SHA:
        raise ValueError('Frozen readout identity/case bank differs')
    selected=[dict({k:v for k,v in spec.items() if k!='window_sample'},original_bank_index=index)
              for index,spec in enumerate(plan['bank']) if spec['role']=='check']
    if len(selected)!=96:raise ValueError('Exactly the already fixed96check patches are required')
    for case in CHECK_CASES:
        rows=[s for s in selected if s['case_id']==case]
        if len(rows)!=8 or [s['slot'] for s in rows]!=list(range(8)):
            raise ValueError('Fixed check case/slot ordering differs')
        for spec in rows:
            expected='GT_centerline_plus_jitter' if spec['slot']<4 else 'CT_uniform'
            if spec['stratum']!=expected:raise ValueError('Frozen patch stratum differs')
    if [s['case_id'] for s in selected]!=[case for case in CHECK_CASES for _ in range(8)]:
        raise ValueError('Original check bank order differs')
    return selected


def verify_sources(root):
    root=Path(root).resolve()
    actual={relative:sha(root/relative) for relative in SOURCES}
    if actual!=SOURCES:raise ValueError('Frozenr3qa model/helper source bytes differ')
    return actual


def plan_spec(args):
    if sha(args.plan)!=PLAN_SHA:raise ValueError('Readout plan SHA differs')
    original=read(args.plan)
    selected=validate_bank(original)
    gate=read(args.gate_config)
    if not gate.get('locked_before_oracle_execution') or any(
        gate.get(k)!=v for k,v in (('readout_plan_sha256',PLAN_SHA),
                                   ('common_weights_sha256',WEIGHT_SHA),('common_head_sha256',HEAD_SHA))):
        raise ValueError('Prospective root gate is not bound to the fixed artifacts')
    sources=verify_sources(args.source_root)
    return dict(schema='GTA-AVT-fixed-local-oracle-plan-v1',
        readout_plan_path=str(Path(args.plan).resolve()),readout_plan_sha256=PLAN_SHA,
        gate_config_path=str(Path(args.gate_config).resolve()),gate_config_sha256=sha(args.gate_config),
        gate=gate,source_root=str(Path(args.source_root).resolve()),source_sha256=sources,
        private_probe_source_sha256=sha(__file__),
        weights=args.weights or original['weights'],weights_sha256=WEIGHT_SHA,
        readout_head=str(Path(args.head).resolve()),readout_head_sha256=HEAD_SHA,
        input_manifest=args.input_manifest or original['manifest'],
        input_manifest_sha256=original['manifest_sha256'],cache_root_override=args.cache_root,
        case_ids=list(CHECK_CASES),patches=selected,modes=list(MODES),
        operator_q0=Q0,neutral_s0=S0,alpha=1.,
        geometry='original hard2/4/8mm bins,16mmwindows onGAAM1.5+4u; no new extraction',
        primary_measure='mean full128cubed Dice of four primary nonempty patches within each case, then mean12cases',
        Dice_ratio_units=True,Dice_difference_pp_conversion=100.,
        Dice_formula='sigmoid(native logits)>=.5; float32(2TP+1e-6)/(pred+GT+1e-6), full128cubed',
        real_CT_metrics='independent descriptive counts/Dice oninput_valid; never replace main score',
        empty_policy='No empty Dice pooled; trueempty reportFP onfullpatch andrealCT and assertthree-routeidentity',
        intervention='within eachpatch, roll eligible lexicographic rows byfloor(N/2); no renormalization',
        noneligible='all unknown,trueempty,incompleteCTwindows remain exactlyuniform.2 inbothoracle fields',
        model_initialization='loadoldcommon intoA, loadreadout, thenB.load_state_dict(A.state_dict,strict=True) preservesGDFhead',
        A_B_check='one no-GT configured frozenforward atalpha1, allcommon outputtensors exactlyequal',
        precision='CUDAAMP forfrozenmodel, FP32externalfields; TF32off; no backward',
        patient_arrays_opened_in_plan_only=False,parameters_updated=False,
        automatic_prediction_stage=False,automatic_training=False,
        preview_only=True,actual_input_or_weight_verification_in_plan_only=False)


def load_model_modules(root):
    root=Path(root).resolve()
    for folder in (root/'code',root/'next_stage'):
        sys.path.insert(0,str(folder))
    helper=importlib.import_module('frozen_readout')
    learning=importlib.import_module('local_learning_model')
    frozen=importlib.import_module('local_model')
    modules={'frozen_readout':'next_stage/frozen_readout.py',
             'local_learning_model':'next_stage/local_learning_model.py',
             'local_model':'next_stage/local_model.py',
             'stage_model':'next_stage/stage_model.py','faithful_model':'code/faithful_model.py',
             'historical':'code/historical/__init__.py','historical.func':'code/historical/func/__init__.py',
             'historical.func.AnatomicalClassifier':'code/historical/func/AnatomicalClassifier.py',
             'historical.func.AnatomicalFusionModules':'code/historical/func/AnatomicalFusionModules.py',
             'historical.func.model_arch_model10':'code/historical/func/model_arch_model10.py'}
    for name,relative in modules.items():
        module=sys.modules.get(name)
        if module is None or Path(module.__file__).resolve()!=root/relative:
            raise ValueError('Loaded model/helper from an unintended source root')
    for name in ('data','geometry','stage_train','inference','next_data',
                 'local_learning_train','local_learning_data'):
        module=sys.modules.get(name)
        if module is not None and getattr(module,'__file__',None):
            path=Path(module.__file__).resolve()
            if root in path.parents:raise RuntimeError('Forbidden scientific-stack import: '+name)
    return helper,learning.LocalLearningModel,frozen.LocalStageModel


def construct_pair(torch,learning_class,frozen_class,common,head):
    a=learning_class('local_scale',alpha=1.,q0=Q0,s0=S0,
                     gdf_seed=20261023,local_seed=20261023)
    a.load_common_state(common)
    a.load_readout_state(head)
    b=frozen_class('local_scale',alpha=1.,q0=Q0,s0=S0,
                   gdf_seed=20261023,local_seed=20261023)
    b.load_state_dict(a.state_dict(),strict=True)
    for model in (a,b):
        model.eval()
        model.requires_grad_(False)
    if a.state_dict().keys()!=b.state_dict().keys() or any(
        not torch.equal(value,b.state_dict()[name]) for name,value in a.state_dict().items()):
        raise AssertionError('Complete A/B initialization differs')
    return a,b


def rng_hashes(torch):
    return dict(python=hashlib.sha256(pickle.dumps(random.getstate(),protocol=4)).hexdigest(),
        numpy=hashlib.sha256(pickle.dumps(np.random.get_state(),protocol=4)).hexdigest(),
        torch_CPU=array_hash(torch.get_rng_state().cpu().numpy()),
        torch_CUDA=[array_hash(x.cpu().numpy()) for x in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_initialized() else [])


def buffer_hash(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.named_buffers()):
        digest.update(name.encode());digest.update(array_hash(value.detach().cpu().numpy()).encode())
    return digest.hexdigest()


def frozen_eval(model):
    if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise AssertionError('Model is not fully frozen/eval')
    if model.q0!=Q0 or model.s0!=S0 or model.alpha!=1.:
        raise AssertionError('Oracle operator coefficients oralpha changed')


def patch_truth(directory,spec,helper):
    mask=np.load(Path(directory)/'mask.npy',mmap_mode='r',allow_pickle=False)
    start=spec['start_zyx']
    slices=tuple(slice(int(s),min(int(s)+128,int(n))) for s,n in zip(start,mask.shape))
    truth=np.asarray(mask[slices],bool)
    truth=np.pad(truth,[(0,128-n) for n in truth.shape])
    return helper.transform_spatial(truth,spec['flips'],spec['rotation'],spec['rotation_axes'])


def patch_scores(torch,logits,truth,input_valid):
    """Legacy threshold/FP32 ratio; trueempty Dice is deliberately null."""
    prediction=(logits.sigmoid()>=.5).detach().cpu().numpy().reshape(128,128,128)
    truth=np.asarray(truth,bool);real=np.asarray(input_valid,bool)
    def region(active):
        pred=prediction & active;gt=truth & active
        tp=int(np.count_nonzero(pred&gt));pn=int(np.count_nonzero(pred));gn=int(np.count_nonzero(gt))
        numerator=torch.tensor(2*tp,dtype=torch.float32)+1e-6
        denominator=torch.tensor(pn+gn,dtype=torch.float32)+1e-6
        return dict(Dice=float(numerator/denominator) if gn else None,
            truth_voxels=gn,prediction_voxels=pn,TP_voxels=tp,
            FP_voxels=int(np.count_nonzero(pred & ~gt)),FN_voxels=int(np.count_nonzero(~pred & gt)))
    return dict(full128=region(np.ones_like(real)),real_CT=region(real),
                true_empty=not bool(truth.any()),empty_prediction=not bool(prediction.any()),
                prediction_sha256=array_hash(prediction)),prediction


def _mean(values):
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def summarize(rows,case_ids=CHECK_CASES):
    cases=[]
    for cid in case_ids:
        selected=[r for r in rows if r['case_id']==cid]
        primary=[r for r in selected if r['slot']<4]
        uniform=[r for r in selected if r['slot']>=4]
        complete=(len(primary)==4 and len(uniform)==4 and
                  sorted(r['slot'] for r in selected)==list(range(8)) and
                  all(not r['true_empty'] for r in primary))
        values={mode:_mean([r['scores'][mode]['full128']['Dice'] for r in primary]) if complete else None
                for mode in MODES}
        diffs={name:(values[x]-values[y])*100. if values[x] is not None and values[y] is not None else None
               for name,x,y in (('correct_minus_permuted_Dice_pp','correct','permuted'),
                                ('correct_minus_constant_Dice_pp','correct','constant'))}
        uniform_nonempty=[r for r in uniform if not r['true_empty']]
        uniform_empty=[r for r in uniform if r['true_empty']]
        u=dict(patches=len(uniform),nonempty_patches=len(uniform_nonempty),trueempty_patches=len(uniform_empty),
            nonempty_full128_Dice={mode:_mean([r['scores'][mode]['full128']['Dice'] for r in uniform_nonempty]) for mode in MODES},
            trueempty_FP_full128_sum={mode:sum(r['scores'][mode]['full128']['FP_voxels'] for r in uniform_empty) for mode in MODES},
            trueempty_FP_real_CT_sum={mode:sum(r['scores'][mode]['real_CT']['FP_voxels'] for r in uniform_empty) for mode in MODES})
        cases.append(dict(case_id=cid,source=cid.rsplit('_',1)[0],primary_patch_count=len(primary),
            complete_primary_case=complete,primary_Dice_ratio=values,**diffs,
            informative_case=any(r['oracle']['eligible_windows']>=2 and r['oracle']['nonconstant_fine'] for r in primary),
            primary_fine_changed_windows=sum(r['oracle']['fine_value_changed_windows'] for r in primary),
            mean_absolute_logit_correct_vs_permuted=_mean([r['mean_abs_logit']['correct_vs_permuted'] for r in primary]) if complete else None,
            mean_absolute_logit_correct_vs_constant=_mean([r['mean_abs_logit']['correct_vs_constant'] for r in primary]) if complete else None,
            CT_uniform=u))
    primary=dict(complete_cases=sum(c['complete_primary_case'] for c in cases),
        informative_cases=sum(c['informative_case'] for c in cases),
        mean_Dice_ratio={mode:_mean([c['primary_Dice_ratio'][mode] for c in cases]) for mode in MODES},
        correct_minus_permuted_mean_Dice_pp=_mean([c['correct_minus_permuted_Dice_pp'] for c in cases]),
        correct_minus_constant_mean_Dice_pp=_mean([c['correct_minus_constant_Dice_pp'] for c in cases]),
        strictly_positive_case_Dice_differences=sum(c['correct_minus_permuted_Dice_pp'] is not None and c['correct_minus_permuted_Dice_pp']>0. for c in cases),
        mean_absolute_logit_correct_vs_permuted=_mean([c['mean_absolute_logit_correct_vs_permuted'] for c in cases]),
        mean_absolute_logit_correct_vs_constant=_mean([c['mean_absolute_logit_correct_vs_constant'] for c in cases]))
    primary['case_Dice_sample_SD_ratio']={mode:float(np.std([c['primary_Dice_ratio'][mode] for c in cases],ddof=1))
        if len(cases)>1 and all(c['primary_Dice_ratio'][mode] is not None for c in cases) else None for mode in MODES}
    sources={source:dict(cases=sum(c['source']==source for c in cases),
        correct_minus_permuted_mean_Dice_pp=_mean([c['correct_minus_permuted_Dice_pp'] for c in cases if c['source']==source]),
        correct_minus_constant_mean_Dice_pp=_mean([c['correct_minus_constant_Dice_pp'] for c in cases if c['source']==source]))
        for source in sorted(set(c['source'] for c in cases))}
    empty=[r for r in rows if r['slot']>=4 and r['true_empty']]
    nonempty=[r for r in rows if r['slot']>=4 and not r['true_empty']]
    uniform=dict(trueempty_patches=len(empty),nonempty_patches=len(nonempty),
        trueempty_FP_full128_sum={m:sum(r['scores'][m]['full128']['FP_voxels'] for r in empty) for m in MODES},
        trueempty_FP_real_CT_sum={m:sum(r['scores'][m]['real_CT']['FP_voxels'] for r in empty) for m in MODES},
        nonempty_full128_patch_mean_Dice={m:_mean([r['scores'][m]['full128']['Dice'] for r in nonempty]) for m in MODES},
        nonempty_defined_cases=sum(c['CT_uniform']['nonempty_patches']>0 for c in cases),
        nonempty_defined_case_mean_Dice={m:_mean([c['CT_uniform']['nonempty_full128_Dice'][m] for c in cases if c['CT_uniform']['nonempty_patches']]) for m in MODES},
        policy='CT-uniform descriptive only; defined subsets explicit; never pooled with primary or emptyDice')
    return dict(primary=primary,cases=cases,sources=sources,CT_uniform=uniform)


def evaluate_gate(summary,gate,invariants):
    p=summary['primary']
    ge=lambda value,threshold:value is not None and value>=threshold
    clauses=dict(complete_cases=p['complete_cases']>=gate['minimum_complete_cases'],
        informative_cases=p['informative_cases']>=gate['minimum_informative_cases'],
        correct_minus_permuted_Dice=ge(p['correct_minus_permuted_mean_Dice_pp'],gate['minimum_correct_minus_repaired_mean_Dice_pp']),
        positive_cases=p['strictly_positive_case_Dice_differences']>=gate['minimum_strictly_positive_case_Dice_differences'],
        mean_absolute_logit_response=ge(p['mean_absolute_logit_correct_vs_permuted'],gate['minimum_correct_vs_repaired_mean_absolute_logit_response']),
        correct_minus_constant_Dice=ge(p['correct_minus_constant_mean_Dice_pp'],gate['minimum_correct_minus_constant_mean_Dice_pp']),
        required_invariants=all(invariants.values()))
    passed=all(clauses.values())
    return dict(passed=passed,clauses=clauses,registered_gate=gate,
        action=gate['passing_action'] if passed else gate['failing_action'],
        automatic_followup_executed=False)


def execute(args,spec):
    import torch
    output=Path(args.output)
    if output.exists():raise FileExistsError('One immutable new oracle output directory required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA execution required; no silent fallback')
    if sha(spec['weights'])!=WEIGHT_SHA or sha(spec['readout_head'])!=HEAD_SHA:
        raise ValueError('Pinned common checkpoint/readout bytes differ')
    if sha(spec['input_manifest'])!=spec['input_manifest_sha256']:
        raise ValueError('Frozen source input manifest differs')
    original=read(args.plan)
    manifest=read(spec['input_manifest'])
    train=set(manifest['splits']['train'])
    if not set(CHECK_CASES)<=train or train & (set(manifest['splits']['val'])|set(manifest['splits']['test'])):
        raise ValueError('The fixed check bank must remain TRAIN-only')
    records={row['id']:row for row in manifest['cases']}
    helper,learning_class,frozen_class=load_model_modules(spec['source_root'])
    output.mkdir(parents=True)
    write_new(output/'execution_plan.json',dict(spec,preview_only=False))
    write_new(output/'execution_started.json',dict(schema='GTA-AVT-fixed-oracle-start-v1',
        time_unix=time.time(),torch_version=torch.__version__,numpy_version=np.__version__,
        GPU=torch.cuda.get_device_name(0),source_sha256=spec['source_sha256'],
        readout_plan_sha256=PLAN_SHA,gate_config_sha256=spec['gate_config_sha256']))
    started=time.perf_counter()
    try:
        inputs={};geometries={};actual_paths=[]
        for cid in CHECK_CASES:
            directory=Path(args.cache_root)/cid if args.cache_root else Path(original['inputs'][cid]['cache_dir'])
            row=records[cid];hashes={}
            for name in ('image.npy','mask.npy','geometry.npz'):
                path=directory/name;actual=sha(path)
                if actual!=row['cache_file_sha256'][name]:raise ValueError('TRAIN cache bytes differ: '+cid+'/'+name)
                if name=='geometry.npz' and actual!=original['inputs'][cid]['geometry_sha256']:
                    raise ValueError('Readout-plan geometry differs: '+cid)
                hashes[name]=dict(path=str(path),bytes=path.stat().st_size,sha256=actual)
                actual_paths.append(str(path))
            image=np.load(directory/'image.npy',mmap_mode='r',allow_pickle=False)
            mask=np.load(directory/'mask.npy',mmap_mode='r',allow_pickle=False)
            if image.shape!=mask.shape or list(image.shape)!=row['shape_zyx']:
                raise ValueError('TRAIN image/mask/manifest grid differs')
            del image,mask
            inputs[cid]=dict(cache_dir=str(directory),files=hashes)
            geometries[cid]=helper._load_geometry(directory)
        torch.set_num_threads(4)
        random.seed(original['seed']);np.random.seed(original['seed']);torch.manual_seed(original['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True,warn_only=True)
        common=torch.load(spec['weights'],map_location='cpu')
        if isinstance(common,dict) and 'model' in common:common=common['model']
        head=torch.load(spec['readout_head'],map_location='cpu')
        a,b=construct_pair(torch,learning_class,frozen_class,common,head)
        del common,head
        a.cuda();b.cuda()
        state_before=helper._state_hash(b);a_before=helper._state_hash(a)
        if state_before!=a_before:raise AssertionError('A/B state digest mismatch')
        buffers_before=buffer_hash(b)
        RNG_before=rng_hashes(torch)
        parity=None;rows=[]
        for index,patch in enumerate(spec['patches']):
            cid=patch['case_id'];directory=inputs[cid]['cache_dir']
            local=helper.prepare_example(directory,geometries[cid],patch,True)
            if helper.target_stats(local)!=patch['target_audit']:
                raise ValueError('Original full-field target audit changed')
            truth=patch_truth(directory,patch,helper)
            correct,permuted,eligible,audit=oracle_fields(local)
            image=torch.from_numpy(local['image'][None]).cuda()
            valid=torch.from_numpy(local['input_valid'][None,None]).cuda()
            if parity is None:
                with torch.no_grad(),torch.cuda.amp.autocast():
                    pa=a(image,input_valid=valid,alpha=1.)
                    pb=b(image,input_valid=valid,alpha=1.,trace_local=True)
                common_keys=set(pb)
                for name in common_keys:
                    xa,xb=pa[name],pb[name]
                    if isinstance(xa,list):same=all(torch.equal(x,y) for x,y in zip(xa,xb))
                    else:same=torch.equal(xa,xb)
                    if not same:raise AssertionError('Frozen configured A/B output differs: '+name)
                if helper._state_hash(a)!=a_before or helper._state_hash(b)!=state_before:
                    raise AssertionError('A/B parity forward modified state')
                if rng_hashes(torch)!=RNG_before:raise AssertionError('A/B eval parity changed RNG')
                parity=dict(passed=True,patch_original_bank_index=patch['original_bank_index'],
                    no_GT_configured_input=True,common_output_keys=sorted(common_keys),
                    full_state_digest_equal=True,outputs_exact=True,alpha=1.,AMP=True)
                del pa,pb,a
                torch.cuda.empty_cache()
            scores={};logits={};predictions={};trace_invariants=[]
            for mode in MODES:
                frozen_eval(b)
                kwargs=dict(input_valid=valid,alpha=1.,trace_local=True)
                if mode=='constant':kwargs['condition_mode']='constant'
                else:
                    field=correct if mode=='correct' else permuted
                    kwargs.update(local_probability=torch.from_numpy(field[None]).cuda(),local_mode='configured')
                with torch.no_grad(),torch.cuda.amp.autocast():
                    observed=b(image,**kwargs)
                score,prediction=patch_scores(torch,observed['logits'],truth,local['input_valid'])
                scores[mode]=score;predictions[mode]=prediction
                logits[mode]=observed['logits'].detach().float().cpu()
                if mode!='constant':
                    trace=b.last_local_trace
                    consumed=trace['consumed_score'].cpu().numpy()[0,0]
                    expected=(correct if mode=='correct' else permuted)[1].copy()
                    expected[~local['ct_support']]=S0
                    if not np.array_equal(consumed,expected):raise AssertionError('Consumed oracle score differs from supplied/fallbackfield')
                    if trace['head_evaluated'] or not trace['oracle_input']:
                        raise AssertionError('Oracle route unexpectedly evaluated a predictor')
                    trace_invariants.append(True)
                if rng_hashes(torch)!=RNG_before:raise AssertionError('Frozen oracle forward changed RNG')
                del observed
            empty=not bool(truth.any())
            empty_identical=not empty or all(np.array_equal(predictions['constant'],predictions[m])
                and torch.equal(logits['constant'],logits[m]) for m in ('correct','permuted'))
            if not empty_identical:raise AssertionError('Trueempty three-route outputs differ')
            response={name:float((logits[x]-logits[y]).abs().mean())
                for name,x,y in (('correct_vs_permuted','correct','permuted'),
                                ('correct_vs_constant','correct','constant'),
                                ('permuted_vs_constant','permuted','constant'))}
            record=dict(case_id=cid,source=patch['source'],slot=patch['slot'],
                original_bank_index=patch['original_bank_index'],stratum=patch['stratum'],
                start_zyx=patch['start_zyx'],flips=patch['flips'],rotation=patch['rotation'],
                rotation_axes=patch['rotation_axes'],intensity_seed=patch['intensity_seed'],
                input_image_sha256=array_hash(local['image']),truth_sha256=array_hash(truth),
                CT_input_valid_sha256=array_hash(local['input_valid']),
                input_real_CT_voxels=int(local['input_valid'].sum()),true_empty=empty,
                oracle=audit,scores=scores,mean_abs_logit=response,
                trueempty_three_routes_identical=empty_identical,
                consumed_score_equals_supplied_with_CT_fallback=all(trace_invariants))
            rows.append(record)
            with (output/'patchwise.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,allow_nan=False)+'\n')
            if (index+1)%8==0:
                print(json.dumps(dict(event='oracle_case_complete',case_id=cid,patches=index+1)),flush=True)
            del image,valid,logits,predictions,local
        state_after=helper._state_hash(b);buffers_after=buffer_hash(b);RNG_after=rng_hashes(torch)
        frozen_eval(b)
        source_after=verify_sources(spec['source_root'])
        source_unchanged=(source_after==spec['source_sha256'] and
            sha(__file__)==spec['private_probe_source_sha256'] and sha(args.plan)==PLAN_SHA and
            sha(args.gate_config)==spec['gate_config_sha256'])
        invariants=dict(TRAIN_only_files=True,no_fitting_or_updates=True,
            model_tensors_unchanged=state_before==state_after,
            BN_buffers_unchanged=buffers_before==buffers_after,RNG_unchanged=RNG_before==RNG_after,
            evaluation_state_unchanged=True,source_and_input_identities_bound=source_unchanged,
            operator_q0_fixed=b.q0==Q0,neutral_s0_fixed=b.s0==S0,
            same_eligible_and_joint_r_multiset=all(r['oracle']['invariant_joint_r_multiset'] and r['oracle']['invariant_eligibility'] for r in rows),
            outside_scores_neutral=all(r['oracle']['invariant_outside_uniform'] for r in rows),
            consumed_oracle_scores_match=all(r['consumed_score_equals_supplied_with_CT_fallback'] for r in rows),
            trueempty_outputs_identical=all(r['trueempty_three_routes_identical'] for r in rows),
            A_B_frozen_configured_parity=parity['passed'])
        if not all(invariants.values()):raise AssertionError('Frozen-state invariant failed')
        summary=summarize(rows)
        verdict=evaluate_gate(summary,spec['gate'],invariants)
        report=dict(schema='GTA-AVT-local-oracle-probe-v1',
            readout_plan_sha256=PLAN_SHA,weights_sha256=WEIGHT_SHA,readout_head_sha256=HEAD_SHA,
            gate_config_sha256=spec['gate_config_sha256'],source_sha256=spec['source_sha256'],
            private_probe_source_sha256=sha(__file__),inputs=inputs,actual_patient_files_opened=actual_paths,
            cases_opened=list(CHECK_CASES),VAL_TEST_patient_files_opened=False,
            params_updated=False,optimizer_created=False,model_state_sha256_before=state_before,
            model_state_sha256_after=state_after,BN_buffers_sha256_before=buffers_before,
            BN_buffers_sha256_after=buffers_after,RNG_before=RNG_before,RNG_after=RNG_after,
            A_B_configured_parity=parity,invariants=invariants,**summary,
            gate=verdict,patch_count=len(rows),primary_patch_count=sum(r['slot']<4 for r in rows),
            patchwise_file='patchwise.jsonl',patchwise_sha256=sha(output/'patchwise.jsonl'),
            primary_scoring=spec['Dice_formula'],Dice_ratio_units=True,pp_difference_conversion=100.,
            actual_runtime=dict(torch=torch.__version__,numpy=np.__version__,GPU=torch.cuda.get_device_name(0),AMP=True,
                CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG')),
            elapsed_seconds=time.perf_counter()-started,
            limits=['One cold common initialization and fixed window/permutation only.',
                'TRAIN data with GToracle intervention; not normal deployment or independent-patient performance.',
                'Half-ring preserveseligible/fullr multiset but changesfield smoothness andlocal correlations.',
                'Negative coldprobe does not refutelearnableGTA co-adaptation.',
                'Passing only permits a separatelyspecifiedfrozenpredictiondiagnostic, never automatictraining.'])
        write_new(output/'analysis.json',report)
        print(json.dumps(dict(event='local_oracle_complete',gate=verdict,primary=summary['primary'])),flush=True)
        return report
    except BaseException as error:
        write_new(output/'failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error),
            elapsed_seconds=time.perf_counter()-started,automatic_retry=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--plan-only',action='store_true')
    mode.add_argument('--execute',action='store_true')
    parser.add_argument('--source-root',required=True)
    parser.add_argument('--plan',required=True,help='Existing immutable readout plan, not a resampled bank')
    parser.add_argument('--head',required=True)
    parser.add_argument('--weights')
    parser.add_argument('--gate-config',required=True)
    parser.add_argument('--input-manifest')
    parser.add_argument('--cache-root')
    parser.add_argument('--output')
    args=parser.parse_args()
    spec=plan_spec(args)
    if not args.execute:
        preview={k:v for k,v in spec.items() if k!='patches'}
        preview['fixed_patch_count']=len(spec['patches'])
        preview['original_bank_indices']=[p['original_bank_index'] for p in spec['patches']]
        print(json.dumps(dict(execute=False,plan=preview),ensure_ascii=False,indent=2))
        return
    if not args.output:parser.error('Explicit execute requires a new --output directory')
    execute(args,spec)


if __name__=='__main__':main()
