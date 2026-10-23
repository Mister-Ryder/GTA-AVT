"""Synthetic-only contracts for the one fixed local oracle probe.

Default checks use NumPy, no patient files/Torch. --torch additionally checks
legacy scoring, and --full-patch128 checks frozen model integration on a purely
synthetic pair. Neither option loads study checkpoints or executes training.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import local_oracle_probe as probe


def field_fixture():
    target=np.zeros((5,32,32,32),np.float32)
    ct=np.zeros((32,)*3,bool);ct[2:30,2:30,2:30]=True
    valid=ct.copy();foreground=np.zeros_like(ct)
    locations=np.asarray([[3,4,5],[5,4,3],[8,12,16],[12,8,16],[16,12,8]])
    values=np.asarray([[.7,.1,.1,.1],[.1,.7,.1,.1],[.2,.3,.4,.1],
                       [.05,.05,.8,.1],[0.,0.,0.,1.]],np.float32)
    foreground[tuple(locations.T)]=True
    target[0,ct & ~foreground]=1.
    target[1:,tuple(locations.T)[0],tuple(locations.T)[1],tuple(locations.T)[2]]=values.T
    # An interior foreground/geometry unknown is not empty and stays neutral.
    foreground[9,9,9]=True;valid[9,9,9]=False;target[:,9,9,9]=0.
    return dict(target=target,ct_support=ct,loss_valid=valid,foreground_window=foreground),locations,values


def field_checks():
    local,locations,values=field_fixture()
    correct,permuted,eligible,audit=probe.oracle_fields(local)
    assert np.array_equal(correct[1:,eligible].T,values)
    # Lexicographic locations are already in increasing flat-index order.
    expected=np.roll(values,2,axis=0)
    assert np.array_equal(permuted[1:,eligible].T,expected)
    assert np.equal(correct[:,~eligible],np.float32(.2)).all()
    assert np.equal(permuted[:,~eligible],np.float32(.2)).all()
    assert np.equal(correct[:,9,9,9],np.float32(.2)).all()
    assert audit['eligible_windows']==5 and audit['GT_unknown_windows']==1
    assert audit['invariant_joint_r_multiset'] and audit['invariant_raw_fine_multiset']
    assert audit['fine_value_changed_windows']>0 and audit['nonconstant_fine']
    # Adversarial 1ulp row sums must be moved without normalizing the rows.
    special=copy.deepcopy(local)
    special['target'][1:,3,4,5]=np.asarray([.33333334,.33333334,.33333334,1e-8],np.float32)
    a,b,e,check=probe.oracle_fields(special)
    assert probe.row_multiset_hash(a[1:,e].T)==probe.row_multiset_hash(b[1:,e].T)
    assert np.array_equal(b[1:,e].T,np.roll(a[1:,e].T,2,axis=0))
    # Empty/unknown/incomplete locations cannot supply a presence intervention.
    empty=dict(target=np.zeros_like(local['target']),ct_support=local['ct_support'],
               loss_valid=local['ct_support'],foreground_window=np.zeros_like(eligible))
    empty['target'][0,empty['ct_support']]=1.
    ec,ep,ee,ea=probe.oracle_fields(empty)
    assert not ee.any() and np.array_equal(ec,ep) and np.equal(ec,np.float32(.2)).all()
    singleton=copy.deepcopy(local)
    singleton['loss_valid'][:]=False;singleton['loss_valid'][3,4,5]=True
    c,p,e,a=probe.oracle_fields(singleton)
    assert a['eligible_windows']==1 and a['half_ring_shift']==0 and np.array_equal(c,p)
    assert not a['nonconstant_fine']
    bad=copy.deepcopy(local);bad['target'][0,3,4,5]=.1
    try:probe.oracle_fields(bad)
    except ValueError:pass
    else:raise AssertionError('Nonempty GT with empty mass was accepted')
    return dict(status='passed',half_ring_full_rows_exact=True,
        joint_multiset_and_fine_multiset_exact=True,no_renormalization_even1ulp=True,
        unknown_empty_border_neutral=True,empty_singleton_degeneracy_reported=True)


def synthetic_rows():
    rows=[]
    for index,cid in enumerate(probe.CHECK_CASES):
        for slot in range(8):
            empty=slot>=6
            values={m:(None if empty else (.5+index*.01+(dict(constant=0.,correct=.002,permuted=0.)[m])))
                    for m in probe.MODES}
            oracle=dict(eligible_windows=10,nonconstant_fine=True,fine_value_changed_windows=6)
            if empty:oracle.update(eligible_windows=0,nonconstant_fine=False,fine_value_changed_windows=0)
            rows.append(dict(case_id=cid,slot=slot,true_empty=empty,oracle=oracle,
                mean_abs_logit=dict(correct_vs_permuted=.002,correct_vs_constant=.003),
                scores={m:dict(full128=dict(Dice=values[m],FP_voxels=7 if empty else 2),
                               real_CT=dict(Dice=values[m],FP_voxels=3 if empty else 1)) for m in probe.MODES}))
    return rows


def aggregate_checks(gate):
    rows=synthetic_rows()
    summary=probe.summarize(rows)
    p=summary['primary']
    assert p['complete_cases']==12 and p['informative_cases']==12
    assert abs(p['correct_minus_permuted_mean_Dice_pp']-.2)<1e-10
    assert abs(p['mean_Dice_ratio']['constant']-.555)<1e-10
    assert p['strictly_positive_case_Dice_differences']==12
    assert summary['CT_uniform']['trueempty_patches']==24
    assert summary['CT_uniform']['nonempty_patches']==24
    assert summary['CT_uniform']['trueempty_FP_full128_sum']['correct']==24*7
    assert summary['CT_uniform']['trueempty_FP_real_CT_sum']['correct']==24*3
    invariants={'synthetic_invariant':True}
    assert probe.evaluate_gate(summary,gate,invariants)['passed']
    # Empty uniform Dice values, even if maliciously filled with1, cannot inflate
    # the primary or the explicit nonempty CT-uniform score.
    changed=copy.deepcopy(rows)
    for r in changed:
        if r['true_empty']:
            for mode in probe.MODES:r['scores'][mode]['full128']['Dice']=1.
    assert probe.summarize(changed)['primary']==p
    assert probe.summarize(changed)['CT_uniform']['nonempty_full128_patch_mean_Dice']==summary['CT_uniform']['nonempty_full128_patch_mean_Dice']
    for missing in (0,4):
        deficient=[r for i,r in enumerate(rows) if i!=missing]
        s=probe.summarize(deficient)
        assert s['primary']['complete_cases']==11
        assert s['primary']['correct_minus_permuted_mean_Dice_pp'] is None
        assert not probe.evaluate_gate(s,gate,invariants)['passed']
    uninformative=copy.deepcopy(rows)
    for r in uninformative:
        if r['case_id'] in probe.CHECK_CASES[:5]:r['oracle']['nonconstant_fine']=False
    assert not probe.evaluate_gate(probe.summarize(uninformative),gate,invariants)['passed']
    assert not probe.evaluate_gate(summary,gate,{'state_unchanged':False})['passed']
    # Gate values are read from the prospective JSON, not embedded as a second
    # implementation threshold. A stronger synthetic threshold must be obeyed.
    demanding=dict(gate,minimum_correct_minus_repaired_mean_Dice_pp=1.)
    assert not probe.evaluate_gate(summary,demanding,invariants)['passed']
    return dict(status='passed',case_equal_four_primary_patches=True,
        legacy_ratio_and_pp_conversion=True,empty_Dice_not_pooled=True,
        trueempty_fullpatch_and_CT_FP_separate=True,missing_primary_not_dropped=True,
        insufficient_support_or_invariant_blocks=True,prospective_JSON_controls_thresholds=True)


def bank_checks():
    bank=[]
    for cid in probe.CHECK_CASES:
        for slot in range(8):bank.append(dict(case_id=cid,slot=slot,role='check',
            stratum='GT_centerline_plus_jitter' if slot<4 else 'CT_uniform',window_sample={'unused':True}))
    plan=dict(check_cases=list(probe.CHECK_CASES),weights_sha256=probe.WEIGHT_SHA,bank=bank)
    checked=probe.validate_bank(plan)
    assert len(checked)==96 and all('window_sample' not in p for p in checked)
    for bad in ('missing','reordered','replaced'):
        copy_plan=copy.deepcopy(plan)
        if bad=='missing':copy_plan['bank'].pop()
        elif bad=='reordered':copy_plan['bank'][0],copy_plan['bank'][1]=copy_plan['bank'][1],copy_plan['bank'][0]
        else:copy_plan['bank'][0]['case_id']='replacement_NOT_A_CASE'
        try:probe.validate_bank(copy_plan)
        except ValueError:pass
        else:raise AssertionError('Altered fixed bank accepted: '+bad)
    return dict(status='passed',exact96_existing_order=True,no_sparse_window_resampling=True,
                missing_reordered_replaced_patches_rejected=True)


def torch_scoring_checks(device):
    import torch
    logits=torch.full((1,1,128,128,128),-10.,device=device)
    truth=np.zeros((128,)*3,bool);truth[:2,:2,:2]=True
    logits[0,0,:1,:2,:2]=10.
    logits[0,0,127,127,127]=10.
    valid=np.ones_like(truth);valid[127]=False
    result,prediction=probe.patch_scores(torch,logits,truth,valid)
    expected=float((torch.tensor(8.,dtype=torch.float32)+1e-6)/(torch.tensor(13.,dtype=torch.float32)+1e-6))
    assert result['full128']['Dice']==expected
    assert result['full128']['FP_voxels']==1 and result['real_CT']['FP_voxels']==0
    assert result['full128']['FN_voxels']==4
    empty,_=probe.patch_scores(torch,logits,np.zeros_like(truth),valid)
    assert empty['full128']['Dice'] is None and empty['real_CT']['Dice'] is None
    # Compare the actual legacy fixture formula on native logits (no early
    # float32 sigmoid cast). This also covers a near-zero half-precision logit.
    native_checks={}
    for dtype,value in (('float32',logits),('float16',logits.half())):
        try:
            predicted=value.sigmoid()>=.5
        except RuntimeError as error:
            message=str(error)
            if value.device.type=='cpu' and dtype=='float16' and ('not implemented' in message.lower() or 'not support' in message.lower()):
                native_checks[dtype]=dict(status='not_supported',reason=message)
                continue
            raise
        gt=torch.from_numpy(truth).to(value.device)[None,None]
        intersection=(predicted&gt).sum().float()
        denominator=predicted.sum().float()+gt.sum().float()
        legacy=float((2*intersection+1e-6)/(denominator+1e-6))
        got,_=probe.patch_scores(torch,value,truth,valid)
        assert got['full128']['Dice']==legacy
        native_checks[dtype]=dict(status='passed',native_sigmoid_formula_exact=True)
    return dict(status='passed',full128_old_formula_exact=True,
        trueempty_Dice_null=True,real_CT_FP_separate=True,
        device=str(logits.device),native_dtype_checks=native_checks)


def full_model_checks(source_root,device):
    import torch
    helper,learning_class,frozen_class=probe.load_model_modules(source_root)
    torch.manual_seed(20261023)
    template=learning_class('local_scale',alpha=1.)
    common={k:v for k,v in template.state_dict().items() if not k.startswith(('local_head.','gdf_head.'))}
    head={k:v.clone() for k,v in template.local_head.state_dict().items()}
    head['weight'][1,0,0,0,0]=.1;head['bias'][1]=.8
    a,b=probe.construct_pair(torch,learning_class,frozen_class,common,head)
    del template,common,head
    a.to(device);b.to(device)
    image=torch.linspace(-1.,1.,2*128**3,device=device).reshape(2,1,128,128,128)
    input_valid=torch.ones_like(image,dtype=torch.bool)
    state=helper._state_hash(b);rng=probe.rng_hashes(torch);BN=probe.buffer_hash(b)
    amp=str(device).startswith('cuda')
    with torch.no_grad(),torch.cuda.amp.autocast(enabled=amp):
        x=a(image,input_valid=input_valid);y=b(image,input_valid=input_valid,trace_local=True)
    for key,value in y.items():
        same=all(torch.equal(xx,yy) for xx,yy in zip(x[key],value)) if isinstance(value,list) else torch.equal(x[key],value)
        assert same,key
    assert helper._state_hash(a)==state and helper._state_hash(b)==state
    del a,x,y
    local,_,_=field_fixture();correct,permuted,eligible,audit=probe.oracle_fields(local)
    # The second sample is a genuinely empty oracle target: it must stay uniform
    # and have identical predictions under all three routes.
    fields={mode:np.stack([field,np.full_like(field,.2)]) for mode,field in [('correct',correct),('permuted',permuted)]}
    out={}
    for mode in probe.MODES:
        kwargs=dict(input_valid=input_valid,trace_local=True)
        if mode=='constant':kwargs['condition_mode']='constant'
        else:kwargs['local_probability']=torch.from_numpy(fields[mode]).to(device)
        with torch.no_grad(),torch.cuda.amp.autocast(enabled=amp):
            out[mode]=b(image,**kwargs)['logits'].detach()
        if mode!='constant':
            score=b.last_local_trace['consumed_score'].cpu().numpy()
            wanted=fields[mode][:,1:2].copy()
            support=helper.ct_support_tensor(input_valid).cpu().numpy()
            wanted[~support]=.2
            assert np.array_equal(score,wanted)
            assert not b.last_local_trace['head_evaluated']
        assert probe.rng_hashes(torch)==rng
    assert torch.equal(out['constant'][1],out['correct'][1])
    assert torch.equal(out['constant'][1],out['permuted'][1])
    assert helper._state_hash(b)==state and probe.buffer_hash(b)==BN
    probe.frozen_eval(b)
    return dict(status='passed',synthetic_only=True,patch128=True,batch2=True,
        AMP=amp,A_B_configured_exact=True,head_and_GDF_initialization_exact=True,
        full_model_state_BN_RNG_unchanged=True,external_score_trace_exact=True,
        trueempty_three_routes_logits_exact=True,patient_checkpoints_loaded=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate-config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--torch',action='store_true')
    parser.add_argument('--full-patch128',action='store_true')
    parser.add_argument('--source-root')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=parser.parse_args()
    if args.full_patch128 and not args.source_root:parser.error('--full-patch128 requires --source-root')
    started=time.perf_counter()
    if args.torch or args.full_patch128:
        import torch
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True,warn_only=True)
        if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    gate=probe.read(args.gate_config)
    report=dict(schema='GTA-AVT-local-oracle-synthetic-QA-v1',status='passed',
        patient_arrays_opened=False,patient_checkpoints_loaded=False,optimizer_created=False,
        field=field_checks(),aggregate=aggregate_checks(gate),fixed_bank=bank_checks(),
        Torch_scoring=torch_scoring_checks(args.device) if args.torch or args.full_patch128 else dict(status='not_run'),
        full_model128=full_model_checks(args.source_root,args.device) if args.full_patch128 else dict(status='not_run'),
        source_sha256={'local_oracle_probe.py':probe.sha(probe.__file__),
                       'local_oracle_probe_checks.py':probe.sha(__file__)},
        gate_config_sha256=probe.sha(args.gate_config),seconds=time.perf_counter()-started)
    probe.write_new(args.output,report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
