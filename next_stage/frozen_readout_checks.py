"""Synthetic-only NumPy QA; --torch additionally checks deployment helpers.

Does not load patient images, checkpoints, or fit a readout. NumPy checks run on
the local workstation; Torch checks are intended for the existing remote stack.
"""
import argparse
import ast
import json
from pathlib import Path
import tempfile

import numpy as np
import frozen_readout as helper
from frozen_readout_fit import semantic_gate, weighted_median, _prior, evaluate


GATE = dict(locked_before_fit=True,
    minimum_relative_raw_MSE_reduction=.1,
    minimum_relative_conditional_MSE_reduction=.1,
    minimum_conditional_MSE_better_cases=8,
    minimum_fine_supported_cases=8,
    minimum_sampled_fine_cells_per_case=16,
    all_three_source_raw_and_conditional_MSE_not_worse=True,
    optimizer_resolved_required=True,
    MAE_is_report_only=True,
    automatic_segmentation_training_authorized=False,
    reference='fixed readout-label diagnostic only; root A protocol governs any paired segmentation continuation')


def numpy_checks(code_root):
    result = {}
    rng = np.random.default_rng(917)
    volume = rng.integers(0, 2, size=(128,)*3, dtype=np.uint8)
    sums = helper._box_sum(volume)
    for z,y,x in rng.integers(0,32,size=(40,3)):
        slices = tuple(slice(max(0,4*int(u)-6),min(128,4*int(u)+10)) for u in (z,y,x))
        assert int(sums[z,y,x]) == int(volume[slices].sum())
    valid = np.ones((128,)*3,bool)
    support = helper.ct_window_support(valid)
    expected = np.zeros((32,)*3,bool)
    expected[2:30,2:30,2:30] = True
    assert np.array_equal(support,expected)
    valid[:10] = False
    support_padded = helper.ct_window_support(valid)
    assert not support_padded[:4].any()
    assert support_padded[4:30,2:30,2:30].all()
    result['boxsum_bruteforce_and_CT_padding'] = 'passed'

    points = np.asarray([[64,64,64],[65,64,64],[66,64,64],[67,64,64]])
    mask = np.zeros((128,)*3,bool)
    mask[tuple(points.T)] = True
    radii = np.asarray([2.,4.,8.,9.])
    target = helper.local_targets(mask,points,radii,np.ones_like(mask))
    assert np.array_equal(target['target'][1:,16,16,16],np.full(4,.25,np.float32))
    assert target['loss_valid'][16,16,16]
    assert target['target'][0,2,2,2] == 1
    unknown = helper.local_targets(mask,np.empty((0,3)),np.empty(0),np.ones_like(mask))
    assert unknown['ct_support'][16,16,16] and not unknown['loss_valid'][16,16,16]
    assert unknown['target'][:,16,16,16].sum() == 0
    invalid = False
    try:
        helper.local_targets(mask,points,[2,4,8,0],np.ones_like(mask))
    except ValueError:
        invalid=True
    assert invalid
    result['hard_boundaries_empty_unknown_invalid'] = 'passed'

    flips=[True,False,True]
    axes=(0,2)
    transformed_points=points.copy()
    for axis,flip in enumerate(flips):
        if flip:
            transformed_points[:,axis]=127-transformed_points[:,axis]
    old=transformed_points.copy()
    transformed_points[:,axes[0]]=127-old[:,axes[1]]
    transformed_points[:,axes[1]]=old[:,axes[0]]
    transformed_mask=helper.transform_spatial(mask,flips,1,axes)
    recalculated=helper.local_targets(transformed_mask,transformed_points,radii,np.ones_like(mask))
    assert np.array_equal(recalculated['target'],helper.transform_spatial(target['target'],flips,1,axes,1))
    for key in ('loss_valid','ct_support','foreground_window','support_nodes'):
        assert np.array_equal(recalculated[key],helper.transform_spatial(target[key],flips,1,axes))
    result['voxel_and_GAAM_target_transform_commute'] = 'passed'

    sample=helper.select_windows(target,20261023,0,0)
    coords=np.asarray(sample['coords_zyx'])
    ipw=np.asarray(sample['inverse_inclusion_weights'])
    assert len(coords) <= 128
    assert np.isclose(ipw.sum(),target['loss_valid'].sum())
    assert sample==helper.select_windows(target,20261023,0,0)
    assert target['loss_valid'][tuple(coords.T)].all()
    # Exhaust all choices in a tiny stratified population: HT expectation equals
    # the full uniform-window objective, not a foreground/background-balanced loss.
    values=np.asarray([1.,3.,5.,10.,20.])
    estimates=[(3*values[f]+2*values[b])/5 for f in range(3) for b in range(3,5)]
    assert np.isclose(np.mean(estimates),values.mean())
    result['fixed_stratified_sampling_IPW_unbiased_fixture'] = 'passed'

    text=(Path(code_root)/'data.py').read_text(encoding='utf-8')
    tree=ast.parse(text)
    definition=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='normalize')
    module=ast.Module(body=[definition],type_ignores=[])
    namespace={'np':np}
    exec(compile(module,'frozen_data_normalize_fixture','exec'),namespace)
    fixture=np.asarray([-1024,-200,-1,0,200,700,1000,1500],np.float32)
    assert np.array_equal(helper.normalize(fixture),namespace['normalize'](fixture))
    result['frozen_data_normalize_function_fixture'] = 'passed'

    identities={
      'Dongyang':['D1','D10','D16','D17','D8','D6','D4','D9','D15','D3','D18','D12'],
      'KiTS':['K3','K17','K2','K11','K4','K19','K6','K20','K12','K16','K9','K14'],
      'Rider':['R7','R6','R1','R18','R17','R8','R4','R12','R5','R15','R3','R11']}
    expected_fit=[source+'_'+sid for source,ids in identities.items() for sid in ids[:8]]
    expected_check=[source+'_'+sid for source,ids in identities.items() for sid in ids[8:]]
    manifest={'splits':{'train':[source+'_'+sid for source,ids in identities.items() for sid in ids],
                        'val':[],'test':[]}}
    fit,check=helper.split_train(manifest,20261023)
    assert fit==expected_fit and check==expected_check
    manifest['splits']['test']=['unrelated_D1']
    try:
        helper.split_train(manifest,20261023)
    except ValueError:
        pass
    else:
        raise AssertionError('Heldout short-ID leakage was not rejected')
    result['prespecified_split_and_overlap_rejection'] = 'passed'

    cases=[]
    for i in range(12):
        cases.append(dict(sampled_fine_positive_windows=16))
    evaluation=dict(cases=cases,valid_nonempty_cases=12,conditional_MSE_better_cases=8,
       macro={k:dict(model_MSE=.08,mean_constant_MSE=.1) for k in ('raw','conditional')},
       sources={s:dict(cases=4,**{k:dict(model_MSE=.08,mean_constant_MSE=.1) for k in ('raw','conditional')})
                for s in ('Dongyang','KiTS','Rider')})
    assert semantic_gate(evaluation,GATE,True)['passed']
    assert not semantic_gate(evaluation,GATE,False)['passed']
    evaluation['macro']['conditional']['model_MSE']=.099
    assert not semantic_gate(evaluation,GATE,True)['passed']
    assert weighted_median(np.asarray([0.,1.,2.]),np.asarray([1.,3.,1.]))==1.
    result['fixed_semantic_gate_and_optimizer_unresolved_blocks'] = 'passed'

    ids=expected_fit+expected_check
    bank=dict(target=[],case_index=[],metric_weight=[],objective_weight=[],
              foreground=[],patch_stratum=[])
    plan=dict(fit_cases=expected_fit,check_cases=expected_check,bank=[])
    for c,case_id in enumerate(ids):
        for slot in range(8):
            for target_row,foreground in (([0,.1,.9,0,0],True),([0,.5,.5,0,0],True),
                                          ([1,0,0,0,0],False),([1,0,0,0,0],False)):
                for key,value in dict(target=target_row,case_index=c,metric_weight=.25,
                     objective_weight=1/(24*8*4),foreground=foreground,
                     patch_stratum=int(slot>=4)).items():
                    bank[key].append(value)
            plan['bank'].append(dict(case_id=case_id,target_audit=dict(fine_positive_cells=2)))
    bank={k:np.asarray(v) for k,v in bank.items()}
    fit_rows=np.flatnonzero(bank['case_index']<24)
    probe_rows=np.flatnonzero(bank['case_index']>=24)
    prior=_prior(bank,fit_rows)
    assert np.isclose(prior['raw_fine_mean'],.3)
    exact=bank['target'].astype(np.float32)
    evaluated=evaluate(bank,exact,probe_rows,plan,prior)
    assert semantic_gate(evaluated,GATE,True)['passed']
    presence_only=exact.copy()
    presence_only[bank['foreground']]=np.asarray([0,.3,.7,0,0])
    no_scale=evaluate(bank,presence_only,probe_rows,plan,prior)
    assert not semantic_gate(no_scale,GATE,True)['passed']
    result['sparse_metric_pipeline_exact_scale_vs_presence_only_fixture'] = 'passed'
    return result


def torch_checks():
    import torch
    from torch.nn import functional as F
    rng_before=torch.get_rng_state().clone()
    head=helper.make_readout_head()
    assert torch.equal(rng_before,torch.get_rng_state())
    assert torch.count_nonzero(head.weight)==0 and torch.count_nonzero(head.bias)==0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(717)
        head.weight.data.normal_()
        head.bias.data.normal_()
        feature=torch.randn(1,64,32,32,32)
    aligned=helper.align_classifier_features(feature)
    coord=torch.tensor([[2,3,4],[16,15,9],[29,29,29]])
    rows=aligned[0,:,coord[:,0],coord[:,1],coord[:,2]].T.contiguous()
    dense=head(aligned)[0,:,coord[:,0],coord[:,1],coord[:,2]].T
    sparse=F.linear(rows,head.weight[:,:,0,0,0],head.bias)
    assert torch.allclose(dense,sparse,rtol=1e-5,atol=1e-5)
    line=torch.arange(32,dtype=torch.float32)
    z,y,x=torch.meshgrid(line,line,line)
    ramp=(100*z+10*y+x)[None,None].expand(1,64,-1,-1,-1).contiguous()
    shifted=helper.align_classifier_features(ramp)
    expected=100*min(16.375,31)+10*min(15.375,31)+min(9.375,31)
    assert abs(float(shifted[0,0,16,15,9])-expected)<.001
    valid=torch.ones(1,1,128,128,128)
    assert np.array_equal(helper.ct_support_tensor(valid)[0,0].numpy(),helper.ct_window_support(valid[0,0].numpy()))
    valid[:,:,:10]=0
    assert np.array_equal(helper.ct_support_tensor(valid)[0,0].numpy(),helper.ct_window_support(valid[0,0].numpy()))
    actual=helper.normal_local_score(head,feature,valid)
    support=helper.ct_support_tensor(valid)
    assert torch.equal(actual[~support],torch.full_like(actual[~support],.2))
    # Convex solver integration on synthetic vectors, never patient features.
    from frozen_readout_fit import fit_convex
    gen=np.random.default_rng(831)
    synthetic=gen.normal(size=(80,64)).astype(np.float32)
    logits=np.stack([synthetic[:,0],-synthetic[:,0],synthetic[:,1],-synthetic[:,1],np.zeros(80)],1)
    logits=np.exp(logits-logits.max(1,keepdims=True))
    target=(logits/logits.sum(1,keepdims=True)).astype(np.float32)
    bank=dict(feature=synthetic,target=target,objective_weight=np.full(80,1/80))
    recipe=dict(lr=1.,lambda_weight_l2=.0001,max_iter=500,requested_max_eval=650,
                tolerance_grad=1e-6,tolerance_change=1e-9,history_size=100,line_search_fn='strong_wolfe')
    w,b,solver=fit_convex(torch,bank,np.arange(80),recipe)
    assert w.dtype==torch.float32 and b.dtype==torch.float32
    assert np.isfinite(solver['objective']) and solver['objective']<np.log(5.)
    assert solver['actual_optimizer_func_evals']>0
    return dict(torch_version=torch.__version__,
        head_zero_init_and_RNG_preservation='passed',
        FP32_Conv1x1_and_sparse_linear_equivalence='passed',
        explicit_nominal_grid_shift='passed',CT_support_numpy_torch_parity='passed',
        normal_inference_CT_only_fallback='passed',
        synthetic_only_convex_solver_integration=solver)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root',required=True)
    parser.add_argument('--torch',action='store_true')
    parser.add_argument('--output',required=True)
    parser.add_argument('--write-gate-config')
    args=parser.parse_args()
    result=dict(schema='frozen-readout-synthetic-QA-v1',patient_data_opened=False,
                patient_head_fitting_performed=False,
                synthetic_solver_fitting_performed=bool(args.torch),
                numpy=numpy_checks(args.code_root),
                torch=torch_checks() if args.torch else dict(status='not_run_torch_not_required_for_numpy_QA'),
                source_sha256=helper._source_map(args.code_root))
    helper.write_json(args.output,result)
    if args.write_gate_config:
        helper.write_json(args.write_gate_config,GATE)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
