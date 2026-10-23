"""Synthetic ordered-radius candidate parity/gradient QA; default is an inert plan.

CPU64 checks exact preclip common gradients. CUDA128/B2 also reports the
baseline's own backward variation; it never claims CUDA training bitwise parity.
Optional one-update-per-arm QA uses the frozen optimizer/AMP retry helper.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
ORIGINAL = '7406a3ba7e5bac1a7d8ee24898cb954ff8576f37fcf5fba0e26ef9a2cb5c43e6'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def execute(args):
    root=Path(args.source_root).resolve()
    if Path(__file__).resolve() != root/'next_stage/ordinal_stage_checks.py':
        raise ValueError('Execute the QA helper from the bound ordinal scientific snapshot')
    for path in (root/'code',root/'next_stage',HERE):sys.path.insert(0,str(path))
    import torch
    from stage_model import StageModel
    from ordinal_stage_model import OrdinalStageModel
    from ordinal_stage_train import stage_objective,runtime_identity,source_inventory,OPERATOR_VARIANT
    source=source_inventory(args.source_manifest,args.source_manifest_sha256)
    def set_seed(value):
        # This structural QA has no sampler/NumPy draws. Avoid importing the
        # trainer's optional medical-image stack until an update is requested.
        torch.manual_seed(value)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(value)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True,warn_only=True)
    require=lambda x,m: None if x else (_ for _ in ()).throw(AssertionError(m))
    torch.set_num_threads(1 if args.device=='cpu' else 4)
    set_seed(20261024);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','CUBLAS profile')
    if args.device=='cuda':
        require(torch.cuda.is_available() and args.patch==128 and args.batch_size==2,'Exact CUDA128/B2 required')
        require(args.warmstart and sha(args.warmstart)==ORIGINAL,'Original constant warmstart required on CUDA')
    set_seed(20261024);baseline=StageModel('constant',alpha=1.)
    rng_after_base=torch.get_rng_state().clone()
    set_seed(20261024);candidate=OrdinalStageModel('constant',alpha=1.)
    require(torch.equal(rng_after_base,torch.get_rng_state()),'new parameters changed initialization RNG')
    common=baseline.state_dict()
    require(set(candidate.state_dict())-set(common)=={'interaction_weight','interaction_bias'},'only declared interaction keys')
    require(candidate.interaction_weight.numel()+candidate.interaction_bias.numel()==256,'256 parameters at128-channel GAAM')
    require(all(torch.equal(candidate.state_dict()[k],v) for k,v in common.items()),'common random tensors differ')
    if args.warmstart:
        from common import load_trusted_checkpoint
        state=load_trusted_checkpoint(args.warmstart,'cpu');state=state['model'] if 'model' in state else state
        report=baseline.load_state_dict(state,strict=False)
        require(not report.unexpected_keys and set(report.missing_keys).issubset({'gdf_head.weight','gdf_head.bias'}),'old warmstart keys')
        candidate.load_common_state(baseline.state_dict())
    # Both algebraic ordinal forms agree for fixed distributions, including
    # empty/coarse one-hot and the uniform-five-class constant definition.
    probability=torch.tensor([[1.,0.,0.,0.,0.],[0.,1.,0.,0.,0.],[0.,0.,1.,0.,0.],
                              [0.,0.,0.,1.,0.],[0.,0.,0.,0.,1.],[.2,.2,.2,.2,.2],
                              [.1,.2,.3,.15,.25]],dtype=torch.float32)
    score=OrdinalStageModel.ordinal_condition(probability)
    cdf=(probability[:,1:2]+probability[:,1:3].sum(1,keepdim=True)+
         probability[:,1:4].sum(1,keepdim=True))/3.
    expected_score=torch.tensor([[0.],[1.],[2./3.],[1./3.],[0.],[.4],[.45]])
    require(torch.allclose(score,cdf,atol=1e-7,rtol=0.),'average ordered CDF identity')
    require(torch.allclose(score,expected_score,atol=1e-7,rtol=0.),'onehot/uniform/mixture ordered score')
    require(not score.requires_grad,'fixture needs no GT conditioning')
    initial_candidate_state={k:v.detach().cpu().clone() for k,v in candidate.state_dict().items()}
    device=torch.device(args.device);baseline.to(device).eval();candidate.to(device).eval()
    p,b=args.patch,args.batch_size
    image=torch.linspace(-1.,1.,b*p**3,device=device).view(b,1,p,p,p)
    axis=torch.arange(p,device=device,dtype=torch.float32)-(p-1)/2.
    ball=(axis[:,None,None]**2+axis[None,:,None]**2+axis[None,None,:]**2)<=(p/5.)**2
    mask=ball[None,None].expand(b,1,p,p,p).float().contiguous()
    scale=torch.tensor([[0.,.7,.1,.1,.1],[0.,.1,.2,.3,.4]],device=device)[:b]
    structure=torch.tensor([[0.,.8,.2],[0.,.3,.7]],device=device)[:b]
    arrays=dict(mask=mask,scale=scale,structure=structure,attribute_valid=torch.ones(b,device=device))
    rng_cpu=torch.get_rng_state().clone()
    rng_cuda=torch.cuda.get_rng_state_all() if args.device=='cuda' else None
    amp=args.device=='cuda'
    # One cheap train-mode forward per model, with identical initial BN buffers
    # and CPU/CUDA RNG restored. No extra backward or optimizer update here.
    initial_buffers={k:v.detach().clone() for k,v in baseline.named_buffers()}
    train_reference=None;train_buffers=None;train_cpu=None;train_cuda=None
    for label,model in (('baseline',baseline),('constant',candidate),('scale_only',candidate)):
        if label!='baseline':model.conditioning=label
        with torch.no_grad():
            for key,value in model.named_buffers():value.copy_(initial_buffers[key])
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:torch.cuda.set_rng_state_all(rng_cuda)
        model.train()
        with torch.no_grad(),torch.cuda.amp.autocast(enabled=amp):
            output=model(image)
        observed={k:output[k].detach().cpu() for k in ('logits','scale_logits','structure_logits','gdf_prediction')}
        buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
        cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all() if amp else None
        if train_reference is None:
            train_reference,train_buffers,train_cpu,train_cuda=observed,buffers,cpu,cuda
        else:
            require(all(torch.equal(observed[k],train_reference[k]) for k in observed),'train-mode zero-init forward exact '+label)
            require(all(torch.equal(buffers[k],train_buffers[k]) for k in buffers),'train-mode BN buffers exact '+label)
            require(torch.equal(cpu,train_cpu),'train-mode CPU RNG exact '+label)
            if amp:require(all(torch.equal(x,y) for x,y in zip(cuda,train_cuda)),'train-mode CUDA RNG exact '+label)
    del output,observed,buffers,train_reference,train_buffers
    for model in (baseline,candidate):
        with torch.no_grad():
            for key,value in model.named_buffers():value.copy_(initial_buffers[key])
        model.eval()
    candidate.conditioning='constant'
    torch.set_rng_state(rng_cpu)
    if rng_cuda is not None:torch.cuda.set_rng_state_all(rng_cuda)
    def run(model):
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:torch.cuda.set_rng_state_all(rng_cuda)
        model.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            output=model(image)
            loss,_=stage_objective(output,arrays,.1,.1,0.)
        loss.backward()
        gradients={k:None if v.grad is None else v.grad.detach().cpu().clone() for k,v in model.named_parameters() if k in common}
        tensors={k:output[k].detach().cpu() for k in ('logits','scale_logits','structure_logits','gdf_prediction')}
        return tensors,gradients,torch.get_rng_state().clone(),torch.cuda.get_rng_state_all() if amp else None
    ref,grads,after_cpu,after_cuda=run(baseline)
    results=[]
    def compare_grad(other):
        exact=True;maximum=0.
        for key,value in grads.items():
            right=other[key];require((value is None)==(right is None),'gradient presence '+key)
            if value is not None:
                require(bool(torch.isfinite(right).all()),'finite common gradient '+key)
                exact=exact and torch.equal(value,right)
                maximum=max(maximum,float((value-right).abs().max()))
        return exact,maximum
    repeat_max=0.
    if amp:
        again,repeated,_,_=run(baseline)
        require(all(torch.equal(ref[k],again[k]) for k in ref),'baseline repeated forward exact')
        _,repeat_max=compare_grad(repeated)
    for conditioning in ('constant','scale_only'):
        candidate.conditioning=conditioning
        observed,actual,cpu,cuda=run(candidate)
        require(all(torch.equal(ref[k],observed[k]) for k in ref),'zero-init forward exact '+conditioning)
        require(torch.equal(cpu,after_cpu),'forward RNG exact')
        if amp:require(all(torch.equal(x,y) for x,y in zip(cuda,after_cuda)),'CUDA RNG exact')
        exact,error=compare_grad(actual)
        if not amp:require(exact,'CPU preclip common gradients must be exact')
        else:require(error<=max(1e-6,2*repeat_max),'candidate gradient exceeds own-repeat numerical background')
        for name in ('interaction_weight','interaction_bias'):
            grad=getattr(candidate,name).grad
            require(grad is not None and bool(torch.isfinite(grad).all()) and bool(grad.count_nonzero()),'nonzero finite '+conditioning+'/'+name)
        trace=candidate.last_interaction_trace
        pscale=candidate.last_condition_trace['scale_head_probability'].float().detach()
        expected_ordinal=pscale[:,1:2]+(2.*pscale[:,2:3]+pscale[:,3:4])/3.
        require(torch.equal(trace['raw_fine'],pscale[:,1:2]),'fine diagnostic remains literal p1')
        require(torch.equal(trace['raw_ordinal'],expected_ordinal),'actual ordinal projection matches current image head')
        expected=torch.full_like(expected_ordinal,.4) if conditioning=='constant' else expected_ordinal
        require(torch.equal(trace['consumed_c'],expected) and not trace['consumed_c'].requires_grad,'actual CT condition consumed')
        require(candidate.last_condition_trace['alpha']==1. and candidate._stage_request is None,'parent request/trace restored')
        results.append(dict(conditioning=conditioning,forward_exact=True,preclip_common_gradient_exact=exact,
                            preclip_common_gradient_max_abs_error=error,weight_bias_gradients_nonzero=True))
    # alpha0 remains original G0 even after signed interaction parameters change.
    with torch.no_grad():candidate.interaction_weight.fill_(.15);candidate.interaction_bias.fill_(-.08)
    with torch.no_grad(),torch.cuda.amp.autocast(enabled=amp):zero=candidate(image,alpha=0.)
    require(torch.equal(zero['logits'].cpu(),ref['logits']),'alpha0 original constant fallback exact')
    for sign in (-1.,1.):
        with torch.no_grad():candidate.interaction_weight.zero_();candidate.interaction_bias.fill_(sign*.1)
        with torch.no_grad(),torch.cuda.amp.autocast(enabled=amp):candidate(image,trace_interaction=True)
        trace=candidate.last_interaction_trace;selected=trace['g0'].abs()>1e-5
        ratio=trace['adapted'][selected]/trace['g0'][selected]
        require(bool((ratio>1.).all()) if sign>0 else bool((ratio<1.).all()),'signed amplification/attenuation')
    with torch.no_grad():candidate.interaction_weight.fill_(.1);candidate.interaction_bias.zero_()
    override=torch.full((b,1),.8,device=device)
    try:candidate(image,consumed_c_override=override)
    except ValueError:pass
    else:raise AssertionError('normal override accepted')
    with torch.no_grad(),torch.cuda.amp.autocast(enabled=amp):candidate(image,consumed_c_override=override,diagnostic=True)
    require(torch.equal(candidate.last_interaction_trace['consumed_c'],override),'diagnostic override consumed')
    require(candidate.last_condition_trace['normal_scale_q0']==.2 and
            torch.equal(candidate.last_condition_trace['supplied_structure'],torch.full_like(candidate.last_condition_trace['supplied_structure'],1./3.)),'q0/SAAM unchanged')
    with torch.no_grad():candidate.interaction_weight.zero_();candidate.interaction_bias.zero_()
    candidate.last_interaction_trace=None
    update_result=None
    if args.optimizer_update:
        from train import optimizer_update_with_amp_retries
        update_result={}
        for conditioning in ('constant','scale_only'):
            # Independent starts: never feed C's updated state/BN/RNG into D.
            candidate.load_state_dict(initial_candidate_state,strict=True)
            candidate.conditioning=conditioning;candidate.train();candidate.set_alpha(1.)
            torch.set_rng_state(rng_cpu)
            if rng_cuda is not None:torch.cuda.set_rng_state_all(rng_cuda)
            require(not bool(candidate.interaction_weight.count_nonzero()) and
                    not bool(candidate.interaction_bias.count_nonzero()),'fresh zero interaction state')
            optimizer=torch.optim.AdamW(candidate.parameters(),lr=.0001,weight_decay=.0001)
            scaler=torch.cuda.amp.GradScaler(enabled=amp)
            original_step=optimizer.step;step_count=[0];step_grad={};group_evidence={}
            parameters=dict(candidate.named_parameters())
            watched_groups={'segmentation_head':'anatomical_output_fusion.3.',
                'scale_head':'classifier.generation_head.',
                'structure_head':'classifier.structure_head.',
                'shared_segmentation_feature':'anatomical_output_fusion.0.',
                'shared_attribute_encoder':'classifier.encoder.',
                'interaction_weight':'interaction_weight','interaction_bias':'interaction_bias'}
            def observed_step(*a,**kw):
                step_count[0]+=1;require(step_count[0]==1,'only one successful step per arm')
                chosen={};before={}
                for group,prefix in watched_groups.items():
                    members=[(name,value) for name,value in parameters.items() if name.startswith(prefix)]
                    require(members and all(value.grad is None or bool(torch.isfinite(value.grad).all())
                        for _,value in members),'actual group gradient finite '+group)
                    selected=[(name,value) for name,value in members if value.grad is not None and bool(value.grad.count_nonzero())]
                    require(selected,'actual nonzero step gradient '+group)
                    name,value=selected[0];chosen[group]=name;before[group]=value.detach().cpu().clone()
                    group_evidence[group]=dict(parameter=name,finite=True,
                        gradient_nonzero_elements=int(value.grad.count_nonzero()),
                        clipped_gradient_absmax=float(value.grad.detach().abs().max()))
                    if group in ('interaction_weight','interaction_bias'):
                        step_grad[group]=int(value.grad.count_nonzero())
                gdf_before={name:value.detach().cpu().clone() for name,value in parameters.items() if name.startswith('gdf_head.')}
                require(gdf_before and all(parameters[name].grad is None for name in gdf_before),'GDF0 parameter gradients must be absent')
                result=original_step(*a,**kw)
                for group,name in chosen.items():
                    after=parameters[name].detach().cpu()
                    require(bool(torch.isfinite(after).all()),'updated parameter finite '+group)
                    changes=int((after!=before[group]).count_nonzero())
                    require(changes>0,'actual parameter change '+group)
                    group_evidence[group].update(changed_elements=changes,
                        parameter_delta_absmax=float((after-before[group]).abs().max()))
                require(all(torch.equal(parameters[name].detach().cpu(),value) for name,value in gdf_before.items()),'Disabled GDF parameters changed')
                return result
            optimizer.step=observed_step
            def loss_forward():
                with torch.cuda.amp.autocast(enabled=amp):return stage_objective(candidate(image),arrays,.1,.1,0.)
            _,update=optimizer_update_with_amp_retries(candidate,optimizer,scaler,loss_forward,gradient_clip=12.,max_attempts=8)
            update.update(successful_steps=step_count[0],actual_step_gradient_nonzero_elements=step_grad,
                          representative_parameter_updates=group_evidence,
                          disabled_gdf_gradients_absent_and_parameters_unchanged=True,
                          initial_state_restored_including_BN=True,initial_RNG_restored=True,fresh_optimizer_and_scaler=True)
            update_result[conditioning]=update
    return dict(schema='GTA-AVT-ordinal-cumulative-interaction-QA-v1',status='passed',synthetic_only=True,study_data_read=False,TEST_files_read=False,formal_training=False,
        device=args.device,patch_size=p,batch_size=b,precision='CUDA_AMP' if amp else 'CPU_FP32',
        initialization_and_CPU_RNG_exact=True,new_parameter_count=256,GAAM_channels=128,zero_init_checks=results,
        train_mode_zero_init_forward_buffers_RNG_exact=True,train_mode_extra_backward_count=0,
        baseline_own_repeat_common_gradient_max_abs_error=repeat_max,
        cuda_gradient_background_rule='max_abs<=max(1e-6,2*baseline_own_repeat_max); not bitwise training claim',
        alpha0_exact=True,signed_effects_checked=True,diagnostic_override_only=True,optimizer_update=update_result,
        successful_optimizer_updates=2 if update_result is not None else 0,
        runtime=runtime_identity(device),operator_variant=OPERATOR_VARIANT, original_warmstart_sha256=ORIGINAL if args.warmstart else None,
        normal_condition_inputs='CT predictions only; no GT',
        ordinal_projection_checked=True, constant_uniform_five_probability_score=.4,
        source_manifest_sha256=args.source_manifest_sha256, source_sha256=source)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',default=str(HERE.parent));parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--patch',type=int,default=64);parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--source-manifest');parser.add_argument('--source-manifest-sha256')
    parser.add_argument('--warmstart');parser.add_argument('--optimizer-update',action='store_true')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--output')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(execute=False,models_created=False,study_data_read=False,device=args.device,
            patch_size=args.patch,batch_size=args.batch_size,formal_training_authorized=False)));return
    if args.batch_size!=2:raise ValueError('QA fixes batch2')
    started=time.time();result=execute(args);result['seconds']=time.time()-started
    if args.output:
        with Path(args.output).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
