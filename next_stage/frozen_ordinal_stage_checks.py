"""Synthetic full-classifier-freeze checks; default inert.

CPU64/B2 or actual CUDA128/B2 AMP. No study data or training run is opened.
CUDA QA requires the original7406 warmstart and two isolated engineering
optimizer updates. Passing this helper alone never authorizes formal training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE=Path(__file__).resolve().parent
ORIGINAL='7406a3ba7e5bac1a7d8ee24898cb954ff8576f37fcf5fba0e26ef9a2cb5c43e6'


def require(value,message):
    if not value:raise AssertionError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def execute(args):
    root=Path(args.source_root).resolve()
    require(Path(__file__).resolve()==root/'next_stage/frozen_ordinal_stage_checks.py','Run from the new bound39-file snapshot')
    for path in (root/'code',root/'next_stage'):sys.path.insert(0,str(path))
    import torch
    from ordinal_stage_model import OrdinalStageModel
    from frozen_ordinal_stage_model import FrozenOrdinalStageModel
    from frozen_ordinal_stage_train import (source_inventory,stage_objective,runtime_identity,
        assert_classifier_frozen,OPERATOR_VARIANT,CLASSIFIER_POLICY,tensor_digest)
    source=source_inventory(args.source_manifest,args.source_manifest_sha256)
    require(args.batch_size==2 and args.patch==(128 if args.device=='cuda' else 64),'CPU64/B2 or CUDA128/B2 only')
    if args.device=='cuda':
        require(torch.cuda.is_available() and args.optimizer_update,'CUDA and two isolated actual updates are required')
        require(args.warmstart and sha(args.warmstart)==ORIGINAL,'Original7406 warmstart required for CUDA QA')
    def seed():
        torch.manual_seed(20261024)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(20261024)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True,warn_only=True)
    torch.set_num_threads(1 if args.device=='cpu' else 4);seed()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','CUBLAS profile differs')
    seed();parent=OrdinalStageModel('constant',alpha=1.);parent_rng=torch.get_rng_state().clone()
    seed();model=FrozenOrdinalStageModel('constant',alpha=1.)
    require(torch.equal(parent_rng,torch.get_rng_state()),'Freeze changed initialization RNG')
    require(set(model.state_dict())==set(parent.state_dict()),'Freeze added/removed state keys')
    require(all(torch.equal(v,parent.state_dict()[k]) for k,v in model.state_dict().items()),'Initial tensors changed')
    require(model.interaction_weight.numel()+model.interaction_bias.numel()==256,'Interaction parameter count changed')
    inherited=('forward','soft_generation','ordinal_condition','interaction_diagnostics','load_common_state')
    require(all(getattr(FrozenOrdinalStageModel,n) is getattr(OrdinalStageModel,n) for n in inherited),'A consumer/forward method was overridden')
    model.load_state_dict(parent.state_dict(),strict=True)
    missing=dict(parent.state_dict());missing.pop(next(k for k in missing if k.startswith('classifier.')))
    try:model.load_state_dict(missing,strict=True)
    except RuntimeError:pass
    else:raise AssertionError('Strict loading accepted missing classifier state')
    common={k:v for k,v in parent.state_dict().items() if k not in ('interaction_weight','interaction_bias')}
    report=model.load_common_state(common)
    require(set(report.missing_keys)=={'interaction_weight','interaction_bias'} and not report.unexpected_keys,'Inherited common loader changed')
    del missing,common,parent
    if args.warmstart:
        require(sha(args.warmstart)==ORIGINAL,'Warmstart identity differs')
        from common import load_trusted_checkpoint
        bundle=load_trusted_checkpoint(args.warmstart,'cpu');state=bundle['model'] if 'model' in bundle else bundle
        model.load_common_state(state);del bundle,state
    require(not bool(model.interaction_weight.count_nonzero()) and not bool(model.interaction_bias.count_nonzero()),'Interaction must start at zero')
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    classifier_initial={k:v.detach().cpu().clone() for k,v in model.classifier.state_dict().items()}
    classifier_initial_digest=tensor_digest(classifier_initial)
    for mode in (True,True,False,True,False,False,True):
        require(model.train(mode) is model and model.training is mode,'train(mode) return/parent mode changed')
        assert_classifier_frozen(model)
        require(model.base_model.training is mode and model.generation_guidance.training is mode and
            model.structure_guidance.training is mode,'A nonclassifier training mode changed')
    require(all(not p.requires_grad for p in model.classifier.parameters()),'Some classifier parameters remain trainable')
    device=torch.device(args.device);model.to(device);amp=args.device=='cuda';p=args.patch;b=args.batch_size
    image=torch.linspace(-1.,1.,b*p**3,device=device).view(b,1,p,p,p)
    axis=torch.arange(p,device=device,dtype=torch.float32)-(p-1)/2.
    ball=(axis[:,None,None]**2+axis[None,:,None]**2+axis[None,None,:]**2)<=(p/5.)**2
    arrays=dict(mask=ball[None,None].expand(b,1,p,p,p).float().contiguous(),
        scale=torch.tensor([[0.,.7,.1,.1,.1],[0.,.1,.2,.3,.4]],device=device),
        structure=torch.tensor([[0.,.8,.2],[0.,.3,.7]],device=device),attribute_valid=torch.ones(b,device=device))
    def classifier_state_equal():
        return all(torch.equal(value.detach().cpu(),classifier_initial[key]) for key,value in model.classifier.state_dict().items())
    def cpu_outputs(output):
        return {key:output[key].detach().cpu() for key in ('logits','scale_logits','structure_logits','gdf_prediction')}
    def classifier_outputs(output):
        return [output[key].detach().cpu() for key in ('generation','structure','location','guidance_map')]+[
            value.detach().cpu() for value in output['encoder_features']]
    model.train();rng_before=torch.get_rng_state().clone();cuda_before=torch.cuda.get_rng_state_all() if amp else None
    with torch.no_grad(),torch.cuda.amp.autocast(enabled=amp):
        first=classifier_outputs(model.classifier(image));second=classifier_outputs(model.classifier(image))
    require(all(torch.equal(a,c) for a,c in zip(first,second)),'Repeated complete classifier outputs differ')
    require(torch.equal(rng_before,torch.get_rng_state()),'Frozen classifier advanced CPU dropout RNG')
    if amp:require(all(torch.equal(a,c) for a,c in zip(cuda_before,torch.cuda.get_rng_state_all())),'Frozen classifier advanced CUDA dropout RNG')
    require(classifier_state_equal(),'Classifier BN/parameter state changed on repeated forward')
    del first,second
    rng_cpu=torch.get_rng_state().clone();rng_cuda=torch.cuda.get_rng_state_all() if amp else None
    def restore(training=True):
        model.load_state_dict(initial,strict=True);model.train(training);model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng_cpu)
        if amp:torch.cuda.set_rng_state_all(rng_cuda)
    reference=None;buffer_reference=None
    for conditioning in ('constant','scale_only'):
        restore();model.conditioning=conditioning
        with torch.no_grad(),torch.cuda.amp.autocast(enabled=amp):output=model(image,alpha=1.)
        observed=cpu_outputs(output);buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
        if reference is None:reference,buffer_reference=observed,buffers
        else:
            require(all(torch.equal(v,reference[k]) for k,v in observed.items()),'Train zero-init C/D outputs differ')
            require(all(torch.equal(v,buffer_reference[k]) for k,v in buffers.items()),'Train zero-init C/D BN state differs')
        require(classifier_state_equal(),'Zero-init full segmentation path changed frozen classifier state')
    restore(False);model.conditioning='scale_only'
    with torch.no_grad(),torch.cuda.amp.autocast(enabled=amp):g0=model(image,alpha=1.)['logits'].detach().clone()
    with torch.no_grad():model.interaction_weight.fill_(.15);model.interaction_bias.fill_(-.08)
    for conditioning in ('constant','scale_only'):
        model.conditioning=conditioning
        with torch.no_grad(),torch.cuda.amp.autocast(enabled=amp):zero=model(image,alpha=0.)['logits']
        require(torch.equal(zero,g0),'Alpha0 fallback differs with nonzero interaction')
    with torch.no_grad():model.interaction_weight.fill_(.1);model.interaction_bias.zero_()
    response=[];g0_trace=[]
    for value in (0.,1.):
        override=torch.full((b,1),value,device=device)
        with torch.no_grad(),torch.cuda.amp.autocast(enabled=amp):
            response.append(model(image,alpha=1.,consumed_c_override=override,diagnostic=True,trace_interaction=True)['logits'].detach().clone())
        g0_trace.append(model.last_interaction_trace['g0'].detach().clone())
    require(torch.equal(*g0_trace),'Diagnostic c changed G0')
    response_max=float((response[0]-response[1]).abs().max());require(response_max>0,'Nonzero slope diagnostic produced no logit response')
    require(classifier_state_equal(),'Capability diagnostic changed frozen classifier state')
    del output,reference,buffer_reference,observed,buffers,g0,response,g0_trace
    updates={}
    if args.optimizer_update:
        from train import optimizer_update_with_amp_retries
        for role,conditioning in (('C_fixed','constant'),('D_fixed','scale_only')):
            restore();model.conditioning=conditioning;model.set_alpha(1.)
            parameters=dict(model.named_parameters());trainable=[v for v in parameters.values() if v.requires_grad]
            classifier_ids={id(v) for v in model.classifier.parameters()}
            require(not classifier_ids.intersection(id(v) for v in trainable),'Optimizer includes frozen classifier')
            optimizer=torch.optim.AdamW(trainable,lr=.0001,weight_decay=.0001);scaler=torch.cuda.amp.GradScaler(enabled=amp)
            original_step=optimizer.step;step_count=[0];evidence={}
            prefixes={'backbone':'base_model.encoders.','GAAM_fusion':'generation_guidance.fusion.',
                'SAAM_fusion':'structure_guidance.fusion.','segmentation_fusion':'anatomical_output_fusion.0.',
                'interaction_weight':'interaction_weight','interaction_bias':'interaction_bias'}
            def actual_step(*args,**kwargs):
                step_count[0]+=1;require(step_count[0]==1,'Only one successful update per arm')
                assert_classifier_frozen(model);chosen={}
                for group,prefix in prefixes.items():
                    members=[(name,value) for name,value in parameters.items() if name.startswith(prefix)]
                    require(members and all(v.grad is None or bool(torch.isfinite(v.grad).all()) for _,v in members),'Nonfinite/missing group '+group)
                    selected=[(name,value) for name,value in members if value.grad is not None and bool(value.grad.count_nonzero())]
                    require(selected,'No nonzero trainable consumer/backbone gradient '+group)
                    name,value=selected[0];chosen[group]=(name,value.detach().cpu().clone())
                    evidence[group]=dict(parameter=name,gradient_nonzero_elements=int(value.grad.count_nonzero()))
                result=original_step(*args,**kwargs)
                for group,(name,before) in chosen.items():
                    changes=int((parameters[name].detach().cpu()!=before).count_nonzero());require(changes>0,'No actual update '+group)
                    evidence[group]['parameter_changed_elements']=changes
                require(classifier_state_equal(),'Optimizer/AMP changed frozen classifier parameter/BN state')
                return result
            optimizer.step=actual_step
            def forward_loss():
                with torch.cuda.amp.autocast(enabled=amp):return stage_objective(model(image),arrays,.1,.1,0.)
            _,update=optimizer_update_with_amp_retries(model,optimizer,scaler,forward_loss,gradient_clip=12.,max_attempts=8)
            require(step_count[0]==1 and classifier_state_equal(),'One-update freeze proof incomplete')
            assert_classifier_frozen(model)
            updates[role]=dict(update,successful_steps=step_count[0],actual_parameter_updates=evidence,
                classifier_full_state_exact=True,classifier_gradients_all_None=True,optimizer_excludes_classifier=True)
    require(classifier_state_equal(),'Final complete classifier state differs')
    return dict(schema='GTA-AVT-frozen-ordinal-classifier-QA-v1',status='passed',synthetic_only=True,
        study_data_read=False,TEST_files_read=False,formal_training=False,device=args.device,patch_size=p,batch_size=b,
        precision='CUDA_AMP' if amp else 'CPU_FP32',runtime=runtime_identity(device),operator_variant=OPERATOR_VARIANT,
        classifier_policy=CLASSIFIER_POLICY,classifier_frozen_eval=True,classifier_state_unchanged=True,
        classifier_gradients_absent=bool(updates),repeat_classifier_outputs_exact=True,optimizer_excludes_classifier=bool(updates),
        classifier_initial_digest=classifier_initial_digest,classifier_final_digest=tensor_digest(model.classifier.state_dict()),
        classifier_state_entries=len(classifier_initial),classifier_parameter_count=sum(p.numel() for p in model.classifier.parameters()),
        total_parameter_count=sum(p.numel() for p in model.parameters()),trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        strict_full_state_load_checked=True,common_state_loader_inherited=True,constructor_RNG_and_tensors_unchanged=True,
        repeated_parent_train_eval_checked=True,complete_classifier_outputs_including_features_guidance_exact=True,
        train_zero_init_C_D_output_and_buffer_parity=True,alpha0_nonzero_parameters_exact=True,
        nonzero_slope_diagnostic_logit_max_abs_response=response_max,capability_response_is_not_anatomical_efficacy=True,
        inherited_forward_consumer_methods=list(inherited),new_parameter_count=256,GAAM_channels=128,
        optimizer_update=updates,successful_optimizer_updates=sum(v['successful_steps'] for v in updates.values()),
        source_manifest_sha256=args.source_manifest_sha256,source_sha256=source,source_files=39,
        original_warmstart_sha256=ORIGINAL if args.warmstart else None,normal_condition_inputs='CT predictions only; no GT')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--output')
    p.add_argument('--source-root',default=str(HERE.parent));p.add_argument('--source-manifest');p.add_argument('--source-manifest-sha256')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--patch',type=int,default=64)
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--warmstart');p.add_argument('--optimizer-update',action='store_true');a=p.parse_args()
    if not a.execute:
        print(json.dumps(dict(execute=False,models_created=False,study_data_read=False,formal_training_authorized=False,
            required_source_files=39,device=a.device,patch_size=a.patch,batch_size=a.batch_size)));return
    started=time.time();result=execute(a);result['seconds']=time.time()-started
    if a.output:
        with Path(a.output).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
