"""Small source/algebra checks; optional --torch tests init/tiny autograd on CPU.

No full128-patch forward, data, checkpoint or optimizer is used. Actual128/B2
CUDA zero-parity and two-update checks remain the deployment QA's task.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PARENT_SHA = '7741821580cf6fded92b2d5033aa4649bdbab51a8f218378525c28533763c942'


def require(value,message):
    if not value:raise AssertionError(message)


def definitions(tree):
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    return cls,{n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}


def literal(tree,name):
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))


def torch_checks(here,new_path,old_path):
    import torch
    from torch.nn import functional as F
    sys.path.insert(0,str(here))
    def load(path,name):
        spec=importlib.util.spec_from_file_location(name,path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    old=load(old_path,'calibre_old_init_check');new=load(new_path,'calibre_basis_init_check')
    for arm in new.ARMS:
        torch.manual_seed(20261030)
        parent=old.CalibreCoadaptStageModel(arm)
        expected={k:v.detach().clone() for k,v in parent.state_dict().items()}
        expected_rng=torch.get_rng_state().clone()
        del parent
        torch.manual_seed(20261030)
        model=new.CalibreBasisStageModel(arm)
        require(torch.equal(torch.get_rng_state(),expected_rng),'Constructor advanced a different RNG sequence')
        require(set(model.state_dict())==set(expected) and all(torch.equal(v,expected[k]) for k,v in model.state_dict().items()),
            'Initial model/buffer tensors differ from774')
        active={n for n,p in model.named_parameters() if p.requires_grad}
        require(active==set(new.EXISTING_PARAMETER_NAMES)|set(new.COEFFICIENT_NAMES) and
            sum(p.numel() for p in model.parameters() if p.requires_grad)==115074,
            'Actual trainable inventory differs')
        require(not any(m.training for m in model.modules()),'BN/Dropout modules escaped eval')
        groups=model.trainable_groups()
        require(sum(p.numel() for p in groups['shared'])==114882 and
                sum(p.numel() for p in groups['gates'])==192,'Optimizer group counts differ')
        del model,expected
    op=torch.nn.Conv3d(64,64,1)
    with torch.no_grad():
        op.weight.copy_(torch.arange(4096,dtype=torch.float32).reshape(64,64,1,1,1)/4096.)
        op.bias.fill_(999.)
    h=(torch.arange(64*8,dtype=torch.float32).reshape(1,64,2,2,2)/100.-1.).requires_grad_()
    p=torch.full((1,3),1./3.)
    D=new.CalibreBasisStageModel.context_direction(h,op,p)
    signed=torch.cat((-h[:,:16],-h[:,16:32],h[:,32:48],h[:,48:64]),1)
    expected=(1-p[:,0,None,None,None,None])*(1+.1*p[:,2,None,None,None,None])*F.conv3d(signed,op.weight,bias=None)
    require(torch.equal(D,expected),'Tiny D formula/bias exclusion differs')
    D.square().mean().backward()
    require(h.grad is not None and op.weight.grad is not None and bool(torch.isfinite(h.grad).all()) and
        bool(torch.isfinite(op.weight.grad).all()) and bool((h.grad!=0).any()) and bool((op.weight.grad!=0).any()),
        'Tiny context must teach BOTH original h and original W')
    require(op.bias.grad is None,'D must not use morph_fusion bias')
    A=torch.tensor([0.,1.,1.,0.]).reshape(1,1,1,1,4)
    R=torch.tensor([0.,-1.,2.,0.]).reshape_as(A)
    for arm in new.ARMS:
        b=torch.zeros(64,requires_grad=True);v=torch.zeros(64,requires_grad=True);w=torch.zeros(64,requires_grad=True)
        zA=torch.zeros_like(A) if arm=='C0' else A
        zR=R if arm=='AR' else torch.zeros_like(R)
        gain=torch.tanh(b[None,:,None,None,None]+v[None,:,None,None,None]*zA+w[None,:,None,None,None]*zR)
        require(torch.count_nonzero(gain)==0,'Zero-init forward residual is not zero')
        gain.sum().backward()
        require(bool((b.grad!=0).all()) and ((torch.count_nonzero(v.grad)==0) if arm=='C0' else bool((v.grad!=0).all())) and
            ((torch.count_nonzero(w.grad)==0) if arm!='AR' else bool((w.grad!=0).all())),
            'Inactive C0/A-only slope gradients differ')
    return ['actual_all_arm_initial_tensor_RNG_identity_and_inventory',
            'tiny_context_h_and_W_autograd_bias_exclusion','tiny_exact_control_and_active_axis_gradients']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--torch',action='store_true');args=parser.parse_args()
    here=Path(__file__).parent;parent=here/'calibre_coadapt_stage_model.py';path=here/'calibre_basis_stage_model.py'
    require(hashlib.sha256(parent.read_bytes()).hexdigest()==PARENT_SHA,'Frozen774 changed')
    oldtext=parent.read_text(encoding='utf-8');text=path.read_text(encoding='utf-8')
    old=ast.parse(oldtext,feature_version=(3,8));tree=ast.parse(text,feature_version=(3,8))
    cls,methods=definitions(tree);_,original=definitions(old);checks=[]
    require(cls.name=='CalibreBasisStageModel' and ast.unparse(cls.bases[0])=='StageModel','Wrong new class/base')
    changed={name for name in methods if ast.dump(methods[name],include_attributes=False)!=ast.dump(original[name],include_attributes=False)}
    require(changed=={'_configure_trainable_parameters','context_direction','soft_structure'},'Other old model methods changed')
    checks.append('forward_conditions_oracle_state_loader_and_init_AST_exact774')
    # Remove the sole h detach edits before comparing both full method ASTs.
    restored=text.replace('h = h.float()','h = h.detach().float()').replace("captured['h'] = inputs[0]\n","captured['h'] = inputs[0].detach()\n")
    restored=restored.replace('Existing local/context branch contrast, with both branch h and W differentiable.',
        'Existing local/context branch contrast, with original W differentiable.')
    _,restore=definitions(ast.parse(restored))
    for name in ('context_direction','soft_structure'):
        require(ast.dump(restore[name],include_attributes=False)==ast.dump(original[name],include_attributes=False),
            'Changed operator math beyond h autograd scope: '+name)
    checks.append('only_two_h_detach_removals_in_consumer_math')
    names=literal(tree,'EXISTING_PARAMETER_NAMES');oldnames=literal(old,'EXISTING_PARAMETER_NAMES')
    require(names[:4]==oldnames and names[4:]==tuple('structure_guidance.morphology_convs.%d.%s'%(i,s)
        for i in range(4) for s in ('weight','bias')),'Eight existing branch names differ')
    require(4*(16*64*27+16)==110656 and 110656+4226==114882 and 114882+192==115074,'Parameter count arithmetic differs')
    config=ast.get_source_segment(text,methods['_configure_trainable_parameters'])
    require('parameter.requires_grad_(name in expected)' in config and 'calibre_arm' not in config,
        'All arms must share parameter inventory')
    checks.append('original_eight_branch_keys_and_shared114882_gate192_inventory')
    context=ast.get_source_segment(text,methods['context_direction'])
    capture=ast.get_source_segment(text,methods['soft_structure'])
    require('operator.weight.float(), bias=None' in context and 'h.detach' not in context and
            "captured['h'] = inputs[0]\n" in capture and 'hook.remove()' in capture,
            'Differentiable h/W or transient capture cleanup differs')
    require(ast.dump(methods['train'],include_attributes=False)==ast.dump(original['train'],include_attributes=False),
        'Eval policy changed')
    checks.append('h_and_W_retained_eval_policy_and_finally_hook_cleanup')
    # Tiny analytic linear-chain finite differences: h(theta) must teach theta,
    # in addition to the fusion W; this does not run a3D model.
    import numpy as np
    theta=np.arange(64.,dtype=np.float64)/100.+.1;x=np.arange(64.,dtype=np.float64)/80.+.2
    signs=np.repeat([-1.,-1.,1.,1.],16);W=np.arange(4096.,dtype=np.float64).reshape(64,64)/4096.
    h=theta*x;factor=(1-1/3)*(1+.1/3);D=factor*(W@(signs*h));eps=1e-6
    t=theta.copy();t[37]+=eps;finite=(factor*(W@(signs*t*x))-D)/eps
    require(np.allclose(finite,factor*W[:,37]*signs[37]*x[37],rtol=1e-7,atol=1e-8) and np.any(finite!=0),
        'Tiny branch-basis derivative differs')
    checks.append('tiny_linear_chain_branch_derivative_finite_difference')
    torch_names=torch_checks(here,path,parent) if args.torch else []
    print(json.dumps(dict(status='PASS',checks=checks,Torch_checks=torch_names,
        Torch_checks_executed=args.torch,full128_forward_executed=False,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),parent774_unchanged=True),indent=2))


if __name__=='__main__':main()
