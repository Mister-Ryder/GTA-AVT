"""Small source/algebra checks; optional --torch uses only tiny feature tensors.

No image, oracle bank, checkpoint, full128 forward or optimizer is read/run.
Actual128/B2 CUDA parity and two-update checks belong to deployment QA.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PARENT_SHA = '6f5e6fcf84600c737fc8844c0fdce6839f980d19d734512686f374cce1a0c294'


def require(value, message):
    if not value:
        raise AssertionError(message)


def methods(text):
    tree = ast.parse(text, feature_version=(3,8))
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef))
    return tree,cls,{n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}


def dump(node):
    return ast.dump(node,include_attributes=False)


def literal(tree,name):
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))


def torch_checks(here, old_path, new_path):
    import torch
    from torch.nn import functional as F
    sys.path.insert(0,str(here))
    def load(path,name):
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    old=load(old_path,'routing_parent_check');new=load(new_path,'routing_new_check')
    for arm in new.ARMS:
        torch.manual_seed(20261030);parent=old.CalibreBasisStageModel(arm)
        state={k:v.detach().clone() for k,v in parent.state_dict().items()}
        rng=torch.get_rng_state().clone();del parent
        torch.manual_seed(20261030);model=new.CalibreRoutingStageModel(arm)
        require(torch.equal(torch.get_rng_state(),rng),'New constructor changed RNG')
        for key,value in model.state_dict().items():
            if key in new.COEFFICIENT_NAMES:
                require(tuple(value.shape)==(3,64) and torch.count_nonzero(value)==0 and
                    torch.equal(value[0],state[key]),'Coefficient shape/zero/old axis mismatch')
            else:
                require(torch.equal(value,state[key]),'Original state changed: '+key)
        active={n for n,p in model.named_parameters() if p.requires_grad}
        groups=model.trainable_groups()
        require(active==set(new.EXISTING_PARAMETER_NAMES)|set(new.COEFFICIENT_NAMES) and len(active)==15,
            'Trainable names/tensor count differ')
        require(sum(p.numel() for p in groups['shared'])==114882 and
            sum(p.numel() for p in groups['gates'])==576,'Actual optimizer groups differ')
        require(not any(m.training for m in model.modules()),'Eval policy escaped')
        del model,state
    op=torch.nn.Conv3d(64,64,1)
    h=torch.linspace(-1.,2.,64*8).reshape(1,64,2,2,2).requires_grad_()
    probability=torch.full((1,3),1./3.)
    D1=new.CalibreRoutingStageModel.context_direction(h,op,probability)
    D2,D3=new.CalibreRoutingStageModel.within_pair_directions(h,op,probability)
    factor=(1-probability[:,0,None,None,None,None])*(1+.1*probability[:,2,None,None,None,None])
    require(torch.equal(D1,old.CalibreBasisStageModel.context_direction(h,op,probability)),
        'Original D1 calculation changed')
    require(torch.equal(D2,factor*F.conv3d(torch.cat((h[:,:16],-h[:,16:32]),1),op.weight[:,:32],bias=None)) and
        torch.equal(D3,factor*F.conv3d(torch.cat((h[:,32:48],-h[:,48:64]),1),op.weight[:,32:],bias=None)),
        'D2/D3 signs/weight slices/factor differ')
    g1=torch.tanh(torch.linspace(-1,1,64)).reshape(1,64,1,1,1)
    zeros=torch.zeros_like(g1)
    F0=torch.randn_like(D1)
    require(torch.equal(F0+.7*g1*D1,(F0+.7*g1*D1)+.7*(1-g1.abs())*(zeros*D2+zeros*D3)),
        'Tiny original-axis nesting failed')
    coefficients=torch.zeros((3,64),requires_grad=True)
    gains=torch.tanh(coefficients).reshape(3,1,64,1,1,1)
    result=F0+gains[0]*D1+(1-gains[0].abs())*(gains[1]*D2+gains[2]*D3)
    result.square().mean().backward()
    require(all(bool((coefficients.grad[i]!=0).any()) for i in range(3)) and
        h.grad is not None and op.weight.grad is not None and op.bias.grad is None,
        'Three axes or original h/W autograd failed; bias must be absent')
    stats=new.CalibreRoutingStageModel.route_statistics(g1,zeros,zeros,1.)
    require(0.<=stats['min']<=stats['max']<=2. and stats['sum_residual_maxabs']<=2**-20,
        'FP32 coefficient diagnostics differ')
    return ['actual_init_RNG_original_state_and_15_tensor_inventory',
            'tiny_D1_exact_D2_D3_bias_free_h_W_autograd_and_axis_nesting']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--torch',action='store_true');args=parser.parse_args()
    here=Path(__file__).parent;old_path=here/'calibre_basis_stage_model.py';new_path=here/'calibre_routing_stage_model.py'
    require(hashlib.sha256(old_path.read_bytes()).hexdigest()==PARENT_SHA,'Frozen6f parent changed')
    old_text=old_path.read_text(encoding='utf-8');new_text=new_path.read_text(encoding='utf-8')
    old_tree,_,old=methods(old_text);tree,cls,new=methods(new_text);checks=[]
    require(cls.name=='CalibreRoutingStageModel' and isinstance(cls.bases[0],ast.Name) and
        cls.bases[0].id=='StageModel','Wrong routing class/base')
    changed={n for n in old if dump(old[n])!=dump(new[n])}
    require(changed=={'__init__','_configure_trainable_parameters','soft_structure'} and
        set(new)-set(old)=={'within_pair_directions','route_statistics'},'Unrelated parent methods changed')
    require(all(dump(new[n])==dump(old[n]) for n in ('forward','context_direction','_oracle',
        'load_common_state','install_moments','train','coadapt_parameter_groups','trainable_groups')),
        'Original oracle/forward/D1/eval/loader/group semantics changed')
    checks.append('original_forward_oracle_D1_loader_eval_and_optimizer_group_AST_exact')
    require(literal(tree,'EXISTING_PARAMETER_NAMES')==literal(old_tree,'EXISTING_PARAMETER_NAMES') and
        literal(tree,'COEFFICIENT_NAMES')==literal(old_tree,'COEFFICIENT_NAMES') and
        114882+3*3*64==115458,'Original names/counts differ')
    require(new_text.count('nn.Parameter(torch.zeros(3, 64, dtype=torch.float32))')==3 and
        'result = F0 + float(alpha) * g1 * D1' in new_text and
        'result = result + float(alpha) * remaining * (g2 * D2 + g3 * D3)' in new_text,
        'Zero init or old-then-new evaluation order differs')
    checks.append('same_12_existing_tensors_576_zero_gates_and_original_axis_order')
    import numpy as np
    rng=np.random.default_rng(20261024)
    g=np.vstack((rng.uniform(-1.,1.,(3,1000)),))
    a=1-np.abs(g[0]);p=np.stack((-g[0]+a*g[1],-g[0]-a*g[1],g[0]+a*g[2],g[0]-a*g[2]))
    for alpha in (0.,.4,1.):
        q=1+alpha*p
        require(np.all(q>=0.) and np.all(q<=2.) and np.allclose(q.sum(0),4.,rtol=0,atol=1e-14),
            'Bound/sum formula failed')
    arbitrary=rng.normal(size=(4,1000));arbitrary-=arbitrary.mean(0)
    arbitrary*=.9/np.max(np.abs(arbitrary),axis=0)
    inverse1=(arbitrary[2]+arbitrary[3])/2
    inverse2=(arbitrary[0]-arbitrary[1])/(2*(1-np.abs(inverse1)))
    inverse3=(arbitrary[2]-arbitrary[3])/(2*(1-np.abs(inverse1)))
    require(np.max(np.abs(np.stack((inverse1,inverse2,inverse3))))<1.,'Interior inverse outside tanh range')
    recovered=np.stack((-inverse1+(1-np.abs(inverse1))*inverse2,
        -inverse1-(1-np.abs(inverse1))*inverse2,inverse1+(1-np.abs(inverse1))*inverse3,
        inverse1-(1-np.abs(inverse1))*inverse3))
    require(np.allclose(recovered,arbitrary,rtol=0,atol=1e-14),'Full zero-sum interior not recovered')
    checks.append('full_zero_sum_interior_inverse_bounds_and_sum4_without_projection')
    W=rng.normal(size=(64,64));h=rng.normal(size=(64,5));factor=(1-1/3)*(1+.1/3)
    signs=np.repeat([-1,-1,1,1],16)
    D=np.stack((factor*(W@(signs[:,None]*h)),
        factor*(W[:,:32]@(np.repeat([1,-1],16)[:,None]*h[:32])),
        factor*(W[:,32:]@(np.repeat([1,-1],16)[:,None]*h[32:]))))
    t=np.zeros(3);target=rng.normal(size=(64,5))
    def objective(t):
        gain=np.tanh(t);return np.sum(target*(gain[0]*D[0]+(1-abs(gain[0]))*(gain[1]*D[1]+gain[2]*D[2])))
    finite=[]
    for axis in range(3):
        plus=t.copy();minus=t.copy();plus[axis]=1e-6;minus[axis]=-1e-6
        finite.append((objective(plus)-objective(minus))/2e-6)
    require(np.allclose(finite,[np.sum(target*d) for d in D],rtol=1e-8,atol=1e-8) and
        all(v!=0 for v in finite),'Zero-start three-axis derivatives differ')
    checks.append('tiny_three_distinct_bias_free_directions_and_zero_start_derivatives')
    A=np.array([0.,1.,1.,0.]);R=np.array([0.,-1.,2.,0.])
    for arm in ('C0','A_only','AR'):
        zA=np.zeros_like(A) if arm=='C0' else A;zR=R if arm=='AR' else np.zeros_like(R)
        require((np.count_nonzero(zA)==0)==(arm=='C0') and
            (np.count_nonzero(zR)==0)==(arm!='AR'),'Control axes differ')
    checks.append('unchanged_C0_A_only_AR_hidden_axis_semantics')
    actual=torch_checks(here,old_path,new_path) if args.torch else []
    print(json.dumps(dict(status='PASS',checks=checks,Torch_checks=actual,
        Torch_checks_executed=args.torch,full128_forward_executed=False,
        source_sha256=hashlib.sha256(new_path.read_bytes()).hexdigest(),frozen6f_unchanged=True),indent=2))


if __name__=='__main__':
    main()
