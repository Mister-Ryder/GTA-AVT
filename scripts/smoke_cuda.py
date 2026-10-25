"""Synthetic actual-size engineering test. Does not read AVT or estimate efficacy."""
import argparse
import json
import time
import torch
from gta_avt.models import AttributePredictor, GTAAVT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--patch', type=int, default=128)
    parser.add_argument('--batch', type=int, default=2)
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required')
    torch.set_num_threads(4)
    torch.manual_seed(20261024)
    device=torch.device('cuda')
    predictor=AttributePredictor(dict(mu=1.,sigma=.5,pi_support=.2))
    # Synthetic noncollapsed conditions exercise all three routing paths.
    with torch.no_grad():
        predictor.local_head.weight.normal_(0,.01)
        predictor.local_head.bias[0]=2.
        predictor.local_head.bias[1]=.3
        predictor.local_head.bias[2]=-2.
    model=GTAAVT(predictor,arm='AR',use_gdf=True).to(device).set_stage('local_adapt').train()
    frozen={k:v.detach().clone() for k,v in model.predictor.state_dict().items()}
    x=torch.randn(args.batch,1,args.patch,args.patch,args.patch,device=device)
    target=(torch.rand_like(x)>.9).float()
    groups=model.parameter_groups()
    optimizer=torch.optim.AdamW([{'params':groups['shared'],'lr':1e-4},
        {'params':groups['gates'],'lr':1e-3},{'params':groups['gdf'],'lr':1e-4}],weight_decay=1e-5)
    scaler=torch.cuda.amp.GradScaler()
    before=model.output_head.weight.detach().clone()
    start=time.monotonic(); torch.cuda.reset_peak_memory_stats()
    successful=0; attempts=0; losses=[]
    while successful<2 and attempts<12:
        attempts+=1
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            output=model(x,alpha=.5)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(output['logits'].float(),target)
        loss=loss+.1*(output['gdf'].float()-target).square().mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite engineering loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],12.)
        old=scaler.get_scale(); scaler.step(optimizer); scaler.update()
        if scaler.get_scale()>=old:
            successful+=1; losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    assert successful==2
    assert all(torch.equal(value,model.predictor.state_dict()[key]) for key,value in frozen.items())
    assert all(p.grad is None for p in model.predictor.parameters())
    assert all(p.grad is None for p in model.guidance_adapter.parameters())
    assert all(not module.training for module in model.modules())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==115491
    assert model.gdf_head[0].weight.grad is not None
    assert model.routing.radius_weight.grad is not None
    assert not torch.equal(before,model.output_head.weight.detach())
    result=dict(status='passed',scope='synthetic engineering only',patch=args.patch,batch=args.batch,
                successful_updates=successful,attempts=attempts,losses=losses,
                predictor_frozen=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                elapsed_seconds=time.monotonic()-start,torch=torch.__version__,
                cuda=torch.version.cuda,device=torch.cuda.get_device_name())
    print(json.dumps(result))


if __name__=='__main__': main()
