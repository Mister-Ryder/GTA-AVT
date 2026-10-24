"""Portable random-initialization staged training with explicit checkpoint lineage.

No training starts at import. A step means a successful optimizer update, not
an attempted AMP iteration. The dataset keys every sample and augmentation by
(seed, epoch, index), so worker prefetch does not change ordered batches or the
resume cursor. Workers are recreated after set_epoch; no worker RNG is resumed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PreparedDataset, load_manifest
from .losses import attribute_loss, segmentation_loss
from .metrics import AttributeMetrics, binary_metrics, summarize_cases
from .models import AttributePredictor, GTAAVT

SCHEMA = 'GTA-AVT-v1-checkpoint-1'


def _json(path, value):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def state_digest(module):
    digest=hashlib.sha256()
    for name,tensor in sorted(module.state_dict().items()):
        value=tensor.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_checkpoint(path, device='cpu'):
    # Only load a checkpoint produced by this project from a trusted source.
    try:
        value=torch.load(str(path),map_location=device,weights_only=False)
    except TypeError:  # PyTorch1.12 deployed on the existing research runtimes.
        value=torch.load(str(path),map_location=device)
    if value.get('schema') != SCHEMA:
        raise ValueError('Unsupported checkpoint schema')
    return value


def _seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def _device(config):
    requested=config.get('device','cuda' if torch.cuda.is_available() else 'cpu')
    device=torch.device(requested)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; explicitly set device=cpu for CPU verification')
    return device


def _parameters(config):
    c=dict(config)
    required=('manifest','output_dir')
    if any(not c.get(k) for k in required):
        raise ValueError('manifest and output_dir are required')
    defaults=dict(seed=31,patch_size=128,batch_size=2,max_steps=10000,lr=1e-3,weight_decay=1e-4,
        num_workers=0,samples_per_epoch=256,checkpoint_every=500,validate_every=500,amp=True,
        availability_weight=1.,radius_weight=1.,dice_weight=1.,bce_weight=1.,lambda_gdf=0.,
        deep_supervision_weight=0.,alpha_ramp_steps=1000,arm='AR',stage='adapt',
        require_attribute_readiness=True)
    for key,value in defaults.items():
        c.setdefault(key,value)
    for key in ('patch_size','batch_size','max_steps','samples_per_epoch','checkpoint_every','validate_every'):
        if type(c[key]) is not int or c[key]<=0:
            raise ValueError(key+' must be a positive integer')
    if c['patch_size']<32 or c['patch_size']%16:
        raise ValueError('patch_size must be >=32 and divisible by16')
    if type(c['num_workers']) is not int or c['num_workers']<0:
        raise ValueError('num_workers must be a nonnegative integer')
    if c['lr']<=0 or c['weight_decay']<0 or c['alpha_ramp_steps']<0:
        raise ValueError('Invalid optimizer/ramp settings')
    for key in ('availability_weight','radius_weight','dice_weight','bce_weight','lambda_gdf','deep_supervision_weight'):
        if not math.isfinite(c[key]) or c[key]<0:
            raise ValueError(key+' must be finite and nonnegative')
    return c


def _output_directory(config):
    output=Path(config['output_dir']).resolve()
    if config.get('resume'):
        if not output.is_dir():
            raise FileNotFoundError('Resume output_dir does not exist')
    else:
        output.mkdir(parents=True,exist_ok=False)
        _json(output/'config.json',config)
    return output


def _to_device(batch,device):
    return {k:v.to(device,non_blocking=False) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def _datasets(config):
    common=dict(manifest=config['manifest'],patch_size=config['patch_size'],seed=config['seed'])
    train=PreparedDataset(split='TRAIN',samples_per_epoch=config['samples_per_epoch'],
                          augment=config.get('augment',True),**common)
    val_cases=[case for case in load_manifest(config['manifest'])['cases'] if case['split'].upper()=='VAL']
    if not val_cases:
        raise ValueError('An explicit VAL split is required; TEST is never used for selection')
    val=PreparedDataset(split='VAL',samples_per_epoch=config.get('val_samples',len(val_cases)*8),
                        augment=False,**common)
    return train,val


def _worker_init(worker_id):
    """Bound CPU oversubscription; item randomness still uses its indexed seed.

    Top-level function permits Windows spawn as well as Linux fork. Global RNGs
    are seeded defensively, but PreparedDataset does not consume them when
    sampling or augmenting. Parent model RNG has a separate loader generator.
    """
    torch.set_num_threads(1)
    worker_seed=torch.initial_seed()%2**32
    random.seed(worker_seed);np.random.seed(worker_seed)


def _loader(dataset,config):
    generator=torch.Generator().manual_seed(int(config['seed']))
    workers=config.get('num_workers',0)
    options=dict(batch_size=config['batch_size'],shuffle=False,num_workers=workers,
                 drop_last=False,generator=generator)
    if workers:
        options.update(worker_init_fn=_worker_init,prefetch_factor=2,persistent_workers=False)
    return DataLoader(dataset,**options)


@torch.no_grad()
def _validate(model,dataset,config,device,kind,alpha):
    model.eval();dataset.set_epoch(0)
    values=[];rows=[]
    attrs=AttributeMetrics(dataset.statistics['pi']) if kind=='pretrain' else None
    for batch in _loader(dataset,config):
        batch=_to_device(batch,device)
        output=model(batch['image']) if kind=='pretrain' else model(batch['image'],alpha=alpha)
        loss=_loss(output,batch,config,kind)
        values.append({k:float(v.detach().cpu()) for k,v in loss.items()})
        if attrs is not None:
            attrs.update(output,batch)
        else:
            pred=output['logits'].sigmoid()>=.5
            for i,case_id in enumerate(batch['case_id']):
                metric=binary_metrics(pred[i,0].cpu().numpy(),batch['label'][i,0].cpu().numpy(),
                                      batch['valid'][i,0].cpu().numpy(),centerline=False)
                rows.append(dict(case_id=str(case_id),**metric))
    losses={k:float(np.mean([row[k] for row in values])) for k in values[0]}
    report={'losses':losses,'scope':'deterministic sampled VAL patches, not native whole-case evaluation',
            'selection_score':_selection_score(losses,config,kind),
            'selection_definition':'attribute BCE+radius MSE' if kind=='pretrain' else
                'common segmentation Dice+BCE and configured common deep supervision; GDF excluded'}
    if attrs is not None:
        report['attributes']=attrs.report()
    else:
        # Average sampled patch metrics per case first, then case equal summary.
        pooled=[]
        for case in sorted({r['case_id'] for r in rows}):
            subset=[r for r in rows if r['case_id']==case]
            pooled.append(dict(case_id=case,**{k:float(np.mean([r[k] for r in subset if r[k] is not None]))
                if any(r[k] is not None for r in subset) else None for k in ('dice','precision','recall')}))
        report.update(per_case=pooled,summary=summarize_cases(pooled))
    return report


def _loss(output,batch,config,kind):
    if kind=='pretrain':
        return attribute_loss(output,batch,availability_weight=config['availability_weight'],
                               radius_weight=config['radius_weight'])
    return segmentation_loss(output,batch,dice_weight=config['dice_weight'],bce_weight=config['bce_weight'],
        gdf_weight=config['lambda_gdf'],deep_supervision_weight=config['deep_supervision_weight'])


def _selection_score(losses,config,kind):
    if kind=='pretrain':
        return float(losses['loss'])
    return float(config['dice_weight']*losses['dice_loss']+
        config['bce_weight']*losses['segmentation_bce']+
        config['deep_supervision_weight']*losses['auxiliary_bce'])


def _alpha(config,step,kind):
    if kind=='pretrain' or config['stage']=='warmup':
        return 0.
    return min(1.,step/max(1,config['alpha_ramp_steps'])) if config['alpha_ramp_steps'] else 1.


def _optimizer(model,config):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],
                             weight_decay=config['weight_decay'])


def _save(path,state):
    temporary=path.with_name(path.name+'.tmp')
    torch.save(state,temporary);os.replace(temporary,path)


def _run(config,kind):
    config=_parameters(config)
    if kind=='train' and (config['stage'] not in ('warmup','adapt') or config['arm'] not in ('C0','A_only','AR')):
        raise ValueError('stage must be warmup/adapt and arm C0/A_only/AR')
    if kind=='train' and config['stage']=='warmup' and config['arm']!='C0':
        raise ValueError('Common random-initialized warmup uses C0')
    if config['stage']=='warmup' and config.get('warmup_checkpoint'):
        raise ValueError('Warmup must start from random S initialization, no warmup_checkpoint')
    device=_device(config);_seed(config['seed'])
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(bool(config.get('deterministic',False)))
    if config.get('deterministic',False):
        if device.type=='cuda':
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    dataset,val=_datasets(config)
    stats=dict(dataset.statistics)
    if _digest(stats)!=_digest(dict(val.statistics)):
        raise ValueError('TRAIN and VAL normalizer metadata differ')
    predictor=AttributePredictor(stats)
    lineage={'predictor_init':'random','segmentation_init':None}
    expected_predictor_digest=None
    if kind=='pretrain':
        model=predictor
        model_config={'type':'AttributePredictor'}
    else:
        if not config.get('predictor_checkpoint'):
            raise ValueError('predictor_checkpoint from independent pretraining is required')
        pck=load_checkpoint(config['predictor_checkpoint'])
        if pck.get('kind')!='pretrain' or _digest(pck['statistics'])!=_digest(stats):
            raise ValueError('Predictor checkpoint stage or TRAIN statistics mismatch')
        readiness=pck.get('validation',{}).get('attributes',{}).get('readiness',{})
        if config['require_attribute_readiness'] and not readiness.get('passed',False):
            raise ValueError('Attribute prerequisite report has not passed; inspect diagnostics before segmentation. Explicit require_attribute_readiness=false permits a diagnostic run only.')
        predictor.load_state_dict(pck['model'],strict=True)
        expected_predictor_digest=state_digest(predictor)
        model_config={'type':'GTAAVT','arm':config['arm'],
                      'use_gdf':bool(config.get('use_gdf',config['lambda_gdf']>0))}
        model=GTAAVT(predictor,arm=model_config['arm'],use_gdf=model_config['use_gdf'])
        lineage.update(predictor_init={'checkpoint_sha256':file_digest(config['predictor_checkpoint']),
            'successful_updates':pck['step'],'predictor_state_sha256':expected_predictor_digest,
            'readiness_passed':readiness.get('passed',False)},segmentation_init='random')
        if config['stage']=='adapt':
            if not config.get('warmup_checkpoint'):
                raise ValueError('Matched adaptation requires one shared C0 warmup_checkpoint')
            warmup=load_checkpoint(config['warmup_checkpoint'])
            if (warmup['kind']!='train' or warmup['config']['stage']!='warmup'
                or warmup['model_config'].get('use_gdf')!=model_config['use_gdf']
                or _digest(warmup['statistics'])!=_digest(stats)
                or warmup['lineage']['predictor_init']['predictor_state_sha256']!=expected_predictor_digest
                or warmup['config']['seed']!=config['seed']
                or warmup['config']['patch_size']!=config['patch_size']):
                raise ValueError('Warmup must match architecture, predictor, seed, spacing/statistics and patch size')
            model.load_state_dict(warmup['model'],strict=True)
            lineage['segmentation_init']={'warmup_checkpoint_sha256':file_digest(config['warmup_checkpoint']),
                'warmup_successful_updates':warmup['step'],'optimizer_reset':True,
                'warmup_state_sha256':state_digest(model)}
    model.to(device)
    optimizer=_optimizer(model,config)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,
        lr_lambda=lambda step:max(0.,1-step/config['max_steps'])**.9)
    use_amp=bool(config['amp'] and device.type=='cuda')
    if hasattr(torch,'amp') and hasattr(torch.amp,'GradScaler'):
        scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
    else:
        scaler=torch.cuda.amp.GradScaler(enabled=use_amp)
    output=_output_directory(config)
    state={'step':0,'attempts':0,'epoch':0,'batch_index':0,'best_score':math.inf,
           'best_ready_score':math.inf}
    validation=None
    signature=_digest({k:v for k,v in config.items() if k not in ('resume','output_dir')})
    if config.get('resume'):
        checkpoint=load_checkpoint(config['resume'])
        if checkpoint['kind']!=kind or checkpoint['config_signature']!=signature:
            raise ValueError('Resume requires identical training configuration (except output_dir/resume)')
        model.load_state_dict(checkpoint['model'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler']);state.update(checkpoint['progress'])
        lineage=checkpoint['lineage'];validation=checkpoint.get('validation');_restore_rng(checkpoint['rng'])
    started=time.monotonic()

    def snapshot():
        if kind=='train' and state_digest(predictor)!=expected_predictor_digest:
            raise RuntimeError('Frozen predictor weights or BatchNorm buffers changed')
        return dict(schema=SCHEMA,kind=kind,config=config,config_signature=signature,
            statistics=stats,model_config=model_config,model=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=_rng(),progress=dict(state),
            step=state['step'],lineage=lineage,validation=validation,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))

    while state['step']<config['max_steps']:
        dataset.set_epoch(state['epoch'])
        loader=_loader(dataset,config)
        model.train()
        for batch_index,batch in enumerate(loader):
            if batch_index<state['batch_index']:
                continue
            batch=_to_device(batch,device)
            optimizer.zero_grad(set_to_none=True)
            alpha=_alpha(config,state['step']+1,kind)
            with torch.autocast(device_type=device.type,enabled=use_amp):
                result=model(batch['image']) if kind=='pretrain' else model(batch['image'],alpha=alpha)
                losses=_loss(result,batch,config,kind)
            scaler.scale(losses['loss']).backward()
            scaler.unscale_(optimizer)
            if config.get('max_grad_norm',0):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],config['max_grad_norm'])
            previous_scale=scaler.get_scale()
            if not use_amp and any(p.grad is not None and not torch.isfinite(p.grad).all()
                                   for p in model.parameters() if p.requires_grad):
                raise FloatingPointError('Nonfinite gradient; optimizer update was not applied')
            scaler.step(optimizer);scaler.update()
            successful=not use_amp or scaler.get_scale()>=previous_scale
            state['attempts']+=1;state['batch_index']=batch_index+1
            if not successful:
                if state['attempts']-state['step']>config.get('max_skipped_updates',100):
                    raise RuntimeError('Too many AMP-overflow skipped updates')
                continue
            state['step']+=1;scheduler.step()
            record={'successful_update':state['step'],'attempt':state['attempts'],'epoch':state['epoch'],
                'alpha':alpha,'lr':optimizer.param_groups[0]['lr'],
                'losses':{k:float(v.detach().cpu()) for k,v in losses.items()}}
            do_val=state['step']%config['validate_every']==0 or state['step']==config['max_steps']
            is_best=False;is_best_ready=False
            if do_val:
                validation=_validate(model,val,config,device,kind,alpha)
                record['validation']=validation
                score=validation['selection_score']
                if score<state['best_score']:
                    state['best_score']=score;is_best=True
                if (kind=='pretrain' and validation['attributes']['readiness']['passed']
                    and score<state['best_ready_score']):
                    state['best_ready_score']=score;is_best_ready=True
                model.train()
            with (output/'history.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            if state['step']%config['checkpoint_every']==0 or do_val:
                checkpoint=snapshot();_save(output/'last.pt',checkpoint)
                if is_best:
                    _save(output/'best.pt',checkpoint)
                if is_best_ready:
                    _save(output/'best_ready.pt',checkpoint)
            if state['step']>=config['max_steps']:
                break
        if state['batch_index']>=len(loader):
            state['epoch']+=1;state['batch_index']=0
    summary={'kind':kind,'stage':config.get('stage') if kind=='train' else 'independent attribute pretraining',
        'successful_updates':state['step'],'attempts':state['attempts'],'elapsed_seconds':time.monotonic()-started,
        'best_validation_loss':state['best_score'],'last_checkpoint':str(output/'last.pt'),
        'best_checkpoint':str(output/'best.pt'),'lineage':lineage,
        'best_ready_checkpoint':str(output/'best_ready.pt') if (output/'best_ready.pt').is_file() else None,
        'validation':validation,'weights_are_newly_trained':True,
        'convergence_claim':'A finite step budget is not evidence of convergence; inspect validation history and readiness.'}
    _json(output/'summary.json',summary)
    return summary


def file_digest(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def pretrain(config):
    return _run(config,'pretrain')


def train(config):
    return _run(config,'train')
