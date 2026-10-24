"""Fixed v1 sequence: independent P, common S warmup, four matched arms."""
import copy
import hashlib
import json
from pathlib import Path
import time

from .engine import pretrain, train, _json, load_checkpoint
from .inference import infer, evaluate


def run_pipeline(config):
    """Run an explicitly declared plan. Failed prerequisites stop, never rescue.

    Each seed is a complete repeat. Within a seed every continuation has the
    same predictor, warmup, sample sequence and update budget. GDF changes only
    its auxiliary weight; its head is present in all matched arms.
    """
    plan=copy.deepcopy(config)
    resume=bool(plan.pop('resume_pipeline',False))
    signature=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    home=Path(plan['output_dir']).resolve()
    state_path=home/'pipeline.json'
    if resume:
        state=json.loads(state_path.read_text(encoding='utf8'))
        if state['plan_sha256']!=signature:
            raise ValueError('Pipeline resume requires the same fixed plan')
    else:
        home.mkdir(parents=True,exist_ok=False)
        state=dict(status='running',plan_sha256=signature,started_unix=time.time(),completed={},current=None)
        _json(home/'plan.json',plan)
    _json(state_path,state)
    common=dict(manifest=plan['manifest'],seed=plan['seed'],device=plan.get('device','cuda'),
        patch_size=128,batch_size=2,lr=.0003,weight_decay=.0001,samples_per_epoch=256,
        val_samples=64,num_workers=plan.get('num_workers',0),amp=True,max_grad_norm=12.,
        checkpoint_every=1000,validate_every=1000,deterministic=False,
        dice_weight=1.,bce_weight=1.,deep_supervision_weight=0.)
    common.update(plan.get('common',{}))

    def execute(name,kind,updates,**extra):
        if name in state['completed']:
            return state['completed'][name]
        state.update(status='running',current=name);_json(state_path,state)
        cfg=dict(common,output_dir=str(home/name),max_steps=updates,**extra)
        last=home/name/'last.pt'
        if resume and last.is_file():
            cfg['resume']=str(last)
        result=(pretrain if kind=='pretrain' else train)(cfg)
        if result['successful_updates']!=updates:
            raise RuntimeError('Fixed stage ended without its declared successful-update budget')
        state['completed'][name]=result;_json(state_path,state)
        return result
    try:
        p=execute('attribute','pretrain',int(plan.get('pretrain_steps',20000)))
        ready=p.get('best_ready_checkpoint')
        if not ready:
            raise RuntimeError('No attribute checkpoint passed the fixed prerequisite report; segmentation was not launched')
        warm=execute('warmup','train',int(plan.get('warmup_steps',10000)),stage='warmup',
            arm='C0',predictor_checkpoint=ready,use_gdf=True,lambda_gdf=0.,require_attribute_readiness=True)
        warmup=warm['last_checkpoint']
        definitions=(('C0','C0',0.),('A_only','A_only',0.),('AR','AR',0.),('AR_GDF','AR',float(plan.get('lambda_gdf',.1))))
        for name,arm,weight in definitions:
            execute(name,'train',int(plan.get('adapt_steps',5000)),stage='adapt',arm=arm,
                predictor_checkpoint=ready,warmup_checkpoint=warmup,use_gdf=True,lambda_gdf=weight,
                alpha_ramp_steps=int(plan.get('alpha_ramp_steps',1000)),lr=float(plan.get('adapt_lr',.0001)),
                require_attribute_readiness=True)
        # Native VAL is evaluated at the common final update, regardless of
        # which checkpoint has the lowest sampled validation loss.
        if plan.get('evaluation_cases'):
            value=json.loads(Path(plan['evaluation_cases']).read_text(encoding='utf8'))
            cases=value['cases'] if isinstance(value,dict) else value
            cases=[row for row in cases if str(row.get('split','VAL')).upper()=='VAL']
            if not cases:
                raise ValueError('Native evaluation requires explicit VAL cases')
            for name,_,_ in definitions:
                key='native_VAL_'+name
                if key in state['completed']:continue
                state.update(current=key);_json(state_path,state)
                predictions=[]
                for case in cases:
                    destination=home/'native_VAL'/name/case['id']/'inference.json'
                    if resume and destination.is_file():
                        result=json.loads(destination.read_text(encoding='utf8'))
                    else:
                        result=infer(dict(checkpoint=str(home/name/'last.pt'),image=case['image'],
                            case_id=case['id'],output_dir=str(home/'native_VAL'/name),device=common['device'],
                            patch_size=common['patch_size'],batch_size=common['batch_size'],overlap=.5,
                            reviewed_override=case.get('intensity_override'),amp=common['amp']))
                    predictions.append(dict(id=case['id'],prediction=result['prediction'],label=case['label']))
                result=evaluate(dict(cases=predictions,output_dir=str(home/'metrics'/key),centerline=True))
                state['completed'][key]=result;_json(state_path,state)
        state.update(status='complete',current=None,finished_unix=time.time());_json(state_path,state)
        return state
    except Exception as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc),failed_unix=time.time())
        _json(state_path,state)
        raise
