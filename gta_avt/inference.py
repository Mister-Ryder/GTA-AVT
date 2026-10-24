"""Whole-volume CT-only inference and separate label-based objective evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import prepare_ct, restore_native, sliding_probability, load_manifest
from .engine import load_checkpoint, file_digest, _json, _device
from .metrics import binary_metrics, summarize_cases
from .models import AttributePredictor, GTAAVT


def model_from_checkpoint(path,device='cpu'):
    checkpoint=load_checkpoint(path)
    if checkpoint['kind']!='train':
        raise ValueError('Whole-volume segmentation requires a segmentation checkpoint')
    config=checkpoint['model_config']
    model=GTAAVT(AttributePredictor(checkpoint['statistics']),arm=config['arm'],use_gdf=config['use_gdf'])
    model.load_state_dict(checkpoint['model'],strict=True)
    return model.to(device).eval(),checkpoint


@torch.no_grad()
def predict_volume(model,image,metadata,*,device,patch_size=128,batch_size=2,overlap=.5,
                   amp=True,condition_mode='predicted'):
    if condition_mode not in ('predicted','radius_constant','neutral','radius_half_ring'):
        raise ValueError('Unknown explicit condition diagnostic')
    use_amp=amp and device.type=='cuda'
    def callback(patches):
        x=torch.from_numpy(patches).to(device)
        with torch.autocast(device_type=device.type,enabled=use_amp):
            override=None
            if condition_mode!='predicted':
                attributes=model.predictor(x)
                a=(attributes['a_logits'].sigmoid()>=.5).float()
                z=attributes['z_radius'].float().clone()
                if condition_mode=='radius_constant':
                    z.zero_()
                elif condition_mode=='neutral':
                    a.zero_();z.zero_()
                else:
                    for index in range(len(z)):
                        supported=a[index].bool()
                        values=z[index][supported]
                        z[index][supported]=torch.roll(values,len(values)//2)
                override={'a':a,'z_radius':z}
            output=model(x,alpha=1.,condition_override=override)
        return output['logits'].float().sigmoid().cpu().numpy()
    return sliding_probability(callback,image,metadata,patch=patch_size,batch_size=batch_size,overlap=overlap)


def infer(config):
    if not config.get('checkpoint') or not config.get('image'):
        raise ValueError('checkpoint and image are required; no label input is accepted')
    prohibited=('label','mask','label_path','roi','geometry')
    if any(config.get(key) for key in prohibited):
        raise ValueError('CT-only inference does not accept annotation or geometry inputs')
    device=_device(config)
    image_path=Path(config['image']).resolve()
    case_id=config.get('case_id',image_path.name.split('.')[0])
    if not isinstance(case_id,str) or not case_id or case_id in ('.','..') or any(c in case_id for c in '/\\'):
        raise ValueError('case_id must be one safe directory name')
    output=Path(config.get('output_dir','predictions'))/case_id
    output.mkdir(parents=True,exist_ok=False)
    override=config.get('reviewed_override')
    if isinstance(override,str):
        override=json.loads(Path(override).read_text(encoding='utf-8'))
    image,metadata=prepare_ct(image_path,case_id=case_id,spacing=1.0,reviewed_override=override)
    model,checkpoint=model_from_checkpoint(config['checkpoint'],device)
    mode=config.get('condition_mode','predicted')
    probability,tiles=predict_volume(model,image,metadata,device=device,
        patch_size=int(config.get('patch_size',128)),batch_size=int(config.get('batch_size',2)),
        overlap=float(config.get('overlap',.5)),amp=bool(config.get('amp',True)),condition_mode=mode)
    destination=output/'prediction.nii.gz'
    restore_native(probability,metadata,destination,threshold=.5)
    probability_path=None
    if config.get('output_probability',False):
        probability_path=output/'probability.nii.gz'
        restore_native(probability,metadata,probability_path)
    report={'case_id':case_id,'prediction':str(destination.resolve()),
        'probability':str(probability_path.resolve()) if probability_path else None,
        'checkpoint_sha256':file_digest(config['checkpoint']),'checkpoint_successful_updates':checkpoint['step'],
        'arm':model.arm,'condition_mode':mode,'tiles':tiles,'threshold':.5,
        'input':'CT only; image-derived crop; 1mm LPS grid; native physical header restored',
        'patch_size':int(config.get('patch_size',128)),'overlap':float(config.get('overlap',.5)),
        'tile_blending':'Gaussian importance weights; distinct from the GDF training auxiliary'}
    _json(output/'inference.json',report)
    return report


def _evaluation_cases(config):
    if config.get('cases'):
        return config['cases']
    if config.get('cases_json'):
        value=json.loads(Path(config['cases_json']).read_text(encoding='utf-8'))
        return value['cases'] if isinstance(value,dict) else value
    if config.get('manifest') and config.get('prediction_dir'):
        manifest=load_manifest(config['manifest'])
        split=config.get('split','VAL').upper()
        cases=[]
        for case in manifest['cases']:
            if case['split'].upper()!=split:
                continue
            label=case.get('label') or case.get('label_path')
            if not label:
                raise ValueError('Evaluation manifest needs explicit native label paths; use cases_json')
            cases.append({'id':case['id'],'label':label,'source':case.get('source'),
                'prediction':str(Path(config['prediction_dir'])/case['id']/'prediction.nii.gz')})
        return cases
    raise ValueError('Provide cases/cases_json or manifest + prediction_dir for explicit evaluation')


def evaluate(config):
    import SimpleITK as sitk
    cases=_evaluation_cases(config)
    output=Path(config['output_dir']).resolve()
    output.mkdir(parents=True,exist_ok=False)
    rows=[];seen=set()
    for case in cases:
        case_id=str(case.get('id',case.get('case_id','')))
        if not case_id or case_id in seen:
            raise ValueError('Evaluation case IDs must be nonempty and unique')
        seen.add(case_id)
        prediction=sitk.ReadImage(str(case['prediction']))
        label=sitk.ReadImage(str(case['label']))
        if (prediction.GetDimension()!=3 or label.GetDimension()!=3 or
            prediction.GetNumberOfComponentsPerPixel()!=1 or label.GetNumberOfComponentsPerPixel()!=1):
            raise ValueError('Scalar 3D prediction/label required for '+case_id)
        if (prediction.GetSize()!=label.GetSize() or
            any(not np.allclose(getattr(prediction,key)(),getattr(label,key)(),rtol=0,atol=1e-5)
                for key in ('GetSpacing','GetOrigin','GetDirection'))):
            raise ValueError('Native prediction/label physical grids differ for '+case_id)
        label_values=sitk.GetArrayFromImage(label)
        if not np.isfinite(label_values).all():
            raise ValueError('Nonfinite label for '+case_id)
        target=label_values>0
        value=sitk.GetArrayFromImage(prediction)
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite prediction for '+case_id)
        pred=value>=.5
        row={'case_id':case_id,'source':case.get('source'),
             **binary_metrics(pred,target,centerline=config.get('centerline',True),
                              spacing_zyx=label.GetSpacing()[::-1])}
        voxel_ml=float(np.prod(label.GetSpacing())/1000.)
        row.update(predicted_volume_ml=float(pred.sum())*voxel_ml,
                   target_volume_ml=float(target.sum())*voxel_ml,
                   fp_volume_ml=row['fp_voxels']*voxel_ml,fn_volume_ml=row['fn_voxels']*voxel_ml)
        rows.append(row)
    report={'scope':config.get('scope','native whole-case evaluation'),
        'split':config.get('split','unspecified'),'scores':'fractions in [0,1]',
        'definitions':{'centerline':'Lee skeleton voxel overlap on each native grid; not physical branch length',
                       'fine_centerline':'GT native Lee skeleton sites with native physical EDT radius <=2mm; report supported voxel count; no sites yields null recall',
                       'dispersion':'population SD across defined cases; worst20 means ceil(20% cases)',
                       'empty':'Both masks empty Dice/clDice=1; undefined precision or recall is null'},
        'per_case':rows,'summary':summarize_cases(rows)}
    _json(output/'metrics.json',report)
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with (output/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    return report
