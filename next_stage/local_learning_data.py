"""Matched old sampler plus local supervision and CT-only padding validity.

The frozen StageData remains the sole image sampler. A short RNG replay recovers
its already selected spatial transforms, without changing its calls or outputs.
GT local validity is a loss mask; input_valid alone may enter the model.
"""
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))
from next_data import StageData
from frozen_readout import local_targets, transform_spatial, ct_window_support


def replay_spatial(data, seed, step, index, patch, foreground_probability):
    """Replay only through spatial augmentation, checking case/start identity."""
    rng=np.random.default_rng(np.random.SeedSequence([seed,step,index]))
    ids=data.manifest['splits']['train']
    case_id=ids[rng.integers(len(ids))]
    item=data.case(case_id,geometry=True)
    shape=np.asarray(item['image'].shape)
    points=item['geometry']['points']
    if len(points) and rng.random()<foreground_probability:
        centre=points[rng.integers(len(points))]+rng.integers(-patch//4,patch//4+1,size=3)
        start=np.clip(centre-patch//2,0,np.maximum(shape-patch,0))
    else:
        start=np.array([rng.integers(max(1,int(n)-patch+1)) for n in shape])
    flips=[bool(rng.random()<.5) for _ in range(3)]
    rotation=int(rng.integers(4))
    axes=((0,1),(0,2),(1,2))[int(rng.integers(3))]
    return case_id,start,flips,rotation,axes


class LocalStageData(StageData):
    def __init__(self, manifest, gdf_root=None):
        super().__init__(manifest,gdf_root=gdf_root)
        spacing=np.asarray(self.manifest['spacing_mm'],dtype=float)
        if not np.all(spacing==1.):
            raise ValueError('This fixed local geometry definition requires the frozen 1mm grid')

    def sample(self, seed, step, index, patch, foreground_probability=.5):
        if patch!=128:
            raise ValueError('The single local revision is defined for 128^3 inputs')
        example=super().sample(seed,step,index,patch,foreground_probability)
        case_id,start,flips,rotation,axes=replay_spatial(
            self,seed,step,index,patch,foreground_probability)
        if case_id!=example['case_id'] or start.tolist()!=example['start']:
            raise AssertionError('Frozen sampler selection no longer matches the spatial replay')
        item=self.case(case_id,geometry=True)
        shape=np.asarray(item['image'].shape)
        slices=tuple(slice(int(s),min(int(s)+patch,int(n))) for s,n in zip(start,shape))
        mask=np.asarray(item['mask'][slices],dtype=bool)
        real_shape=mask.shape
        mask=np.pad(mask,[(0,patch-int(n)) for n in real_shape])
        input_valid=np.zeros((patch,)*3,dtype=bool)
        input_valid[tuple(slice(0,n) for n in real_shape)]=True
        geometry=item['geometry']
        local=local_targets(mask,geometry['points']-start[None],geometry['radius'],input_valid,
                            geometry_valid=bool(np.asarray(geometry.get('geometry_valid',True)).item()))
        target=transform_spatial(local['target'],flips,rotation,axes,spatial_start=1)
        valid=transform_spatial(local['loss_valid'],flips,rotation,axes)
        input_valid=transform_spatial(input_valid,flips,rotation,axes)
        # This check concerns CT coverage only, never a GT-derived inference gate.
        expected_support=transform_spatial(local['ct_support'],flips,rotation,axes)
        if not np.array_equal(ct_window_support(input_valid),expected_support):
            raise AssertionError('CT window grid does not commute with the frozen augmentation')
        example.update(local_target=np.ascontiguousarray(target,dtype=np.float32),
                       local_valid=np.ascontiguousarray(valid[None],dtype=bool),
                       input_valid=np.ascontiguousarray(input_valid[None],dtype=bool))
        return example


def batch(data,seed,step,batch_size,patch,foreground_probability):
    examples=[data.sample(seed,step,index,patch,foreground_probability) for index in range(batch_size)]
    names=('image','mask','scale','structure','attribute_valid','coordinate',
           'gdf_target','gdf_valid','local_target','local_valid','input_valid')
    return {name:np.stack([x[name] for x in examples]) for name in names}
