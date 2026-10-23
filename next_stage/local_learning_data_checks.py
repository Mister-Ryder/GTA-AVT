"""Independent frozen-sampler and transformed-geometry fixtures for local A."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import numpy as np

from data_checks import _fixture, _close_data, _same_array
from next_data import StageData
from local_learning_data import LocalStageData, replay_spatial
from frozen_readout import local_targets


def transform_points(points, flips, rotation, axes):
    # Independent integer coordinate transform, rather than rotating target maps.
    result=points.copy()
    for axis,flip in enumerate(flips):
        if flip:result[:,axis]=127-result[:,axis]
    a,b=axes
    for _ in range(rotation):
        previous=result.copy()
        result[:,a]=127-previous[:,b]
        result[:,b]=previous[:,a]
    return result


def run():
    compared=padding=unknown=0
    local_seconds=old_seconds=0.
    with tempfile.TemporaryDirectory(prefix='GTA_local_A_QA_') as temporary:
        root=Path(temporary)
        for number,(shape,valid_geometry) in enumerate([
                ((144,138,132),True),((91,107,83),True),((91,107,83),False)]):
            manifest,_,field,_=_fixture(root,'QA_'+str(number),shape,valid_geometry)
            metadata=json.loads(manifest.read_text());metadata['spacing_mm']=1.
            manifest.write_text(json.dumps(metadata),encoding='utf-8')
            old=StageData(manifest,gdf_root=field)
            new=LocalStageData(manifest,gdf_root=field)
            try:
                for step in range(6):
                    begin=time.perf_counter()
                    reference=old.sample(20261023,step,step%2,128,.5)
                    old_seconds+=time.perf_counter()-begin
                    begin=time.perf_counter()
                    candidate=new.sample(20261023,step,step%2,128,.5)
                    local_seconds+=time.perf_counter()-begin
                    for key in reference:
                        if key in ('case_id','start'):
                            assert reference[key]==candidate[key]
                        else:
                            _same_array(reference[key],candidate[key],key)
                    cid,start,flips,rotation,axes=replay_spatial(new,20261023,step,step%2,128,.5)
                    geom=new.case(cid,geometry=True)['geometry']
                    points=transform_points(geom['points']-start[None],flips,rotation,axes)
                    direct=local_targets(candidate['mask'][0]>0,points,geom['radius'],
                                         candidate['input_valid'][0],geometry_valid=valid_geometry)
                    assert np.array_equal(candidate['local_target'],direct['target'])
                    assert np.array_equal(candidate['local_valid'][0],direct['loss_valid'])
                    assert set(new.opened)<=set(new.manifest['splits']['train'])
                    assert set(new.gdf_opened)<=set(new.manifest['splits']['train'])
                    padding+=int(not candidate['input_valid'].all())
                    unknown+=int(np.sum(direct['ct_support'] & direct['foreground_window'] & ~direct['loss_valid']))
                    compared+=1
            finally:
                _close_data(old);_close_data(new)
    assert compared==18 and padding==12 and unknown>0
    return dict(status='passed',synthetic_samples=compared,padding_samples=padding,
                unknown_local_windows_retained=unknown,
                old_fields_and_image_bytes_identical=True,
                target_maps_equal_independent_transformed_point_recomputation=True,
                study_data_opened=False,models_run=False,
                synthetic_old_sampler_seconds=old_seconds,
                synthetic_local_sampler_seconds=local_seconds,
                estimated_extra_CPU_seconds_per_sample=(local_seconds-old_seconds)/compared)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    args=parser.parse_args()
    result=run()
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
