"""Five small synthetic checks; no actual CT, labels, model or endpoint IO."""
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock


def require(value,message):
    if not value:
        raise AssertionError(message)


def main():
    import numpy as np
    import SimpleITK as sitk
    path=Path(__file__).with_name('continuous_ct_input.py')
    ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,8))
    response=subprocess.run([sys.executable,'-B',str(path),'--image','DOES_NOT_EXIST.nrrd'],
        check=True,capture_output=True,text=True)
    require(json.loads(response.stdout)['status']=='inert','Default CLI performed input work')
    spec=importlib.util.spec_from_file_location('continuous_ct_input_check',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    prepare,data,_=module._algorithms();checks=['default_inert_and_python38_API']
    with tempfile.TemporaryDirectory(prefix='continuous_ct_input_synthetic_') as temporary:
        home=Path(temporary)
        raw=np.full((9,52,60),-1000.25,np.float32)
        raw[:,20:34,20:40]=300.6
        original=sitk.GetImageFromArray(raw)
        original.SetSpacing((.9,1.1,2.5));original.SetOrigin((13.25,-22.5,7.75))
        original.SetDirection((-1.,0.,0.,0.,-1.,0.,0.,0.,1.))
        CT=home/'synthetic_CT.nrrd';sitk.WriteImage(original,str(CT))
        original_reader=sitk.ReadImage;reads=[]
        def read_CT_only(filename,*args,**kwargs):
            require(Path(filename).resolve()==CT.resolve(),'Unexpected image/label file read')
            reads.append(str(filename));return original_reader(filename,*args,**kwargs)
        with mock.patch.object(sitk,'ReadImage',side_effect=read_CT_only), \
                mock.patch.object(prepare,'discover',side_effect=AssertionError('No discovery allowed')), \
                mock.patch.object(prepare,'sparse_geometry',side_effect=AssertionError('No geometry allowed')):
            record=module.prepare_ct(CT,'synthetic_case',home/'cache')
        require(len(reads)==1 and set(p.name for p in (home/'cache').iterdir())=={'image.npy','record.json'},
            'Preparation accessed extra input or created non-image cache files')
        corrected,_=prepare.correct_stored_intensity(original)
        oriented=sitk.DICOMOrient(corrected,'LPS');grid=prepare.reference_grid(oriented,1.)
        resampled=sitk.Resample(oriented,grid,sitk.Transform(),sitk.sitkLinear,-1024.,sitk.sitkFloat32)
        array=sitk.GetArrayFromImage(resampled);crop=prepare.image_body_crop(array,1.)
        expected=np.clip(np.rint(array[crop]),-1024,2000).astype(np.int16)
        value=module.load_image(home/'cache'/'record.json',mmap_mode=None)
        buffer=io.BytesIO();np.save(buffer,expected,allow_pickle=False)
        require(np.array_equal(value,expected) and (home/'cache'/'image.npy').read_bytes()==buffer.getvalue(),
            'Original rint/clip/int16/cache bytes differ')
        require(record['full_shape_zyx']==list(array.shape) and
            record['crop_start_zyx']==[s.start for s in crop] and record['resampled_spacing']==(1.,1.,1.),
            'Original CT-only crop/grid metadata differ')
        try:
            module.prepare_ct(CT,'synthetic_case',home/'bad_override',{'image_sha256':'0'*64})
        except ValueError:
            pass
        else:
            raise AssertionError('Wrong original reviewed-override identity accepted')
        require(not (home/'bad_override').exists(),'Rejected input created output')
        checks.append('one_CT_read_no_label_geometry_and_original_cache_bytes_at_explicit_1mm')
        tiny=np.arange(3*4*5,dtype=np.int16).reshape(3,4,5)
        tiny_record=dict(record,shape_zyx=list(tiny.shape),full_shape_zyx=[21,31,41],crop_start_zyx=[2,3,4])
        batches=list(module.patch_batches(tiny,tiny_record,patch=8,batch_size=2))
        starts,batch,locations=batches[0]
        padded=np.pad(tiny.astype(np.float32),[(0,5),(0,4),(0,3)],constant_values=-1024)
        expected_location=2.*(np.array([2,3,4])+4.)/(np.array([21,31,41])-1.)-1.
        require(len(batches)==1 and starts==[(0,0,0)] and
            np.array_equal(batch[0,0],data.normalize(padded)) and
            np.array_equal(locations[0],expected_location.astype(np.float32)) and batch[0,0,-1,-1,-1]==-1.,
            'Original air padding/normalization/full-grid coordinates differ')
        checks.append('air_padding_original_normalize_and_coordinate_formula')
        large=np.zeros((17,18,19),np.int16)
        large_record=dict(record,shape_zyx=list(large.shape),full_shape_zyx=list(large.shape),crop_start_zyx=[0,0,0])
        observed=[]
        def constant_callback(CT,coordinates):
            require(CT.dtype==np.float32 and coordinates.dtype==np.float32 and CT.shape[1:]==(1,8,8,8),
                'Callback input contract differs')
            observed.extend(coordinates.tolist())
            return np.full((len(CT),1,8,8,8),.25,np.float32)
        probability,tiles=module.sliding_probability(constant_callback,large,large_record,patch=8,batch_size=2)
        require(probability.shape==large.shape and np.array_equal(probability,np.full_like(probability,.25)) and
            tiles==4*4*4 and len(observed)==tiles,'Full-crop Gaussian tile fusion/coverage differs')
        checks.append('CT_callback_all_tiles_Gaussian_fusion_without_GT_selection')
        with mock.patch.object(sitk,'ReadImage',side_effect=AssertionError('Native restore must need only saved header')):
            restored=module.native_probability(np.full_like(value,.5,dtype=np.float32),record)
            binary=module.native_prediction(np.full_like(value,.5,dtype=np.float32),record)
        require(restored.GetSize()==original.GetSize() and restored.GetOrigin()==original.GetOrigin() and
            restored.GetSpacing()==original.GetSpacing() and restored.GetDirection()==original.GetDirection(),
            'Native CT header not restored exactly')
        restored_array=sitk.GetArrayFromImage(restored)
        require(np.array_equal(sitk.GetArrayFromImage(binary),restored_array>=.5),
            'Native fixed0.5 threshold differs')
        checks.append('saved_native_header_only_recovery_and_original_fixed_threshold')
    print(json.dumps(dict(status='PASS',checks=checks,synthetic_only=True,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),real_data_read=False,
        model_or_predictor_executed=False),indent=2))


if __name__=='__main__':
    main()
