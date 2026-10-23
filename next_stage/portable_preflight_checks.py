"""Pure synthetic portable-preflight guards; no Torch, CUDA or study files.

Subprocess fixtures deliberately emulate false QA reports to test rejection;
they are not the scientific QA and never produce a real passed preflight.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import portable_preflight as guard


def put(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def fixture(root):
    import numpy as np
    project, cache, gdf = root/'source', root/'cache', root/'gdf'
    required = ('code/data.py','code/train.py','code/geometry.py','code/faithful_model.py',
                'code/prepare.py','code/common.py','code/intensity.py',
                'next_stage/stage_train.py','next_stage/stage_model.py','next_stage/stage_checks.py',guard.MODEL_QA_PATH,
                'next_stage/data_checks.py','next_stage/next_data.py')
    for relative in required:
        path=project/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('# Synthetic source fixture, no actual model.\n',encoding='utf-8')
    here=Path(__file__).resolve().parent
    for name in ('gdf.py','prepare_gdf.py'):
        shutil.copy2(here/name,project/'next_stage'/name)
    directory=cache/'TRAIN';directory.mkdir(parents=True)
    image=np.arange(1320,dtype=np.int16).reshape(12,11,10)
    mask=np.zeros(image.shape,np.uint8);mask[3:9,4:8,4:7]=1
    points=np.array([[4,5,5],[7,6,5]],np.int32);radius=np.array([2.,3.],np.float32)
    np.save(directory/'image.npy',image);np.save(directory/'mask.npy',mask)
    np.savez(directory/'geometry.npz',points=points,radius=radius,structure=np.array([1,2],np.uint8),geometry_valid=np.array(True))
    hashes={n:guard.digest(directory/n) for n in ('image.npy','mask.npy','geometry.npz')}
    record=dict(id='TRAIN',cache_dir=str(directory),shape_zyx=list(image.shape),spacing_mm=1.,
                geometry_training_only=True,cache_file_sha256=hashes,
                preprocessing_source_sha256={n:guard.digest(project/'code'/n)
                    for n in ('prepare.py','geometry.py','common.py','intensity.py')})
    sentinel=root/'HELD_OUT_MUST_NOT_OPEN'
    manifest=dict(complete=True,spacing_mm=1.,cases=[record,dict(id='VAL',cache_dir=str(sentinel/'VAL')),
                     dict(id='TEST',cache_dir=str(sentinel/'TEST'))],splits=dict(train=['TRAIN'],val=['VAL'],test=['TEST']))
    path=root/'manifest.json';put(path,manifest)
    out=gdf/'TRAIN'/'gdf.npy';out.parent.mkdir(parents=True)
    field=np.zeros(image.shape,np.float16);coords=np.argwhere(mask)
    distance=((coords[:,None,:]-points[None,:,:])**2).sum(2)
    field[tuple(coords.T)]=(np.log1p(np.exp(-.5*distance/radius[None,:]**2).sum(1))/2.5).astype(np.float16)
    np.save(out,field)
    row=dict(id='TRAIN',complete=True,target_file=str(out.resolve()),
             target_file_sha256=guard.digest(out),target_file_bytes=out.stat().st_size,dtype='float16',
             input_file_sha256={n:hashes[n] for n in ('mask.npy','geometry.npz')},shape_zyx=list(image.shape),
             spacing_zyx_mm=[1.,1.,1.],fixed_train_field_scale=2.5,foreground_geometry_valid=True)
    producer=dict(complete=True,cases=[row],fixed_train_field_scale=2.5,
                  source_code_sha256={n:guard.digest(project/'next_stage'/n) for n in ('gdf.py','prepare_gdf.py')})
    put(gdf/'manifest.json',producer)
    return project,cache,gdf,path,manifest,producer,sentinel


def run():
    import numpy as np
    checks=[]
    with tempfile.TemporaryDirectory(prefix='gta_portable_preflight_QA_') as temporary:
        root=Path(temporary)
        # -S removes third-party site packages: a successful default plan cannot
        # depend on importing Numpy/Torch or on opening the missing manifest.
        output=root/'PLAN_MUST_NOT_CREATE'
        result=subprocess.run([sys.executable,'-I','-S',str(Path(guard.__file__).resolve()),
            '--manifest',str(root/'NO_MANIFEST'),'--project',str(root/'NO_PROJECT'),'--output',str(output)],
            capture_output=True,text=True)
        assert result.returncode==0 and json.loads(result.stdout)['execute'] is False and not output.exists()
        checks.append('default_plan_works_without_site_packages_or_manifest')
        project,cache,gdf,path,manifest,producer,sentinel=fixture(root)
        base=guard.inspect_data(path,project)
        assert base['fully_rehashed_array_files']==3 and not base['unexpected_array_open_attempts']
        assert not sentinel.exists();checks.append('TRAIN_full_byte_header_geometry_guard_no_heldout_open')
        data=guard.inspect_data(path,project,cache,gdf,True,2)
        assert data['fully_rehashed_array_files']==4 and data['cases'][0]['gdf']['numerical_samples_checked']==2
        checks.append('same_native_path_overrides_and_analytic_CPU64_samples')

        def reject(function,label):
            try:function()
            except (ValueError,FileNotFoundError,KeyError):checks.append(label)
            else:raise AssertionError('Invalid synthetic input accepted: '+label)

        # Correct relocated bytes must not certify the unchanged trainer, which
        # still consumes cache_dir/target_file from its original manifests.
        relocated=copy.deepcopy(manifest);relocated['cases'][0]['cache_dir']=str(root/'MISSING_OLD_CACHE'/'TRAIN')
        put(path,relocated)
        assert not Path(relocated['cases'][0]['cache_dir']).exists()
        assert guard.digest(cache/'TRAIN/image.npy')==relocated['cases'][0]['cache_file_sha256']['image.npy']
        reject(lambda:guard.inspect_data(path,project,cache),'correct_relocated_cache_bytes_not_training_ready')
        refused=root/'relocated_cache_execute'
        reject(lambda:guard.main(['--manifest',str(path),'--project',str(project),'--output',str(refused),
                                  '--cache-root',str(cache),'--execute']),'execute_relocated_cache_no_passed_receipt')
        assert not (refused/'receipt.json').exists() and not (refused/'stage_checks_actual.json').exists()
        assert 'Cache override changes' in guard.read(refused/'failure.json')['error']
        put(path,manifest)
        relocated=copy.deepcopy(producer);relocated['cases'][0]['target_file']=str(root/'MISSING_OLD_GDF'/'TRAIN/gdf.npy')
        put(gdf/'manifest.json',relocated)
        assert not Path(relocated['cases'][0]['target_file']).exists()
        assert guard.digest(gdf/'TRAIN/gdf.npy')==relocated['cases'][0]['target_file_sha256']
        reject(lambda:guard.inspect_data(path,project,cache,gdf,True),'correct_relocated_GDF_bytes_not_training_ready')
        refused=root/'relocated_GDF_execute'
        reject(lambda:guard.main(['--manifest',str(path),'--project',str(project),'--output',str(refused),
                                  '--gdf-root',str(gdf),'--relocate-gdf-targets','--execute']),
               'execute_relocated_GDF_no_passed_receipt')
        assert not (refused/'receipt.json').exists() and not (refused/'stage_checks_actual.json').exists()
        assert 'GDF relocation changes' in guard.read(refused/'failure.json')['error']
        put(gdf/'manifest.json',producer)

        for change,label in ((lambda m:m['splits']['val'].append('TRAIN'),'split_overlap'),
                             (lambda m:m['cases'][0].update(shape_zyx=[1,2,3]),'grid_mismatch'),
                             (lambda m:m['cases'][0].update(cache_dir=m['cases'][1]['cache_dir']),'heldout_directory_alias')):
            altered=copy.deepcopy(manifest);change(altered);put(path,altered)
            reject(lambda:guard.inspect_data(path,project),label)
        put(path,manifest)
        image=cache/'TRAIN/image.npy';previous=image.read_bytes();image.write_bytes(previous[:-1]+bytes([previous[-1]^1]))
        reject(lambda:guard.inspect_data(path,project),'changed_actual_image_SHA');image.write_bytes(previous)
        altered=copy.deepcopy(manifest);floating=np.arange(1320,dtype=np.float32).reshape(12,11,10);floating[0,0,0]=np.nan
        np.save(image,floating);altered['cases'][0]['cache_file_sha256']['image.npy']=guard.digest(image);put(path,altered)
        reject(lambda:guard.inspect_data(path,project),'nonfinite_CT_with_resealed_manifest');image.write_bytes(previous);put(path,manifest)
        altered=copy.deepcopy(manifest);altered['cases'][0]['preprocessing_source_sha256']['geometry.py']='0'*64;put(path,altered)
        reject(lambda:guard.inspect_data(path,project),'cache_preprocessing_source_mismatch');put(path,manifest)
        for value,label in ((None,'missing_preprocessing_inventory'),({},'empty_preprocessing_inventory'),
                            ({'geometry.py':manifest['cases'][0]['preprocessing_source_sha256']['geometry.py']},
                             'partial_preprocessing_inventory')):
            altered=copy.deepcopy(manifest)
            if value is None:altered['cases'][0].pop('preprocessing_source_sha256')
            else:altered['cases'][0]['preprocessing_source_sha256']=value
            put(path,altered);reject(lambda:guard.inspect_data(path,project),label)
        put(path,manifest)
        maskfile=cache/'TRAIN/mask.npy';previous=maskfile.read_bytes();mask=np.load(maskfile);mask[0,0,0]=255;np.save(maskfile,mask)
        altered=copy.deepcopy(manifest);altered['cases'][0]['cache_file_sha256']['mask.npy']=guard.digest(maskfile);put(path,altered)
        reject(lambda:guard.inspect_data(path,project),'nonbinary_mask_even_with_resealed_manifest')
        maskfile.write_bytes(previous);put(path,manifest)
        for change,label in ((lambda m:m['cases'].append(dict(m['cases'][0],id='VAL')),'GDF_heldout_case'),
                             (lambda m:m['cases'][0]['input_file_sha256'].update({'geometry.npz':'0'*64}),'GDF_geometry_source_mismatch'),
                             (lambda m:m['cases'][0].update(target_file_sha256='0'*64),'GDF_actual_target_SHA'),
                             (lambda m:m['source_code_sha256'].update({'gdf.py':'0'*64}),'GDF_producer_source_mismatch')):
            altered=copy.deepcopy(producer);change(altered);put(gdf/'manifest.json',altered)
            reject(lambda:guard.inspect_data(path,project,cache,gdf,True),label)
        put(gdf/'manifest.json',producer)
        target=gdf/'TRAIN/gdf.npy';previous=target.read_bytes();np.save(target,np.zeros((12,11,10),np.float16))
        altered=copy.deepcopy(producer);altered['cases'][0]['target_file_sha256']=guard.digest(target);put(gdf/'manifest.json',altered)
        byte_only=guard.inspect_data(path,project,cache,gdf,True)
        assert byte_only['gdf_numerical_scope'].startswith('Not performed')
        reject(lambda:guard.inspect_data(path,project,cache,gdf,True,2),'numeric_samples_reject_semantically_wrong_resealed_target')
        target.write_bytes(previous);put(gdf/'manifest.json',producer)
        original=target.read_bytes();field=np.load(target);candidate=np.unique(np.random.default_rng(np.random.SeedSequence([20261023,0,1])).integers(field.size,size=128))
        mask=np.load(maskfile);bad_index=candidate[mask.reshape(-1)[candidate]==0][0];field.reshape(-1)[bad_index]=.25;np.save(target,field)
        altered=copy.deepcopy(producer);altered['cases'][0]['target_file_sha256']=guard.digest(target);put(gdf/'manifest.json',altered)
        reject(lambda:guard.inspect_data(path,project,cache,gdf,True,2),'numeric_samples_reject_nonzero_background')
        target.write_bytes(original);put(gdf/'manifest.json',producer)
        np.save(target,np.zeros((132,10),np.float16));altered=copy.deepcopy(producer)
        altered['cases'][0].update(target_file_sha256=guard.digest(target),target_file_bytes=target.stat().st_size);put(gdf/'manifest.json',altered)
        reject(lambda:guard.inspect_data(path,project,cache,gdf,True),'GDF_resealed_wrong_header_shape')
        target.write_bytes(original);put(gdf/'manifest.json',producer)
        # Valid empty geometry is a known-zero target, not falsely invalid just
        # because the source geometry_valid scalar is True with zero seeds.
        np.save(maskfile,np.zeros((12,11,10),np.uint8))
        np.savez(cache/'TRAIN/geometry.npz',points=np.empty((0,3),np.int32),radius=np.empty(0,np.float32),
                 structure=np.empty(0,np.uint8),geometry_valid=np.array(True))
        empty_manifest=copy.deepcopy(manifest)
        for name in ('mask.npy','geometry.npz'):empty_manifest['cases'][0]['cache_file_sha256'][name]=guard.digest(cache/'TRAIN'/name)
        put(path,empty_manifest);np.save(target,np.zeros((12,11,10),np.float16));empty_gdf=copy.deepcopy(producer)
        empty_gdf['cases'][0].update(foreground_geometry_valid=False,target_file_sha256=guard.digest(target),
            input_file_sha256={n:empty_manifest['cases'][0]['cache_file_sha256'][n] for n in ('mask.npy','geometry.npz')})
        put(gdf/'manifest.json',empty_gdf)
        empty_audit=guard.inspect_data(path,project,cache,gdf,True,2)
        assert empty_audit['cases'][0]['centerline_points']==0 and empty_audit['cases'][0]['gdf']['known_background_samples_checked']==2
        checks.append('empty_geometry_true_scalar_no_seeds_known_background_valid')

        # Deliberately false synthetic QA children: all must be rejected. The
        # production function invokes the explicitly bound QA file at actual
        # CUDA128 batch2, with FP32+AMP phases. All false children below fail.
        baseline=dict(status='passed',device='cuda',patch_size=128,batch_size=2,
            autocast_modes_checked=[False,True],study_data_read=False,model_update_count=0,
            amp_backward_manual_loss_scale=1024.,
            phases=[dict(name=name,status='passed') for name in guard.MODEL_QA_PHASES],
            backend_settings=dict(matmul_tf32=False,cudnn_tf32=False,cudnn_benchmark=False,
                cudnn_deterministic=True,deterministic_algorithms=True,deterministic_warn_only=True,
                cublas_workspace_config=':4096:8'))
        variants=[('exit_nonzero',1,{},False),('wrong_device',0,dict(device='cpu'),False),
                  ('wrong_patch',0,dict(patch_size=32),False),('wrong_batch',0,dict(batch_size=1),False),
                  ('wrong_qa_SHA',0,dict(qa_file_sha256='0'*64),False),
                  ('wrong_source_binding',0,dict(source_sha256={}),False),
                  ('missing_model_phase',0,dict(phases=baseline['phases'][:-1]),False),
                  ('wrong_backend',0,dict(backend_settings=dict(baseline['backend_settings'],matmul_tf32=True)),False),
                  ('source_drift',0,{},True)]
        for label,exitcode,changes,drift in variants:
            binding=root/(label+'_binding.json')
            script="import sys,json,pathlib\np=pathlib.Path(sys.argv[sys.argv.index('--output')+1])\n"
            script+="source=json.loads(pathlib.Path(%r).read_text())\n"%str(binding)
            script+="payload=%r\npayload.update(qa_file_path=%r,qa_file_sha256=source[%r],source_sha256=source)\n"%(baseline,guard.MODEL_QA_PATH,guard.MODEL_QA_PATH)
            script+="payload.update(%r)\np.write_text(json.dumps(payload))\n"%changes
            if drift:script+="pathlib.Path(%r).write_text('synthetic changed source')\n"%str(project/'code/data.py')
            script+='raise SystemExit(%d)\n'%exitcode
            (project/guard.MODEL_QA_PATH).write_text(script,encoding='utf-8')
            inventory=guard.source_inventory(project);folder=root/label;folder.mkdir()
            put(binding,inventory)
            reject(lambda:guard.run_qa(project,folder,sys.executable,inventory),'QA_'+label)
            actual=guard.read(folder/'stage_checks_actual.json')
            assert actual['exit_code']==exitcode and actual['source_sha256']==inventory
            assert actual['qa_file_path']==guard.MODEL_QA_PATH and actual['qa_file_sha256']==inventory[guard.MODEL_QA_PATH]
            assert actual['command'][actual['command'].index('--batch-size')+1]=='2'
        # Immutable output refusal occurs before any QA execution.
        occupied=root/'occupied';occupied.mkdir();(occupied/'marker').write_text('preserve')
        try:guard.main(['--manifest',str(path),'--project',str(project),'--output',str(occupied),'--execute'])
        except FileExistsError:checks.append('immutable_output_refused_before_execution')
        else:raise AssertionError('Existing output overwritten')
        assert (occupied/'marker').read_text()=='preserve'
        assert 'torch' not in sys.modules and not sentinel.exists()
    return dict(status='passed',synthetic_guard_checks=len(checks),checks=checks,torch_imported=False,
                source_sha256={'portable_preflight.py':guard.digest(guard.__file__),
                               'portable_preflight_checks.py':guard.digest(__file__)},
                CUDA_executed=False,study_arrays_read=False,real_passed_preflight_receipt_created=False,
                limitations='Synthetic arrays and deliberately false QA subprocess fixtures only; no actual model/CUDA pass is claimed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    args=parser.parse_args();result=run()
    if args.output:guard.write_new(args.output,result)
    print(json.dumps(result))
