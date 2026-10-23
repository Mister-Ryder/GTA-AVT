"""Portable global-stage preflight; metadata plan unless --execute.

Explicit execution rehashes TRAIN arrays and runs this source's model CUDA128
and independent synthetic sampler checks in separate observed subprocesses.
VAL/TEST case files are never opened. Local-learning mode is not supported.
GDF checking is full byte/header/provenance validation; optional numerical
samples check consistency with this source's CPU64 reference, not independence.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time

MODEL_QA_PATH = 'next_stage/stage_checks_memory_safe.py'
MODEL_QA_PHASES = [
    'parity_%s_amp%s' % (name, amp)
    for amp in (False,True) for name in ('legacy','constant','alpha0')
] + ['%s_amp%s' % (name, amp)
     for amp in (False,True) for name in ('override','detached','differentiable','supervised','gdf')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    path = Path(path)
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            'File changed while hashing: '+str(path))
    return h.hexdigest()


def checked_sha(path, expected):
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected),
            'Expected SHA256 is missing/malformed: '+str(path))
    actual = digest(path)
    require(actual == expected, 'Actual byte identity differs: '+str(path))
    return actual


def source_inventory(project):
    project = Path(project).resolve()
    result = {}
    for folder in ('code', 'next_stage'):
        root = project/folder
        require(root.is_dir(), 'Source directory missing: '+str(root))
        for path in sorted(root.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            if path.suffix == '.py' or ('historical' in path.parts and path.suffix.lower() in ('.txt','.json','.md')):
                path.resolve().relative_to(project)
                result[path.relative_to(project).as_posix()] = digest(path)
    required = {'code/data.py','code/train.py','code/geometry.py','code/faithful_model.py',
                'next_stage/stage_train.py','next_stage/stage_model.py',
                'next_stage/stage_checks.py',MODEL_QA_PATH,'next_stage/data_checks.py','next_stage/next_data.py'}
    require(required.issubset(result), 'Incomplete executable global source snapshot')
    return result


def manifest_metadata(path):
    manifest = read(path)
    require(manifest.get('complete') is True and isinstance(manifest.get('cases'), list),
            'A complete list-case cache manifest is required')
    rows = manifest['cases']
    cases = {r['id']: r for r in rows}
    require(len(cases) == len(rows), 'Duplicate manifest case ID')
    splits = {s:manifest['splits'].get(s, []) for s in ('train','val','test')}
    require(splits['train'] and splits['val'], 'Nonempty TRAIN/VAL metadata is required')
    all_ids = []
    for name, ids in splits.items():
        require(isinstance(ids, list) and len(set(ids)) == len(ids), 'Invalid/duplicate '+name+' IDs')
        for cid in ids:
            require(isinstance(cid, str) and cid not in ('.','..') and '/' not in cid and '\\' not in cid,
                    'Case ID must be a single path component')
        all_ids.extend(ids)
    require(len(set(all_ids)) == len(all_ids), 'Split overlap')
    require(set(all_ids) == set(cases), 'Split/case metadata coverage differs')
    return manifest, cases, splits


def close_map(array):
    handle = getattr(array, '_mmap', None)
    if handle is not None:
        handle.close()


def inspect_data(manifest_path, project, cache_root=None, gdf_root=None,
                 relocate_gdf=False, sample_points=0, seed=20261023):
    """TRAIN-only byte/header validation. Does not import Torch or run QA."""
    import numpy as np
    manifest, cases, splits = manifest_metadata(manifest_path)
    require(type(sample_points) is int and 0 <= sample_points <= 64,
            'GDF sample points must be an integer between0 and64')
    cache_root = Path(cache_root).resolve() if cache_root else None
    dirs = {cid:(cache_root/cid if cache_root else Path(row['cache_dir'])).resolve()
            for cid,row in cases.items()}
    for cid in splits['train']:
        require(dirs[cid] == Path(cases[cid]['cache_dir']).resolve(),
                'Cache override changes the path the trainer will read; explicitly prepare a new manifest first: '+cid)
    train_dirs = [dirs[cid] for cid in splits['train']]
    heldout_dirs = [dirs[c] for c in splits['val']+splits['test']]
    require(len(set(train_dirs)) == len(train_dirs), 'TRAIN cache directories alias each other')
    require(not any(a == b or a in b.parents or b in a.parents for a in train_dirs for b in heldout_dirs),
            'TRAIN cache directory overlaps held-out case metadata')
    allowed_arrays = set()
    for cid in splits['train']:
        for name in ('image.npy','mask.npy','geometry.npz'):
            path = (dirs[cid]/name).resolve()
            path.relative_to(dirs[cid])
            allowed_arrays.add(str(path))
    gdf, fields, gdf_manifest_sha, reference = None, {}, None, None
    if gdf_root:
        gdf_root = Path(gdf_root).resolve()
        gdf = read(gdf_root/'manifest.json')
        gdf_manifest_sha = digest(gdf_root/'manifest.json')
        require(gdf.get('complete') is True and isinstance(gdf.get('cases'), list),
                'Complete list-case GDF manifest required')
        fields = {r['id']:r for r in gdf['cases']}
        require(len(fields) == len(gdf['cases']) and set(fields) == set(splits['train']),
                'GDF must cover exactly TRAIN, with no duplicates or held-out cases')
        for name, expected in gdf['source_code_sha256'].items():
            require(name in ('prepare_gdf.py','gdf.py'), 'Unexpected GDF producer source name')
            checked_sha(Path(project)/'next_stage'/name, expected)
        require(set(gdf['source_code_sha256']) == {'prepare_gdf.py','gdf.py'},
                'GDF producer source inventory incomplete')
        scale = float(gdf['fixed_train_field_scale'])
        require(np.isfinite(scale) and scale > 0, 'Invalid fixed TRAIN field scale')
        for cid,row in fields.items():
            trainer_path = (gdf_root/row['target_file']).resolve()
            target = Path(row['target_file'])
            if relocate_gdf:
                target = gdf_root/cid/PurePosixPath(str(row['target_file']).replace('\\','/')).name
            elif not target.is_absolute():
                target = gdf_root/target
            target = target.resolve()
            require(target == trainer_path,
                    'GDF relocation changes target_file consumed by StageData; explicitly prepare a new manifest first: '+cid)
            target.relative_to((gdf_root/cid).resolve())
            row['_actual_path'] = target
            allowed_arrays.add(str(target))
        if sample_points:
            spec = importlib.util.spec_from_file_location('portable_gdf_reference', Path(project)/'next_stage/gdf.py')
            reference = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = reference
            spec.loader.exec_module(reference)
    else:
        require(not sample_points, 'Numerical GDF sampling requires --gdf-root')
    tracker = dict(active=True,array_open_events=0,unexpected_array_open_attempts=0)

    def audit(event, args):
        if event != 'open' or not tracker['active'] or not args or not isinstance(args[0], (str,bytes)):
            return
        path = Path(os.fsdecode(args[0]))
        if path.suffix.lower() not in ('.npy','.npz'):
            return
        tracker['array_open_events'] += 1
        if str(path.resolve()) not in allowed_arrays:
            tracker['unexpected_array_open_attempts'] += 1
            raise ValueError('Array outside explicit TRAIN file set was opened: '+str(path))

    sys.addaudithook(audit)
    reports, files, total_bytes = [], 0, 0
    try:
        for ordinal,cid in enumerate(splits['train']):
            row, directory = cases[cid], dirs[cid]
            require(row.get('geometry_training_only', True) is True, 'Not TRAIN-only geometry: '+cid)
            hashes = {n:checked_sha(directory/n,row['cache_file_sha256'].get(n))
                      for n in ('image.npy','mask.npy','geometry.npz')}
            preprocessing = row.get('preprocessing_source_sha256')
            require(isinstance(preprocessing,dict) and set(preprocessing) == {
                'prepare.py','geometry.py','common.py','intensity.py'},
                'Complete preprocessing source SHA inventory required: '+cid)
            for name,expected in preprocessing.items():
                checked_sha(Path(project)/'code'/name,expected)
            files += 3
            total_bytes += sum((directory/n).stat().st_size for n in hashes)
            image = mask = target = None
            try:
                image = np.load(directory/'image.npy',mmap_mode='r',allow_pickle=False)
                mask = np.load(directory/'mask.npy',mmap_mode='r',allow_pickle=False)
                shape = tuple(row['shape_zyx'])
                require(len(shape) == 3 and all(type(x) is int and x > 0 for x in shape)
                        and image.shape == mask.shape == shape, 'Image/mask/manifest grids differ: '+cid)
                require(image.dtype.kind in 'iuf' and mask.dtype == np.uint8,
                        'Expected numeric CT and frozen uint8 mask: '+cid)
                if image.dtype.kind == 'f':
                    for begin in range(0,image.size,2_000_000):
                        require(bool(np.isfinite(image.reshape(-1)[begin:begin+2_000_000]).all()),
                                'CT contains nonfinite values: '+cid)
                for begin in range(0,mask.size,2_000_000):
                    block = mask.reshape(-1)[begin:begin+2_000_000]
                    require(bool(((block == 0)|(block == 1)).all()), 'Mask contains nonbinary values: '+cid)
                with np.load(directory/'geometry.npz',allow_pickle=False) as archive:
                    points, radius, structure = [np.asarray(archive[n]) for n in ('points','radius','structure')]
                    valid = bool(np.asarray(archive['geometry_valid']).item()) if 'geometry_valid' in archive else True
                require(points.ndim == 2 and points.shape[1] == 3 and points.dtype.kind in 'iu'
                        and radius.shape == structure.shape == (len(points),)
                        and structure.dtype.kind in 'iu' and bool(np.isin(structure,[1,2,255]).all())
                        and bool(np.isfinite(radius).all()) and bool((radius > 0).all()),
                        'Malformed centerline geometry: '+cid)
                require(bool((points >= 0).all()) and bool((points < np.asarray(shape)).all()),
                        'Centerline point outside mask grid: '+cid)
                require(not len(points) or bool(mask[tuple(points.T)].all()), 'Centerline outside foreground: '+cid)
                spacing = np.asarray(row.get('spacing_mm',manifest['spacing_mm']),dtype=float)
                if spacing.ndim == 0:
                    spacing = np.repeat(spacing,3)
                require(spacing.shape == (3,) and bool(np.isfinite(spacing).all()) and bool((spacing > 0).all()),
                        'Invalid physical spacing: '+cid)
                record = dict(id=cid,cache_dir=str(directory),actual_input_sha256=hashes,shape_zyx=list(shape),
                              image_dtype=str(image.dtype),mask_dtype=str(mask.dtype),spacing_zyx_mm=spacing.tolist(),
                              centerline_points=len(points),geometry_valid=valid,
                              image_finiteness=('all_floating_values_scanned' if image.dtype.kind == 'f' else 'finite_integer_dtype'))
                if gdf:
                    field = fields[cid]
                    require(field.get('complete') is True and field['input_file_sha256'] == {
                        n:hashes[n] for n in ('mask.npy','geometry.npz')}, 'GDF source geometry bytes differ: '+cid)
                    require(field['foreground_geometry_valid'] == (valid and len(points) > 0) and field['shape_zyx'] == list(shape)
                            and np.array_equal(np.asarray(field['spacing_zyx_mm']),spacing)
                            and field['fixed_train_field_scale'] == scale, 'GDF physical grid/validity/scale differs: '+cid)
                    target_sha = checked_sha(field['_actual_path'],field['target_file_sha256'])
                    target = np.load(field['_actual_path'],mmap_mode='r',allow_pickle=False)
                    require(target.shape == shape and str(target.dtype) == field['dtype']
                            and target.dtype in (np.dtype('float16'),np.dtype('float32'))
                            and field['_actual_path'].stat().st_size == field['target_file_bytes'],
                            'GDF target header/size differs: '+cid)
                    record['gdf'] = dict(path=str(field['_actual_path']),sha256=target_sha,
                                         numerical_samples_checked=0,unknown_foreground_not_numerically_verified=True)
                    files += 1
                    total_bytes += field['_actual_path'].stat().st_size
                    if sample_points and valid and len(points):
                        rng = np.random.default_rng(np.random.SeedSequence([seed,ordinal]))
                        chosen = points[rng.choice(len(points),min(sample_points,len(points)),replace=False)]
                        geometry = reference.CumulativeRadiusGeometry(points,radius,spacing,True,
                            reference.GaussianFieldSpec(field_scale=scale))
                        expected = np.log1p(geometry.evaluate_sum(chosen))/scale
                        observed = np.asarray(target[tuple(chosen.T)],dtype=np.float64)
                        rounded = expected.astype(target.dtype)
                        tolerance = np.abs(np.spacing(rounded).astype(np.float64))*.501+1e-7
                        require(bool(np.isfinite(observed).all()) and bool((np.abs(observed-expected) <= tolerance).all()),
                                'GDF sampled foreground disagrees with CPU64 reference: '+cid)
                        record['gdf'].update(numerical_samples_checked=len(chosen),
                            max_sample_abs_error=float(np.abs(observed-expected).max()))
                    if sample_points:
                        rng = np.random.default_rng(np.random.SeedSequence([seed,ordinal,1]))
                        candidate = np.unique(rng.integers(mask.size,size=max(64,sample_points*64)))
                        background = candidate[mask.reshape(-1)[candidate] == 0][:sample_points]
                        values = np.asarray(target.reshape(-1)[background],dtype=np.float64)
                        require(bool(np.isfinite(values).all()) and bool((values == 0).all()),
                                'Sampled known-background GDF values are not zero: '+cid)
                        record['gdf']['known_background_samples_checked'] = len(background)
                reports.append(record)
            finally:
                for array in (image,mask,target):
                    if array is not None:
                        close_map(array)
    finally:
        tracker['active'] = False
    return dict(train_case_count=len(splits['train']),train_ids=splits['train'],validation_ids=splits['val'],
        heldout_case_files_opened=False,unexpected_array_open_attempts=tracker['unexpected_array_open_attempts'],
        array_open_events=tracker['array_open_events'],fully_rehashed_array_files=files,fully_rehashed_array_bytes=total_bytes,
        manifest_sha256=digest(manifest_path),gdf_manifest_sha256=gdf_manifest_sha,cases=reports,
        gdf_numerical_scope=('Centerline seed samples against same-source CPU64 all-seed reference plus fixed background samples; not independent derivation or full-field validation'
                             if sample_points else 'Not performed; GDF validation covers actual bytes, headers and source/grid bindings only'))


def run_qa(project, output, python, source):
    jobs = [dict(name='stage_checks',source_path=MODEL_QA_PATH,
                 command=[python,str(project/MODEL_QA_PATH),
                     '--device','cuda','--patch','128','--batch-size','2','--output',str(output/'stage_checks.json')]),
            dict(name='data_checks',source_path='next_stage/data_checks.py',command=[python,str(project/'next_stage/data_checks.py'),
                     '--project',str(project),'--output',str(output/'data_checks.json')])]
    executions = []
    env = dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for job in jobs:
        result = subprocess.run(job['command'],cwd=project,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        observed = dict(command=job['command'],exit_code=result.returncode,
                        stdout=result.stdout,stderr=result.stderr,source_sha256=source,
                        qa_file_path=job['source_path'],qa_file_sha256=source[job['source_path']],
                        cublas_workspace_config=env['CUBLAS_WORKSPACE_CONFIG'])
        write_new(output/(job['name']+'_actual.json'),observed)
        require(result.returncode == 0, 'Actual QA subprocess exited nonzero: '+job['name'])
        path = output/(job['name']+'.json')
        require(path.is_file(), 'QA exited0 without a result file: '+job['name'])
        qa = read(path)
        require(qa.get('status') == 'passed', 'Actual QA result did not pass: '+job['name'])
        if job['name'] == 'stage_checks':
            require(qa.get('device') == 'cuda' and qa.get('patch_size') == 128 and qa.get('batch_size') == 2
                    and qa.get('autocast_modes_checked') == [False,True]
                    and qa.get('study_data_read') is False and qa.get('model_update_count') == 0
                    and qa.get('amp_backward_manual_loss_scale') == 1024., 'CUDA128 batch2 QA scope differs')
            require(qa.get('qa_file_path') == MODEL_QA_PATH
                    and qa.get('qa_file_sha256') == source[MODEL_QA_PATH]
                    and qa.get('source_sha256') == source, 'Model QA executable/source binding differs')
            phases = qa.get('phases',[])
            require(len(phases) == len(MODEL_QA_PHASES)
                    and {row['name'] for row in phases} == set(MODEL_QA_PHASES)
                    and all(row.get('status') == 'passed' for row in phases), 'Model QA phases incomplete')
            backend = qa.get('backend_settings',{})
            require(backend.get('matmul_tf32') is False and backend.get('cudnn_tf32') is False
                    and backend.get('cudnn_benchmark') is False and backend.get('cudnn_deterministic') is True
                    and backend.get('deterministic_algorithms') is True
                    and backend.get('deterministic_warn_only') in (True,None)
                    and backend.get('cublas_workspace_config') == ':4096:8', 'Model QA backend profile differs')
        else:
            require(qa.get('synthetic_only') is True and qa.get('torch_used') is False
                    and qa.get('test_file_open_attempts') == 0, 'Sampler QA scope differs')
            require(all(source.get(n) == value for n,value in qa['source_sha256'].items()),
                    'Sampler QA actual source differs')
        require(source_inventory(project) == source, 'Source changed during actual QA')
        executions.append(dict(name=job['name'],actual_execution_receipt_sha256=digest(output/(job['name']+'_actual.json')),
                               result_sha256=digest(path),observed_exit_code=result.returncode,
                               qa_file_path=job['source_path'],qa_file_sha256=source[job['source_path']]))
    return executions


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--project',default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--output',required=True,help='New independent preflight directory; never overwritten')
    p.add_argument('--python',default=sys.executable,help='Interpreter for both observed QA subprocesses')
    p.add_argument('--cache-root',help='Only accepted when resolved TRAIN paths equal the original manifest cache_dir paths')
    p.add_argument('--gdf-root')
    p.add_argument('--relocate-gdf-targets',action='store_true',help='Only accepted when GDF_ROOT/CASE/basename equals the trainer original target_file path')
    p.add_argument('--gdf-sample-points',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20261023)
    p.add_argument('--expected-manifest-sha256')
    p.add_argument('--expected-source-inventory',help='Optional JSON SHA map, or object with source_sha256')
    p.add_argument('--mode',choices=('global','local'),default='global')
    p.add_argument('--execute',action='store_true')
    args = p.parse_args(argv)
    if args.mode != 'global':
        p.error('Local-learning preflight is not implemented; no local passed receipt can be produced')
    if not args.execute:
        print(json.dumps(dict(execute=False,mode='global',local_supported=False,manifest=args.manifest,
            project=args.project,output=args.output,QA_python=args.python,TRAIN_arrays_read=False,
            torch_imported=False,CUDA_executed=False,QA_scope='Current-source memory-safe model CUDA128 batch2 FP32+AMP plus synthetic data_checks',
            model_QA_file_path=MODEL_QA_PATH,model_QA_batch_size=2,
            GDF_byte_validation=bool(args.gdf_root),GDF_CPU64_samples_per_case=args.gdf_sample_points,
            relocation_policy='Overrides cannot alter trainer-consumed paths; explicitly revise manifests first if moving data',
            eventual_receipt='OUTPUT/receipt.json, only after actual data checks and observed QA passes',test_used=False)))
        return
    project, output = Path(args.project).resolve(), Path(args.output).resolve()
    for folder in ('code','next_stage'):
        require((project/folder).resolve() not in (output, *output.parents), 'Output must be outside scientific source directories')
    output.mkdir(parents=True,exist_ok=False)
    started = time.time()
    try:
        source = source_inventory(project)
        if args.expected_source_inventory:
            expected = read(args.expected_source_inventory)
            expected = expected.get('source_sha256',expected)
            require(source == expected, 'Current source differs from explicitly pinned inventory')
        manifest_sha = digest(args.manifest)
        if args.expected_manifest_sha256:
            checked_sha(args.manifest,args.expected_manifest_sha256)
        write_new(output/'started.json',dict(source_sha256=source,manifest_sha256=manifest_sha,
            command=sys.argv,preflight_tool_sha256=digest(__file__),mode='global',test_used=False))
        data = inspect_data(args.manifest,project,args.cache_root,args.gdf_root,
                            args.relocate_gdf_targets,args.gdf_sample_points,args.sample_seed)
        write_new(output/'TRAIN_data_audit.json',data)
        qa = run_qa(project,output,args.python,source)
        require(source_inventory(project) == source and digest(args.manifest) == manifest_sha,
                'Source or manifest changed before receipt completion')
        if args.gdf_root:
            require(digest(Path(args.gdf_root)/'manifest.json') == data['gdf_manifest_sha256'],
                    'GDF manifest changed before receipt completion')
        receipt = dict(schema='GTA-AVT-portable-global-preflight-v2',status='passed',mode='global',
            source_sha256=source,manifest_sha256=manifest_sha,train_ids=data['train_ids'],validation_ids=data['validation_ids'],
            actual_TRAIN_array_files_fully_rehashed=data['fully_rehashed_array_files'],
            TRAIN_data_audit_sha256=digest(output/'TRAIN_data_audit.json'),actual_QA=qa,
            actual_QA_subprocess_interpreter=args.python,local_learning_supported=False,
            gdf_numerical_scope=data['gdf_numerical_scope'],allowed_splits=['train','val'],test_used=False,
            heldout_case_files_opened=False,preflight_tool_sha256=digest(__file__),elapsed_seconds=time.time()-started,
            limitations=['VAL metadata only; VAL/native source image bytes were not rehashed.',
                        'A receipt binds observed bytes and QA exits; it is not an independent remote-execution attestation.',
                        'Training must use this same source, data and declared QA runtime; future changes are not certified.'])
        write_new(output/'receipt.json',receipt)
        print(json.dumps(dict(status='passed',receipt=str(output/'receipt.json'),sha256=digest(output/'receipt.json'))))
    except BaseException as error:
        write_new(output/'failure.json',dict(status='failed',error=type(error).__name__+': '+str(error),
                                           elapsed_seconds=time.time()-started,passed_receipt_created=False))
        raise


if __name__ == '__main__':
    main()
