"""Fixed TRAIN-only boundary/separability diagnostic; no optimization or GT model input.

Default is inert and imports only stdlib. Coordinates and physical proxy labels
come from the original fixed bank, while the model consumes normalized CT only.
"""
import argparse
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
BANK_SHA = '9be42890017341e4cc32fb0f17688d859a13e84643f95cd21e347462043419d6'
WEIGHT_SHA = '7406a3ba7e5bac1a7d8ee24898cb954ff8576f37fcf5fba0e26ef9a2cb5c43e6'
MAP_SHA = 'f311c9fbe6ea55be70e72ed220ab0562da08b3ee267863fc0d78509a30c3654e'
SETTINGS = dict(patch_size=128, cell_size=16, shell_mm=2., halo_voxels=2,
    FOV_censor_voxels=2, minimum_shell_voxels=16, HU_clip=[-200, 1000],
    batch_size=2, AMP=True, probability_threshold=.5, spacing_mm=1.,
    conditioning='constant', alpha=0., q0=.2, s0=.2, augmentation=False)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + '\n').encode('utf-8')


def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value))


def rank_auc(inside_hist, outside_hist):
    """P(HU_inside>HU_outside)+.5P(tie), exact integer pair count."""
    ni, no = sum(inside_hist), sum(outside_hist)
    if not ni or not no:
        return None
    twice_wins, below = 0, 0
    for inside, outside in zip(inside_hist, outside_hist):
        twice_wins += int(inside) * (2 * below + int(outside))
        below += int(outside)
    return twice_wins / float(2 * ni * no)


def scalar_summary(values):
    values = [value for value in values if value is not None]
    return dict(support=len(values), mean=statistics.mean(values) if values else None,
                within_patch_SD=statistics.pstdev(values) if values else None)


def state_sha(model):
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().contiguous().cpu()
        result.update(name.encode()); result.update(str(tensor.dtype).encode())
        result.update(str(tuple(tensor.shape)).encode()); result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def metadata(request):
    require(request.get('schema') == 'GTA-AVT-boundary-attribute-probe-request-v1', 'Wrong request schema')
    require(request['settings'] == SETTINGS, 'Fixed diagnostic settings differ')
    require(request['bank_sha256'] == BANK_SHA and sha(request['bank']) == BANK_SHA, 'Old bank differs')
    require(request['weights_sha256'] == WEIGHT_SHA and sha(request['weights']) == WEIGHT_SHA, 'Common7406 differs')
    require(request['source_manifest_sha256'] == MAP_SHA and sha(request['source_manifest']) == MAP_SHA,
            'Exact source42 map required')
    source = read(request['source_manifest'])['source_sha256']
    require(len(source) == 42, 'Expected source42')
    root = Path(request['source_root']).resolve()
    for name, digest in source.items():
        rel = PurePosixPath(name)
        require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in name and ':' not in name,
                'Unsafe source path')
        path = root.joinpath(*rel.parts)
        require(path.resolve() == path and path.is_file() and sha(path) == digest, 'Source differs: ' + name)
    require(sha(request['manifest']) == request['manifest_sha256'], 'Input manifest differs')
    manifest, bank = read(request['manifest']), read(request['bank'])
    train = manifest['splits']['train']; selected = request['case_ids']
    require(len(train) == 36 and bank['case_ids'] == train and len(bank['patches']) == 576, 'TRAIN bank identity differs')
    require(len(selected) == 12 and len(set(selected)) == 12 and set(selected) <= set(train), 'Exactly12 TRAIN cases required')
    require(not set(selected) & (set(manifest['splits']['val']) | set(manifest['splits']['test'])), 'Non-TRAIN access forbidden')
    require([(r['case_id'], r['slot']) for r in bank['patches']] == [(c, s) for c in train for s in range(16)],
            'Original bank order differs')
    records = {row['id']: row for row in manifest['cases']}
    for case in selected:
        require(records[case]['cache_file_sha256'] == bank['expected_TRAIN_cache_sha256'][case], 'Cache identity differs')
    return root, source, manifest, bank, records


def cached_inputs(request, records):
    result = {}
    for case in request['case_ids']:
        home = Path(records[case]['cache_dir']).resolve()
        result[case] = {}
        for name in ('image.npy', 'mask.npy', 'geometry.npz'):
            path = home / name
            expected = records[case]['cache_file_sha256'][name]
            require(path.is_file() and path.resolve() == path and sha(path) == expected, 'TRAIN cache differs: ' + case + '/' + name)
            result[case][name] = dict(sha256=expected, bytes=path.stat().st_size)
    return result


def patch_arrays(np, ndi, image, mask, start):
    """EDT on the actual neighboring mask, then crop; do not EDT each cell."""
    shape = np.asarray(image.shape); start = np.asarray(start)
    lo = np.maximum(start - 2, 0); hi = np.minimum(start + 130, shape)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    halo = np.asarray(mask[slices], dtype=bool)
    pad = [(int(a - b), int(c - d)) for a, b, c, d in zip(lo, start - 2, start + 130, hi)]
    halo = np.pad(halo, pad, constant_values=False)
    require(halo.shape == (132, 132, 132), 'Halo shape differs')
    inside = ndi.distance_transform_edt(halo)[2:130, 2:130, 2:130]
    outside = ndi.distance_transform_edt(~halo)[2:130, 2:130, 2:130]
    slices = tuple(slice(int(a), min(int(a) + 128, int(n))) for a, n in zip(start, shape))
    raw = np.asarray(image[slices]); truth = np.asarray(mask[slices], dtype=bool)
    require(raw.dtype.kind in 'iu' and raw.shape == truth.shape, 'Expected integral raw cached HU')
    pad = [(0, 128 - n) for n in raw.shape]
    raw = np.pad(raw, pad, constant_values=-1024); truth = np.pad(truth, pad)
    real = np.ones((128, 128, 128), bool); uncensored = real.copy()
    for axis in range(3):
        coords = start[axis] + np.arange(128)
        form = [1, 1, 1]; form[axis] = 128
        real &= (coords < shape[axis]).reshape(form)
        uncensored &= ((coords >= 2) & (coords < shape[axis] - 2)).reshape(form)
    inner = uncensored & truth & (inside > 0) & (inside <= 2.)
    outer = uncensored & ~truth & (outside > 0) & (outside <= 2.)
    return raw, truth, real, uncensored, inner, outer


def boundary_values(np, raw, truth, logits, inner, outer, binary=None):
    ni, no = int(inner.sum()), int(outer.sum())
    row = dict(inside_shell_count=ni, outside_shell_count=no, reliable=ni >= 16 and no >= 16,
               support_missing_reason=None, AUC=None, signed_separation=None, S=None, B=None,
               boundary_error=None, boundary_BCE=None, boundary_bias=None, inside_FNR=None, outside_FPR=None)
    if not row['reliable']:
        row['support_missing_reason'] = ('inside_and_outside_shell_lt16' if ni < 16 and no < 16 else
            'inside_shell_lt16' if ni < 16 else 'outside_shell_lt16')
        return row
    clipped = np.clip(raw, -200, 1000).astype(np.int16) + 200
    hist_i = np.bincount(clipped[inner], minlength=1201)
    hist_o = np.bincount(clipped[outer], minlength=1201)
    auc = rank_auc(hist_i.tolist(), hist_o.tolist()); signed = 2. * auc - 1.
    # Production passes the actual torch FP32 sigmoid >=.5 decision, preserving
    # probability rounding near zero. Logits are retained for stable BCE only.
    binary = (logits >= 0.) if binary is None else binary
    fnr = float(np.mean(~binary[inner])); fpr = float(np.mean(binary[outer]))
    inside_bce = np.logaddexp(0., -logits[inner].astype(np.float64)).mean()
    outside_bce = np.logaddexp(0., logits[outer].astype(np.float64)).mean()
    row.update(AUC=auc, signed_separation=signed, S=abs(signed), B=1. - abs(signed),
        boundary_error=.5 * (fnr + fpr), boundary_BCE=float(.5 * (inside_bce + outside_bce)),
        boundary_bias=.5 * (fnr - fpr), inside_FNR=fnr, outside_FPR=fpr)
    return row


def cells(np, geometry, spec, arrays, logits, binary):
    raw, truth, real, uncensored, inner, outer = arrays
    point = geometry['points']; start = np.asarray(spec['start_zyx'])
    relative = point - start
    valid_points = ((relative >= 0) & (relative < 128)).all(1)
    point_ids = np.full(len(point), -1, dtype=np.int32)
    groups = relative[valid_points] // 16
    point_ids[valid_points] = groups[:, 0] * 64 + groups[:, 1] * 8 + groups[:, 2]
    rows = []
    for z in range(8):
        for y in range(8):
            for x in range(8):
                index = z * 64 + y * 8 + x
                sl = tuple(slice(v * 16, (v + 1) * 16) for v in (z, y, x))
                row = dict(case_id=spec['case_id'], source=spec['source'], slot=spec['slot'],
                    stratum=spec['stratum'], cell_index=index, cell_zyx=[z, y, x],
                    real_voxels=int(real[sl].sum()), censored_voxels=int((real[sl] & ~uncensored[sl]).sum()))
                row.update(boundary_values(np, raw[sl], truth[sl], logits[sl], inner[sl], outer[sl], binary[sl]))
                row['cell_FG_fraction'] = float(truth[sl][real[sl]].mean()) if row['real_voxels'] else None
                chosen = point_ids == index; radius = geometry['radius'][chosen]
                row.update(local_skeleton_points=int(chosen.sum()), radius_CDF=None,
                           GT_local_ordinal=None, GT_local_fine=None, structure_target=None)
                if len(radius):
                    cdf = [float(np.mean(radius <= threshold)) for threshold in (2., 4., 8.)]
                    row.update(radius_CDF=cdf, GT_local_ordinal=sum(cdf) / 3., GT_local_fine=cdf[0])
                    structure = geometry['structure'][chosen]; structure = structure[structure != 255]
                    if len(structure):
                        counts = np.bincount(structure, minlength=3)
                        row['structure_target'] = (counts / counts.sum()).tolist()
                rows.append(row)
    return rows


def execute(args, request):
    root, source, manifest, bank, records = metadata(request)
    output = Path(args.output); require(not output.exists(), 'Fresh output required')
    before_inputs = cached_inputs(request, records)
    import numpy as np
    import torch
    from scipy import ndimage as ndi
    for path in (root / 'code', root / 'next_stage'):
        sys.path.insert(0, str(path))
    modules = {name: importlib.import_module(name) for name in ('data', 'geometry', 'stage_model')}
    for name, rel in (('data', 'code/data.py'), ('geometry', 'code/geometry.py'), ('stage_model', 'next_stage/stage_model.py')):
        require(Path(modules[name].__file__).resolve() == (root / rel).resolve(), 'Wrong imported scientific module')
    require(torch.cuda.is_available(), 'CUDA diagnostic required')
    torch.set_num_threads(4); torch.manual_seed(20261023); torch.cuda.manual_seed_all(20261023)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = modules['stage_model'].StageModel('constant', alpha=0., q0=.2, s0=.2)
    weights = torch.load(request['weights'], map_location='cpu')
    loaded = model.load_state_dict(weights, strict=False); del weights
    require(set(loaded.missing_keys) == {'gdf_head.weight', 'gdf_head.bias'} and not loaded.unexpected_keys,
            'Unexpected common checkpoint shape')
    model.cuda().eval(); before_state = state_sha(model)
    head = {}
    if request.get('head_reference'):
        require(sha(request['head_reference']) == request['head_reference_sha256'], 'Head reference differs')
        head = {(r['case_id'], r['slot']): r['normalized_CT_bytes_sha256'] for r in read(request['head_reference'])['rows']}
    output.mkdir(); (output / 'cases').mkdir()
    with (output / 'request.json').open('xb') as stream:
        stream.write(Path(args.request).read_bytes())
    patch_count, cell_count = 0, 0
    with (output / 'patches.jsonl').open('xb') as patch_stream:
        for case in request['case_ids']:
            home = Path(records[case]['cache_dir'])
            image = np.load(home / 'image.npy', mmap_mode='r', allow_pickle=False)
            mask = np.load(home / 'mask.npy', mmap_mode='r', allow_pickle=False)
            with np.load(home / 'geometry.npz', allow_pickle=False) as value:
                geometry = {key: value[key] for key in value.files}
            require(image.shape == mask.shape and list(image.shape) == bank['headers'][case]['shape_zyx'], 'Cache shape differs')
            require(bool(np.asarray(geometry.get('geometry_valid', True)).item()), 'Invalid geometry')
            specs = [row for row in bank['patches'] if row['case_id'] == case]
            with gzip.open(str(output / 'cases' / (case + '.cells.jsonl.gz')), 'xb', compresslevel=6) as stream:
                for offset in range(0, 16, 2):
                    pending, inputs = [], []
                    for spec in specs[offset:offset + 2]:
                        arrays = patch_arrays(np, ndi, image, mask, spec['start_zyx'])
                        normalized = np.ascontiguousarray(modules['data'].normalize(arrays[0].astype(np.float32))[None], dtype=np.float32)
                        ct_sha = hashlib.sha256(normalized.tobytes()).hexdigest()
                        require(not head or head[(case, spec['slot'])] == ct_sha, 'Fixed normalized CT differs')
                        inputs.append(normalized); pending.append((spec, arrays, ct_sha))
                    tensor = torch.from_numpy(np.stack(inputs)).cuda()
                    with torch.inference_mode(), torch.cuda.amp.autocast():
                        predicted = model(tensor)['logits'].float()
                    require(bool(torch.isfinite(predicted).all()), 'Nonfinite model logits')
                    logits = predicted[:, 0].cpu().numpy()
                    binary = (torch.sigmoid(predicted[:, 0]) >= .5).cpu().numpy(); del predicted, tensor
                    for (spec, arrays, ct_sha), predicted, predicted_binary in zip(pending, logits, binary):
                        rows = cells(np, geometry, spec, arrays, predicted, predicted_binary)
                        for row in rows:
                            stream.write(encoded(row))
                        scale, structure = modules['geometry'].patch_histograms(geometry, spec['start_zyx'], [128] * 3)
                        valid = bool(scale.sum() > 0 and not (arrays[1].any() and scale[0] == 1.))
                        whole = boundary_values(np, arrays[0], arrays[1], predicted, arrays[4], arrays[5], predicted_binary)
                        patch = dict(case_id=case, source=spec['source'], slot=spec['slot'], stratum=spec['stratum'],
                            role=spec['role'], start_zyx=spec['start_zyx'], normalized_CT_bytes_sha256=ct_sha,
                            scale_target=scale.tolist(), structure_target=structure.tolist(), attribute_valid=valid,
                            known_nonempty=bool(valid and arrays[1].any()), valid_empty=bool(valid and not arrays[1].any()),
                            GT_ordinal=float(scale[1] + (2. * scale[2] + scale[3]) / 3.) if valid else None,
                            GT_fine=float(scale[1]) if valid else None, cells=512, reliable_cells=sum(r['reliable'] for r in rows),
                            whole_patch_boundary=whole, cell_summary={key: scalar_summary([r[key] for r in rows])
                                for key in ('S', 'B', 'signed_separation', 'boundary_error', 'boundary_BCE', 'boundary_bias', 'GT_local_ordinal')})
                        patch_stream.write(encoded(patch)); patch_count += 1; cell_count += len(rows)
    after_state = state_sha(model); require(before_state == after_state, 'Model state mutated')
    after_inputs = cached_inputs(request, records); require(before_inputs == after_inputs, 'TRAIN cache changed')
    require(sha(request['weights']) == WEIGHT_SHA, 'Common weights changed')
    for name, digest in source.items():
        require(sha(root / name) == digest, 'Scientific source changed')
    write(output / 'inputs.json', dict(schema='GTA-AVT-boundary-attribute-probe-inputs-v1',
        request_sha256=sha(args.request), source_manifest_sha256=MAP_SHA, source_sha256=source,
        bank_sha256=BANK_SHA, weights_sha256=WEIGHT_SHA, case_ids=request['case_ids'],
        fit_cases=bank['fit_cases'], probe_cases=bank['probe_cases'], cache_before=before_inputs,
        cache_after=after_inputs, model_state_before=before_state, model_state_after=after_state,
        no_VAL_or_TEST_arrays=True, no_optimization=True))
    files = {p.relative_to(output).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
             for p in sorted(output.rglob('*')) if p.is_file()}
    require(patch_count == 192 and cell_count == 192 * 512, 'Incomplete fixed bank diagnostic')
    summary = dict(schema='GTA-AVT-boundary-attribute-probe-v1', status='verified_complete',
        case_ids=request['case_ids'], patch_count=patch_count, cell_count=cell_count,
        files=files, request_sha256=sha(args.request), settings=SETTINGS, model_state_unchanged=True,
        no_optimization=True, no_VAL_or_TEST_arrays=True, gpu_max_memory_allocated=int(torch.cuda.max_memory_allocated()),
        elapsed_seconds=time.monotonic() - args.started)
    write(output / 'summary.json', summary)
    return dict(status=summary['status'], cases=12, patches=patch_count, cells=cell_count,
                summary_sha256=sha(output / 'summary.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request'); parser.add_argument('--output'); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='inert', settings=SETTINGS))); return
    require(args.request and args.output, 'Request and fresh output required')
    args.started = time.monotonic()
    print(json.dumps(execute(args, read(args.request)), allow_nan=False))


if __name__ == '__main__':
    main()
