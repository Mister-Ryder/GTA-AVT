"""Convert official AVT NRRD pairs to a reproducible, image-defined crop cache."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from common import atomic_json, load_json, sha256, source_hashes
from geometry import sparse_geometry, SCALE_NAMES, STRUCTURE_NAMES
from intensity import detect_ct_offset


def discover(root):
    cases = []
    for label in sorted(Path(root).rglob('*.seg.nrrd')):
        image = label.with_name(label.name.replace('.seg.nrrd', '.nrrd'))
        if not image.exists():
            raise FileNotFoundError('Unpaired segmentation: ' + str(label))
        relative = label.relative_to(root)
        cohort = next((x for x in relative.parts if x in ('Dongyang', 'KiTS', 'Rider')), relative.parts[0])
        name = image.name[:-5]
        cases.append({'id': cohort + '_' + name, 'cohort': cohort,
                      'image': str(image.resolve()), 'label': str(label.resolve()), 'source_case': name})
    if len(cases) != 56 or len({x['id'] for x in cases}) != 56:
        raise ValueError('Expected 56 uniquely paired AVT cases, got ' + str(len(cases)))
    return cases


def resolve_split(split, cases):
    lookup = {case['id']: case['id'] for case in cases}
    for case in cases:
        for alias in (case['source_case'], case['cohort'] + '/' + case['source_case'],
                      case['cohort'] + ':' + case['source_case']):
            if alias in lookup and lookup[alias] != case['id']:
                continue
            lookup[alias] = case['id']
    output = {}
    source = split.get('splits', split)
    for name in ('train', 'val', 'test'):
        values = source.get(name, source.get('validation') if name == 'val' else None)
        if values is None:
            raise ValueError('Split JSON must contain train, val and test case IDs')
        output[name] = [lookup[x if isinstance(x, str) else x['id']] for x in values]
    flat = sum(output.values(), [])
    if len(flat) != 56 or len(set(flat)) != 56:
        raise ValueError('Patient splits must partition the 56 cases exactly once')
    return output


def reference_grid(image, spacing):
    output = sitk.Image([int(round((n - 1) * s / spacing)) + 1 for n, s in zip(image.GetSize(), image.GetSpacing())], sitk.sitkFloat32)
    output.SetOrigin(image.GetOrigin())
    output.SetDirection(image.GetDirection())
    output.SetSpacing([spacing] * 3)
    return output


def image_body_crop(array, spacing):
    # All z slices are retained; x/y bounds use only the largest >-600 HU body.
    body = array > -600
    components, count = ndi.label(body)
    if count:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        bounds = ndi.find_objects(components)[sizes.argmax() - 1]
    else:
        bounds = tuple(slice(0, n) for n in array.shape)
    margin = int(np.ceil(10. / spacing))
    return (slice(0, array.shape[0]),) + tuple(slice(max(0, x.start - margin), min(n, x.stop + margin))
                                               for x, n in zip(bounds[1:], array.shape[1:]))


def correct_stored_intensity(image, reviewed_override=None):
    """CT-only air evidence, excluding negative padding sentinels.

AVT combines HU and +1024 stored values. Negative Rider padding alone does not
establish HU. Unknown/ambiguous cases fail instead of silently clipping vessels.
"""
    source = sitk.GetArrayFromImage(image)
    audit = detect_ct_offset(source)
    if reviewed_override is not None:
        if not audit['requires_review']:
            raise ValueError('Override unexpectedly targets an automatic CT decision; review audit drift')
        audit.update(original_detector_requires_review=True, original_detector_rule=audit['rule'],
                     offset_subtract=float(reviewed_override['offset_subtract']), requires_review=False,
                     rule='source-hash-locked CT-only reviewed exception', reviewed_override=reviewed_override)
    if audit['requires_review']:
        raise ValueError('Ambiguous stored intensity; inspect CT histogram before preprocessing')
    offset = audit['offset_subtract']
    corrected = source.astype(np.float32) - offset
    corrected[source <= -1500] = -1024.
    result = sitk.GetImageFromArray(corrected)
    result.CopyInformation(image)
    audit.update(padding_sentinel_cutoff=-1500., decision_used_labels=False)
    return result, audit


def prepare_case(case, output, spacing, train, reviewed_override=None, overrides_sha256=None):
    directory = output / case['id']
    directory.mkdir(exist_ok=True)
    record_path = directory / 'record.json'
    if record_path.exists():
        record = load_json(record_path)
        if record['spacing_mm'] != spacing or record['geometry_training_only'] != train:
            raise ValueError('Existing cache protocol differs; use a fresh output')
        if record.get('offset_overrides_sha256') != overrides_sha256:
            raise ValueError('Reviewed intensity-override manifest changed; use a fresh output')
        expected_sources = {name: value for name, value in source_hashes().items() if name in ('prepare.py', 'geometry.py', 'common.py', 'intensity.py')}
        if record.get('preprocessing_source_sha256') != expected_sources:
            raise ValueError('Cache preprocessing implementation changed; use a fresh output')
        if sha256(case['image']) != record['image_sha256'] or sha256(case['label']) != record['label_sha256']:
            raise ValueError('Cache source image or segmentation changed')
        for filename in ('image.npy', 'mask.npy') + (('geometry.npz',) if train else ()):
            if not (directory / filename).exists():
                raise ValueError('Incomplete cache record: ' + str(directory))
            if record.get('cache_file_sha256', {}).get(filename) != sha256(directory / filename):
                raise ValueError('Cache bytes changed or are unverified: ' + str(directory / filename))
        return record
    started = time.time()
    image_native = sitk.ReadImage(case['image'], sitk.sitkFloat32)
    label_native = sitk.ReadImage(case['label'])
    if image_native.GetDimension() != 3 or label_native.GetDimension() != 3:
        raise ValueError('Only scalar 3D CT/mask images are supported')
    if reviewed_override is not None and sha256(case['image']) != reviewed_override['image_sha256']:
        raise ValueError('Reviewed CT offset exception source hash mismatch: ' + case['id'])
    corrected, intensity_audit = correct_stored_intensity(image_native, reviewed_override)
    image = sitk.DICOMOrient(corrected, 'LPS')
    grid = reference_grid(image, spacing)
    ct = sitk.Resample(image, grid, sitk.Transform(), sitk.sitkLinear, -1024., sitk.sitkFloat32)
    mask = sitk.Resample(label_native > 0, grid, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    array, target = sitk.GetArrayFromImage(ct), sitk.GetArrayFromImage(mask)
    if not target.any():
        raise ValueError('Empty or physically misaligned mask: ' + case['id'])
    crop = image_body_crop(array, spacing)
    if target[crop].sum() != target.sum():
        raise ValueError('Image-only crop excludes annotated vessel; inspect source before changing protocol: ' + case['id'])
    ct_crop = np.clip(np.rint(array[crop]), -1024, 2000).astype(np.int16)
    mask_crop = target[crop].astype(np.uint8)
    np.save(directory / 'image.npy', ct_crop)
    np.save(directory / 'mask.npy', mask_crop)
    roundtrip_qc = None
    if train:
        native_mask = sitk.GetArrayFromImage(label_native) > 0
        native_spacing = np.array(label_native.GetSpacing())[::-1]
        native_radius = ndi.distance_transform_edt(native_mask, sampling=native_spacing).astype(np.float32)
        radius_image = sitk.GetImageFromArray(native_radius)
        radius_image.CopyInformation(label_native)
        resampled_radius = sitk.GetArrayFromImage(sitk.Resample(radius_image, grid, sitk.Transform(), sitk.sitkLinear, 0., sitk.sitkFloat32))
        geometry = sparse_geometry(mask_crop, np.array([spacing] * 3), radius_field=resampled_radius[crop])
        np.savez_compressed(directory / 'geometry.npz', **geometry)
        recovered = sitk.GetArrayFromImage(sitk.Resample(mask, label_native, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)) > 0
        from skimage.morphology import skeletonize
        native_skeleton = skeletonize(native_mask, method='lee').astype(bool)
        thin = native_skeleton & (native_radius <= 2.)
        roundtrip_qc = {'native_foreground_voxels': int(native_mask.sum()),
                        'resampled_foreground_voxels': int(target.sum()),
                        'native_roundtrip_dice': float(2. * (recovered & native_mask).sum() / max(int(recovered.sum() + native_mask.sum()), 1)),
                        'native_centerline_coverage': float(recovered[native_skeleton].mean()) if native_skeleton.any() else None,
                        'native_centerline_radius_le2mm_coverage': float(recovered[thin].mean()) if thin.any() else None,
                        'native_thin_centerline_support': int(thin.sum())}
    record = dict(case, spacing_mm=spacing, shape_zyx=list(ct_crop.shape),
                  full_shape_zyx=list(array.shape), crop_start_zyx=[x.start for x in crop],
                  crop_stop_zyx=[x.stop for x in crop], cache_dir=str(directory.resolve()),
                  resampled_origin=grid.GetOrigin(), resampled_direction=grid.GetDirection(),
                  resampled_spacing=grid.GetSpacing(), resampled_size=grid.GetSize(),
                  original_origin=image_native.GetOrigin(), original_direction=image_native.GetDirection(),
                  original_spacing=image_native.GetSpacing(), original_size=image_native.GetSize(),
                  image_sha256=sha256(case['image']), label_sha256=sha256(case['label']),
                  preprocessing_source_sha256={name: value for name, value in source_hashes().items() if name in ('prepare.py', 'geometry.py', 'common.py', 'intensity.py')},
                  intensity_audit=intensity_audit, native_roundtrip_qc=roundtrip_qc,
                  offset_overrides_sha256=overrides_sha256,
                  cache_file_sha256={name: sha256(directory / name) for name in
                                      ('image.npy', 'mask.npy') + (('geometry.npz',) if train else ())},
                  geometry_training_only=train, foreground_voxels=int(target.sum()),
                  preprocessing_seconds=time.time() - started)
    atomic_json(record_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--spacing', type=float, default=1.5)
    parser.add_argument('--limit', type=int, default=0, help='Diagnostic only; incomplete cache cannot train formally')
    parser.add_argument('--offset-overrides', help='Reviewed CT-only exceptions locked to original image SHA256')
    args = parser.parse_args()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cases = discover(args.data_root)
    split = resolve_split(load_json(args.split), cases)
    overrides, overrides_sha256 = {}, None
    if args.offset_overrides:
        override_manifest = load_json(args.offset_overrides)
        if override_manifest.get('detector_sha256') != sha256(Path(__file__).parent / 'intensity.py'):
            raise ValueError('Reviewed offset exceptions were produced with a different detector')
        overrides = {x['pipeline_id']: x for x in override_manifest['overrides']}
        if len(overrides) != len(override_manifest['overrides']) or not set(overrides).issubset({x['id'] for x in cases}):
            raise ValueError('Unknown or duplicate CT offset exceptions')
        if any(x['offset_subtract'] not in (0., 1024.) for x in overrides.values()):
            raise ValueError('Unsupported reviewed offset')
        overrides_sha256 = sha256(args.offset_overrides)
    records = []
    # Training records first permit prompt pipeline validation while later cases process.
    cases.sort(key=lambda x: (x['id'] not in split['train'], x['id']))
    for case in cases[:args.limit or None]:
        record = prepare_case(case, output, args.spacing, case['id'] in split['train'],
                              overrides.get(case['id']), overrides_sha256)
        records.append(record)
        print(json.dumps({'prepared': case['id'], 'shape': record['shape_zyx'],
                          'seconds': record['preprocessing_seconds']}), flush=True)
        atomic_json(output / 'manifest.json', {'protocol': 'GTA-AVT-v1', 'split_sha256': sha256(args.split),
                    'spacing_mm': args.spacing, 'splits': split, 'cases': records,
                    'complete': len(records) == 56, 'scale_labels': SCALE_NAMES, 'structure_labels': STRUCTURE_NAMES,
                    'offset_overrides_sha256': overrides_sha256,
                    'crop_definition': 'largest CT body >-600 HU xy box plus10mm; retain full z'})


if __name__ == '__main__':
    main()
