"""Portable CT-only preprocessing and TRAIN-derived geometric supervision.

Files containing labels are opened only by ``prepare_dataset``. ``prepare_ct``
and native restoration accept no segmentation, geometry or annotated ROI.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .intensity import detect_ct_offset


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalize(image):
    """Historical AVT HU window, identical during training and inference."""
    return (np.clip(np.asarray(image, np.float32), -200., 1000.) - 200.) / 400.


def index_avt(root, output, split_file=None):
    """Index official NRRD pairs; leave split unset for explicit assignment.

    The generated manifest cannot be prepared until every split is assigned.
    No random hold-out set is silently constructed.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    rows = []
    for label in sorted(root.rglob('*.seg.nrrd')):
        image = label.with_name(label.name.replace('.seg.nrrd', '.nrrd'))
        if not image.is_file():
            raise FileNotFoundError(f'Unpaired segmentation: {label}')
        relative = label.relative_to(root)
        cohort = next((p for p in relative.parts if p in ('Dongyang', 'KiTS', 'Rider')), relative.parts[0])
        case_id = cohort + '_' + image.name[:-5]
        rows.append(dict(id=case_id, image=str(image), label=str(label), split=None, source=cohort))
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('No paired cases or duplicate case IDs')
    if split_file is not None:
        assigned = json.loads(Path(split_file).read_text(encoding='utf-8-sig'))
        assigned = assigned.get('splits', assigned)
        lookup = {r['id']: r for r in rows}
        for row in rows:
            basename = Path(row['image']).name[:-5]
            for alias in (basename, row['source'] + '/' + basename, row['source'] + ':' + basename):
                if alias in lookup and lookup[alias] is not row:
                    raise ValueError(f'Ambiguous split alias: {alias}')
                lookup[alias] = row
        seen = set()
        for name in ('TRAIN', 'VAL', 'TEST'):
            values = assigned.get(name.lower(), assigned.get(name))
            if name == 'VAL' and values is None:
                values = assigned.get('validation')
            if values is None:
                raise ValueError('Split file must explicitly contain train, val and test')
            for value in values:
                key = value if isinstance(value, str) else value['id']
                if key not in lookup or lookup[key]['id'] in seen:
                    raise ValueError(f'Unknown or duplicate split case: {key}')
                lookup[key]['split'] = name
                seen.add(lookup[key]['id'])
        if len(seen) != len(rows):
            raise ValueError('Split file must partition all indexed cases exactly once')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(schema='gta-avt-cases-v1', cases=rows,
                  split_instruction='Assign each split explicitly to TRAIN, VAL or TEST before prepare.')
    _json(output, result)
    return result


def _source_cases(path):
    path = Path(path).resolve()
    obj = json.loads(path.read_text(encoding='utf-8'))
    cases = obj if isinstance(obj, list) else obj['cases']
    ids, images, labels = set(), set(), set()
    rows = []
    for original in cases:
        row = dict(original)
        case_id = row.get('id')
        if not isinstance(case_id, str) or not case_id or any(c in case_id for c in '/\\') or case_id in ('.', '..'):
            raise ValueError('Case IDs must be safe, unique directory names')
        split = row.get('split')
        if not isinstance(split, str) or split.upper() not in ('TRAIN', 'VAL', 'TEST'):
            raise ValueError(f'Explicit TRAIN/VAL/TEST split required for {case_id}')
        row['split'] = split.upper()
        for key, seen in (('image', images), ('label', labels)):
            source = Path(row[key])
            source = (path.parent / source).resolve() if not source.is_absolute() else source.resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            if source in seen:
                raise ValueError(f'Duplicate {key} source: {source}')
            seen.add(source)
            row[key] = str(source)
        if case_id in ids:
            raise ValueError(f'Duplicate case ID: {case_id}')
        ids.add(case_id)
        rows.append(row)
    if not rows or not any(r['split'] == 'TRAIN' for r in rows):
        raise ValueError('At least one explicitly assigned TRAIN case is required')
    return rows


def reference_grid(image, spacing):
    import SimpleITK as sitk
    grid = sitk.Image([int(round((n - 1) * s / spacing)) + 1
                       for n, s in zip(image.GetSize(), image.GetSpacing())], sitk.sitkFloat32)
    grid.SetOrigin(image.GetOrigin())
    grid.SetDirection(image.GetDirection())
    grid.SetSpacing([float(spacing)] * 3)
    return grid


def image_body_crop(array, spacing):
    from scipy import ndimage as ndi
    components, count = ndi.label(array > -600.)
    if count:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        bounds = ndi.find_objects(components)[sizes.argmax() - 1]
    else:
        bounds = tuple(slice(0, n) for n in array.shape)
    margin = int(np.ceil(10. / spacing))
    return (slice(0, array.shape[0]),) + tuple(
        slice(max(0, b.start - margin), min(n, b.stop + margin))
        for b, n in zip(bounds[1:], array.shape[1:]))


def _correct_image(original, source_path, reviewed_override=None):
    import SimpleITK as sitk
    source = sitk.GetArrayFromImage(original)
    if not np.isfinite(source).all():
        raise ValueError('Nonfinite CT intensities are unsupported')
    audit = detect_ct_offset(source)
    if reviewed_override is not None:
        if not audit['requires_review']:
            raise ValueError('Override targets an automatic intensity decision; inspect detector drift')
        if reviewed_override.get('image_sha256') != _sha(source_path):
            raise ValueError('Intensity override must match this image SHA256')
        offset = reviewed_override.get('offset_subtract')
        if offset not in (0, 0., 1024, 1024.):
            raise ValueError('Reviewed offset must be 0 or 1024')
        audit.update(offset_subtract=float(offset), requires_review=False,
                     rule='source-hash-locked CT-only reviewed exception')
    if audit['requires_review']:
        raise ValueError('Ambiguous CT intensity convention; provide a reviewed image-hash-locked override')
    corrected = source.astype(np.float32) - audit['offset_subtract']
    corrected[source <= -1500.] = -1024.
    image = sitk.GetImageFromArray(corrected)
    image.CopyInformation(original)
    valid = sitk.GetImageFromArray((source > -1500.).astype(np.uint8))
    valid.CopyInformation(original)
    audit.update(padding_sentinel_cutoff=-1500., decision_used_labels=False)
    return image, valid, audit


def prepare_ct(image_path, output_dir=None, *, case_id=None, spacing=1.0, reviewed_override=None):
    """Return integer HU cache and native-restoration metadata from CT alone."""
    import SimpleITK as sitk
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Positive finite spacing required')
    path = Path(image_path).resolve()
    original = sitk.ReadImage(str(path), sitk.sitkFloat32)
    if original.GetDimension() != 3 or original.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError('One scalar 3D CT image required')
    corrected, native_valid, audit = _correct_image(original, path, reviewed_override)
    image = sitk.DICOMOrient(corrected, 'LPS')
    grid = reference_grid(image, spacing)
    resampled = sitk.Resample(image, grid, sitk.Transform(), sitk.sitkLinear, -1024., sitk.sitkFloat32)
    valid_image = sitk.Resample(native_valid, grid, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    array = sitk.GetArrayFromImage(resampled)
    crop = image_body_crop(array, spacing)
    cached = np.clip(np.rint(array[crop]), -1024, 2000).astype(np.int16)
    valid = sitk.GetArrayFromImage(valid_image)[crop].astype(np.uint8)
    metadata = dict(schema='gta-avt-ct-v1', id=case_id or path.stem, spacing_mm=float(spacing),
        shape_zyx=list(cached.shape), full_shape_zyx=list(array.shape),
        crop_start_zyx=[s.start for s in crop], crop_stop_zyx=[s.stop for s in crop],
        resampled_origin=grid.GetOrigin(), resampled_direction=grid.GetDirection(),
        resampled_spacing=grid.GetSpacing(), resampled_size=grid.GetSize(),
        original_origin=original.GetOrigin(), original_direction=original.GetDirection(),
        original_spacing=original.GetSpacing(), original_size=original.GetSize(),
        image_sha256=_sha(path), intensity_audit=audit,
        crop_definition='largest CT component >-600HU XY bbox plus10mm; fullZ',
        label_required=False, geometry_required=False)
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        np.save(output / 'image.npy', cached, allow_pickle=False)
        np.save(output / 'valid.npy', valid, allow_pickle=False)
        _json(output / 'record.json', metadata)
    else:
        # Internal caller can use validity without serializing a large JSON array.
        metadata['_valid_array'] = valid
    return cached, metadata


def restore_native(probability, metadata, output_path=None, threshold=None):
    """Restore native physical header; apply optional threshold after resampling."""
    import SimpleITK as sitk
    probability = np.asarray(probability, np.float32)
    if list(probability.shape) != metadata['shape_zyx'] or not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError('Finite cropped probabilities in [0,1] required')
    full = np.zeros(metadata['full_shape_zyx'], np.float32)
    crop = tuple(slice(a, b) for a, b in zip(metadata['crop_start_zyx'], metadata['crop_stop_zyx']))
    full[crop] = probability
    image = sitk.GetImageFromArray(full)
    image.SetOrigin(metadata['resampled_origin'])
    image.SetDirection(metadata['resampled_direction'])
    image.SetSpacing(metadata['resampled_spacing'])
    reference = sitk.Image([int(n) for n in metadata['original_size']], sitk.sitkFloat32)
    reference.SetOrigin(metadata['original_origin'])
    reference.SetDirection(metadata['original_direction'])
    reference.SetSpacing(metadata['original_spacing'])
    native = sitk.Resample(image, reference, sitk.Transform(), sitk.sitkLinear, 0., sitk.sitkFloat32)
    if threshold is not None:
        if not 0. <= threshold <= 1.:
            raise ValueError('Threshold must be in [0,1]')
        native = sitk.Cast(native >= threshold, sitk.sitkUInt8)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(native, str(output_path), True)
    return native


def _load_geometry(directory):
    with np.load(Path(directory) / 'geometry.npz', allow_pickle=False) as stream:
        return {key: stream[key] for key in stream.files}


def train_statistics(rows, output_dir, *, sample_per_case=4096, gdf_sample_per_case=256, seed=20261024):
    """Case-equal population moments of fixed TRAIN-only samples.

    Radius nodes must satisfy the same partial oracle as supervision. Availability
    is sampled uniformly on the nominal half-resolution grid, including A=0.
    Gaussian RMS uses fixed foreground samples, not a per-volume maximum.
    """
    from .geometry import radius_at_queries, CumulativeRadiusGeometry, background_stencil_certificate
    radius_means, radius_squares, availability, gdf_squares = [], [], [], []
    used = []
    for row_index, row in enumerate(sorted((r for r in rows if r['split'] == 'TRAIN'), key=lambda r: r['id'])):
        directory = Path(output_dir) / row['cache_dir']
        mask = np.load(directory / 'mask.npy', mmap_mode='r', allow_pickle=False).astype(bool)
        real = np.load(directory / 'valid.npy', mmap_mode='r', allow_pickle=False).astype(bool)
        geometry = _load_geometry(directory)
        spacing = np.repeat(float(row['spacing_mm']), 3)
        rng = np.random.default_rng(np.random.SeedSequence([seed, row_index]))
        grid = np.maximum(np.asarray(mask.shape) // 2, 1)
        total = int(np.prod(grid))
        chosen = rng.choice(total, min(sample_per_case, total), replace=False)
        indices = np.stack(np.unravel_index(chosen, grid), axis=1)
        queries = 2. * indices + .5
        result = radius_at_queries(geometry, queries, mask.shape, spacing)
        valid_grid = background_stencil_certificate(~real, queries)
        measured = result['R_valid'] & valid_grid
        if valid_grid.any():
            availability.append(float(measured[valid_grid].mean()))
        # Radius moments use reliable physical skeleton samples to avoid the
        # rare-support instability of uniform background-heavy grid samples.
        points = np.asarray(geometry['points'])
        node_result = radius_at_queries(geometry, points, mask.shape, spacing) if len(points) else None
        eligible = np.flatnonzero(node_result['R_valid']) if node_result is not None else np.empty(0, int)
        if len(eligible):
            chosen_nodes = rng.choice(eligible, min(sample_per_case, len(eligible)), replace=False)
            values = node_result['rawR'][chosen_nodes]
            radius_means.append(float(values.mean()))
            radius_squares.append(float(np.mean(values ** 2)))
        foreground = np.flatnonzero(mask & real)
        if len(foreground) and bool(geometry['geometry_valid']) and len(points):
            sample = rng.choice(foreground, min(gdf_sample_per_case, len(foreground)), replace=False)
            queries = np.stack(np.unravel_index(sample, mask.shape), axis=1)
            gaussians = CumulativeRadiusGeometry(points, geometry['radius'], spacing)
            gdf_squares.append(float(np.mean(np.log1p(gaussians.evaluate_sum(queries)) ** 2)))
        used.append(row['id'])
    if not radius_means or not availability or not gdf_squares:
        raise ValueError('TRAIN has no reliable radius, real-grid availability or supported Gaussian target')
    mu = float(np.mean(radius_means))
    sigma = float(np.sqrt(max(float(np.mean(radius_squares)) - mu * mu, 0.)))
    # Degenerate synthetic/constant-radius data have an explicit safe scale.
    sigma = max(sigma, 1e-6)
    return dict(mu=mu, sigma=sigma, pi=float(np.clip(np.mean(availability), 1e-6, 1. - 1e-6)),
        gdf_scale=float(np.sqrt(np.mean(gdf_squares))), training_case_ids=used,
        definition='TRAIN-only case-equal moments; reliable node logR; uniform nominal half-grid A; foreground logG RMS',
        seed=int(seed), sample_per_case=int(sample_per_case), gdf_sample_per_case=int(gdf_sample_per_case))


def _gaussian_target(geometry, mask, real, scale, options):
    from .geometry import CumulativeRadiusGeometry, GaussianFieldSpec
    spec = GaussianFieldSpec(field_scale=scale,
        query_chunk=int(options.get('gdf_query_chunk', 512)), center_chunk=int(options.get('gdf_center_chunk', 1024)))
    backend = options.get('gdf_backend', 'cpu')
    if backend not in ('cpu', 'cuda'):
        raise ValueError('gdf_backend must be cpu or cuda')
    cls = CumulativeRadiusGeometry
    if backend == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Explicit CUDA GDF requested but unavailable; no silent fallback')
        class CudaGeometry(CumulativeRadiusGeometry):
            def evaluate_sum(self, query_zyx):
                query = torch.as_tensor(np.asarray(query_zyx), dtype=torch.float64, device='cuda')
                centers = torch.as_tensor(self.points_zyx, dtype=torch.float64, device='cuda')
                spacing = torch.as_tensor(self.spacing_mm, dtype=torch.float64, device='cuda')
                variance = torch.as_tensor(self.sigma_mm ** 2, dtype=torch.float64, device='cuda')
                result = torch.zeros(len(query), dtype=torch.float64, device='cuda')
                with torch.no_grad():
                    for qb in range(0, len(query), self.spec.query_chunk):
                        qe = min(qb + self.spec.query_chunk, len(query))
                        for cb in range(0, len(centers), self.spec.center_chunk):
                            ce = min(cb + self.spec.center_chunk, len(centers))
                            delta = (query[qb:qe, None] - centers[None, cb:ce]) * spacing
                            result[qb:qe] += torch.exp(-.5 * delta.square().sum(-1) / variance[None, cb:ce]).sum(1)
                return result.cpu().numpy()
        cls = CudaGeometry
    field = cls(geometry['points'], geometry['radius'], (1., 1., 1.), bool(geometry['geometry_valid']), spec)
    return field.target(mask, input_valid=real)


def prepare_dataset(cases_json, output_dir, spacing=1.0, geometry_options=None):
    """Prepare fresh portable cache; freeze statistics using TRAIN cases only."""
    import SimpleITK as sitk
    from scipy import ndimage as ndi
    from .geometry import sparse_geometry, CumulativeRadiusGeometry, GaussianFieldSpec
    if float(spacing) != 1.0:
        raise ValueError('This v1 architecture/attribute definition requires 1mm spacing')
    options = {} if geometry_options is None else dict(geometry_options)
    allowed = {'sample_per_case', 'gdf_sample_per_case', 'seed', 'gdf_query_chunk', 'gdf_center_chunk', 'gdf_backend'}
    if set(options) - allowed:
        raise ValueError(f'Unknown geometry options: {set(options) - allowed}')
    cases = _source_cases(cases_json)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True)
    rows = []
    for case in cases:
        directory = output / case['id']
        image, metadata = prepare_ct(case['image'], directory, case_id=case['id'], spacing=spacing,
                                     reviewed_override=case.get('intensity_override'))
        native_label = sitk.ReadImage(case['label'])
        if native_label.GetDimension() != 3 or native_label.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError('One scalar 3D label required')
        native_array = sitk.GetArrayFromImage(native_label)
        if not np.isfinite(native_array).all():
            raise ValueError('Nonfinite label')
        reference = sitk.Image([int(n) for n in metadata['resampled_size']], sitk.sitkFloat32)
        reference.SetOrigin(metadata['resampled_origin'])
        reference.SetDirection(metadata['resampled_direction'])
        reference.SetSpacing(metadata['resampled_spacing'])
        target = sitk.GetArrayFromImage(sitk.Resample(native_label > 0, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8))
        crop = tuple(slice(a, b) for a, b in zip(metadata['crop_start_zyx'], metadata['crop_stop_zyx']))
        mask = target[crop].astype(np.uint8)
        if not mask.any() or int(mask.sum()) != int(target.sum()):
            raise ValueError(f'Empty, misaligned or CT-crop-excluded label: {case["id"]}')
        valid = np.load(directory / 'valid.npy', allow_pickle=False).astype(bool)
        if (mask.astype(bool) & ~valid).any():
            raise ValueError(f'Label overlaps CT padding: {case["id"]}')
        native_radius = ndi.distance_transform_edt(native_array > 0, sampling=np.asarray(native_label.GetSpacing())[::-1]).astype(np.float32)
        radius_image = sitk.GetImageFromArray(native_radius)
        radius_image.CopyInformation(native_label)
        radius = sitk.GetArrayFromImage(sitk.Resample(radius_image, reference, sitk.Transform(), sitk.sitkLinear, 0., sitk.sitkFloat32))[crop]
        geometry = sparse_geometry(mask, np.repeat(spacing, 3), radius_field=radius)
        np.save(directory / 'mask.npy', mask, allow_pickle=False)
        np.savez_compressed(directory / 'geometry.npz', **geometry)
        rows.append(dict(id=case['id'], split=case['split'], cache_dir=case['id'],
                         spacing_mm=float(spacing), shape_zyx=list(mask.shape), source=case.get('source'),
                         label_sha256=_sha(case['label']), image_sha256=metadata['image_sha256']))
    statistics = train_statistics(rows, output, **{k: v for k, v in options.items() if k in {'sample_per_case', 'gdf_sample_per_case', 'seed'}})
    # Every field uses the same frozen TRAIN scalar; held-out cases never affect it.
    for row in rows:
        directory = output / row['cache_dir']
        mask = np.load(directory / 'mask.npy', allow_pickle=False).astype(bool)
        real = np.load(directory / 'valid.npy', allow_pickle=False).astype(bool)
        geometry = _load_geometry(directory)
        gdf, gdf_valid, audit = _gaussian_target(geometry, mask, real, statistics['gdf_scale'], options)
        np.save(directory / 'gdf.npy', gdf, allow_pickle=False)
        np.save(directory / 'gdf_valid.npy', gdf_valid.astype(np.uint8), allow_pickle=False)
        _json(directory / 'gdf_audit.json', audit)
    manifest = dict(schema='gta-avt-prepared-v1', spacing_mm=float(spacing), cases=rows, statistics=statistics,
        target_definition='native physical EDT sampled on 1mm Lee skeleton; partial pointwise logR; operational measurement availability',
        a_definition='1 iff partial radius pipeline emits a measurement; 0 elsewhere on real grid; padding excluded',
        gdf_definition='mask*log1p(sum_all_case_centerline_Gaussians_sigma_native_radius)/fixed_TRAIN_RMS',
        statistics_use_heldout=False, source_cases_sha256=_sha(cases_json))
    _json(output / 'manifest.json', manifest)
    return manifest


def _transfer_array(source, target, link_mode='hardlink_or_copy'):
    """Reference verified immutable inputs without altering their bytes."""
    import os
    import shutil
    if link_mode == 'symlink':
        os.symlink(str(Path(source).resolve(strict=True)), str(target))
    elif link_mode == 'hardlink_or_copy':
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
    else:
        raise ValueError('link_mode must be hardlink_or_copy or symlink')


def import_prepared_cache(cache_manifest, output_dir, *, cache_root=None, raw_root=None,
                          gdf_manifest=None, geometry_options=None, include_test=False,
                          link_mode='hardlink_or_copy'):
    """Reuse verified original 1mm arrays without a private checkpoint/bank.

    Original TRAIN geometry and masks must match their manifest hashes. CT
    padding validity is independently restored from native CT. Missing VAL
    geometry is reconstructed with the same native-EDT pipeline. An optional
    completed Gaussian cache supplies both its TRAIN-only RMS and targets,
    after checking case identity, target bytes and geometry/mask hashes.
    ``link_mode='symlink'`` avoids cross-filesystem copies; its verified source
    arrays must remain available and immutable for the lifetime of this cache.
    Held-out Gaussian foreground is explicitly unknown in this efficient path;
    ``prepare_dataset`` remains the complete raw-data path for all targets.
    """
    import SimpleITK as sitk
    from scipy import ndimage as ndi
    from .geometry import sparse_geometry
    legacy_path = Path(cache_manifest).resolve()
    if link_mode not in ('hardlink_or_copy', 'symlink'):
        raise ValueError('link_mode must be hardlink_or_copy or symlink')
    legacy = json.loads(legacy_path.read_text(encoding='utf-8'))
    if legacy.get('complete') is not True or float(legacy.get('spacing_mm', 0)) != 1.:
        raise ValueError('A completed original 1mm cache is required')
    source_rows = {r['id']: r for r in legacy['cases']}
    if len(source_rows) != len(legacy['cases']):
        raise ValueError('Duplicate source case IDs')
    assignments = {}
    for split in ('train', 'val', 'test'):
        for case_id in legacy['splits'][split]:
            if case_id in assignments or case_id not in source_rows:
                raise ValueError('Overlapping/unknown original split case')
            assignments[case_id] = split.upper()
    if set(assignments) != set(source_rows):
        raise ValueError('Original splits must partition the source manifest')
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh imported-cache directory')
    output.mkdir(parents=True)
    options = dict(geometry_options or {})
    rows, input_hashes = [], {}

    def transfer(source, target):
        _transfer_array(source, target, link_mode)

    def raw_path(value):
        path = Path(value)
        if raw_root is not None:
            parts = path.parts
            if 'extracted' not in parts:
                raise ValueError('raw_root relocation requires original extracted/ path component')
            path = Path(raw_root).joinpath(*parts[parts.index('extracted') + 1:])
        return path

    for case_id, split in assignments.items():
        if split == 'TEST' and not include_test:
            continue
        record = source_rows[case_id]
        directory = output / case_id
        directory.mkdir()
        source = Path(cache_root) / case_id if cache_root is not None else Path(record['cache_dir'])
        hashes = {}
        for filename in ('image.npy', 'mask.npy'):
            actual = _sha(source / filename)
            if actual != record['cache_file_sha256'][filename]:
                raise ValueError(f'Original array hash differs: {case_id}/{filename}')
            hashes[filename] = actual
            transfer(source / filename, directory / filename)
        image_path = raw_path(record['image'])
        if _sha(image_path) != record['image_sha256']:
            raise ValueError(f'Native CT hash differs: {case_id}')
        audit = record['intensity_audit']
        if audit.get('requires_review') or audit.get('offset_subtract') not in (0., 1024.):
            raise ValueError(f'Original intensity decision unresolved: {case_id}')
        native_image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
        source_array = sitk.GetArrayFromImage(native_image)
        if not np.isfinite(source_array).all():
            raise ValueError('Native CT contains nonfinite values')
        valid_native = sitk.GetImageFromArray((source_array > -1500.).astype(np.uint8))
        valid_native.CopyInformation(native_image)
        reference = sitk.Image([int(n) for n in record['resampled_size']], sitk.sitkFloat32)
        reference.SetOrigin(record['resampled_origin'])
        reference.SetDirection(record['resampled_direction'])
        reference.SetSpacing(record['resampled_spacing'])
        crop = tuple(slice(a, b) for a, b in zip(record['crop_start_zyx'], record['crop_stop_zyx']))
        valid = sitk.GetArrayFromImage(sitk.Resample(valid_native, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8))[crop].astype(bool)
        mask = np.load(directory / 'mask.npy', mmap_mode='r', allow_pickle=False).astype(bool)
        if tuple(valid.shape) != tuple(mask.shape) or (mask & ~valid).any():
            raise ValueError(f'Padding/mask physical alignment differs: {case_id}')
        np.save(directory / 'valid.npy', valid.astype(np.uint8), allow_pickle=False)
        if (source / 'geometry.npz').is_file():
            actual = _sha(source / 'geometry.npz')
            if actual != record['cache_file_sha256']['geometry.npz']:
                raise ValueError(f'Original geometry hash differs: {case_id}')
            transfer(source / 'geometry.npz', directory / 'geometry.npz')
            hashes['geometry.npz'] = actual
        else:
            label_path = raw_path(record['label'])
            if _sha(label_path) != record['label_sha256']:
                raise ValueError(f'Native label hash differs: {case_id}')
            label = sitk.ReadImage(str(label_path))
            native_radius = ndi.distance_transform_edt(sitk.GetArrayFromImage(label) > 0,
                sampling=np.asarray(label.GetSpacing())[::-1]).astype(np.float32)
            radius_image = sitk.GetImageFromArray(native_radius)
            radius_image.CopyInformation(label)
            radius = sitk.GetArrayFromImage(sitk.Resample(radius_image, reference, sitk.Transform(), sitk.sitkLinear, 0., sitk.sitkFloat32))[crop]
            geometry = sparse_geometry(mask, (1., 1., 1.), radius_field=radius)
            np.savez_compressed(directory / 'geometry.npz', **geometry)
            hashes['geometry.npz'] = _sha(directory / 'geometry.npz')
        metadata = {k: record[k] for k in ('shape_zyx', 'full_shape_zyx', 'crop_start_zyx', 'crop_stop_zyx',
            'resampled_origin', 'resampled_direction', 'resampled_spacing', 'resampled_size',
            'original_origin', 'original_direction', 'original_spacing', 'original_size')}
        metadata.update(schema='gta-avt-ct-v1', id=case_id, spacing_mm=1., intensity_audit=audit,
                        image_sha256=record['image_sha256'], label_required=False, geometry_required=False)
        _json(directory / 'record.json', metadata)
        rows.append(dict(id=case_id, split=split, cache_dir=case_id, spacing_mm=1., shape_zyx=list(mask.shape),
                         source=record.get('cohort'), image_sha256=record['image_sha256'], label_sha256=record['label_sha256']))
        input_hashes[case_id] = hashes
        print(json.dumps(dict(event='cache_import_case', id=case_id, split=split, geometry_reused=(source / 'geometry.npz').is_file())), flush=True)
    statistics = train_statistics(rows, output, **{k: v for k, v in options.items() if k in {'sample_per_case', 'gdf_sample_per_case', 'seed'}})
    gdf_rows = {}
    if gdf_manifest is not None:
        gdf = json.loads(Path(gdf_manifest).read_text(encoding='utf-8'))
        train_ids = {r['id'] for r in rows if r['split'] == 'TRAIN'}
        if not gdf.get('complete') or set(gdf['training_case_ids']) != train_ids or gdf.get('heldout_case_files_opened') is not False:
            raise ValueError('Gaussian cache must be completed with the same TRAIN-only identity')
        scale = float(gdf['fixed_train_field_scale'])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid Gaussian TRAIN RMS')
        gdf_rows = {r['id']: r for r in gdf['cases']}
        statistics['gdf_scale'] = scale
        statistics['gdf_scale_source'] = dict(manifest_sha256=_sha(gdf_manifest), normalization=gdf.get('normalization'))
    for row in rows:
        directory = output / row['cache_dir']
        mask = np.load(directory / 'mask.npy', mmap_mode='r', allow_pickle=False).astype(bool)
        real = np.load(directory / 'valid.npy', mmap_mode='r', allow_pickle=False).astype(bool)
        geometry = _load_geometry(directory)
        if row['split'] == 'TRAIN' and row['id'] in gdf_rows:
            item = gdf_rows[row['id']]
            for filename in ('mask.npy', 'geometry.npz'):
                if item['input_file_sha256'][filename] != input_hashes[row['id']][filename]:
                    raise ValueError(f'Gaussian/source target geometry differs: {row["id"]}')
            source = Path(item['target_file'])
            if not source.is_file():
                source = Path(gdf_manifest).parent / row['id'] / 'gdf.npy'
            if _sha(source) != item['target_file_sha256']:
                raise ValueError('Gaussian target bytes differ')
            array = np.load(source, mmap_mode='r', allow_pickle=False)
            if array.shape != mask.shape or not np.isfinite(array).all():
                raise ValueError('Gaussian target shape/nonfinite mismatch')
            transfer(source, directory / 'gdf.npy')
            gvalid = real & (~mask | (bool(geometry['geometry_valid']) and len(geometry['points']) > 0))
            np.save(directory / 'gdf_valid.npy', gvalid.astype(np.uint8), allow_pickle=False)
        elif row['split'] == 'TRAIN':
            target, gvalid, audit = _gaussian_target(geometry, mask, real, statistics['gdf_scale'], options)
            np.save(directory / 'gdf.npy', target, allow_pickle=False)
            np.save(directory / 'gdf_valid.npy', gvalid.astype(np.uint8), allow_pickle=False)
            _json(directory / 'gdf_audit.json', audit)
        else:
            # Missing held-out fields are unknown, never fabricated zero targets.
            zero = np.lib.format.open_memmap(directory / 'gdf.npy', mode='w+', dtype=np.float16, shape=mask.shape)
            zero[:] = 0.
            zero.flush()
            del zero
            np.save(directory / 'gdf_valid.npy', (real & ~mask).astype(np.uint8), allow_pickle=False)
            _json(directory / 'gdf_audit.json', dict(status='foreground_not_prepared', foreground_unknown=True))
    manifest = dict(schema='gta-avt-prepared-v1', spacing_mm=1., cases=rows, statistics=statistics,
        imported_array_link_mode=link_mode,
        original_cache_manifest_sha256=_sha(legacy_path), original_array_hashes=input_hashes,
        statistics_use_heldout=False, heldout_gdf_foreground_prepared=False,
        a_definition='1 iff partial radius pipeline emits a measurement; 0 elsewhere on real grid; padding excluded',
        gdf_definition='mask*log1p(sum_all_case_centerline_Gaussians_sigma_native_radius)/fixed_TRAIN_RMS')
    _json(output / 'manifest.json', manifest)
    return manifest


def load_manifest(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('schema') != 'gta-avt-prepared-v1':
        raise ValueError('Prepared v1 manifest required')
    for case in value['cases']:
        directory = Path(case['cache_dir'])
        case['cache_dir'] = str(directory.resolve() if directory.is_absolute() else (path.parent / directory).resolve())
    return value


class PreparedDataset:
    """Stateless case/patch sampling; all controls receive identical batches."""
    def __init__(self, manifest, split='TRAIN', patch_size=128, samples_per_epoch=1000, seed=0, augment=False):
        self.manifest = load_manifest(manifest)
        self.split = split.upper()
        self.statistics = self.manifest['statistics']
        self.cases = [c for c in self.manifest['cases'] if c['split'] == split.upper()]
        if not self.cases:
            raise ValueError(f'No {split} cases')
        if not isinstance(patch_size, int) or patch_size < 16 or patch_size % 16:
            raise ValueError('Patch size must be a positive multiple of16')
        if samples_per_epoch < 1:
            raise ValueError('Positive samples_per_epoch required')
        self.patch_size, self.samples_per_epoch = patch_size, int(samples_per_epoch)
        self.seed, self.augment, self.epoch = int(seed), bool(augment), 0
        self._opened = {}

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.samples_per_epoch

    def _case(self, row):
        if row['id'] not in self._opened:
            directory = Path(row['cache_dir'])
            result = {key: np.load(directory / f'{key}.npy', mmap_mode='r', allow_pickle=False)
                      for key in ('image', 'mask', 'valid', 'gdf', 'gdf_valid')}
            result['geometry'] = _load_geometry(directory)
            self._opened[row['id']] = result
        return self._opened[row['id']]

    def __getitem__(self, index):
        import torch
        from .geometry import query_grid, radius_at_queries, background_stencil_certificate
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        row = self.cases[int(rng.integers(len(self.cases))) if self.split == 'TRAIN' else int(index) % len(self.cases)]
        item = self._case(row)
        patch = self.patch_size
        shape = np.asarray(item['image'].shape)
        points = item['geometry']['points']
        if len(points) and rng.random() < .5:
            centre = points[rng.integers(len(points))] + rng.integers(-patch // 4, patch // 4 + 1, size=3)
            start = np.clip(centre - patch // 2, 0, np.maximum(shape - patch, 0))
        else:
            start = np.asarray([rng.integers(max(1, int(n) - patch + 1)) for n in shape])
        slices = tuple(slice(int(s), min(int(s) + patch, int(n))) for s, n in zip(start, shape))
        arrays = {}
        for key, source in (('image', 'image'), ('label', 'mask'), ('valid', 'valid'), ('gdf', 'gdf'), ('gdf_valid', 'gdf_valid')):
            array = np.asarray(item[source][slices], np.float32)
            arrays[key] = np.pad(array, [(0, patch - n) for n in array.shape], constant_values=-1024. if key == 'image' else 0.)
        queries = query_grid(start, (patch // 2,) * 3, (patch,) * 3)
        spacing = np.repeat(row['spacing_mm'], 3)
        observed_bounds = np.stack((start - .5, np.minimum(start + patch, shape) - .5)) * spacing
        oracle = radius_at_queries(item['geometry'], queries, shape, spacing,
                                   observed_bounds_mm=observed_bounds)
        local_queries = query_grid((0, 0, 0), (patch // 2,) * 3, (patch,) * 3)
        real = background_stencil_certificate(~arrays['valid'].astype(bool), local_queries)
        measured = oracle['R_valid'] & real
        r = np.zeros(measured.shape, np.float32)
        r[measured] = ((oracle['rawR'][measured] - self.statistics['mu']) / self.statistics['sigma']).astype(np.float32)
        arrays.update(a=measured.astype(np.float32), r=r, r_valid=measured.astype(np.float32), a_valid=real.astype(np.float32))
        arrays['image'] = normalize(arrays['image'])
        if self.augment:
            for axis in range(3):
                if rng.random() < .5:
                    arrays = {k: np.flip(v, axis) for k, v in arrays.items()}
            rotation = int(rng.integers(4))
            axes = ((0, 1), (0, 2), (1, 2))[int(rng.integers(3))]
            arrays = {k: np.rot90(v, rotation, axes) for k, v in arrays.items()}
            arrays['image'] = arrays['image'] * rng.uniform(.9, 1.1) + rng.uniform(-.1, .1)
            if rng.random() < .15:
                arrays['image'] += rng.normal(0., .03, arrays['image'].shape).astype(np.float32)
        result = {key: torch.from_numpy(np.ascontiguousarray(value[None], dtype=np.float32)) for key, value in arrays.items()}
        result['case_id'] = row['id']
        return result


def tile_starts(size, patch, overlap):
    if size <= patch:
        return [0]
    stride = max(1, int(patch * (1. - overlap)))
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def sliding_probability(predict_probability, image, metadata, patch=128, batch_size=2, overlap=.5):
    """CT-only whole-crop tiles; Gaussian blending is independent of GDF loss."""
    if patch <= 0 or batch_size <= 0 or not 0. <= overlap < 1.:
        raise ValueError('Positive patch/batch and overlap in [0,1) required')
    shape = np.asarray(image.shape)
    if list(shape) != metadata['shape_zyx']:
        raise ValueError('Image/metadata shape mismatch')
    allocation = tuple(np.maximum(shape, patch))
    probability, weights = np.zeros(allocation, np.float32), np.zeros(allocation, np.float32)
    coordinates = np.arange(patch, dtype=np.float32) - (patch - 1.) / 2.
    gaussian = np.exp(-.5 * (coordinates / (patch / 8.)) ** 2)
    weight = np.maximum(gaussian[:, None, None] * gaussian[None, :, None] * gaussian[None, None, :], 1e-5)
    starts = list(itertools.product(*(tile_starts(int(n), patch, overlap) for n in np.maximum(shape, patch))))
    for offset in range(0, len(starts), batch_size):
        chosen = starts[offset:offset + batch_size]
        samples = []
        for start in chosen:
            slices = tuple(slice(s, min(s + patch, int(n))) for s, n in zip(start, shape))
            array = np.asarray(image[slices], np.float32)
            array = np.pad(array, [(0, patch - n) for n in array.shape], constant_values=-1024.)
            samples.append(normalize(array)[None])
        prediction = np.asarray(predict_probability(np.ascontiguousarray(samples, dtype=np.float32)), np.float32)
        if prediction.shape == (len(chosen), 1, patch, patch, patch):
            prediction = prediction[:, 0]
        if prediction.shape != (len(chosen), patch, patch, patch) or not np.isfinite(prediction).all() or (prediction < 0).any() or (prediction > 1).any():
            raise ValueError('Callback must return finite patch probabilities in [0,1]')
        for start, value in zip(chosen, prediction):
            slices = tuple(slice(s, s + patch) for s in start)
            probability[slices] += value * weight
            weights[slices] += weight
    probability /= np.maximum(weights, 1e-8)
    return probability[tuple(slice(0, int(n)) for n in shape)], len(starts)
