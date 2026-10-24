"""CPU-only known-radius straight-tube calibration; default inert.

No real case, VAL/TEST, fitted correction, threshold search or model is used.
Distances are in mm. This measures a raster/EDT/Lee proxy, not a scanner PSF,
clinical diameter or segmentation/adaptation benefit.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

SETTINGS = dict(native_spacing_ZYX_mm=[[.625, 1., 1.], [2.993, 1., 1.], [5., 1., 1.]],
    corresponding_spacing_XYZ_mm=[[1., 1., .625], [1., 1., 2.993], [1., 1., 5.]],
    radii_mm=[2., 4., 8.], directions_ZYX=dict(axial_z=[1., 0., 0.], oblique_111=[1., 1., 1.]),
    axis_phase_in_native_voxels_ZYX=[[0., 0., 0.], [.5, .5, .5], [.37, -.23, .41]],
    minimum_FOV_half_extent_mm=32., evaluation_axis_half_length_mm=12., analytic_axis_step_mm=.25,
    Lee_grid_spacing_ZYX_mm=[1., 1., 1.], Lee_mask_resampling='NN from native voxel-centre mask, same physical origin, constant-background outside FOV',
    sampling_methods=['true_analytic_axis', 'Lee_1mm_from_native_NN'],
    tube='Infinite continuous straight cylinder rasterized by voxel-centre inclusion; evaluate only interior axis segment',
    radius_proxy='Native physical EDT to background voxel centres, sampled trilinearly at requested physical locations',
    raw_arrays_saved=False)
DECISION_POLICY = dict(automatic_training_or_threshold_selection=False,
    missing_Lee='Keep all configurations and record empty/no-central-support as unavailable, never as a negative anatomical label',
    scale_or_phase_dependence='Report complete radius/orientation/phase tables; do not fit an affine correction or select a favourable phase/cohort after reading results',
    next_prerequisite='A local radius oracle requires a defensible support/measurement domain and a matched conditional-utility experiment with jointly learned consumption; small calibration error alone is not a continuation gate. A cold fixed-direction gradient projection is not required for a jointly learned consumer',
    stop='If the intended local calibre distinction is confounded by sampling/interpolation/skeleton displacement, stop treating that proxy as precise calibre until the definition is prospectively revised',
    limits='No native-PSF estimate, clinical radius truth, real annotation reliability, CT predictability, dynamic-GTA efficacy or threshold robustness claim')


def require(value, message):
    if not value:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def csv_bytes(rows):
    stream = io.StringIO(newline=''); writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def geometry(np, spacing, direction, phase, radius):
    spacing = np.asarray(spacing, dtype=np.float64)
    unit = np.asarray(direction, dtype=np.float64); unit /= np.linalg.norm(unit)
    half_steps = np.ceil(SETTINGS['minimum_FOV_half_extent_mm'] / spacing).astype(int)
    shape = 2 * half_steps + 1; origin = -half_steps * spacing
    centre = np.asarray(phase, dtype=np.float64) * spacing
    axes = [origin[i] + np.arange(shape[i]) * spacing[i] for i in range(3)]
    q = [axes[i].reshape(tuple(shape[i] if j == i else 1 for j in range(3))) - centre[i] for i in range(3)]
    axial = sum(q[i] * unit[i] for i in range(3))
    radial2 = np.maximum(sum(v * v for v in q) - axial * axial, 0.)
    mask = radial2 <= radius ** 2
    require(mask.any() and (~mask).any(), 'Analytic tube must have foreground and background')
    require(SETTINGS['minimum_FOV_half_extent_mm'] - SETTINGS['evaluation_axis_half_length_mm'] - np.max(np.abs(centre)) >
        radius + np.linalg.norm(spacing), 'Insufficient halo for fixed central evaluation')
    return mask, spacing, origin, centre, unit


def support_geometry(np, points, centre, unit, radius):
    relative = points - centre
    axial = relative @ unit
    off_axis = np.linalg.norm(relative - axial[:, None] * unit[None, :], axis=1)
    ideal_clearance = np.maximum(radius - off_axis, 0.)
    return axial, off_axis, ideal_clearance


def samples(np, ndi, skeletonize, mask, spacing, origin, centre, unit):
    n = int(round(2 * SETTINGS['evaluation_axis_half_length_mm'] / SETTINGS['analytic_axis_step_mm'])) + 1
    t = np.linspace(-SETTINGS['evaluation_axis_half_length_mm'], SETTINGS['evaluation_axis_half_length_mm'], n)
    analytic = centre[None, :] + t[:, None] * unit[None, :]
    extent = (np.asarray(mask.shape) - 1) * spacing
    shape = np.ceil(extent).astype(int) + 1
    z, y, x = np.meshgrid(*(np.arange(v, dtype=np.float64) for v in shape), indexing='ij')
    coordinates = np.stack((z / spacing[0], y / spacing[1], x / spacing[2]))
    one_mm_mask = ndi.map_coordinates(mask.astype(np.uint8), coordinates, order=0, mode='constant', cval=0, prefilter=False).astype(bool)
    skeleton = skeletonize(one_mm_mask, method='lee').astype(bool)
    all_Lee = np.argwhere(skeleton).astype(np.float64) + origin
    axial = (all_Lee - centre) @ unit
    selected = np.abs(axial) <= SETTINGS['evaluation_axis_half_length_mm']
    Lee = all_Lee[selected]
    return [('true_analytic_axis', analytic, len(analytic)), ('Lee_1mm_from_native_NN', Lee, len(all_Lee))], one_mm_mask.shape


def point_measurements(np, ndi, tree, edt, points, spacing, origin, centre, unit, radius, method):
    if not len(points):
        return []
    axial, offset, clearance = support_geometry(np, points, centre, unit, radius)
    if method == 'true_analytic_axis':
        # Analytic construction defines these samples on the exact continuous axis.
        offset[:] = 0.; clearance[:] = radius
    nearest_centres = tree.query(points, workers=1)[0]
    proxy = ndi.map_coordinates(edt, ((points - origin) / spacing).T, order=1, mode='constant', cval=0., prefilter=False)
    rows = []
    for i in range(len(points)):
        row = dict(sample_index=i, axial_coordinate_mm=float(axial[i]), off_axis_mm=float(offset[i]),
            outside_continuous_cylinder=bool(offset[i] > radius), ideal_local_clearance_mm=float(clearance[i]),
            exact_nearest_native_background_centre_distance_mm=float(nearest_centres[i]), EDT_proxy_mm=float(proxy[i]),
            background_centre_minus_ideal_clearance_mm=float(nearest_centres[i] - clearance[i]),
            EDT_interpolation_minus_exact_background_centre_mm=float(proxy[i] - nearest_centres[i]),
            clearance_minus_true_axis_radius_mm=float(clearance[i] - radius),
            EDT_proxy_minus_true_axis_radius_mm=float(proxy[i] - radius))
        decomposition = (row['background_centre_minus_ideal_clearance_mm'] +
            row['EDT_interpolation_minus_exact_background_centre_mm'] + row['clearance_minus_true_axis_radius_mm'])
        row['decomposition_rounding_residual_mm'] = row['EDT_proxy_minus_true_axis_radius_mm'] - decomposition
        require(all(not isinstance(v, float) or math.isfinite(v) for v in row.values()), 'Nonfinite analytic measurement')
        rows.append(row)
    return rows


def describe(np, rows):
    fields = ('off_axis_mm', 'EDT_proxy_mm', 'EDT_proxy_minus_true_axis_radius_mm',
        'background_centre_minus_ideal_clearance_mm', 'EDT_interpolation_minus_exact_background_centre_mm',
        'clearance_minus_true_axis_radius_mm', 'decomposition_rounding_residual_mm')
    result = {}
    for field in fields:
        values = np.asarray([r[field] for r in rows], dtype=np.float64)
        for suffix, value in (('mean', values.mean() if len(values) else None),
            ('median', np.median(values) if len(values) else None),
            ('MAE', np.abs(values).mean() if len(values) else None),
            ('max_abs', np.abs(values).max() if len(values) else None)):
            result[field + '_' + suffix] = float(value) if value is not None else None
    result['outside_continuous_cylinder_samples'] = sum(r['outside_continuous_cylinder'] for r in rows)
    return result


def execute(output):
    output = Path(output); require(not output.exists(), 'Fresh output directory required')
    import time
    started = time.monotonic()
    import numpy as np
    import scipy
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    import skimage
    from skimage.morphology import skeletonize
    rows, summaries = [], []
    phantom = 0
    for spacing in SETTINGS['native_spacing_ZYX_mm']:
        for name, direction in SETTINGS['directions_ZYX'].items():
            for radius in SETTINGS['radii_mm']:
                for phase_id, phase in enumerate(SETTINGS['axis_phase_in_native_voxels_ZYX']):
                    phantom += 1
                    mask, spacing_a, origin, centre, unit = geometry(np, spacing, direction, phase, radius)
                    edt = ndi.distance_transform_edt(mask, sampling=spacing_a)
                    background = np.argwhere(~mask).astype(np.float64) * spacing_a + origin
                    tree = cKDTree(background)
                    methods, Lee_shape = samples(np, ndi, skeletonize, mask, spacing_a, origin, centre, unit)
                    identity = dict(phantom_id=phantom, spacing_z_mm=spacing[0], spacing_y_mm=spacing[1], spacing_x_mm=spacing[2],
                        axis=name, radius_mm=radius, phase_id=phase_id,
                        phase_z_native_voxels=phase[0], phase_y_native_voxels=phase[1], phase_x_native_voxels=phase[2])
                    for method, points, all_support in methods:
                        measurements = point_measurements(np, ndi, tree, edt, points, spacing_a, origin, centre, unit, radius, method)
                        rows.extend(dict(**identity, sample_method=method, **r) for r in measurements)
                        summaries.append(dict(**identity, sample_method=method, sampled_points=len(points),
                            total_axis_or_Lee_points=all_support, unavailable_reason=None if len(points) else 'no_central_Lee_support',
                            native_shape_ZYX='x'.join(map(str, mask.shape)), Lee_1mm_shape_ZYX='x'.join(map(str, Lee_shape)),
                            **describe(np, measurements)))
    require(phantom == 54 and len(summaries) == 108, 'Fixed complete analytic design required')
    payload = {'points.csv': csv_bytes(rows), 'configurations.csv': csv_bytes(summaries)}
    analysis = dict(schema='GTA-AVT-analytic-physical-radius-calibration-v1', settings=SETTINGS, decision_policy=DECISION_POLICY,
        configurations=54, method_configuration_rows=108, raw_point_rows=len(rows),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        packages=dict(numpy=np.__version__, scipy=scipy.__version__, skimage=skimage.__version__),
        no_real_case_or_VAL_TEST_data=True, no_model_or_training=True, correction_or_threshold_fitted=False,
        explanation='Background-centre vs ideal-clearance gap and interpolation gap are separated; at outside-cylinder Lee samples the former is descriptive, not a pure boundary-discretization bias. Lee axis displacement includes native rasterization, NN resampling and thinning, not Lee-only causal error.',
        metrics= summaries, elapsed_seconds=time.monotonic() - started)
    payload['analysis.json'] = encode(analysis)
    payload['manifest.json'] = encode(dict(schema='GTA-AVT-analytic-radius-calibration-manifest-v1',
        files={k: dict(sha256=hashlib.sha256(v).hexdigest(), bytes=len(v)) for k, v in payload.items()},
        no_real_case_data=True, no_training_authorized=True))
    output.mkdir(parents=True)
    for name, value in payload.items():
        (output / name).write_bytes(value)
    return dict(status='complete', configurations=54, methods=2, points=len(rows),
        analysis_sha256=hashlib.sha256(payload['analysis.json']).hexdigest(), no_training_authorized=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output'); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='inert', settings=SETTINGS, decision_policy=DECISION_POLICY, real_data_read=False, outputs_written=False))); return
    require(args.output, 'Explicit fresh output required')
    print(json.dumps(execute(args.output)))


if __name__ == '__main__':
    main()
