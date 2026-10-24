"""Training-only sparse geometric targets; no target enters model.forward.

Radius is the native physical EDT sampled at resampled Lee skeleton nodes.
Degree >=3 clusters are candidates. A connected 3 mm core must have at least
three incident arms supported to 9 mm inside a local 12 mm neighbourhood.
Rejected candidate neighbourhoods are UNKNOWN and excluded from supervision.
These are local skeleton-graph labels, not validated anatomical bifurcations.
Lee thinning can lose branches under grid phase changes; this filter does not
recover them. Nonempty masks with empty skeletons have invalid geometry.
PCA nonlinearity is retained as a diagnostic and is NOT called curvature or used
as supervision. True branch-path curvature belongs to the independent audit.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

RADIUS_EDGES_MM = np.array([2., 4., 8.])
STRUCTURE_NAMES = ['empty', 'nonjunction', 'junction_neighbourhood']
SCALE_NAMES = ['empty', 'radius_le_2mm', 'radius_2_to_4mm', 'radius_4_to_8mm', 'radius_gt_8mm']
UNKNOWN_STRUCTURE = 255
JUNCTION_CORE_MM = 3.
JUNCTION_ARM_EXTENSION_MM = 6.
JUNCTION_OUTER_MM = 12.
JUNCTION_NEIGHBOURHOOD_MM = 5.


def _supported_junctions(skeleton, points, spacing):
    """Audit local incident graph arms after contracting a connected core.

    A nearby but graph-disconnected skeleton segment cannot join the core.
    Short/cropped/undersampled arms cause ambiguity, not a nonjunction label.
    Residual loops may create multiple arms; no anatomical identity is inferred.
    """
    neighbours = np.ones((3, 3, 3), np.uint8)
    degrees = ndi.convolve(skeleton.astype(np.uint8), neighbours, mode='constant') - skeleton
    candidates = skeleton & (degrees >= 3)
    regions, count = ndi.label(candidates, neighbours)
    centres = []
    arm_counts = []
    physical = points * spacing
    for label in range(1, count + 1):
        seed = regions == label
        seed_points = np.argwhere(seed)
        centres.append(seed_points.mean(0))
        distances = cKDTree(seed_points * spacing).query(physical)[0]
        selected = distances <= JUNCTION_OUTER_MM
        local_points = points[selected]
        lower = np.maximum(local_points.min(0) - 1, 0)
        upper = np.minimum(local_points.max(0) + 2, skeleton.shape)
        slices = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
        local_skeleton = skeleton[slices]
        distance_field = np.full(local_skeleton.shape, np.inf, np.float32)
        distance_field[tuple((local_points - lower).T)] = distances[selected]
        possible_core = local_skeleton & (distance_field <= JUNCTION_CORE_MM)
        core_components, _ = ndi.label(possible_core, neighbours)
        seed_components = np.unique(core_components[seed[slices]])
        core = np.isin(core_components, seed_components[seed_components > 0])
        arms = local_skeleton & (~core) & (distance_field <= JUNCTION_OUTER_MM)
        arm_components, _ = ndi.label(arms, neighbours)
        adjacent = ndi.binary_dilation(core, structure=neighbours) & arms
        supported = 0
        for arm in np.unique(arm_components[adjacent]):
            if arm > 0 and np.max(distance_field[arm_components == arm]) >= JUNCTION_CORE_MM + JUNCTION_ARM_EXTENSION_MM:
                supported += 1
        arm_counts.append(supported)
    return (np.asarray(centres, np.float32).reshape(-1, 3), np.asarray(arm_counts, np.int16),
            regions[tuple(points.T)].astype(np.int32))


def sparse_geometry(mask, spacing, radius_field=None):
    mask = np.asarray(mask, dtype=bool)
    spacing = np.asarray(spacing, dtype=float)
    if mask.ndim != 3 or spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Expected a 3D mask and three finite positive spacings')
    skeleton = skeletonize(mask, method='lee').astype(bool)
    points = np.argwhere(skeleton)
    mask_nonempty = bool(mask.any())
    invalid = mask_nonempty and not len(points)
    audit = {
        'geometry_valid': np.asarray(not invalid, np.uint8),
        'mask_nonempty': np.asarray(mask_nonempty, np.uint8),
        'skeleton_empty_nonempty_mask': np.asarray(invalid, np.uint8),
        'skeleton_point_count': np.asarray(len(points), np.int32),
        'junction_parameters_mm': np.asarray([JUNCTION_CORE_MM, JUNCTION_ARM_EXTENSION_MM,
                                              JUNCTION_OUTER_MM, JUNCTION_NEIGHBOURHOOD_MM], np.float32),
        'junction_centres_zyx': np.empty((0, 3), np.float32),
        'junction_supported_arm_counts': np.empty(0, np.int16),
        'junction_candidate_ids_at_points': np.zeros(len(points), np.int32),
        'candidate_junction_count': np.asarray(0, np.int32),
        'supported_junction_count': np.asarray(0, np.int32),
        'unknown_junction_count': np.asarray(0, np.int32),
        'unknown_structure_point_count': np.asarray(0, np.int32),
    }
    if not len(points):
        return {'points': np.empty((0, 3), np.int32), 'radius': np.empty(0, np.float32),
                'structure': np.empty(0, np.uint8), 'local_pca_nonlinearity': np.empty(0, np.float32), **audit}
    radius = (ndi.distance_transform_edt(mask, sampling=spacing) if radius_field is None else radius_field)[tuple(points.T)].astype(np.float32)
    centres, arm_counts, candidate_ids = _supported_junctions(skeleton, points, spacing)
    supported = arm_counts >= 3
    physical = points * spacing
    tree = cKDTree(physical)
    positive_seed = np.zeros(len(points), dtype=bool)
    negative_seed = np.zeros(len(points), dtype=bool)
    candidate_points = candidate_ids > 0
    if candidate_points.any():
        positive_seed[candidate_points] = supported[candidate_ids[candidate_points] - 1]
        negative_seed[candidate_points] = ~supported[candidate_ids[candidate_points] - 1]
    # Use distance to all cluster voxels, not only its centroid: elongated
    # clusters must not leave their own rejected seed voxels labelled valid.
    junction_distance = (cKDTree(physical[positive_seed]).query(physical)[0]
                         if positive_seed.any() else np.full(len(points), np.inf))
    ambiguous_distance = (cKDTree(physical[negative_seed]).query(physical)[0]
                          if negative_seed.any() else np.full(len(points), np.inf))
    curvature = np.zeros(len(points), np.float32)
    for index, neighbours in enumerate(tree.query_ball_point(physical, 7.5)):
        if len(neighbours) < 4:
            continue
        local = physical[neighbours]
        covariance = np.cov(local.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        curvature[index] = max(0., 1. - eigenvalues[-1] / max(eigenvalues.sum(), 1e-12))
    structure = np.ones(len(points), dtype=np.uint8)
    structure[junction_distance <= JUNCTION_NEIGHBOURHOOD_MM] = 2
    # Ambiguity wins if candidate neighbourhoods overlap: no rejected region
    # silently becomes a positive or a straight/nonjunction training target.
    structure[ambiguous_distance <= JUNCTION_NEIGHBOURHOOD_MM] = UNKNOWN_STRUCTURE
    audit.update(junction_centres_zyx=centres, junction_supported_arm_counts=arm_counts,
                 junction_candidate_ids_at_points=candidate_ids,
                 candidate_junction_count=np.asarray(len(centres), np.int32),
                 supported_junction_count=np.asarray(supported.sum(), np.int32),
                 unknown_junction_count=np.asarray((~supported).sum(), np.int32),
                 unknown_structure_point_count=np.asarray((structure == UNKNOWN_STRUCTURE).sum(), np.int32))
    return {'points': points.astype(np.int32), 'radius': radius, 'structure': structure,
            'local_pca_nonlinearity': curvature, **audit}


def patch_histograms(geometry, start, size):
    if not bool(np.asarray(geometry.get('geometry_valid', True)).item()):
        return np.zeros(5, np.float32), np.zeros(3, np.float32)
    points = geometry['points']
    inside = ((points >= np.asarray(start)) & (points < np.asarray(start) + np.asarray(size))).all(axis=1)
    if not inside.any():
        return np.array([1., 0., 0., 0., 0.], np.float32), np.array([1., 0., 0.], np.float32)
    scale = np.bincount(np.searchsorted(RADIUS_EDGES_MM, geometry['radius'][inside], side='left') + 1,
                        minlength=5).astype(np.float32)
    structure_values = geometry['structure'][inside]
    structure_values = structure_values[structure_values != UNKNOWN_STRUCTURE]
    structure = np.bincount(structure_values, minlength=3).astype(np.float32)
    return scale / scale.sum(), structure / structure.sum() if structure.sum() else structure


"""Partial, continuous-valued GT radius oracle; pure array functions, no I/O.

This is a diagnostic target, never a CT prediction. Sparse native-EDT samples
cannot certify a complete dense radius field or identify every background
point. Unknowns remain NaN with explicit validity/state arrays. Nearest-node
association is continuous-valued, not guaranteed spatially continuous.
"""
import json

UNKNOWN = 255
KNOWN_BACKGROUND = 0
AVAILABLE_RADIUS = 1
REASONS = dict(outside_observed_domain=1, invalid_geometry=2, no_nodes=4,
    no_local_radius_support=8, invalid_node_radius=16, junction_or_unknown_structure=32,
    radius_ball_FOV_censored=64, nonidentifiable_nearest_tie=128,
    conflicting_background_certificate=256)
DEFINITION = dict(schema='GTA-AVT-local-continuous-radius-oracle-v1',
    radius='log(radius_mm / 1mm); stored radius is native physical EDT sampled at resampled Lee skeleton nodes, not clinical lumen radius',
    association='Nearest physical skeleton node; distance<=its radius; equal-distance differing radii or labels are unknown',
    support='Only known nonjunction nodes (structure==1), positive finite radius, entire radius ball strictly inside the observed domain',
    ambiguity='Any covering stored junction/unknown or FOV-censored radius ball also makes the query unknown',
    absent_support='No sparse-node support is UNKNOWN, not a vessel-negative label',
    background='Optional GT-mask interpolation-stencil certificate, shared identically by A-only and A+radius; not a model input',
    grid='Explicit grid_shape; align_corners=False nominal sample sites, not actual receptive-field centers',
    comparison='A-only and A+radius must use identical support, unknown handling and background certificates; only radius values may differ',
    limits='No dense radius cache assumed, no graph-depth proxy, no bins/quantiles, no window histogram, no CT generalization claim')


def _require(value, message):
    if not value:
        raise ValueError(message)


def query_grid(start_zyx, grid_shape, patch_shape=(128, 128, 128)):
    """Return nominal voxel-center coordinates [Z,Y,X,3] in cache coordinates."""
    import numpy as np
    start = np.asarray(start_zyx, np.float64)
    shape = np.asarray(grid_shape)
    patch = np.asarray(patch_shape, np.float64)
    _require(start.shape == shape.shape == patch.shape == (3,) and np.isfinite(start).all() and
        np.isfinite(patch).all() and (patch > 0.).all() and np.issubdtype(shape.dtype, np.integer) and (shape > 0).all(),
        'Explicit finite start, positive grid_shape and patch_shape required')
    axes = [start[i] + (np.arange(int(shape[i])) + .5) * patch[i] / shape[i] - .5 for i in range(3)]
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)


def background_stencil_certificate(mask, query_zyx):
    """Conservative all-contributing-voxel GT-background certificate.

    This uses GT only to certify background for the ORACLE comparison. A
    foreground or mixed stencil is not turned into a radius-negative label.
    Caller must additionally enforce the real observed-domain/FOV bounds.
    """
    import numpy as np
    mask = np.asarray(mask, bool); q = np.asarray(query_zyx, np.float64)
    _require(mask.ndim == 3 and q.shape[-1] == 3 and np.isfinite(q).all(), '3D mask and finite queries required')
    flat = q.reshape(-1, 3); lower = np.floor(flat).astype(np.int64); upper = np.ceil(flat).astype(np.int64)
    valid = ((lower >= 0) & (upper < np.asarray(mask.shape))).all(1)
    result = valid.copy()
    for z in (0, 1):
        for y in (0, 1):
            for x in (0, 1):
                index = np.column_stack((upper[:, 0] if z else lower[:, 0],
                    upper[:, 1] if y else lower[:, 1], upper[:, 2] if x else lower[:, 2]))
                safe = np.clip(index, 0, np.asarray(mask.shape) - 1)
                result &= ~mask[tuple(safe.T)]
    return result.reshape(q.shape[:-1])


def radius_at_queries(geometry, query_zyx, cache_shape_zyx, spacing_zyx_mm,
                      observed_bounds_mm=None, known_background=None):
    """Evaluate a partial oracle at explicit queries, without loading any data.

    observed_bounds_mm is optional [lower,upper] in the SAME cache-axis-mm
    frame as points*spacing. It intersects the conservative cache-center
    domain. Supply tighter native-label/FOV bounds if known; never enlarge
    this domain to fill padding. known_background is an optional boolean
    certificate with the query-grid shape, not an inferred default.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    q = np.asarray(query_zyx, np.float64)
    spacing = np.asarray(spacing_zyx_mm, np.float64); shape = np.asarray(cache_shape_zyx)
    _require(q.ndim >= 2 and q.shape[-1] == 3 and np.isfinite(q).all() and
        spacing.shape == shape.shape == (3,) and np.isfinite(spacing).all() and (spacing > 0.).all() and
        np.issubdtype(shape.dtype, np.integer) and (shape > 0).all(), 'Valid queries, cache shape and physical spacing required')
    output_shape = q.shape[:-1]; flat = q.reshape(-1, 3); physical = flat * spacing
    lower, upper = np.zeros(3), (shape - 1) * spacing
    if observed_bounds_mm is not None:
        bounds = np.asarray(observed_bounds_mm, np.float64)
        _require(bounds.shape == (2, 3) and np.isfinite(bounds).all() and (bounds[1] >= bounds[0]).all(), 'Invalid observed-domain bounds')
        lower = np.maximum(lower, bounds[0]); upper = np.minimum(upper, bounds[1])
    _require((upper >= lower).all(), 'Empty observed-domain intersection')
    real = ((physical >= lower) & (physical <= upper)).all(1)
    count = len(flat); state = np.full(count, UNKNOWN, np.uint8); reason = np.zeros(count, np.uint16)
    reason[~real] |= REASONS['outside_observed_domain']
    nearest = np.full(count, -1, np.int64); distance = np.full(count, np.nan)
    value = np.full(count, np.nan); radius_value = np.full(count, np.nan)
    points = np.asarray(geometry['points']); radii = np.asarray(geometry['radius'], np.float64)
    structure = np.asarray(geometry['structure'])
    _require(points.ndim == 2 and points.shape[1] == 3 and np.issubdtype(points.dtype, np.integer) and
        np.isfinite(points).all() and radii.shape == structure.shape == (len(points),) and
        ((points >= 0) & (points < shape)).all(), 'Malformed sparse geometry coordinates or node arrays')
    _require('geometry_valid' in geometry, 'Explicit cached geometry_valid is required')
    valid_geometry = bool(np.asarray(geometry['geometry_valid']).item())
    if not valid_geometry:
        reason[real] |= REASONS['invalid_geometry']
    elif not len(points):
        reason[real] |= REASONS['no_nodes']
    else:
        p = points.astype(np.float64) * spacing
        tree = cKDTree(p); k = min(2, len(p)); d, indices = tree.query(physical, k=k)
        if k == 1:
            d = d[:, None]; indices = indices[:, None]
        nearest[:] = indices[:, 0]; distance[:] = d[:, 0]
        finite_radius = np.isfinite(radii) & (radii > 0.)
        safe_ball = finite_radius & ((p - radii[:, None] > lower) & (p + radii[:, None] < upper)).all(1)
        r = radii[nearest]; types = structure[nearest]
        reason[real & ~finite_radius[nearest]] |= REASONS['invalid_node_radius']
        reason[real & finite_radius[nearest] & (distance > r)] |= REASONS['no_local_radius_support']
        reason[real & (types != 1)] |= REASONS['junction_or_unknown_structure']
        reason[real & ~safe_ball[nearest]] |= REASONS['radius_ball_FOV_censored']
        eligible = real & finite_radius[nearest] & safe_ball[nearest] & (types == 1) & (distance <= r)
        # At lattice-aligned query sites, ties are common. Equal-radius ties
        # identify the same VALUE; unequal values are not silently averaged.
        if k == 2:
            tol = 16. * np.finfo(np.float64).eps * np.maximum(1., distance)
            tied = eligible & (np.abs(d[:, 1] - distance) <= tol)
            for index in np.flatnonzero(tied):
                candidates = np.asarray(tree.query_ball_point(physical[index], distance[index] + tol[index]), np.int64)
                exact = np.linalg.norm(p[candidates] - physical[index], axis=1)
                candidates = candidates[np.abs(exact - distance[index]) <= tol[index]]
                identifiable = ((structure[candidates] == 1).all() and safe_ball[candidates].all() and
                    finite_radius[candidates].all() and (radii[candidates] == r[index]).all())
                if not identifiable:
                    eligible[index] = False; reason[index] |= REASONS['nonidentifiable_nearest_tie']
        # A nearby rejected or junction region cannot become nonjunction by
        # choosing a different node whose physical support overlaps it.
        bad = finite_radius & ((structure != 1) | ~safe_ball)
        if bad.any():
            bad_ids = np.flatnonzero(bad); bad_tree = cKDTree(p[bad])
            reach = float(radii[bad].max())
            for index in np.flatnonzero(eligible):
                candidates = bad_ids[np.asarray(bad_tree.query_ball_point(physical[index], reach), np.int64)]
                if len(candidates):
                    covers = np.linalg.norm(p[candidates] - physical[index], axis=1) <= radii[candidates]
                    candidates = candidates[covers]
                    if len(candidates):
                        eligible[index] = False
                        if (structure[candidates] != 1).any():
                            reason[index] |= REASONS['junction_or_unknown_structure']
                        if (~safe_ball[candidates]).any():
                            reason[index] |= REASONS['radius_ball_FOV_censored']
        state[eligible] = AVAILABLE_RADIUS
        radius_value[eligible] = r[eligible]; value[eligible] = np.log(r[eligible])
    if known_background is not None:
        background = np.asarray(known_background)
        _require(background.shape == output_shape and background.dtype == np.bool_, 'Background certificate must be an explicit boolean query-grid array')
        background = background.reshape(-1) & real
        conflict = background & (state == AVAILABLE_RADIUS)
        state[background & ~conflict] = KNOWN_BACKGROUND
        state[conflict] = UNKNOWN; value[conflict] = np.nan; radius_value[conflict] = np.nan
        reason[conflict] |= REASONS['conflicting_background_certificate']
    a_valid = state != UNKNOWN; r_valid = state == AVAILABLE_RADIUS
    raw_a = np.full(count, np.nan, np.float32); raw_a[a_valid] = r_valid[a_valid].astype(np.float32)
    arrays = dict(rawA=raw_a, rawR=value, radius_mm=radius_value, A_valid=a_valid,
        R_valid=r_valid, measurable_support=r_valid.copy(), state=state, reason=reason,
        nearest_node_index=nearest, nearest_node_distance_mm=distance)
    return dict(**{key: array.reshape(output_shape) for key, array in arrays.items()},
        definition=dict(DEFINITION), reason_codes=dict(REASONS),
        domain_bounds_mm=np.stack((lower, upper)), background_certificate_used=known_background is not None)


def patch_oracle(geometry, start_zyx, grid_shape, cache_shape_zyx, spacing_zyx_mm,
                 observed_bounds_mm=None, known_background=None, patch_shape=(128, 128, 128)):
    """Explicit grid required: SAAM64^3 is nominal .5+2u for128 inputs.

    The classifier's64x32^3 feature tensor is a different grid from the
    decoder SAAM64x64^3 tensor. Neither nominal grid defines actual RF centers.
    """
    queries = query_grid(start_zyx, grid_shape, patch_shape)
    result = radius_at_queries(geometry, queries, cache_shape_zyx, spacing_zyx_mm,
        observed_bounds_mm=observed_bounds_mm, known_background=known_background)
    result.update(grid_shape=tuple(grid_shape), patch_shape=tuple(patch_shape))
    return result




"""Physical cumulative Gaussian-field supervision for the next AVT pilot.

This preserves the legacy sum-of-centreline-kernels target family. The raw sum
is NOT bounded by one. We explicitly supervise log1p(sum) / fixed_train_scale
with a positive softplus prediction. This is a declared AVT adaptation, not a
byte-identical reproduction of the legacy Gaussian target or output head.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianFieldSpec:
    sigma_scale: float = 1.0
    # Numerical protection, not a claimed resolvable or clinical vessel size.
    sigma_floor_mm: float = 1e-3
    # One scalar fixed from TRAIN, never a per-case/patch maximum.
    field_scale: float = 1.0
    query_chunk: int = 512
    center_chunk: int = 1024

    def __post_init__(self):
        for name in ("sigma_scale", "sigma_floor_mm", "field_scale"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if self.query_chunk <= 0 or self.center_chunk <= 0:
            raise ValueError("Chunk sizes must be positive")


def _triple(value, name, positive=False):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(name + " must have three finite z,y,x values")
    if positive and (result <= 0).any():
        raise ValueError(name + " must be positive")
    return result


class CumulativeRadiusGeometry:
    """Complete-case centreline kernels; reuse this object for many patches."""

    def __init__(self, points_zyx, radius_mm, spacing_mm=(1., 1., 1.),
                 geometry_valid=True, spec=None):
        self.spec = GaussianFieldSpec() if spec is None else spec
        self.spacing_mm = _triple(spacing_mm, "spacing_mm", positive=True)
        points = np.asarray(points_zyx, dtype=np.float64)
        radius = np.asarray(radius_mm, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_zyx must have shape [N,3]")
        if radius.shape != (len(points),):
            raise ValueError("radius_mm must have one value per centreline point")
        if not np.isfinite(points).all() or not np.isfinite(radius).all():
            raise ValueError("Geometry must be finite")
        if len(radius) and (radius <= 0).any():
            raise ValueError("Physical centreline radii must be strictly positive")
        if len(points) and len(np.unique(points, axis=0)) != len(points):
            raise ValueError("Centreline seeds must be unique; duplicates alter sum")
        self.points_zyx = points
        self.radius_mm = radius
        self.sigma_mm = np.maximum(radius * self.spec.sigma_scale,
                                   self.spec.sigma_floor_mm)
        self.geometry_valid = bool(geometry_valid)

    def evaluate_sum(self, query_zyx):
        """Exact all-seed sum at query coordinates, with bounded CPU workspace.

        No distance truncation, nearest-seed assignment or max approximation.
        Complexity is N_query * N_center kernels; only foreground queries are
        needed by target(). All distance/kernel arithmetic is float64 here.
        """
        query = np.asarray(query_zyx, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 3:
            raise ValueError("query_zyx must have shape [N,3]")
        if not np.isfinite(query).all():
            raise ValueError("Queries must be finite")
        if not self.geometry_valid:
            raise ValueError("Cannot evaluate an invalid geometry")
        output = np.zeros(len(query), dtype=np.float64)
        for begin in range(0, len(query), self.spec.query_chunk):
            stop = min(begin + self.spec.query_chunk, len(query))
            query_part = query[begin:stop]
            total = np.zeros(len(query_part), dtype=np.float64)
            for cb in range(0, len(self.points_zyx), self.spec.center_chunk):
                ce = min(cb + self.spec.center_chunk, len(self.points_zyx))
                delta = (query_part[:, None, :] -
                         self.points_zyx[None, cb:ce, :]) * self.spacing_mm
                distance_squared = np.einsum("qcd,qcd->qc", delta, delta)
                variance = self.sigma_mm[None, cb:ce] ** 2
                total += np.exp(-.5 * distance_squared / variance).sum(axis=1)
            output[begin:stop] = total
        return output

    def target(self, mask, start_zyx=(0, 0, 0), input_valid=None):
        """Return float32 transformed field, boolean validity and target audit.

        mask is an UN-AUGMENTED 3D patch in the complete-case geometry grid.
        start_zyx locates it in the full case. Build/crop before augmentation;
        then apply the same flips/rotations to mask, target and validity.
        Complete-case seeds outside the crop remain available, so merely
        cropping does not alter a physical voxel's supervised field.

        True empty background has zero field. Foreground with missing geometry
        is unknown and excluded; background remains known zero. input_valid
        independently removes CT padding. Targets are never network inputs.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 3 or any(n <= 0 for n in mask.shape):
            raise ValueError("mask must be a nonempty-shape 3D array")
        start = _triple(start_zyx, "start_zyx")
        if input_valid is None:
            valid = np.ones(mask.shape, dtype=bool)
        else:
            valid = np.asarray(input_valid, dtype=bool)
            if valid.shape != mask.shape:
                raise ValueError("input_valid shape must match mask")
            valid = valid.copy()
        target = np.zeros(mask.shape, dtype=np.float32)
        foreground = mask & valid
        flat = np.flatnonzero(foreground)
        audit = {
            "target_definition": "physical-cumulative-log1p-Gaussian-v1",
            "sigma_scale": float(self.spec.sigma_scale),
            "sigma_floor_mm": float(self.spec.sigma_floor_mm),
            "fixed_train_field_scale": float(self.spec.field_scale),
            "spacing_zyx_mm": self.spacing_mm.tolist(),
            "start_zyx": start.tolist(),
            "complete_case_centerline_points": len(self.points_zyx),
            "foreground_voxels_in_input": int(len(flat)),
            "kernel_evaluations": int(len(flat) * len(self.points_zyx)),
            "geometry_valid": self.geometry_valid,
            "target_used_as_model_input": False,
            "foreground_unknown_voxels": 0,
            "kernel_truncation": None,
        }
        raw_sum_min, raw_sum_max = None, None
        if len(flat) and (not self.geometry_valid or not len(self.points_zyx)):
            valid[foreground] = False
            audit["foreground_unknown_voxels"] = int(len(flat))
            audit["status"] = "foreground_geometry_unknown"
        elif len(flat):
            for begin in range(0, len(flat), self.spec.query_chunk):
                indices = flat[begin:begin + self.spec.query_chunk]
                coords = np.stack(np.unravel_index(indices, mask.shape), axis=1)
                raw = self.evaluate_sum(coords + start)
                values = np.log1p(raw) / self.spec.field_scale
                target.ravel()[indices] = values.astype(np.float32)
                raw_sum_min = (float(raw.min()) if raw_sum_min is None
                               else min(raw_sum_min, float(raw.min())))
                raw_sum_max = (float(raw.max()) if raw_sum_max is None
                               else max(raw_sum_max, float(raw.max())))
            audit["status"] = "supported_foreground"
        else:
            audit["status"] = "true_empty_or_no_valid_foreground"
        audit.update(
            valid_voxels=int(valid.sum()),
            known_background_voxels=int((valid & ~mask).sum()),
            target_min=float(target.min()), target_max=float(target.max()),
            raw_sum_foreground_min=raw_sum_min,
            raw_sum_foreground_max=raw_sum_max,
            foreground_target_mean=(float(target[mask & valid].mean())
                                    if (mask & valid).any() else None),
        )
        return target, valid, audit


def gaussian_target(mask, points_zyx, radius_mm, spacing_mm=(1., 1., 1.),
                    start_zyx=(0, 0, 0), input_valid=None,
                    geometry_valid=True, spec=None):
    """Convenience wrapper; reuse CumulativeRadiusGeometry for many patches."""
    geometry = CumulativeRadiusGeometry(points_zyx, radius_mm, spacing_mm,
                                        geometry_valid, spec)
    return geometry.target(mask, start_zyx, input_valid)


def gdf_mse(prediction, target, valid, foreground=None, normalization="voxel"):
    """FP32 MSE of an already-softplus prediction, per valid case.

    Inputs have identical [B,1,D,H,W] shapes. "voxel" is the first candidate,
    closest to historical whole-volume MSE. "balanced" explicitly averages
    foreground/background means when each exists and must be predeclared.
    Unknown samples are excluded from the case mean; an all-unknown batch has
    graph-connected zero loss. Positive predictions and targets may exceed one.
    """
    import torch
    if prediction.ndim != 5 or prediction.shape[1] != 1:
        raise ValueError("Expected B,1,D,H,W prediction")
    if target.shape != prediction.shape or valid.shape != prediction.shape:
        raise ValueError("GDF prediction, target and validity shapes must match")
    if normalization not in ("voxel", "balanced"):
        raise ValueError("Unknown GDF MSE normalization")
    if foreground is not None and foreground.shape != prediction.shape:
        raise ValueError("foreground shape must match prediction")
    if normalization == "balanced" and foreground is None:
        raise ValueError("Balanced GDF MSE requires a foreground mask")
    weights = valid.float()
    safe_target = torch.where(weights > 0, target.float(),
                              torch.zeros_like(target, dtype=torch.float32))
    squared = (prediction.float() - safe_target).square()
    axes = (1, 2, 3, 4)
    counts = weights.sum(axes)
    per_case = (squared * weights).sum(axes) / counts.clamp_min(1.)
    valid_case = (counts > 0).to(squared.dtype)
    details = {
        "gdf_normalization": normalization,
        "gdf_prediction_transform": "already_softplus_in_model",
        "gdf_valid_cases": int(valid_case.sum().detach()),
        "gdf_valid_voxels": int(counts.sum().detach()),
    }
    if foreground is not None:
        fg = weights * foreground.float()
        bg = weights * (1. - foreground.float())
        fg_count, bg_count = fg.sum(axes), bg.sum(axes)
        fg_mean = (squared * fg).sum(axes) / fg_count.clamp_min(1.)
        bg_mean = (squared * bg).sum(axes) / bg_count.clamp_min(1.)
        fg_present = (fg_count > 0).to(squared.dtype)
        bg_present = (bg_count > 0).to(squared.dtype)
        details.update(
            gdf_foreground_mse=float(
                ((fg_mean * fg_present).sum() /
                 fg_present.sum().clamp_min(1.)).detach()),
            gdf_background_mse=float(
                ((bg_mean * bg_present).sum() /
                 bg_present.sum().clamp_min(1.)).detach()),
            gdf_foreground_cases=int(fg_present.sum().detach()),
            gdf_background_cases=int(bg_present.sum().detach()),
        )
        if normalization == "balanced":
            per_case = ((fg_mean * fg_present + bg_mean * bg_present) /
                        (fg_present + bg_present).clamp_min(1.))
    loss = (per_case * valid_case).sum() / valid_case.sum().clamp_min(1.)
    details["gdf_mse"] = float(loss.detach())
    return loss, details
