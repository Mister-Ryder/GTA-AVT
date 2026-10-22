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
