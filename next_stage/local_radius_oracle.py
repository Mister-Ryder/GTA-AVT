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


if __name__ == '__main__':
    print(json.dumps(dict(status='inert', file_IO=False, model_inference=False, definition=DEFINITION)))
