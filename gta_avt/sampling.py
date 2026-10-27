"""Deterministic TRAIN patch-source sampling; geometric targets are unchanged.

Candidate pools choose patch anchors, not new attribute labels. A selected
background/boundary anchor does not make the whole patch background/boundary.
Pools are bounded, deterministic and cached once per case by PreparedDataset.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np


CATEGORIES = ('vessel', 'fine', 'junction', 'boundary', 'hard_background', 'uniform')
OPTIONS = {'weights', 'fine_radius_mm', 'hard_background_hu', 'candidate_stride', 'max_candidates'}


def validate_sampling(sampling):
    """Return a canonical explicit policy; None preserves the legacy sampler."""
    if sampling is None:
        return None
    if not isinstance(sampling, dict):
        raise ValueError('sampling must be a dictionary or None')
    if 'weights' in sampling:
        if set(sampling) - OPTIONS:
            raise ValueError('Unknown sampling option')
        weights = sampling['weights']
        options = dict(sampling)
    else:
        weights = sampling
        options = {}
    if not isinstance(weights, dict) or set(weights) != set(CATEGORIES):
        raise ValueError('Sampling weights must explicitly contain the six known categories')
    values = []
    for key in CATEGORIES:
        value = weights[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or value < 0:
            raise ValueError('Sampling weights must be finite and nonnegative')
        values.append(float(value))
    if not math.isclose(math.fsum(values), 1., rel_tol=0., abs_tol=1e-8):
        raise ValueError('Sampling weights must sum to one')
    result = dict(weights=dict(zip(CATEGORIES, values)), fine_radius_mm=2.,
                  hard_background_hu=150., candidate_stride=4, max_candidates=50000)
    for key in ('fine_radius_mm', 'hard_background_hu'):
        value = options.get(key, result[key])
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(key + ' must be finite')
        result[key] = float(value)
    if result['fine_radius_mm'] <= 0:
        raise ValueError('fine_radius_mm must be positive')
    for key in ('candidate_stride', 'max_candidates'):
        value = options.get(key, result[key])
        if type(value) is not int or value < 1:
            raise ValueError(key + ' must be a positive integer')
        result[key] = value
    if result['max_candidates'] > 50000:
        raise ValueError('max_candidates must not exceed the bounded pool limit of 50000')
    return result


def _bounded(points, maximum):
    points = np.asarray(points, dtype=np.int64).reshape(-1, 3)
    if len(points) > maximum:
        # Cover the entire deterministic point ordering, rather than keeping
        # only the first spatial block or consuming per-patch random numbers.
        points = points[np.linspace(0, len(points) - 1, maximum, dtype=np.int64)]
    return np.ascontiguousarray(points)


def _lattice_points(selected, stride, maximum):
    flat = np.flatnonzero(selected)
    if len(flat) > maximum:
        flat = flat[np.linspace(0, len(flat) - 1, maximum, dtype=np.int64)]
    if not len(flat):
        return np.empty((0, 3), dtype=np.int64)
    return np.stack(np.unravel_index(flat, selected.shape), axis=1).astype(np.int64) * stride


def build_candidates(geometry, mask, image, valid, sampling):
    """Build six source pools without full-volume EDT/dilation.

    Vessel/fine/junction anchors use existing centreline geometry. Boundary
    anchors are real cells on either side of a label change on the coarse
    stride lattice. FOV/invalid-cell edges are not treated as vessel boundaries.
    Hard-background anchors are real, nonlumen cells above a fixed HU threshold.
    """
    policy = validate_sampling(sampling)
    if policy is None:
        raise ValueError('Explicit sampling policy required to build candidate pools')
    mask, image, valid = np.asarray(mask), np.asarray(image), np.asarray(valid)
    if mask.ndim != 3 or mask.shape != image.shape or mask.shape != valid.shape:
        raise ValueError('Sampling needs aligned 3D image, mask and validity arrays')
    stride, maximum = policy['candidate_stride'], policy['max_candidates']
    points = np.asarray(geometry['points'])
    radius = np.asarray(geometry['radius'])
    structure = np.asarray(geometry['structure'])
    if points.ndim != 2 or points.shape[1] != 3 or radius.shape != (len(points),) or structure.shape != (len(points),):
        raise ValueError('Centreline point, radius and structure arrays must align')
    if not np.isfinite(points).all() or not np.equal(points, points.astype(np.int64)).all():
        raise ValueError('Centreline sampling points must be finite integer voxel coordinates')
    points = points.astype(np.int64)
    keep = ((points >= 0) & (points < np.asarray(mask.shape))).all(axis=1)
    if not bool(np.asarray(geometry.get('geometry_valid', True)).item()):
        keep[:] = False
    supported = np.zeros(len(points), dtype=bool)
    if keep.any():
        index = tuple(points[keep].T)
        supported[keep] = valid[index].astype(bool) & mask[index].astype(bool)
    pools = dict(
        vessel=_bounded(points[supported], maximum),
        fine=_bounded(points[supported & np.isfinite(radius) & (radius > 0) &
                             (radius <= policy['fine_radius_mm'])], maximum),
        junction=_bounded(points[supported & (structure == 2)], maximum))
    coarse_mask = mask[::stride, ::stride, ::stride].astype(bool)
    coarse_real = valid[::stride, ::stride, ::stride].astype(bool)
    boundary = np.zeros(coarse_mask.shape, dtype=bool)
    for axis in range(3):
        left, right = [slice(None)] * 3, [slice(None)] * 3
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        changed = (coarse_mask[left] != coarse_mask[right]) & coarse_real[left] & coarse_real[right]
        boundary[left] |= changed
        boundary[right] |= changed
    pools['boundary'] = _lattice_points(boundary & coarse_real, stride, maximum)
    hard = (coarse_real & ~coarse_mask &
            (image[::stride, ::stride, ::stride] >= policy['hard_background_hu']))
    pools['hard_background'] = _lattice_points(hard, stride, maximum)
    pools['uniform'] = np.empty((0, 3), dtype=np.int64)
    return pools


def sample_patch_start(rng, candidates, shape, patch_size, sampling):
    """Draw one source; missing sources fall back to vessel, then uniform.

    Geometry anchors use the original +/-quarter-patch jitter. Boundary and
    hard-background anchors are kept as the centre before legal FOV clipping.
    Metadata describes the selected anchor source, not patch purity.
    """
    policy = validate_sampling(sampling)
    if policy is None:
        raise ValueError('Explicit sampling policy required')
    shape = np.asarray(shape, dtype=np.int64)
    if shape.shape != (3,) or (shape < 1).any() or type(patch_size) is not int or patch_size < 1:
        raise ValueError('Positive 3D shape and patch_size required')
    requested = str(rng.choice(CATEGORIES, p=[policy['weights'][key] for key in CATEGORIES]))
    actual, reason = requested, ''
    if actual != 'uniform' and not len(candidates[actual]):
        actual = 'vessel' if len(candidates['vessel']) else 'uniform'
        reason = 'no_' + requested + '_candidates;using_' + actual
    anchor = None
    if actual == 'uniform':
        start = np.asarray([rng.integers(max(1, int(n) - patch_size + 1)) for n in shape], dtype=np.int64)
    else:
        pool = candidates[actual]
        anchor = pool[int(rng.integers(len(pool)))].copy()
        centre = anchor.copy()
        if actual in ('vessel', 'fine', 'junction'):
            centre += rng.integers(-patch_size // 4, patch_size // 4 + 1, size=3)
        start = np.clip(centre - patch_size // 2, 0, np.maximum(shape - patch_size, 0))
    return start, dict(requested_category=requested, actual_category=actual,
                       fallback_reason=reason, anchor_zyx=anchor)
