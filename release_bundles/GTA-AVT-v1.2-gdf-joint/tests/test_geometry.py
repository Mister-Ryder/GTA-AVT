import numpy as np

from gta_avt.geometry import (
    CumulativeRadiusGeometry, GaussianFieldSpec, radius_at_queries,
    background_stencil_certificate, query_grid,
)


def geometry(points, radii, structure=None):
    return dict(points=np.asarray(points, np.int32), radius=np.asarray(radii, np.float32),
                structure=np.ones(len(points), np.uint8) if structure is None else np.asarray(structure, np.uint8),
                geometry_valid=np.uint8(1))


def test_radius_physical_support_fov_and_tie():
    value = geometry([[4, 4, 4]], [2.])
    out = radius_at_queries(value, np.asarray([[4, 4, 4], [4, 4, 7], [-1, 4, 4]]), (10, 10, 10), (1, 1, 1))
    assert out['R_valid'].tolist() == [True, False, False]
    np.testing.assert_allclose(out['rawR'][0], np.log(2.))
    censored = radius_at_queries(geometry([[1, 4, 4]], [1.]), np.asarray([[1, 4, 4]]), (10, 10, 10), (1, 1, 1))
    assert not censored['R_valid'].any(), 'Ball touching FOV is unknown'
    tied = radius_at_queries(geometry([[4, 4, 3], [4, 4, 5]], [1., 2.]), np.asarray([[4, 4, 4]]), (10, 10, 10), (1, 1, 1))
    assert not tied['R_valid'].any(), 'Different values at equal distance cannot be averaged'


def test_overlapping_junction_and_real_stencil():
    out = radius_at_queries(geometry([[5, 5, 4], [5, 5, 7]], [2., 2.], [1, 2]),
                            np.asarray([[5, 5, 5]]), (12, 12, 12), (1, 1, 1))
    assert not out['R_valid'].any()
    padding = np.zeros((4, 4, 4), bool)
    padding[2] = True
    real = background_stencil_certificate(padding, np.asarray([[.5, .5, .5], [1.5, .5, .5], [3.5, .5, .5]]))
    assert real.tolist() == [True, False, False]
    np.testing.assert_allclose(query_grid((0, 0, 0), (2, 2, 2), (4, 4, 4))[0, 0, 0], [.5, .5, .5])


def test_all_center_gaussian_physical_distance_and_crop_invariance():
    field = CumulativeRadiusGeometry([[3, 3, 3], [3, 3, 5]], [1., 2.], (2., 1., 1.),
                                    spec=GaussianFieldSpec(field_scale=2.))
    raw = field.evaluate_sum(np.asarray([[3, 3, 3], [4, 3, 3]]))
    np.testing.assert_allclose(raw, [1. + np.exp(-.5), np.exp(-2.) + np.exp(-1.)], rtol=1e-12)
    mask = np.zeros((8, 8, 8), bool)
    mask[3, 3, 3] = True
    target, valid, _ = field.target(mask)
    cropped, _, _ = field.target(mask[2:6, 2:6, 2:6], start_zyx=(2, 2, 2))
    np.testing.assert_allclose(target[3, 3, 3], np.log1p(raw[0]) / 2.)
    np.testing.assert_array_equal(cropped, target[2:6, 2:6, 2:6])
    assert valid.all() and target[0, 0, 0] == 0.


def test_unknown_foreground_gaussian_excluded_background_retained():
    mask = np.zeros((3, 3, 3), bool)
    mask[1, 1, 1] = True
    field = CumulativeRadiusGeometry(np.empty((0, 3)), np.empty(0), geometry_valid=False)
    target, valid, _ = field.target(mask)
    assert not valid[1, 1, 1] and valid[0, 0, 0]
    assert np.count_nonzero(target) == 0
