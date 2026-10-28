"""Synthetic checks of patch-source selection, not dataset training."""
import json
from unittest.mock import patch

import numpy as np
import pytest

from gta_avt.data import PreparedDataset
from gta_avt.sampling import CATEGORIES, build_candidates, sample_patch_start, validate_sampling


def weights(**values):
    return {key: float(values.get(key, 0.)) for key in CATEGORIES}


def synthetic_arrays():
    shape = (32, 32, 32)
    mask = np.zeros(shape, np.uint8)
    mask[10:23, 10:23, 10:23] = 1
    image = np.full(shape, -1000., np.float32)
    image[mask > 0] = 350.
    image[4:9, 4:9, 4:9] = 300.
    valid = np.ones(shape, np.uint8)
    valid[:4] = 0
    image[:4] = 700.
    geometry = dict(points=np.asarray([[12, 12, 12], [16, 16, 16], [20, 20, 20]], np.int32),
        radius=np.asarray([1.5, 5., 3.], np.float32), structure=np.asarray([1, 1, 2], np.uint8),
        geometry_valid=np.asarray(1, np.uint8))
    return mask, image, valid, geometry


def make_cache(tmp_path):
    mask, image, valid, geometry = synthetic_arrays()
    directory = tmp_path / 'case'; directory.mkdir()
    for key, value in dict(mask=mask, image=image, valid=valid,
                           gdf=mask.astype(np.float32), gdf_valid=valid).items():
        np.save(directory / (key + '.npy'), value)
    np.savez(directory / 'geometry.npz', **geometry)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema='gta-avt-prepared-v1.1',
        statistics=dict(mu=0., sigma=1., pi=.1, pi_support=.1, gdf_scale=1.),
        cases=[dict(id='train', split='TRAIN', cache_dir='case', spacing_mm=1.),
               dict(id='val', split='VAL', cache_dir='case', spacing_mm=1.)])), encoding='utf8')
    return manifest


def test_policy_validation_is_explicit_and_bounded():
    mixture = weights(vessel=.4, fine=.2, junction=.1, boundary=.1, hard_background=.1, uniform=.1)
    assert validate_sampling(None) is None
    assert validate_sampling(mixture)['weights'] == mixture
    configured = validate_sampling(dict(weights=mixture, candidate_stride=2, max_candidates=3))
    assert configured['candidate_stride'] == 2 and configured['max_candidates'] == 3
    invalid = [dict(mixture, typo=0.), {'vessel': 1.}, weights(vessel=.8),
               weights(vessel=1.1, uniform=-.1), weights(vessel=float('nan')),
               dict(weights=mixture, unknown=True), dict(weights=mixture, candidate_stride=0),
               dict(weights=mixture, candidate_stride=True), dict(weights=mixture, max_candidates=50001),
               dict(weights=mixture, fine_radius_mm=0), dict(weights=mixture, hard_background_hu=float('inf'))]
    for policy in invalid:
        with pytest.raises(ValueError): validate_sampling(policy)


def test_source_probabilities_and_determinism():
    mixture = weights(vessel=.1, fine=.2, junction=.15, boundary=.15, hard_background=.2, uniform=.2)
    pools = {key: np.asarray([[16, 16, 16]], dtype=np.int64) for key in CATEGORIES}
    rng = np.random.default_rng(937)
    records = [sample_patch_start(rng, pools, (64, 64, 64), 16, mixture)[1] for _ in range(6000)]
    for category in CATEGORIES:
        observed = sum(row['requested_category'] == category for row in records) / len(records)
        assert abs(observed - mixture[category]) < .025
    left, right = np.random.default_rng(21), np.random.default_rng(21)
    for _ in range(30):
        a, am = sample_patch_start(left, pools, (64, 64, 64), 16, mixture)
        b, bm = sample_patch_start(right, pools, (64, 64, 64), 16, mixture)
        np.testing.assert_array_equal(a, b)
        assert am['requested_category'] == bm['requested_category']
        assert am['actual_category'] == am['requested_category']
        assert am['fallback_reason'] == ''


def test_missing_sources_fall_back_without_claiming_missing_anatomy():
    pools = {key: np.empty((0, 3), dtype=np.int64) for key in CATEGORIES}
    pools['vessel'] = np.asarray([[16, 16, 16]], dtype=np.int64)
    _, record = sample_patch_start(np.random.default_rng(1), pools, (32, 32, 32), 16, weights(junction=1))
    assert record['requested_category'] == 'junction' and record['actual_category'] == 'vessel'
    assert record['fallback_reason'] == 'no_junction_candidates;using_vessel'
    pools['vessel'] = np.empty((0, 3), dtype=np.int64)
    start, record = sample_patch_start(np.random.default_rng(1), pools, (9, 11, 13), 16, weights(fine=1))
    assert record['actual_category'] == 'uniform'
    assert record['fallback_reason'] == 'no_fine_candidates;using_uniform'
    np.testing.assert_array_equal(start, (0, 0, 0))


def test_candidate_semantics_and_bounded_spatial_pools():
    mask, image, valid, geometry = synthetic_arrays()
    policy = dict(weights=weights(uniform=1), candidate_stride=4)
    pools = build_candidates(geometry, mask, image, valid, policy)
    np.testing.assert_array_equal(pools['fine'], [[12, 12, 12]])
    np.testing.assert_array_equal(pools['junction'], [[20, 20, 20]])
    assert len(pools['boundary']) > 0 and len(pools['hard_background']) > 0
    hard = tuple(pools['hard_background'].T)
    assert valid[hard].all() and not mask[hard].any() and (image[hard] >= 150.).all()
    boundary = pools['boundary']
    assert valid[tuple(boundary.T)].all()
    # A coarse lattice boundary is near a true cubical label transition; it is
    # not a full-volume EDT or a new pseudo-anatomical supervision target.
    from scipy.ndimage import distance_transform_edt
    distance_to_opposite = np.where(mask, distance_transform_edt(mask), distance_transform_edt(~mask.astype(bool)))
    assert (distance_to_opposite[tuple(boundary.T)] <= 4).all()
    for category in ('boundary', 'hard_background'):
        start, row = sample_patch_start(np.random.default_rng(3), pools, mask.shape, 16, weights(**{category: 1}))
        anchor = row['anchor_zyx']
        assert row['actual_category'] == category
        np.testing.assert_array_equal(start, np.clip(anchor - 8, 0, 16))
        assert ((anchor >= start) & (anchor < start + 16)).all()
    capped = build_candidates(geometry, mask, image, valid,
                             dict(weights=weights(uniform=1), candidate_stride=1, max_candidates=2))
    assert all(len(points) <= 2 for points in capped.values())


def test_invalid_fov_edge_is_not_a_boundary_and_invalid_geometry_is_not_a_vessel():
    mask = np.ones((12, 12, 12), np.uint8)
    valid = np.ones_like(mask); valid[:4] = 0
    geometry = dict(points=np.asarray([[5, 5, 5]], np.int32), radius=np.ones(1),
                    structure=np.asarray([2]), geometry_valid=np.asarray(0))
    pools = build_candidates(geometry, mask, np.full(mask.shape, 300., np.float32), valid, weights(uniform=1))
    assert all(len(points) == 0 for points in pools.values())


def test_dataset_metadata_cache_and_stateless_epoch_sampling(tmp_path):
    torch = pytest.importorskip('torch')
    manifest = make_cache(tmp_path)
    policy = weights(vessel=.2, fine=.2, junction=.2, boundary=.2, hard_background=.1, uniform=.1)
    from gta_avt.sampling import build_candidates as builder
    with patch('gta_avt.sampling.build_candidates', wraps=builder) as build:
        dataset = PreparedDataset(manifest, patch_size=16, samples_per_epoch=30, seed=13, sampling=policy)
        first, repeated = dataset[0], dataset[0]
        assert build.call_count == 1
        for key in ('image', 'label', 'valid', 'support', 'r', 'r_valid', 'support_valid', 'gdf', 'gdf_valid'):
            assert torch.equal(first[key], repeated[key])
        for key in ('sampling_requested', 'sampling_actual', 'sampling_fallback'):
            assert isinstance(first[key], str) and first[key] == repeated[key]
        assert first['sampling_requested'] in CATEGORIES
        epoch0 = [dataset[i]['sampling_requested'] for i in range(20)]
        dataset.set_epoch(1)
        epoch1 = [dataset[i]['sampling_requested'] for i in range(20)]
        assert epoch0 != epoch1
        dataset.set_epoch(0)
        assert epoch0 == [dataset[i]['sampling_requested'] for i in range(20)]
        assert build.call_count == 1
        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0)))
        assert len(batch['sampling_requested']) == 2
        assert (batch['r_valid'] <= batch['support_valid']).all()
        assert (batch['r'][batch['r_valid'] == 0] == 0).all()


def test_val_rejects_curriculum_and_keeps_legacy_sampling(tmp_path):
    manifest = make_cache(tmp_path)
    with pytest.raises(ValueError, match='TRAIN-only'):
        PreparedDataset(manifest, split='VAL', patch_size=16, sampling=weights(vessel=1))
    dataset = PreparedDataset(manifest, split='VAL', patch_size=16, samples_per_epoch=4, seed=0)
    assert dataset.sampling is None
    assert dataset[0]['case_id'] == 'val'
    assert dataset[0]['sampling_requested'] in ('vessel', 'uniform')
    assert dataset[0]['sampling_fallback'] == ''
