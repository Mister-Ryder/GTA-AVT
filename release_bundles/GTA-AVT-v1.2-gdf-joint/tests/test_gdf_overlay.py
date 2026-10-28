import hashlib
import json

import numpy as np
import pytest

from gta_avt.geometry import bounded_gaussian_target
from gta_avt.data import load_manifest, prepare_bounded_gdf_overlay, PreparedDataset


def test_raw_shape_recovered_before_minmax_not_log_compression():
    raw = np.asarray([0., 1., 3., 10.], np.float64).reshape(1, 1, 4)
    cached = (np.log1p(raw) / 2.87).astype(np.float32)
    target, audit = bounded_gaussian_target(cached, 2.87, np.ones(raw.shape, bool))
    np.testing.assert_allclose(target, .5 + .5 * raw / 10., atol=2e-8)
    assert audit['schema'] == 'gdf-joint-bounded-v1'
    assert target.dtype == np.float32 and target.min() == .5 and target.max() == 1.
    assert not np.allclose(target, .5 + .5 * cached / cached.max())


def test_invalid_extrema_excluded_and_filled_neutral():
    source = np.asarray([0., 1., 2., 999.], np.float32).reshape(1, 1, 4)
    valid = np.asarray([1, 1, 1, 0], bool).reshape(source.shape)
    result, audit = bounded_gaussian_target(source, 2., valid)
    assert result[0, 0, 0] == .5 and result[0, 0, 2] == 1.
    assert result[0, 0, 3] == .5 and audit['raw_log_max'] == 4.


@pytest.mark.parametrize('valid', [True, False])
def test_degenerate_whole_case_is_finite_neutral(valid):
    result, audit = bounded_gaussian_target(np.ones((3, 4, 5), np.float32), 2.,
                                            np.full((3, 4, 5), valid))
    assert np.all(result == .5) and audit['degenerate']


def test_huge_log_fields_do_not_overflow():
    source = np.asarray([0., 4000., 4001.], np.float32).reshape(1, 1, 3)
    result, _ = bounded_gaussian_target(source, 2., np.ones(source.shape, bool))
    assert np.isfinite(result).all() and result.min() == .5 and result.max() == 1.
    np.testing.assert_allclose(result[0, 0, 1], .5 + .5 * np.exp(-2.), atol=3e-8)


def test_raw_nonzero_min_is_handled():
    raw = np.asarray([2., 3., 8.], np.float64).reshape(1, 1, 3)
    result, _ = bounded_gaussian_target(np.log1p(raw), 1., np.ones(raw.shape, bool))
    np.testing.assert_allclose(result, .5 + .5 * (raw - 2.) / 6., atol=2e-8)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -1.])
def test_reject_nonfinite_or_negative_cache(bad):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        bounded_gaussian_target(np.full((1, 1, 1), bad), 1., np.zeros((1, 1, 1), bool))


def _source(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    cases = []
    for case_id, split in [('A', 'TRAIN'), ('B', 'VAL')]:
        directory = source / case_id
        directory.mkdir()
        mask = np.zeros((9, 11, 13), np.uint8)
        mask[2:7, 5, 6] = 1
        raw = mask.astype(np.float32)
        raw[4, 5, 6] = 3.
        arrays = dict(image=np.full(mask.shape, 300, np.int16), mask=mask,
                      valid=np.ones(mask.shape, np.uint8), gdf=np.log1p(raw) / 2.,
                      gdf_valid=np.ones(mask.shape, np.uint8))
        for key, value in arrays.items():
            np.save(directory / (key + '.npy'), value)
        np.savez(directory / 'geometry.npz', points=np.argwhere(mask).astype(np.int32),
                 radius=np.ones(int(mask.sum()), np.float32), structure=np.ones(int(mask.sum()), np.uint8),
                 geometry_valid=np.uint8(1))
        cases.append(dict(id=case_id, split=split, cache_dir=case_id, spacing_mm=1.))
    cases.append(dict(id='TEST_NOT_OPENED', split='TEST', cache_dir='missing'))
    manifest = source / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema='gta-avt-prepared-v1.1', cases=cases,
                         statistics=dict(mu=0., sigma=1., pi_support=.1, gdf_scale=2.))), encoding='utf-8')
    return manifest


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_overlay_reuses_inputs_excludes_test_and_resumes_verified(tmp_path):
    source = _source(tmp_path)
    hashes = {p: _hash(p) for p in source.parent.rglob('*') if p.is_file()}
    output = tmp_path / 'overlay'
    manifest = prepare_bounded_gdf_overlay(source, output, chunk_depth=2)
    assert len(manifest['cases']) == 2 and manifest['excluded_test_case_count'] == 1
    assert manifest['gdf_target']['statistics_fitted_on_val'] is False
    assert manifest['statistics'] == json.loads(source.read_text())['statistics']
    assert not list(output.rglob('image.npy')) and not list(output.rglob('geometry.npz'))
    for path, value in hashes.items():
        assert _hash(path) == value, 'All old cache arrays and manifest remain unchanged'
    loaded = load_manifest(output / 'manifest.json')
    assert loaded['cases'][0]['cache_dir'] == str((source.parent / 'A').resolve())
    assert loaded['cases'][0]['gdf_file'] == str((output / 'A' / 'gdf.npy').resolve())
    original = _hash(output / 'A' / 'gdf.npy')
    prepare_bounded_gdf_overlay(source, output, chunk_depth=1, resume=True)
    assert _hash(output / 'A' / 'gdf.npy') == original
    with pytest.raises(FileExistsError):
        prepare_bounded_gdf_overlay(source, output)


def test_padded_dataset_reads_overlay_and_keeps_valid_mask(tmp_path):
    torch = pytest.importorskip('torch')
    source = _source(tmp_path)
    output = tmp_path / 'overlay'
    prepare_bounded_gdf_overlay(source, output)
    new = PreparedDataset(output / 'manifest.json', patch_size=16, samples_per_epoch=1, seed=5)[0]
    old = PreparedDataset(source, patch_size=16, samples_per_epoch=1, seed=5)[0]
    for key in ['image', 'label', 'valid', 'gdf_valid', 'support', 'r', 'r_valid']:
        assert torch.equal(new[key], old[key]), 'Overlay must not change sampling or attribute inputs'
    assert new['gdf'].min().item() == .5 and new['gdf'].max().item() == 1.
    assert new['gdf'][new['gdf_valid'] == 0].unique().tolist() == [.5]
    assert old['gdf'][old['gdf_valid'] == 0].unique().tolist() == [0.]


def test_wholecase_transform_is_chunk_invariant_and_not_patch_refit():
    source = np.linspace(0., 3., 64, dtype=np.float32).reshape(4, 4, 4)
    valid = np.ones(source.shape, bool)
    complete, _ = bounded_gaussian_target(source, 2., valid, chunk_depth=1)
    other, _ = bounded_gaussian_target(source, 2., valid, chunk_depth=3)
    np.testing.assert_array_equal(complete, other)
    refit, _ = bounded_gaussian_target(source[:2], 2., valid[:2])
    assert complete[:2].max() < 1. and refit.max() == 1.


def test_resume_refuses_modified_source(tmp_path):
    source = _source(tmp_path)
    output = tmp_path / 'overlay'
    prepare_bounded_gdf_overlay(source, output)
    array_file = source.parent / 'A' / 'gdf.npy'
    array = np.load(array_file)
    array[2, 5, 6] = .8
    np.save(array_file, array)
    with pytest.raises(ValueError, match='immutable source/target bytes'):
        prepare_bounded_gdf_overlay(source, output, resume=True)
