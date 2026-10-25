import json

import numpy as np
import pytest

from gta_avt.data import (
    PreparedDataset, index_avt, normalize, prepare_ct, restore_native,
    sliding_probability, train_statistics, _transfer_array,
)
from gta_avt.intensity import detect_ct_offset


def _cache(path):
    path.mkdir()
    shape = (9, 11, 13)
    mask = np.zeros(shape, np.uint8)
    mask[2:7, 5, 6] = 1
    image = np.full(shape, -1000, np.int16)
    image[:, 3:8, 3:10] = 300
    valid = np.ones(shape, np.uint8)
    for key, value in dict(image=image, mask=mask, valid=valid, gdf=mask.astype(np.float32), gdf_valid=valid).items():
        np.save(path / f'{key}.npy', value)
    np.savez(path / 'geometry.npz', points=np.argwhere(mask).astype(np.int32),
             radius=np.ones(int(mask.sum()), np.float32), structure=np.ones(int(mask.sum()), np.uint8), geometry_valid=np.uint8(1))
    return shape


def test_hu_normalization_and_ambiguous_offset():
    np.testing.assert_allclose(normalize(np.asarray([-1000., -200., 200., 1000., 3000.])), [-1., -1., 0., 2., 2.])
    assert detect_ct_offset(np.full((20, 20, 20), 350.))['requires_review']
    source = np.full((20, 20, 20), -1000.)
    source[:, 5:15, 5:15] = 350.
    assert detect_ct_offset(source)['offset_subtract'] == 0.
    shifted = source + 1024.
    assert detect_ct_offset(shifted)['offset_subtract'] == 1024.


def test_train_only_statistics_and_padded_dataset(tmp_path):
    torch = pytest.importorskip('torch')
    shape = _cache(tmp_path / 'TRAIN_A')
    rows = [dict(id='TRAIN_A', split='TRAIN', cache_dir='TRAIN_A', spacing_mm=1.),
            dict(id='VAL_NEVER_OPEN', split='VAL', cache_dir='missing', spacing_mm=1.)]
    stats = train_statistics(rows, tmp_path, sample_per_case=1000, gdf_sample_per_case=5, seed=7)
    assert stats['training_case_ids'] == ['TRAIN_A'] and stats['mu'] == 0.
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema='gta-avt-prepared-v1', cases=rows, statistics=stats)), encoding='utf-8')
    dataset = PreparedDataset(manifest, patch_size=16, samples_per_epoch=2, seed=4)
    first, second = dataset[0], dataset[0]
    for key in ('image', 'label', 'valid', 'support', 'r', 'r_valid', 'support_valid', 'gdf', 'gdf_valid'):
        assert torch.equal(first[key], second[key])
        assert torch.isfinite(first[key]).all()
    assert first['image'].shape == (1, 16, 16, 16)
    assert first['support'].shape == (1, 8, 8, 8)
    assert first['valid'].sum() == np.prod(shape)
    assert first['valid'][0, 12, 12, 12] == 0
    assert (first['r_valid'] <= first['support_valid']).all()
    assert 'a' not in first and 'a_valid' not in first, 'Geometry validity is no longer a model attribute'
    assert (first['r'][first['r_valid'] == 0] == 0).all()
    assert first['support_valid'].sum() > first['r_valid'].sum()
    assert first['support'].sum() > 0 and first['support'].max() < 1, 'Thin lumen has fractional occupancy'


def test_ct_only_native_header_and_padding(tmp_path):
    sitk = pytest.importorskip('SimpleITK')
    array = np.full((16, 18, 20), -1000., np.float32)
    array[:, 5:13, 5:15] = 300.
    array[0] = -2000.
    original = sitk.GetImageFromArray(array)
    original.SetOrigin((9., -4., 8.))
    original.SetSpacing((1.2, .9, 1.5))
    original.SetDirection((-1., 0., 0., 0., -1., 0., 0., 0., 1.))
    image_file = tmp_path / 'ct.nrrd'
    sitk.WriteImage(original, str(image_file))
    image, metadata = prepare_ct(image_file, tmp_path / 'ct_only')
    assert metadata['label_required'] is False
    assert not np.load(tmp_path / 'ct_only' / 'valid.npy')[0].any()
    probability, tiles = sliding_probability(lambda x: np.full((len(x), 1, 16, 16, 16), .25, np.float32), image, metadata, patch=16, batch_size=2)
    np.testing.assert_allclose(probability, .25, atol=1e-6)
    assert tiles > 1
    restored = restore_native(probability, metadata)
    assert restored.GetOrigin() == original.GetOrigin()
    assert restored.GetDirection() == original.GetDirection()
    assert restored.GetSpacing() == original.GetSpacing()
    assert restored.GetSize() == original.GetSize()


def test_index_requires_explicit_split_and_duplicate_protection(tmp_path):
    source = tmp_path / 'KiTS'
    source.mkdir()
    (source / 'K1.nrrd').write_bytes(b'image')
    (source / 'K1.seg.nrrd').write_bytes(b'label')
    output = tmp_path / 'cases.json'
    indexed = index_avt(tmp_path, output)
    assert indexed['cases'][0]['split'] is None
    split = tmp_path / 'split.json'
    split.write_text(json.dumps(dict(train=['K1'], val=[], test=[])), encoding='utf-8')
    assigned = index_avt(tmp_path, tmp_path / 'assigned.json', split_file=split)
    assert assigned['cases'][0]['split'] == 'TRAIN'
    with pytest.raises(FileExistsError):
        index_avt(tmp_path, output)


def test_symlink_array_reuse_preserves_source(tmp_path):
    import hashlib
    source = tmp_path / 'immutable.npy'
    np.save(source, np.arange(12, dtype=np.int16).reshape(3, 4))
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path / 'linked.npy'
    try:
        _transfer_array(source, target, link_mode='symlink')
    except OSError as error:
        pytest.skip(f'Symlinks unavailable on this platform: {error}')
    assert target.is_symlink() and target.resolve() == source.resolve()
    np.testing.assert_array_equal(np.load(source), np.load(target))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
    with pytest.raises(ValueError):
        _transfer_array(source, tmp_path / 'bad.npy', link_mode='invalid')
