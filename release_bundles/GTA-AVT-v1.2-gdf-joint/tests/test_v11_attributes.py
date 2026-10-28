"""Support semantics, partial regression and non-background-only GDF checks."""
import json
import numpy as np
import pytest

from gta_avt.data import support_occupancy, train_statistics, prepare_dataset, import_prepared_cache


def test_occupancy_real_normalization_and_padding():
    mask = np.zeros((3, 2, 2), bool)
    real = np.zeros_like(mask)
    real[0, 0, :2] = True
    mask[0, 0, 0] = True
    # Annotation in padding is ignored, rather than manufactured as real foreground.
    mask[2, 1, 1] = True
    value, valid = support_occupancy(mask, real)
    np.testing.assert_array_equal(valid[:, 0, 0], [True, False])
    np.testing.assert_allclose(value[:, 0, 0], [.5, 0.])


def test_support_prior_is_train_only_and_distinct_from_radius_emission(tmp_path):
    cache = tmp_path / 'T'; cache.mkdir()
    mask = np.zeros((10, 10, 10), np.uint8); mask[2:8, 4:6, 4:6] = 1
    np.save(cache / 'mask.npy', mask); np.save(cache / 'valid.npy', np.ones_like(mask))
    points = np.asarray([[4, 4, 4], [5, 4, 4]], np.int32)
    np.savez(cache / 'geometry.npz', points=points, radius=np.ones(2, np.float32),
             structure=np.ones(2, np.uint8), geometry_valid=np.uint8(1))
    rows = [dict(id='T', split='TRAIN', cache_dir='T', spacing_mm=1.),
            dict(id='DO_NOT_OPEN', split='VAL', cache_dir='absent', spacing_mm=1.)]
    stats = train_statistics(rows, tmp_path, sample_per_case=125, gdf_sample_per_case=4)
    assert stats['training_case_ids'] == ['T']
    assert stats['pi_support'] == pytest.approx(mask.mean())
    assert stats['pi_support'] > stats['pi'], 'Lumen occupancy is not the sparse measurement event'


def test_radius_valid_excludes_all_background_cells_but_retains_fractional_lumen(tmp_path):
    torch = pytest.importorskip('torch')
    from gta_avt.data import PreparedDataset
    cache=tmp_path/'T';cache.mkdir();shape=(16,16,16)
    mask=np.zeros(shape,np.uint8);mask[8,8,8]=1
    for name,value in dict(image=np.zeros(shape,np.int16),mask=mask,valid=np.ones(shape,np.uint8),
                           gdf=mask.astype(np.float32),gdf_valid=np.ones(shape,np.uint8)).items():
        np.save(cache/(name+'.npy'),value)
    # Deliberately oversized resampled measurement ball tests the background intersection.
    np.savez(cache/'geometry.npz',points=np.asarray([[8,8,8]],np.int32),radius=np.asarray([3.],np.float32),
             structure=np.ones(1,np.uint8),geometry_valid=np.uint8(1))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(schema='gta-avt-prepared-v1',
        cases=[dict(id='T',split='TRAIN',cache_dir='T',spacing_mm=1.)],
        statistics=dict(mu=0.,sigma=1.,pi_support=.01,gdf_scale=1.))))
    sample=PreparedDataset(manifest,patch_size=16,samples_per_epoch=1,seed=7)[0]
    assert not sample['r_valid'][sample['support']==0].any()
    assert sample['support'].max()==.125 and sample['r_valid'].sum()>0


def _native_case(tmp_path, name):
    sitk = pytest.importorskip('SimpleITK')
    image = np.full((18, 18, 18), -1000., np.float32); image[:, 3:15, 3:15] = 300.
    mask = np.zeros_like(image, np.uint8); mask[3:15, 6:11, 6:11] = 1
    ip, lp = tmp_path / (name + '.nrrd'), tmp_path / (name + '.seg.nrrd')
    sitk.WriteImage(sitk.GetImageFromArray(image), str(ip))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(lp))
    return dict(id=name, image=str(ip), label=str(lp), split='TRAIN' if name == 'T' else 'VAL')


def test_raw_and_import_prepare_real_val_foreground_gdf(tmp_path):
    from gta_avt.data import _sha
    rows = [_native_case(tmp_path, 'T'), _native_case(tmp_path, 'V')]
    cases = tmp_path / 'cases.json'; cases.write_text(json.dumps(dict(cases=rows)))
    prepared = tmp_path / 'raw'
    raw = prepare_dataset(cases, prepared, geometry_options=dict(sample_per_case=50, gdf_sample_per_case=4))
    assert raw['heldout_gdf_foreground_prepared'] is True
    assert raw['statistics']['training_case_ids'] == ['T']
    mask = np.load(prepared / 'V' / 'mask.npy').astype(bool)
    assert np.load(prepared / 'V' / 'gdf_valid.npy')[mask].all()
    assert np.load(prepared / 'V' / 'gdf.npy')[mask].min() > 0
    legacy_rows=[]
    for row in rows:
        record=json.loads((prepared / row['id'] / 'record.json').read_text())
        record.update(row, cache_dir=str(prepared / row['id']), cohort='synthetic',
                      image_sha256=_sha(row['image']), label_sha256=_sha(row['label']),
                      cache_file_sha256={name:_sha(prepared / row['id'] / name)
                                         for name in ('image.npy','mask.npy','geometry.npz')})
        legacy_rows.append(record)
    legacy = tmp_path / 'legacy.json'
    legacy.write_text(json.dumps(dict(complete=True, spacing_mm=1., cases=legacy_rows,
                                     splits=dict(train=['T'],val=['V'],test=[]))))
    imported_path=tmp_path / 'imported'
    imported=import_prepared_cache(legacy, imported_path,
        geometry_options=dict(sample_per_case=50,gdf_sample_per_case=4))
    assert imported['heldout_gdf_foreground_prepared'] is True
    for split in ('T','V'):
        np.testing.assert_allclose(np.load(imported_path / split / 'gdf.npy'),
                                   np.load(prepared / split / 'gdf.npy'))


def _torch_fixture():
    torch = pytest.importorskip('torch')
    shape=(1,1,1,2,4)
    support=torch.zeros(shape); support.flatten()[:4]=1
    valid=torch.ones(shape, dtype=torch.bool)
    measured=torch.zeros_like(valid); measured.flatten()[:4]=True
    radius=torch.full(shape,float('nan')); radius.flatten()[:4]=torch.tensor([-1.,0.,1.,2.])
    batch=dict(support=support,support_valid=valid,r=radius,r_valid=measured,case_id=['one'])
    logits=torch.where(support.bool(),torch.tensor(10.),torch.tensor(-10.)).requires_grad_()
    mu=torch.zeros(shape); mu.flatten()[:4]=torch.tensor([-1.,0.,1.,2.]); mu=mu.requires_grad_()
    return torch,batch,dict(support_logits=logits,radius_mean=mu)


def test_radius_unknown_targets_never_become_zero_measurements():
    torch,batch,out=_torch_fixture()
    from gta_avt.losses import attribute_loss
    with torch.no_grad(): out['radius_mean'].flatten()[4:]=2.
    result=attribute_loss(out,batch)
    assert torch.isfinite(result['loss']) and float(result['radius_mse']) == 0.
    result['loss'].backward()
    assert torch.equal(out['radius_mean'].grad.flatten()[4:],torch.zeros(4))
    assert 'radius_nll' not in result and 'unknown_prior_kl' not in result


def test_unknown_radius_has_no_loss_or_gradient_regardless_of_background_count():
    torch=pytest.importorskip('torch')
    from gta_avt.losses import attribute_loss
    for background_count in (2,100):
        shape=(1,1,1,1,background_count+1)
        support=torch.zeros(shape); support.flatten()[0]=1.
        mu=torch.zeros(shape,requires_grad=True)
        with torch.no_grad():mu.flatten()[0]=2.
        out=dict(support_logits=torch.zeros(shape),radius_mean=mu)
        batch=dict(support=support,support_valid=torch.ones(shape,dtype=torch.bool),
                   r=torch.full(shape,float('nan')),r_valid=torch.zeros(shape,dtype=torch.bool))
        result=attribute_loss(out,batch)
        assert float(result['radius_mse']) == 0.
        result['loss'].backward()
        assert torch.equal(mu.grad,torch.zeros_like(mu))


def test_gdf_foreground_background_balancing_and_unknown_nulls():
    torch=pytest.importorskip('torch')
    from gta_avt.losses import segmentation_loss
    from gta_avt.geometry import gdf_mse
    shape=(1,1,1,1,10)
    label=torch.zeros(shape); label.flatten()[0]=1
    pred=torch.full(shape,1.); pred.flatten()[0]=.75; pred.requires_grad_()
    valid=torch.ones(shape,dtype=torch.bool)
    out=dict(logits=torch.zeros(shape),gdf=pred)
    batch=dict(label=label,valid=valid,gdf=torch.full(shape,.5),gdf_valid=valid)
    result=segmentation_loss(out,batch,dice_weight=0,bce_weight=0,gdf_weight=1)
    assert float(result['gdf_mse']) == pytest.approx(.15625)
    assert result['gdf_fg_mse']==.0625 and result['gdf_bg_mse']==.25
    reference,details=gdf_mse(pred,batch['gdf'],valid,label,normalization='balanced')
    assert float(reference)==float(result['gdf_mse'])
    result['loss'].backward();assert pred.grad.flatten()[0]>pred.grad.flatten()[1]
    batch['gdf_valid']=torch.zeros_like(valid);batch['gdf']=torch.full(shape,float('nan'))
    missing=segmentation_loss(out,batch,dice_weight=0,bce_weight=0,gdf_weight=1)
    assert float(missing['gdf_mse'])==0. and missing['gdf_fg_mse'] is None and missing['gdf_bg_mse'] is None


def test_nan_padding_is_removed_before_segmentation_auxiliary_and_gdf_arithmetic():
    torch=pytest.importorskip('torch')
    from gta_avt.losses import segmentation_loss
    shape=(1,1,2,2,2)
    for wholly_unknown in (False,True):
        valid=torch.zeros(shape,dtype=torch.bool)
        if not wholly_unknown:valid.flatten()[0]=True
        def leaf(value):
            out=torch.full(shape,float('nan'));out[valid]=value
            return out.requires_grad_()
        logits,aux,gdf=leaf(.4),leaf(-.1),leaf(.5)
        label=torch.full(shape,float('nan'));label[valid]=1.
        target=torch.full(shape,float('nan'));target[valid]=.75
        out=dict(logits=logits,aux_logits=[aux],gdf=gdf)
        batch=dict(label=label,valid=valid,gdf=target,gdf_valid=valid)
        result=segmentation_loss(out,batch,gdf_weight=.1,deep_supervision_weight=.1)
        assert torch.isfinite(result['loss'])
        result['loss'].backward()
        for value in (logits,aux,gdf):
            assert torch.isfinite(value.grad).all()
            assert (value.grad[~valid]==0).all()
        if wholly_unknown:assert float(result['loss'])==0.


def test_readiness_rejects_accurate_radius_with_no_anatomical_overlap():
    torch,batch,out=_torch_fixture()
    from gta_avt.metrics import AttributeMetrics
    with torch.no_grad(): out['support_logits'].neg_()
    meter=AttributeMetrics(.5);meter.update(out,batch);report=meter.report()
    assert report['per_case'][0]['radius_mse']==0.
    assert not report['readiness']['passed']
    assert not report['readiness']['tests']['every_case_has_true_support_overlap']
    assert not report['readiness']['tests']['every_case_has_measured_consumer_overlap']


def test_readiness_requires_usable_correct_radius_and_reports_soft_unknown_mass():
    _,batch,out=_torch_fixture()
    from gta_avt.metrics import AttributeMetrics
    meter=AttributeMetrics(.5);meter.update(out,batch);report=meter.report()
    assert report['readiness']['passed']
    row=report['per_case'][0]
    assert row['consumer_overlap_sites']==4 and row['consumer_unmeasured_fraction']==0.
    assert row['soft_consumed_radius_mse']==0.
    assert row['soft_support_mass_on_unmeasured_fraction'] > 0.
    assert 'reliability_error_correlation' not in row


def test_consumption_metrics_do_not_invent_radius_error_on_unknown_foreground():
    torch,batch,out=_torch_fixture()
    from gta_avt.metrics import AttributeMetrics
    with torch.no_grad():
        batch['support'].flatten()[4]=1.
        out['support_logits'].flatten()[4]=10.
        out['radius_mean'].flatten()[4]=3.
    meter=AttributeMetrics(.5);meter.update(out,batch);row=meter.report()['per_case'][0]
    assert row['radius_mse']==0. and row['consumer_radius_mse']==0.
    assert row['consumer_unmeasured_foreground_sites']==1
    assert row['consumer_unmeasured_fraction']==pytest.approx(.2)
    assert row['unmeasured_radius_axis_fraction'] > 0.
