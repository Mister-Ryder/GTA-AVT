"""Scientific contracts for the isolated repaired Gaussian suffix (CPU synthetic)."""
import copy
import json
from pathlib import Path
import tempfile

import pytest
import torch

from gta_avt.models import GTAAVT, AttributePredictor, MODEL_CONFIG
from gta_avt.losses import segmentation_loss
from gta_avt import engine
from gta_avt.gdf_joint import migrate_condition, choose_lambda, arm_config, PLAN_SCHEMA, LEGACY_SCHEMA


STATS = dict(mu=1., sigma=.5, pi_support=.1)


@pytest.fixture(autouse=True)
def small_cpu_pool():
    torch.set_num_threads(2)


def model():
    torch.manual_seed(21)
    return GTAAVT(AttributePredictor(STATS), arm='AR', use_gdf=True)


def test_logit_head_geometry_and_shared_gradient_path():
    m = model().set_stage('full_joint').train()
    assert tuple(m.gdf_head[0].weight.shape) == (1, 2, 3, 3, 3)
    assert sum(p.numel() for p in m.gdf_head.parameters()) == 55
    before = {n: t.clone() for n, t in m.predictor.state_dict().items()}
    bn = {n: t.clone() for n, t in m.state_dict().items() if 'running_' in n or 'num_batches_tracked' in n}
    out = m(torch.randn(1, 1, 32, 32, 32))
    assert out['segmentation_logits'].shape[1] == 2
    assert bool(((out['gdf'] >= 0) & (out['gdf'] <= 1)).all())
    assert torch.equal(out['logits'], out['segmentation_logits'][:, 1:2] - out['segmentation_logits'][:, :1])
    (out['gdf'] - .8).square().mean().backward()
    assert m.output_head.weight.grad is not None and float(m.output_head.weight.grad.abs().sum()) > 0
    first_encoder = next(p for n, p in m.named_parameters() if n.startswith('backbone.encoders.0') and p.ndim == 5)
    assert first_encoder.grad is not None and float(first_encoder.grad.abs().sum()) > 0
    assert all(p.grad is None and not p.requires_grad for p in m.predictor.parameters())
    torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=.0001).step()
    assert all(torch.equal(t, m.predictor.state_dict()[n]) for n, t in before.items())
    assert all(torch.equal(t, m.state_dict()[n]) for n, t in bn.items())


def test_shared_head_warmup_cannot_change_segmentation_parameters():
    m = model().set_stage('gdf_head').train()
    before = {n: t.clone() for n, t in m.state_dict().items() if not n.startswith('gdf_head.')}
    out = m(torch.randn(1, 1, 32, 32, 32))
    out['gdf'].square().mean().backward()
    assert all(p.grad is None for n, p in m.named_parameters() if not n.startswith('gdf_head.'))
    torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=.001).step()
    assert all(torch.equal(t, m.state_dict()[n]) for n, t in before.items())


def test_lambda_zero_control_has_no_auxiliary_gradient():
    m = model().set_stage('full_joint')
    out = m(torch.randn(1, 1, 32, 32, 32))
    batch = dict(label=torch.ones_like(out['logits']), valid=torch.ones_like(out['logits'], dtype=torch.bool),
        gdf=torch.full_like(out['logits'], .75), gdf_valid=torch.ones_like(out['logits'], dtype=torch.bool))
    segmentation_loss(out, batch, gdf_weight=0., bce_class_balance='airway_ratio')['loss'].backward()
    assert all(p.grad is None for p in m.gdf_head.parameters())
    assert m.output_head.weight.grad is not None


def test_loss_rejects_unbounded_targets_and_excludes_invalid_nan():
    logits = torch.zeros(1, 1, 2, 2, 2, requires_grad=True)
    output = dict(logits=logits, gdf=logits.sigmoid())
    batch = dict(label=torch.ones_like(logits), valid=torch.ones_like(logits, dtype=torch.bool),
        gdf=torch.full_like(logits, 1.1), gdf_valid=torch.ones_like(logits, dtype=torch.bool))
    with pytest.raises(ValueError, match='within'):
        segmentation_loss(output, batch, gdf_weight=1.)
    batch['gdf'].fill_(.75); batch['gdf'][0, 0, 0, 0, 0] = float('nan')
    batch['gdf_valid'][0, 0, 0, 0, 0] = False
    loss = segmentation_loss(output, batch, gdf_weight=1.)['loss']
    assert torch.isfinite(loss)


def fixture_plan(tmp):
    manifest = tmp / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema='gta-avt-prepared-v1', statistics=STATS, cases=[])))
    return dict(schema=PLAN_SCHEMA, manifest=str(manifest), output_dir=str(tmp / 'run'),
        condition_checkpoint=str(tmp / 'condition.pt'), condition_checkpoint_sha256='0'*64,
        selected_base_sha256='a'*64, legacy_source_manifest_sha256='b'*64,
        evaluation_cases=str(tmp / 'val.json'), seed=20261032, device='cpu', num_workers=0,
        calibration_batches=8)


def write_legacy(plan):
    m = model()
    state = copy.deepcopy(m.state_dict())
    state['gdf_head.0.weight'] = torch.randn(1, 32, 1, 1, 1)
    state['gdf_head.0.bias'] = torch.randn(1)
    checkpoint = dict(schema=LEGACY_SCHEMA, kind='train', config=dict(phase='condition', stage='local_adapt',
        arm='AR', seed=20261032, patch_size=128, validation_seed=0, max_steps=10000), step=10000,
        statistics=STATS, model_config=dict(revision='20261027-ratio-r2'), model=state,
        pair=dict(selected_base_sha256=plan['selected_base_sha256'], attribute_readiness_passed=True,
            predictor_state_sha256=engine.state_digest(m.predictor), model_state_sha256=engine._tensor_digest(state.items())))
    torch.save(checkpoint, plan['condition_checkpoint'])
    plan['condition_checkpoint_sha256'] = engine.file_digest(plan['condition_checkpoint'])
    return checkpoint


def test_migration_imports_exact_ps_resets_only_head_and_rejects_reuse(tmp_path):
    plan = fixture_plan(tmp_path); legacy = write_legacy(plan)
    output = tmp_path / 'converted.pt'
    report = migrate_condition(plan, output)
    new = engine.load_checkpoint(output)
    assert report['new_head_parameters'] == 55
    assert new['schema'] == engine.SCHEMA
    assert new['model_config']['revision'] == MODEL_CONFIG['revision']
    assert tuple(new['model']['gdf_head.0.weight'].shape) == (1, 2, 3, 3, 3)
    assert all(torch.equal(t, new['model'][n]) for n, t in legacy['model'].items() if not n.startswith('gdf_head.'))
    assert not any(k in new for k in ('optimizer', 'scheduler', 'scaler'))
    with pytest.raises(FileExistsError): migrate_condition(plan, output)
    plan['condition_checkpoint_sha256'] = 'c'*64
    with pytest.raises(ValueError, match='SHA256'): migrate_condition(plan, tmp_path / 'bad.pt')


def test_migration_rejects_post_gdf_or_partial_parent(tmp_path):
    plan = fixture_plan(tmp_path); legacy = write_legacy(plan)
    legacy['config']['phase'] = 'geometry'
    torch.save(legacy, plan['condition_checkpoint'])
    plan['condition_checkpoint_sha256'] = engine.file_digest(plan['condition_checkpoint'])
    with pytest.raises(ValueError, match='clean'):
        migrate_condition(plan, tmp_path / 'bad.pt')


def test_training_plan_pair_matches_except_gdf_and_output(tmp_path):
    plan = fixture_plan(tmp_path)
    for phase, updates, shared_lr in (('geometry', 20000, .0001), ('refine', 10000, .00005)):
        control = arm_config(plan, 'AR', phase, 'same-parent.pt', 50.)
        gaussian = arm_config(plan, 'AR_GDF', phase, 'same-parent.pt', 50.)
        expected = {'output_dir', 'lambda_gdf', 'gdf_weight_start', 'gdf_weight_end'}
        assert {k for k in control if control[k] != gaussian[k]} == expected
        assert control['stage'] == 'full_joint' and control['max_steps'] == updates
        assert control['local_shared_lr'] == shared_lr
        assert control['parent_checkpoint'] == gaussian['parent_checkpoint']


def test_train_only_lambda_rule_uses_loss_and_gradient_guard():
    rows = [dict(seg_loss=1., gdf_loss=.01, seg_s_grad_norm=1., gdf_s_grad_norm=.1) for _ in range(8)]
    result = choose_lambda(rows)
    assert result['lambda_peak'] == pytest.approx(20.)
    assert result['predicted_median_weighted_shared_gradient_ratio'] == pytest.approx(2.)
    for row in rows: row['gdf_loss'] = 1.
    assert choose_lambda(rows)['lambda_peak'] == pytest.approx(5.)
    rows[0]['gdf_s_grad_norm'] = 0.
    with pytest.raises(FloatingPointError): choose_lambda(rows)
