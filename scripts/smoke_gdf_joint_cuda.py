"""Two synthetic 128-cubed/B2 full-S AMP updates, explicitly invoked only.

This is engineering QA, not AVT training or evidence of method performance.
Run after existing GPU jobs and their native evaluations have completed.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from gta_avt.models import GTAAVT, AttributePredictor
from gta_avt.losses import segmentation_loss
from gta_avt.engine import state_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA device required')
    torch.set_num_threads(2); torch.manual_seed(20261032); torch.cuda.manual_seed_all(20261032)
    model = GTAAVT(AttributePredictor(dict(mu=1.366666007637428, sigma=.7268629347959835,
        pi_support=.004302116418128005)), arm='AR', use_gdf=True).cuda().set_stage('full_joint').train()
    frozen = state_digest(model.predictor)
    image = torch.randn(2, 1, 128, 128, 128, device='cuda')
    label = torch.zeros_like(image); label[:, :, 48:80, 48:80, 48:80] = 1.
    batch = dict(label=label, valid=torch.ones_like(label, dtype=torch.bool),
        gdf=.5 + .25 * label, gdf_valid=torch.ones_like(label, dtype=torch.bool))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0001, weight_decay=.00001)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.)
    successful, attempts, rows = 0, 0, []
    started = time.monotonic()
    while successful < 2:
        attempts += 1
        if attempts > 8: raise RuntimeError('Synthetic QA failed to obtain two finite successful updates')
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', enabled=True):
            output = model(image)
            losses = segmentation_loss(output, batch, gdf_weight=1., bce_class_balance='airway_ratio')
        scaler.scale(losses['loss']).backward(); scaler.unscale_(optimizer)
        previous = scaler.get_scale()
        head_grad = float(model.output_head.weight.grad.detach().float().norm())
        encoder = next(p for n, p in model.named_parameters() if n.startswith('backbone.encoders.0') and p.ndim == 5)
        encoder_grad = float(encoder.grad.detach().float().norm())
        scaler.step(optimizer); scaler.update()
        if scaler.get_scale() < previous: continue
        successful += 1
        if head_grad <= 0 or encoder_grad <= 0: raise RuntimeError('Shared segmentation gradient path missing')
        rows.append(dict(loss=float(losses['loss'].detach()), output_head_gradient_norm=head_grad,
            first_encoder_gradient_norm=encoder_grad))
    if state_digest(model.predictor) != frozen: raise RuntimeError('Frozen P changed during synthetic full-S updates')
    torch.cuda.synchronize()
    result = dict(status='PASS', scope='synthetic engineering QA only; no AVT experiment',
        patch_size=128, batch_size=2, successful_updates=successful, attempts=attempts,
        amp=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), elapsed_seconds=time.monotonic() - started,
        gaussian_head_parameters=55, predictor_unchanged=True, measurements=rows)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__': main()
