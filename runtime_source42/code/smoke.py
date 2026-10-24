"""Meaningful GPU/CPU fixtures before any expensive run; no AVT test access."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
import SimpleITK as sitk
from common import atomic_json
from model import GTAAVT
from losses import objective
from metrics import binary_metrics
from inference import native_images, tile_starts
from intensity import detect_ct_offset


def native_fixture():
    with tempfile.TemporaryDirectory(prefix='gta_avt_fixture_') as directory:
        directory = Path(directory)
        array = np.zeros((11, 13, 15), np.uint8)
        array[3:8, 4:9, 5:10] = 1
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((.8, 1.2, 2.))
        image.SetOrigin((14., -7., 28.))
        image.SetDirection((0., -1., 0., 1., 0., 0., 0., 0., 1.))
        image_path, label_path = directory / 'image.nrrd', directory / 'label.nrrd'
        sitk.WriteImage(image, str(image_path))
        sitk.WriteImage(image, str(label_path))
        record = {'image': str(image_path), 'label': str(label_path), 'full_shape_zyx': list(array.shape),
                  'crop_start_zyx': [0, 0, 0], 'crop_stop_zyx': list(array.shape),
                  'resampled_origin': image.GetOrigin(), 'resampled_direction': image.GetDirection(),
                  'resampled_spacing': image.GetSpacing()}
        probability = array.astype(np.float32)
        probability[0, 0, 0] = 1.
        p, t = native_images(probability, record)
        assert np.array_equal(sitk.GetArrayFromImage(p), probability)
        assert np.array_equal(sitk.GetArrayFromImage(t), array)
        assert p.GetDirection() == image.GetDirection() and p.GetSpacing() == image.GetSpacing()
        metrics = binary_metrics(sitk.GetArrayFromImage(p), sitk.GetArrayFromImage(t), (2., 1.2, .8))
        assert metrics['fp_voxels'] == 1 and metrics['fn_voxels'] == 0
    one = np.zeros((9, 9, 9), np.uint8)
    two = one.copy()
    one[3, 3, 3], two[5, 3, 3] = 1, 1
    shifted = binary_metrics(one, two, (2., 1.5, .5))
    assert shifted['dice'] == 0. and shifted['assd_mm'] == 4. and shifted['hd95_mm'] == 4.
    even = np.zeros((12, 12, 24), bool)
    even[4:8, 4:8, 2:22] = True
    identity = binary_metrics(even, even, (1., 1., 1.))
    assert identity['dice'] == 1. and identity['assd_mm'] == 0.
    assert identity['centerline_recall'] is None or identity['centerline_recall'] == 1.
    return {'native_orientation_anisotropic_origin': 'passed', 'false_positive_retained': 'passed',
            'known_translation_4mm': 'passed', 'empty_skeleton_not_scored_as_failure': 'passed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--patch-size', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--output')
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    results = native_fixture()
    for offset in (0., 1024.):
        sample = np.full((48, 64, 64), -1000. + offset, np.float32)
        sample[8:40, 12:52, 12:52] = 250. + offset
        sample[0] = -2000.
        detected = detect_ct_offset(sample)
        assert detected['offset_subtract'] == offset, detected
    for sample in (np.full((32, 32, 32), 50., np.float32),
                   np.random.default_rng(0).uniform(10., 100., (32, 32, 32))):
        assert detect_ct_offset(sample)['requires_review']
    results['stored_intensity_offsets_with_negative_padding'] = 'passed'
    for size in (32, 96, 112, 200):
        starts = tile_starts(size, args.patch_size, .5)
        cover = np.zeros(max(size, args.patch_size), bool)
        for start in starts:
            cover[start:start + args.patch_size] = True
        assert cover.all()
    results['sliding_window_full_coverage'] = 'passed'
    backbones = []
    for method in ('baseline', 'gta_constant', 'gta_joint'):
        torch.manual_seed(20261022)
        m = GTAAVT(method, args.width)
        backbones.append({key: value.clone() for key, value in m.backbone.state_dict().items()})
    assert all(torch.equal(backbones[0][key], other[key]) for other in backbones[1:] for key in backbones[0])
    results['paired_backbone_initialization'] = 'passed'
    del backbones, m
    torch.manual_seed(20261022)
    model = GTAAVT('gta_joint', args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    p, b = args.patch_size, args.batch_size
    image = torch.randn((b, 1, p, p, p), device=device)
    z = torch.arange(p, device=device) - (p - 1.) / 2.
    mask = (z[None, :, None] ** 2 + z[None, None, :] ** 2 < (p / 8.) ** 2).float()[None].expand(b, 1, p, p, p)
    targets = {'image': image, 'mask': mask,
               'scale': torch.tensor([[0., .1, .3, .4, .2]] * b, device=device),
               'structure': torch.tensor([[0., .7, .3]] * b, device=device),
               'attribute_valid': torch.ones(b, device=device), 'coordinate': torch.zeros((b, 3), device=device)}
    times = []
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    for step in range(args.steps):
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            output = model(image, targets['coordinate'])
            loss, details = objective(output, targets)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12.)
        assert torch.isfinite(norm)
        scaler.step(optimizer)
        scaler.update()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - started)
        print(json.dumps({'step': step + 1, 'seconds': times[-1], 'loss': float(loss)}), flush=True)
    # Structural condition path must influence segmentation after an update;
    # auxiliary-head gradients alone would not prove conditional adaptation.
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
        output = model(image, targets['coordinate'])
        gradients = torch.autograd.grad(output['logits'].float().mean(),
                                        (output['scale_logits'], output['structure_logits']), retain_graph=False)
    gradient_sizes = [float(x.abs().sum()) for x in gradients]
    assert all(x > 0. for x in gradient_sizes), gradient_sizes
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
        predicted = model(image, targets['coordinate'])['logits'].float()
        constant = model(image, targets['coordinate'], condition_mode='constant')['logits'].float()
    difference = float((predicted - constant).abs().mean())
    assert difference > 0., difference
    results.update(synthetic_optimizer_finite='passed', segmentation_gradient_to_soft_conditions=gradient_sizes,
                   predicted_vs_constant_mean_absolute_logit_difference=difference,
                   seconds_per_step=times, steady_step_seconds=float(np.median(times[1:] or times)),
                   peak_allocated_gpu_gib=torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == 'cuda' else None,
                   torch_version=torch.__version__, patch_size=p, batch_size=b, width=args.width,
                   parameters=sum(x.numel() for x in model.parameters()), device=str(device))
    if args.output:
        atomic_json(args.output, results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
