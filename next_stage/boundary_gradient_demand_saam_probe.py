"""Fixed TRAIN-bank SAAM64 gradient-demand measurement; no update.

G0 remains numerically unchanged. Only a temporary detached G0 activation leaf
requires gradients. Per-patch spatial moments identify the tangent demand for
the already-fixed CT condition axes, without storing CT or gradient fields.
"""
import argparse
import hashlib
import importlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import time
import types

sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
HELPER_SHA = '016157cf52f7e9d9b7593435d5557dc55166881e94c59ad70bc68857186c8604'
MODEL_SHA = '949a1efd1713a661c73762d7cef387be01205a93e425fba60d2515af4e798cf2'
HEAD_SHA = 'b61885b06fd6413efc0cf86bb664f69aa1a4e9f7fb2a26a32c373a71ce5e11bb'
NORMALIZATION_SHA = 'b1282feb482ec89fd048d556527299404cc4d1e6a0f1b206b7001ee6edc88842'
PROBE_SITE = 'SAAM64_second_decoder_soft_structure_output'
RECIPE = dict(probe_site=PROBE_SITE, channels=64, patch_size=128, batch_size=2, AMP=True, alpha=0., augmentation=False,
    loss='Original main Dice+BCE; padded target and full128^3 domain identical to the pilot',
    initial_loss_scale=65536., maximum_attempts=8, retry='Same pair, halve scale; no optimizer, update or skipped sample',
    derivative='leaf.grad * batch_size / loss_scale restores per-patch loss derivative',
    axes=['1', 'zA', 'zAB'], demand='r=G0*dL_per_patch/dG0',
    spatial='Actual64-channel SAAM grid, all spatial cells including padding; no pooling of r',
    moments='CPU F64 per patch: X X^T/M, X r^T/M, channel mean r^2',
    no_optimization=True, no_VAL_or_TEST_arrays=True)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def load(path, expected, name):
    require(sha(path) == expected, 'Pinned helper/model differs: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def moments(np, g0, gradient, zA, zAB):
    """All arrays for ONE patch; preserve the actual spatial grid."""
    require(g0.shape == gradient.shape and g0.shape[0] == 64 and zA.shape == zAB.shape == (1,) + g0.shape[1:], 'Moment shapes differ')
    f = np.asarray(g0, dtype=np.float64).reshape(64, -1)
    g = np.asarray(gradient, dtype=np.float64).reshape(64, -1)
    x = np.stack((np.ones(f.shape[1], dtype=np.float64), np.asarray(zA, dtype=np.float64).reshape(-1),
        np.asarray(zAB, dtype=np.float64).reshape(-1)))
    require(np.isfinite(f).all() and np.isfinite(g).all() and np.isfinite(x).all(), 'Nonfinite moment input')
    r = f * g; M = f.shape[1]
    value = dict(gram=x @ x.T / M, cross=x @ r.T / M, rr=np.mean(r * r, axis=1), cross_additive=x @ g.T / M,
        grad_rr=np.mean(g * g, axis=1), exactzeroG0_grad_rr=np.mean(g * g * (f == 0.), axis=1),
        G0_rr=np.mean(f * f, axis=1))
    require(all(np.isfinite(v).all() for v in value.values()), 'Nonfinite moment output')
    return value, M


def execute(args):
    started = time.monotonic(); helper = load(args.helper, HELPER_SHA, 'gradient_boundary_helper')
    request = helper.read(args.request); root, source, manifest, bank, records = helper.metadata(request)
    require(sha(args.heads) == args.heads_sha256 == HEAD_SHA and sha(args.field_model) == args.field_model_sha256 == MODEL_SHA,
        'Original independent heads and949 model required')
    require(sha(args.normalization) == args.normalization_sha256 == NORMALIZATION_SHA, 'Original pilot normalization required')
    normal = helper.read(args.normalization)
    require(normal['fit_cases'] == bank['fit_cases'] and normal['cells'] == 24 * 16 * 512 and normal['no_GT_mask_used'] is True and
        normal['common_to_all_arms'] is True and math.isfinite(normal['mean']) and math.isfinite(normal['population_std']) and normal['population_std'] > 0,
        'Fixed fit24 CT-only normalization differs')
    output = Path(args.output); require(not output.exists(), 'Fresh gradient output required')
    active_cases = request['case_ids'][:1] if args.mode == 'qa' else request['case_ids']
    active_request = dict(request, case_ids=active_cases)
    before_cache = helper.cached_inputs(active_request, records)
    import numpy as np
    import torch
    from torch.nn import functional as F
    for path in (root / 'code', root / 'next_stage'):
        sys.path.insert(0, str(path))
    modules = {name: importlib.import_module(name) for name in ('data', 'losses')}
    for name, module in modules.items():
        require(Path(module.__file__).resolve() == root / 'code' / (name + '.py'), 'Scientific import escaped source42')
    field = load(args.field_model, MODEL_SHA, 'gradient_boundary_field_model')
    require(torch.cuda.is_available(), 'Actual CUDA gradient measurement required')
    torch.set_num_threads(4); torch.manual_seed(20261023); torch.cuda.manual_seed_all(20261023)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    heads = torch.load(args.heads, map_location='cpu'); common = torch.load(request['weights'], map_location='cpu')
    model = field.BoundaryFieldStageModel('AB', alpha=0.)
    model.load_common_state(common); model.install_heads(heads, normal['mean'], normal['population_std'])
    del heads, common
    model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    require(not any(p.requires_grad for p in model.parameters()) and not any(m.training for m in model.modules()), 'All model parameters/buffers must stay frozen/eval')
    state_before = helper.state_sha(model)
    cpu_rng = torch.get_rng_state().clone(); cuda_rng = [r.clone() for r in torch.cuda.get_rng_state_all()]
    original_structure = model.soft_structure
    model._gradient_leaf = None
    def leaf_structure(self, feature, attribute, probability):
        value = original_structure(feature, attribute, probability)
        require(not value.requires_grad and self._stage_request['alpha'] == 0., 'Original SAAM G0 must be frozen and alpha0')
        leaf = value.detach().requires_grad_(True)
        self._gradient_leaf = leaf
        return leaf
    leaf_method = types.MethodType(leaf_structure, model)
    output.mkdir(); (output / 'cases').mkdir()
    (output / 'request.json').write_bytes(Path(args.request).read_bytes())
    (output / 'normalization.json').write_bytes(Path(args.normalization).read_bytes())
    helper.write(output / 'intent.json', dict(schema='GTA-AVT-boundary-gradient-demand-saam-intent-v1', mode=args.mode, probe_site=PROBE_SITE,
        worker_sha256=sha(__file__), request_sha256=sha(args.request), helper_sha256=HELPER_SHA, field_model_sha256=MODEL_SHA,
        heads_sha256=HEAD_SHA, normalization_sha256=NORMALIZATION_SHA, recipe=RECIPE, active_cases=active_cases,
        no_optimization=True, no_VAL_or_TEST_arrays=True))
    case_outputs, qa, total, all_attempts = {}, None, 0, []
    try:
        for cid in active_cases:
            home = Path(records[cid]['cache_dir'])
            image = np.load(home / 'image.npy', mmap_mode='r', allow_pickle=False)
            mask = np.load(home / 'mask.npy', mmap_mode='r', allow_pickle=False)
            require(image.shape == mask.shape and list(image.shape) == bank['headers'][cid]['shape_zyx'], 'TRAIN cached shape differs')
            specs = [r for r in bank['patches'] if r['case_id'] == cid]
            if args.mode == 'qa':
                specs = specs[:2]
            records_out, arrays_out = [], {key: [] for key in ('gram', 'cross', 'rr', 'cross_additive', 'grad_rr', 'exactzeroG0_grad_rr', 'G0_rr')}
            sizes = []
            for offset in range(0, len(specs), 2):
                pair = specs[offset:offset + 2]; images, truths, coordinates, descriptions = [], [], [], []
                require(len(pair) == 2, 'Exact B2 pairs required')
                for spec in pair:
                    start = spec['start_zyx']; sl = tuple(slice(v, min(v + 128, n)) for v, n in zip(start, image.shape))
                    raw = np.asarray(image[sl], dtype=np.float32); truth = np.asarray(mask[sl], dtype=bool)
                    pad = [(0, 128 - n) for n in raw.shape]
                    normalized = np.ascontiguousarray(modules['data'].normalize(np.pad(raw, pad, constant_values=-1024))[None], dtype=np.float32)
                    images.append(normalized); truths.append(np.pad(truth, pad)[None])
                    coordinate = 2. * (np.asarray(records[cid]['crop_start_zyx']) + np.asarray(start) + 64.) / np.maximum(np.asarray(records[cid]['full_shape_zyx']) - 1., 1.) - 1.
                    coordinates.append(coordinate.astype(np.float32))
                    descriptions.append(dict(case_id=cid, source=spec['source'], slot=spec['slot'], stratum=spec['stratum'],
                        normalized_CT_bytes_sha256=hashlib.sha256(normalized.tobytes()).hexdigest(),
                        real_CT_voxels=int(np.prod(raw.shape)), padded_CT_voxels=128 ** 3 - int(np.prod(raw.shape))))
                tensor = torch.from_numpy(np.stack(images)).cuda()
                target = torch.from_numpy(np.stack(truths)).cuda().float()
                coordinate = torch.from_numpy(np.stack(coordinates)).cuda()
                model.soft_structure = original_structure
                with torch.no_grad(), torch.cuda.amp.autocast():
                    baseline_output = model(tensor, coordinate=coordinate, alpha=0.)
                    baseline = baseline_output['logits'].detach().float()
                del baseline_output
                model.soft_structure = leaf_method
                attempts = []; accepted = False
                for attempt in range(8):
                    scale = 65536. / (2. ** attempt); model._gradient_leaf = None
                    with torch.cuda.amp.autocast():
                        prediction = model(tensor, coordinate=coordinate, alpha=0., trace_interaction=True)
                        loss = modules['losses'].dice_bce(prediction['logits'], target)
                    leaf = model._gradient_leaf
                    require(leaf is not None and leaf.is_leaf and leaf.requires_grad and leaf.shape[1] == 64, 'Missing64-channel SAAM activation leaf')
                    require(torch.equal(baseline, prediction['logits'].detach().float()), 'Leaf forward is not bitwise equal to ordinary alpha0 G0')
                    finite_loss = bool(torch.isfinite(loss))
                    scaled_loss = loss.float() * scale
                    if finite_loss:
                        scaled_loss.backward()
                    finite_gradient = leaf.grad is not None and bool(torch.isfinite(leaf.grad).all())
                    attempts.append(dict(attempt=attempt + 1, loss_scale=scale, finite_loss=finite_loss, finite_leaf_gradient=finite_gradient))
                    if finite_loss and finite_gradient:
                        accepted = True; break
                    model._gradient_leaf = None; model.last_interaction_trace = None
                    del scaled_loss, prediction, loss, leaf
                require(accepted, 'Nonfinite activation gradient after all8 same-pair attempts; technical failure, no sample skipped')
                require(all(p.grad is None for p in model.parameters()), 'A frozen parameter received a gradient')
                trace = model.last_interaction_trace
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                    rawA = trace['own_A'].double() * model.boundary_B_fit_constant
                    fullA = F.interpolate(rawA, size=leaf.shape[2:], mode='trilinear', align_corners=False)
                    zA = ((fullA - model.boundary_field_mean) / model.boundary_field_std).float()
                    fullAB = F.interpolate(trace['own_raw_c8'].double(), size=leaf.shape[2:], mode='trilinear', align_corners=False)
                    zAB = ((fullAB - model.boundary_field_mean) / model.boundary_field_std).float()
                g0_np = leaf.detach().cpu().numpy().astype(np.float64)
                # Divide the scaled gradient and undo B2 mean exactly in F64.
                gradient_np = leaf.grad.detach().cpu().numpy().astype(np.float64) * (2. / scale)
                zA_np = zA.detach().cpu().numpy(); zAB_np = zAB.detach().cpu().numpy()
                require(np.isfinite(gradient_np).all(), 'Unscaled gradient is nonfinite')
                pair_gradient_nonzero = bool(np.count_nonzero(gradient_np))
                if qa is None:
                    require(pair_gradient_nonzero, 'First actual pair has no activation gradient')
                    qa = dict(schema='GTA-AVT-boundary-gradient-demand-saam-engineering-QA-v1',
                        engineering_qa_passed=True, mode=args.mode, probe_site=PROBE_SITE, channels=64, patch_size=128, batch_size=2, AMP=True,
                        ordinary_G0_vs_leaf_logits_bitwise_equal=True, finite_nonzero_leaf_gradient=True,
                        no_parameter_gradients=True, successful_loss_scale=scale,
                        attempts=attempts, G0_shape=list(leaf.shape),
                        ordinary_logits_sha256=hashlib.sha256(baseline.cpu().numpy().tobytes()).hexdigest())
                for index, description in enumerate(descriptions):
                    value, M = moments(np, g0_np[index], gradient_np[index], zA_np[index], zAB_np[index])
                    for key in arrays_out:
                        arrays_out[key].append(value[key])
                    sizes.append(M)
                    records_out.append(dict(**description, G0_shape=list(leaf.shape[1:]), M=M,
                        batch_loss=float(loss.detach()), loss_scale=scale, attempts=attempts,
                        leaf_forward_bitwise_equal=True, leaf_gradient_nonzero=bool(np.count_nonzero(gradient_np[index])),
                        zA_mean=float(zA_np[index].astype(np.float64).mean()), zA_population_std=float(zA_np[index].astype(np.float64).std()),
                        zAB_mean=float(zAB_np[index].astype(np.float64).mean()), zAB_population_std=float(zAB_np[index].astype(np.float64).std())))
                total += 2; all_attempts.extend(attempts)
                model._gradient_leaf = None; model.last_interaction_trace = None
                del prediction, loss, scaled_loss, leaf, trace, g0_np, gradient_np, zA_np, zAB_np, zA, zAB, fullA, fullAB, rawA, baseline, tensor, target, coordinate
                print(helper.encoded(dict(progress_patches=total, mode=args.mode, case_id=cid)).decode().strip(), flush=True)
            arrays_out = {key: np.stack(value).astype(np.float64) for key, value in arrays_out.items()}
            arrays_out.update(slot=np.asarray([r['slot'] for r in records_out], dtype=np.int64), M=np.asarray(sizes, dtype=np.int64))
            with (output / 'cases' / (cid + '.moments.npz')).open('xb') as stream:
                np.savez_compressed(stream, **arrays_out)
            helper.write(output / 'cases' / (cid + '.json'), dict(schema='GTA-AVT-boundary-gradient-demand-saam-case-v1',
                case_id=cid, probe_site=PROBE_SITE, channels=64, records=records_out, axes=['1', 'zA', 'zAB'], moments_are_spatial_means=True,
                r_spatially_pooled_before_moments=False, no_gradient_or_CT_arrays_saved=True))
            case_outputs[cid] = dict(patches=len(records_out), slots=[r['slot'] for r in records_out],
                G0_shapes=[r['G0_shape'] for r in records_out], cells=sizes)
            del image, mask, arrays_out
    finally:
        model.soft_structure = original_structure; model._gradient_leaf = None
    state_after = helper.state_sha(model); require(state_before == state_after, 'Frozen model/head/buffer state changed')
    require(torch.equal(cpu_rng, torch.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all())), 'Frozen measurement changed RNG state')
    require(not any(p.requires_grad or p.grad is not None for p in model.parameters()) and not any(m.training for m in model.modules()), 'Frozen parameter/mode policy changed')
    after_cache = helper.cached_inputs(active_request, records); require(before_cache == after_cache, 'TRAIN cache bytes changed')
    for name, digest in source.items():
        require(sha(root / name) == digest, 'Scientific source42 changed')
    require(sha(args.heads) == HEAD_SHA and sha(args.field_model) == MODEL_SHA and sha(args.normalization) == NORMALIZATION_SHA and
        sha(args.helper) == HELPER_SHA and sha(request['weights']) == helper.WEIGHT_SHA, 'Bound model/helper/head/moments changed')
    qa.update(full_model_state_unchanged=True, parameter_or_optimizer_updates=0, RNG_unchanged=True,
        state_before=state_before, state_after=state_after)
    helper.write(output / 'qa.json', qa)
    helper.write(output / 'inputs.json', dict(schema='GTA-AVT-boundary-gradient-demand-saam-inputs-v1',
        probe_site=PROBE_SITE, channels=64, source_manifest_sha256=helper.MAP_SHA, bank_sha256=helper.BANK_SHA, common_weights_sha256=helper.WEIGHT_SHA,
        heads_sha256=HEAD_SHA, normalization_sha256=NORMALIZATION_SHA, field_model_sha256=MODEL_SHA,
        helper_sha256=HELPER_SHA, worker_sha256=sha(__file__), request_sha256=sha(args.request),
        case_ids=active_cases, requested_case_ids=request['case_ids'], fit_cases=bank['fit_cases'], probe_cases=bank['probe_cases'],
        source_sha256=source, cache_before=before_cache, cache_after=after_cache,
        model_state_before=state_before, model_state_after=state_after,
        no_optimization=True, no_VAL_or_TEST_arrays=True, RNG_unchanged=True))
    files = {p.relative_to(output).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(output.rglob('*')) if p.is_file()}
    helper.write(output / 'summary.json', dict(schema='GTA-AVT-boundary-gradient-demand-saam-probe-v1', status='verified_complete',
        mode=args.mode, probe_site=PROBE_SITE, channels=64, case_ids=active_cases, cases=case_outputs, patches=total, files=files,
        engineering_qa_passed=True, recipe=RECIPE, source_manifest_sha256=helper.MAP_SHA, bank_sha256=helper.BANK_SHA,
        common_weights_sha256=helper.WEIGHT_SHA, heads_sha256=HEAD_SHA, normalization_sha256=NORMALIZATION_SHA,
        field_model_sha256=MODEL_SHA, worker_sha256=sha(__file__),
        all_model_state_unchanged=True, no_optimization=True, no_VAL_or_TEST_arrays=True,
        attempts=len(all_attempts), same_pair_scale_reductions=sum(a['attempt'] > 1 for a in all_attempts),
        elapsed_seconds=time.monotonic() - started, maximum_cuda_memory_allocated=int(torch.cuda.max_memory_allocated())))
    return dict(status='verified_complete', mode=args.mode, probe_site=PROBE_SITE, engineering_qa_passed=True, patches=total,
        summary_sha256=sha(output / 'summary.json'))


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request'); parser.add_argument('--heads'); parser.add_argument('--heads-sha256')
    parser.add_argument('--field-model'); parser.add_argument('--field-model-sha256')
    parser.add_argument('--normalization'); parser.add_argument('--normalization-sha256')
    parser.add_argument('--helper', default=str(Path(__file__).with_name('boundary_attribute_probe.py')))
    parser.add_argument('--mode', choices=('qa', 'run'), default='qa'); parser.add_argument('--output')
    parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='inert', recipe=RECIPE, files_read=False, weights_loaded=False, GPU_used=False, outputs_written=False))); return
    require(all((args.request, args.heads, args.heads_sha256, args.field_model, args.field_model_sha256,
        args.normalization, args.normalization_sha256, args.helper, args.output)), 'Explicit bound inputs and fresh output required')
    print(json.dumps(execute(args)))


if __name__ == '__main__':
    main()
