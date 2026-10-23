"""Synthetic local-revision contracts; no study files or real training.

CPU single-thread synthetic optimizer steps test exact common-state/RNG parity.
CUDA has exact forward assertions but only reports backward self-repeat noise;
it never claims bitwise-equivalent multistep CUDA training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _common_state(model):
    return {key: value for key, value in model.state_dict().items()
            if not key.startswith("local_head.")}


def _assert_equal_state(a, b):
    import torch
    aa, bb = _common_state(a), _common_state(b)
    assert aa.keys() == bb.keys()
    assert all(torch.equal(aa[key], bb[key]) for key in aa), "Common state/buffer mismatch"


def _assert_equal_output(a, b):
    import torch
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], list):
            assert len(a[key]) == len(b[key])
            assert all(torch.equal(x, y) for x, y in zip(a[key], b[key])), key
        else:
            assert torch.equal(a[key], b[key]), key


def _synthetic_loss(output):
    return (output["logits"].float().square().mean()
            + .1 * sum(x.float().square().mean() for x in output["aux_logits"])
            + .2 * output["scale_logits"].float().square().mean()
            + .2 * output["structure_logits"].float().square().mean()
            + .01 * output["gdf_prediction"].float().square().mean())


def _rng_state():
    import torch
    return (torch.get_rng_state().clone(),
            [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [])


def _set_rng(state):
    import torch
    torch.set_rng_state(state[0])
    if state[1]:
        torch.cuda.set_rng_state_all(state[1])


def _same_rng(a, b):
    import torch
    return torch.equal(a[0], b[0]) and len(a[1]) == len(b[1]) and all(
        torch.equal(x, y) for x, y in zip(a[1], b[1]))


def cpu_common_parity(steps=3):
    import torch
    from stage_model import StageModel
    from local_model import LocalStageModel
    torch.set_num_threads(1)
    torch.manual_seed(20261023)
    reference = StageModel("constant")
    common_rng = _rng_state()
    torch.manual_seed(20261023)
    local = LocalStageModel("local_scale", alpha=0.)
    assert _same_rng(common_rng, _rng_state()), "Readout construction changed RNG"
    _assert_equal_state(reference, local)
    common_names = [name for name, value in reference.named_parameters() if value.requires_grad]
    local_names = [name for name, value in local.named_parameters() if value.requires_grad]
    assert common_names == local_names, "Optimizer parameter registration/order differs"
    assert sum(x.numel() for x in local.local_head.parameters()) == 325
    assert not any(x.requires_grad for x in local.local_head.parameters())
    calls = []
    hook = local.local_head.register_forward_hook(lambda *unused: calls.append(1))
    image = torch.linspace(-1., 1., 2 * 16 ** 3).reshape(2, 1, 16, 16, 16)
    for mode in (False, True):
        reference.train(mode)
        local.train(mode)
        assert not local.local_head.training
        state = _rng_state()
        with torch.no_grad():
            expected = reference(image)
            after_reference = _rng_state()
            _set_rng(state)
            observed = local(image)
        assert _same_rng(after_reference, _rng_state())
        _assert_equal_output(expected, observed)
        _assert_equal_state(reference, local)
    reference.train()
    local.train()
    ropt = torch.optim.AdamW([p for p in reference.parameters() if p.requires_grad], lr=.0001)
    lopt = torch.optim.AdamW([p for p in local.parameters() if p.requires_grad], lr=.0001)
    for step in range(steps):
        # Exercise dropout, legacy classifier supervision, segmentation, GDF,
        # old BN buffers, and common AdamW state on a synthetic 16^3 input.
        batch = image + .01 * step
        state = _rng_state()
        ropt.zero_grad(set_to_none=True)
        a = reference(batch)
        aloss = _synthetic_loss(a)
        aloss.backward()
        after_reference = _rng_state()
        _set_rng(state)
        lopt.zero_grad(set_to_none=True)
        b = local(batch, condition_mode="constant" if step % 2 else None)
        bloss = _synthetic_loss(b)
        bloss.backward()
        assert torch.equal(aloss, bloss)
        _assert_equal_output(a, b)
        assert _same_rng(after_reference, _rng_state())
        for (an, ap), (bn, bp) in zip(reference.named_parameters(), local.named_parameters()):
            assert an == bn
            assert (ap.grad is None) == (bp.grad is None), an
            if ap.grad is not None:
                assert torch.equal(ap.grad, bp.grad), "CPU gradient " + an
        ropt.step()
        lopt.step()
        _assert_equal_state(reference, local)
        for p, q in zip(ropt.param_groups[0]["params"], lopt.param_groups[0]["params"]):
            assert ropt.state[p].keys() == lopt.state[q].keys()
            for key, value in ropt.state[p].items():
                other = lopt.state[q][key]
                assert torch.equal(value, other) if torch.is_tensor(value) else value == other
    hook.remove()
    assert not calls, "Inactive readout was executed"
    return dict(status="passed", synthetic_cpu_updates_per_model=steps,
                patch_size=16, batch_size=2, threads=1,
                common_init_buffers_outputs_gradients_updates_optimizer_rng_exact=True,
                common_optimizer_parameter_order_exact=True, inactive_head_calls=len(calls))


def local_field_contracts(device="cpu"):
    import torch
    from frozen_readout import (align_classifier_features, ct_support_tensor,
                                make_readout_head, readout_probabilities)
    from local_model import local_intervention, LocalStageModel
    # Affine feature fields have an analytic trilinear interpolation value.
    u = torch.arange(32, device=device, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(u, u, u)
    affine = zz + 2. * yy + 3. * xx
    feature = affine[None, None].expand(1, 64, -1, -1, -1).contiguous().requires_grad_()
    aligned = align_classifier_features(feature)
    expected = (u.clamp(max=30.625) + .375)
    # Border padding clips the final sample to31, not31.375.
    expected = expected[:, None, None] + 2. * expected[None, :, None] + 3. * expected[None, None, :]
    assert torch.allclose(aligned[0, 0], expected, atol=4e-5, rtol=1e-6)
    known = torch.ones((1, 1, 128, 128, 128), dtype=torch.bool, device=device)
    support = ct_support_tensor(known)
    assert int(support.sum()) == 28 ** 3
    assert bool(support[:, :, 2:30, 2:30, 2:30].all())
    known[:, :, 64, 64, 64] = False
    assert int(ct_support_tensor(known).sum()) == 28 ** 3 - 4 ** 3
    head = make_readout_head().to(device)
    head.requires_grad_(False)
    with torch.no_grad():
        probability = readout_probabilities(head, feature.detach())
    assert torch.allclose(probability, torch.full_like(probability, .2))
    # A deliberately anticorrelated q/r fixture demonstrates that preserving q
    # and the r multiset does not preserve the consumed raw fine-score mean.
    p = torch.tensor([[.1, .81, .03, .03, .03], [.9, .01, .03, .03, .03]],
                     device=device).T.reshape(1, 5, 1, 1, 2)
    eligible = torch.ones((1, 1, 1, 1, 2), dtype=torch.bool, device=device)
    repaired_s, repaired_p, mapping = local_intervention(p, eligible, "r_repaired")
    assert torch.equal(p[:, :1], repaired_p[:, :1])
    assert torch.allclose(repaired_p.sum(1), torch.ones_like(repaired_p[:, 0]))
    assert torch.allclose(repaired_s.flatten(), torch.tensor([.09, .09], device=device), atol=1e-6)
    assert torch.equal(mapping.flatten(), torch.tensor([1, 0], device=device))
    raw_repaired, _, _ = local_intervention(p, eligible, "s_repaired")
    assert torch.equal(raw_repaired.sort().values, p[:, 1:2].sort().values)
    assert torch.equal(raw_repaired.flatten(), p[:, 1:2].flatten().flip(0))
    mean_s, mean_p, _ = local_intervention(p, eligible, "r_mean")
    assert torch.allclose(mean_p[:, 1:] / (1. - p[:, :1]),
                          torch.tensor([.5, 1. / 6., 1. / 6., 1. / 6.], device=device)[None, :, None, None, None].expand(1, 4, 1, 1, 2), atol=1e-6)
    assert torch.allclose(mean_s.flatten(), torch.tensor([.45, .05], device=device), atol=1e-6)
    # Unknown/border destinations are unchanged by explicit interventions.
    subset = eligible.clone()
    subset[..., 1] = False
    unchanged, _, _ = local_intervention(p, subset, "r_repaired")
    assert torch.equal(unchanged, p[:, 1:2])
    # Normal inference explicitly rejects a diagnostic GT validity mask before
    # executing any model work (the rejection does not require a128^3 forward).
    torch.manual_seed(19)
    local = LocalStageModel("local_scale", alpha=1.).to(device)
    try:
        local(torch.zeros((1, 1, 16, 16, 16), device=device),
              local_diagnostic_valid=eligible)
    except ValueError as error:
        assert "Normal local inference" in str(error)
    else:
        raise AssertionError("Normal inference accepted a GT diagnostic mask")
    return dict(status="passed", device=device,
                affine_alignment_max_abs_error=float((aligned[0, 0] - expected).abs().max()),
                complete_window_count=28 ** 3, one_unknown_voxel_excludes=4 ** 3,
                q_preserved_raw_mean_original=float(p[:, 1:2].mean()),
                q_preserved_raw_mean_repaired=float(repaired_s.mean()),
                synthetic_intervention_mapping="eligible lexicographic half-ring",
                normal_gt_mask_rejected=True)


def full_patch_contracts(device="cuda"):
    """Optional actual128^3 integration/gradient-route check; synthetic only."""
    import torch
    from stage_model import StageModel
    from local_model import LocalStageModel
    torch.manual_seed(113)
    reference = StageModel("constant").to(device).eval()
    local = LocalStageModel("local_scale", alpha=1.).to(device).eval()
    local.load_common_state(reference.state_dict())
    image = torch.linspace(-1., 1., 128 ** 3, device=device).reshape(1, 1, 128, 128, 128)
    modes = (False, True) if str(device).startswith("cuda") else (False,)
    deltas = []
    for amp in modes:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
            a = reference(image)
            b = local(image, trace_local=True)
            zero = local(image, alpha=0.)
        _assert_equal_output(a, b)
        _assert_equal_output(a, zero)
        # Nonuniform head weights produce a real spatial condition. Its values
        # must stay detached while segmentation still trains shared features.
        with torch.no_grad():
            local.local_head.weight.zero_()
            local.local_head.bias.copy_(torch.tensor([0., .8, -.1, -.2, -.3], device=device))
            local.local_head.weight[1, 0, 0, 0, 0] = .03
        gradients = []
        for unused in range(2):
            local.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                out = local(image, trace_local=True)
                loss = out["logits"].float().square().mean()
            (loss * (1024. if amp else 1.)).backward()
            assert local.last_local_trace["probability_detached"]
            assert all(p.grad is None for p in local.local_head.parameters())
            assert all(p.grad is None or not bool(p.grad.count_nonzero())
                       for p in local.classifier.generation_head.parameters())
            assert any(p.grad is not None and bool(p.grad.count_nonzero())
                       for p in local.classifier.encoder.parameters())
            assert any(p.grad is not None and bool(p.grad.count_nonzero())
                       for p in local.generation_guidance.parameters())
            gradients.append({name: p.grad.detach().clone() for name, p in local.named_parameters()
                              if p.grad is not None})
        assert gradients[0].keys() == gradients[1].keys()
        deltas.append(dict(autocast=amp, self_repeat_gradient_max_abs=max(
            float((gradients[0][key] - gradients[1][key]).abs().max()) for key in gradients[0])))
        # Restore zero head before the next AMP mode's constant-forward check.
        with torch.no_grad():
            local.local_head.weight.zero_()
            local.local_head.bias.zero_()
    return dict(status="passed", device=device, patch_size=128,
                controlled_forward_exact=True, frozen_detached_gradient_route=True,
                cuda_multistep_bitwise_claim=False, repeated_backward_noise=deltas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--full-patch128", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Refusing to overwrite a previous QA receipt")
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU substitution")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    started = time.time()
    source_names = ("local_model.py", "local_checks.py", "frozen_readout.py", "stage_model.py")
    sources = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in source_names}
    report = dict(schema="GTA-AVT-local-revision-synthetic-QA-v1", status="passed",
                  study_data_read=False, real_training_started=False,
                  source_sha256=sources,
                  runtime=dict(executable=sys.executable, python=sys.version,
                               torch=torch.__version__, command_argv=sys.argv,
                               requested_device=args.device),
                  cpu_common_parity=cpu_common_parity(),
                  field_contracts=local_field_contracts(args.device),
                  full_patch128=(full_patch_contracts(args.device) if args.full_patch128
                                 else dict(status="not_run", reason="--full-patch128 omitted")),
                  cuda_multistep_bitwise_claim=False,
                  seconds=time.time() - started)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
