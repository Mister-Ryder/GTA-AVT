"""Small model contract checks; no optimizer update and no study data access."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

# Must be set before importing Torch/initializing CUDA, matching stage_train.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
for path in (CODE, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def model_checks(device="cpu", patch=32):
    import torch
    from faithful_model import FaithfulGTAAVT
    from stage_model import StageModel

    torch.set_num_threads(2)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("The parity QA requires the declared :4096:8 CUBLAS profile")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Frozen train.set_seed uses warn_only=True because CUDA interpolation
    # backward has known non-deterministic paths. Forward exact parity remains
    # a hard assertion; this is not an output tolerance relaxation.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(20261023)
    legacy = FaithfulGTAAVT("faithful_soft_constant")
    legacy_state = {k: v.detach().clone() for k, v in legacy.state_dict().items()}
    rng_after_legacy = torch.get_rng_state().clone()
    torch.manual_seed(20261023)
    constant = StageModel("constant")
    assert torch.equal(torch.get_rng_state(), rng_after_legacy)
    assert set(constant.state_dict()) - set(legacy_state) == {"gdf_head.weight", "gdf_head.bias"}
    assert all(torch.equal(constant.state_dict()[k], v) for k, v in legacy_state.items())
    routed = StageModel("scale_only", detach_conditions=True, alpha=0.)
    routed.load_state_dict(constant.state_dict(), strict=True)
    legacy.to(device).eval()
    constant.to(device).eval()
    routed.to(device).eval()
    image = torch.linspace(-1., 1., patch ** 3, device=device).view(1, 1, patch, patch, patch)
    amp_modes = (False, True) if str(device).startswith("cuda") else (False,)
    for use_amp in amp_modes:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            reference = legacy(image)
            fixed = constant(image)
            alpha0 = routed(image)
        assert torch.equal(reference["logits"], fixed["logits"]), "constant AMP parity"
        assert torch.equal(fixed["logits"], alpha0["logits"]), "alpha0 AMP parity"
        assert all(torch.equal(a, b) for a, b in zip(
            reference["aux_logits"], fixed["aux_logits"])), "constant DS AMP parity"
        assert all(torch.equal(a, b) for a, b in zip(
            fixed["aux_logits"], alpha0["aux_logits"])), "alpha0 DS AMP parity"
    assert fixed["gdf_prediction"].shape == fixed["logits"].shape
    assert bool(torch.isfinite(fixed["gdf_prediction"]).all())
    assert bool((fixed["gdf_prediction"] >= 0.).all())

    # Diagnostic interventions must preserve a valid simplex, fixed q when
    # requested, and the nonempty composition when only q changes.
    probability = torch.tensor([[.7, .1, .05, .1, .05],
                                [.1, .4, .2, .1, .2]], device=device)
    fixedq = StageModel._intervene(probability, "qfixed_rpred", .2)
    uniformr = StageModel._intervene(probability, "qpred_runiform", .2)
    assert torch.allclose(fixedq.sum(1), torch.ones(2, device=device))
    assert torch.allclose(fixedq[:, 0], torch.full((2,), .2, device=device))
    assert torch.allclose(fixedq[:, 1:] / .8,
                          probability[:, 1:] / (1. - probability[:, :1]))
    assert torch.equal(uniformr[:, 0], probability[:, 0])
    assert torch.allclose(uniformr[:, 1:],
                          (1. - probability[:, :1]).expand(-1, 4) / 4.)

    override = {"scale": torch.tensor([[.01, .95, .01, .01, .02]], device=device)}
    routed.set_alpha(1.)
    head_parameters = list(routed.classifier.generation_head.parameters())
    for use_amp in amp_modes:
        gradient_scale = 1024. if use_amp else 1.
        routed.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = routed(image, condition_override=override)
            assert bool(torch.isfinite(output["logits"]).all())
            detached = routed(image)
        # Detached probabilities cannot train that head from segmentation;
        # feature/guidance segmentation gradients remain active.
        (detached["logits"].float().square().mean() * gradient_scale).backward()
        assert all(p.grad is None or not bool(p.grad.count_nonzero()) for p in head_parameters)
        assert any(p.grad is not None and bool(p.grad.count_nonzero())
                   for p in routed.classifier.encoder.parameters())
        routed.detach_conditions = False
        routed.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            differentiable = routed(image)
        (differentiable["logits"].float().square().mean() * gradient_scale).backward()
        assert any(p.grad is not None and bool(p.grad.count_nonzero()) for p in head_parameters)
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in head_parameters)
        routed.detach_conditions = True
        routed.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            supervised = routed(image)
        (supervised["scale_logits"][:, 1].float().sum() * gradient_scale).backward()
        assert any(p.grad is not None and bool(p.grad.count_nonzero()) for p in head_parameters)
        routed.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            field_output = routed(image)
        (field_output["gdf_prediction"].square().mean() * gradient_scale).backward()
        assert bool(routed.gdf_head.weight.grad.count_nonzero())
        assert bool(torch.isfinite(routed.gdf_head.weight.grad).all())
        assert any(p.grad is not None and bool(p.grad.count_nonzero())
                   for p in routed.anatomical_output_fusion[0].parameters())
    return dict(
        status="passed", device=device, patch_size=patch, model_update_count=0,
        study_data_read=False, autocast_modes_checked=list(amp_modes),
        amp_backward_manual_loss_scale=1024.,
        backend_settings=dict(
            matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
            cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
            cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
            deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
            deterministic_warn_only=getattr(
                torch, "is_deterministic_algorithms_warn_only_enabled", lambda: None)(),
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG")),
        checks=["legacy_parameter_and_rng_parity", "constant_logit_exact_parity",
                "alpha0_exact_constant_fallback", "gdf_shape_finite_nonnegative",
                "q_r_simplex_and_composition", "detached_probability_segmentation_route",
                "attribute_supervision_head_gradient", "gdf_shared_feature_gradient"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--patch", type=int, default=32)
    parser.add_argument("--output")
    args = parser.parse_args()
    started = time.time()
    result = model_checks(args.device, args.patch)
    result["seconds"] = time.time() - started
    if args.output:
        from common import atomic_json
        path = Path(args.output)
        if path.exists():
            raise ValueError("Check output already exists")
        atomic_json(path, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
