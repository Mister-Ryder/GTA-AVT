"""Sequential model QA with bounded graph lifetime; no updates or study data.

This is a NEW QA implementation, not a change to the frozen scientific model
or the older stage_checks.py. CUDA preflight uses actual batch2, patch128,
FP32 and AMP. A hardware OOM remains a failed check; there is no scope fallback.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# The declared forward-parity profile must exist before importing Torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
for path in (PROJECT / "code", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def source_inventory(project=PROJECT):
    """Same executable/historical-file scope as the global trainer."""
    result = {}
    for folder in ("code", "next_stage"):
        for path in sorted((Path(project) / folder).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix == ".py" or ("historical" in path.parts and path.suffix.lower() in (".txt", ".json", ".md")):
                path.resolve().relative_to(Path(project).resolve())
                result[path.relative_to(project).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def isolated_phase(name, body, cleanup, reports, memory=None):
    """Only JSON scalars may escape; graph-owning locals end before cleanup.

    The check body must be a function (not code in the enclosing QA loop).
    CPython releases its output/loss locals on return. Cleanup then clears
    parameter grads and the allocator cache before the next function starts.
    CPU detached parity snapshots are the sole explicitly retained tensors.
    """
    started = time.time()
    if memory:
        memory("reset")
    print(json.dumps(dict(event="phase_started", name=name)), flush=True)
    succeeded = False
    try:
        result = body()
        json.dumps(result, allow_nan=False)  # Reject accidental tensor/graph returns.
        succeeded = True
        return result
    finally:
        cleanup()
        row = dict(name=name, status="passed" if succeeded else "failed",
                   seconds=time.time() - started)
        if memory:
            row.update(memory("read"))
        reports.append(row)
        print(json.dumps(dict(event="phase_finished", **row)), flush=True)


def assert_exact_parity(reference, fixed, alpha0, equal):
    """All logits and both DS tensors must compare exactly, without tolerance."""
    assert equal(reference["logits"], fixed["logits"]), "constant AMP parity"
    assert equal(fixed["logits"], alpha0["logits"]), "alpha0 AMP parity"
    assert len(reference["aux_logits"]) == len(fixed["aux_logits"]) == len(alpha0["aux_logits"]) == 2
    assert all(equal(a, b) for a, b in zip(reference["aux_logits"], fixed["aux_logits"])), "constant DS AMP parity"
    assert all(equal(a, b) for a, b in zip(fixed["aux_logits"], alpha0["aux_logits"])), "alpha0 DS AMP parity"


def model_checks(device="cpu", patch=32, batch_size=2):
    if batch_size != 2:
        raise ValueError("This QA explicitly requires actual batch2; no batch fallback")
    if patch <= 0 or patch % 16:
        raise ValueError("QA patch must be positive and divisible by16")
    before_source = source_inventory()
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
    for model in (legacy, constant, routed):
        model.eval()
    del legacy_state
    image = torch.linspace(-1., 1., batch_size * patch ** 3, device=device).view(
        batch_size, 1, patch, patch, patch)
    amp_modes = (False, True) if str(device).startswith("cuda") else (False,)
    phases, snapshots = [], {}

    def cuda_memory(action):
        if not str(device).startswith("cuda"):
            return {}
        torch.cuda.synchronize(device)
        if action == "reset":
            torch.cuda.reset_peak_memory_stats(device)
            return {}
        return dict(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                    cuda_allocated_after_cleanup_bytes=torch.cuda.memory_allocated(device),
                    cuda_reserved_after_cleanup_bytes=torch.cuda.memory_reserved(device))

    def cleanup(model, offload=False):
        model.zero_grad(set_to_none=True)
        if offload:
            model.to("cpu")
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    def snapshot(model, name, use_amp):
        # Each forward completes before another model is placed on the GPU.
        model.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            output = model(image)
        if "gdf_prediction" in output:
            assert output["gdf_prediction"].shape == output["logits"].shape
            assert bool(torch.isfinite(output["gdf_prediction"]).all())
            assert bool((output["gdf_prediction"] >= 0.).all())
        value = dict(logits=output["logits"].detach().to("cpu"),
                     aux_logits=[x.detach().to("cpu") for x in output["aux_logits"]])
        assert value["logits"].device.type == "cpu" and not value["logits"].requires_grad
        assert all(x.device.type == "cpu" and not x.requires_grad for x in value["aux_logits"])
        snapshots[name] = value
        return dict(logits_shape=list(output["logits"].shape), logits_dtype=str(output["logits"].dtype))

    for use_amp in amp_modes:
        for name, model in (("legacy", legacy), ("constant", constant), ("alpha0", routed)):
            isolated_phase("parity_%s_amp%s" % (name, use_amp),
                lambda m=model, n=name: snapshot(m, n, use_amp),
                lambda m=model: cleanup(m, offload=True), phases, cuda_memory)
        reference, fixed, alpha0 = [snapshots[n] for n in ("legacy", "constant", "alpha0")]
        assert_exact_parity(reference, fixed, alpha0, torch.equal)
        snapshots.clear()
        del reference, fixed, alpha0
    # No further legacy/constant forward is needed; they remain CPU-only.
    del legacy, constant
    gc.collect()

    probability = torch.tensor([[.7, .1, .05, .1, .05], [.1, .4, .2, .1, .2]], device=device)
    fixedq = StageModel._intervene(probability, "qfixed_rpred", .2)
    uniformr = StageModel._intervene(probability, "qpred_runiform", .2)
    assert torch.allclose(fixedq.sum(1), torch.ones(2, device=device))
    assert torch.allclose(fixedq[:, 0], torch.full((2,), .2, device=device))
    assert torch.allclose(fixedq[:, 1:] / .8, probability[:, 1:] / (1. - probability[:, :1]))
    assert torch.equal(uniformr[:, 0], probability[:, 0])
    assert torch.allclose(uniformr[:, 1:], (1. - probability[:, :1]).expand(-1, 4) / 4.)
    del probability, fixedq, uniformr

    routed.to(device)
    routed.set_alpha(1.)
    head_parameters = list(routed.classifier.generation_head.parameters())
    override = {"scale": torch.tensor([[.01, .95, .01, .01, .02]], device=device).repeat(batch_size, 1)}
    for use_amp in amp_modes:
        gradient_scale = 1024. if use_amp else 1.

        def check_override():
            # The old override assertion never used backward: do not save a graph.
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                output = routed(image, condition_override=override)
                assert bool(torch.isfinite(output["logits"]).all())
            return dict(logits_finite=True)

        def check_detached():
            routed.detach_conditions = True
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = routed(image)
            loss = output["logits"].float().square().mean() * gradient_scale
            del output  # Drop unused DS/GDF outputs before this one backward.
            loss.backward()
            assert all(p.grad is None or not bool(p.grad.count_nonzero()) for p in head_parameters)
            assert any(p.grad is not None and bool(p.grad.count_nonzero()) for p in routed.classifier.encoder.parameters())
            return dict(attribute_head_segmentation_gradient_zero=True, encoder_gradient_nonzero=True)

        def check_differentiable():
            routed.detach_conditions = False
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = routed(image)
            loss = output["logits"].float().square().mean() * gradient_scale
            del output
            loss.backward()
            assert any(p.grad is not None and bool(p.grad.count_nonzero()) for p in head_parameters)
            assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in head_parameters)
            return dict(attribute_head_segmentation_gradient_nonzero_finite=True)

        def check_supervised():
            routed.detach_conditions = True
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = routed(image)
            loss = output["scale_logits"][:, 1].float().sum() * gradient_scale
            # In particular release the unused segmentation graph BEFORE backward.
            del output
            loss.backward()
            assert any(p.grad is not None and bool(p.grad.count_nonzero()) for p in head_parameters)
            return dict(attribute_supervision_head_gradient_nonzero=True)

        def check_field():
            routed.detach_conditions = True
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = routed(image)
            loss = output["gdf_prediction"].square().mean() * gradient_scale
            del output
            loss.backward()
            assert bool(routed.gdf_head.weight.grad.count_nonzero())
            assert bool(torch.isfinite(routed.gdf_head.weight.grad).all())
            assert any(p.grad is not None and bool(p.grad.count_nonzero()) for p in routed.anatomical_output_fusion[0].parameters())
            return dict(gdf_head_gradient_nonzero_finite=True, gdf_shared_feature_gradient_nonzero=True)

        for name, body in (("override", check_override), ("detached", check_detached),
                           ("differentiable", check_differentiable), ("supervised", check_supervised),
                           ("gdf", check_field)):
            isolated_phase("%s_amp%s" % (name, use_amp), body,
                           lambda: cleanup(routed), phases, cuda_memory)
    assert source_inventory() == before_source, "Source changed during model QA"
    return dict(status="passed", device=device, patch_size=patch, batch_size=batch_size,
        model_update_count=0, study_data_read=False, autocast_modes_checked=list(amp_modes),
        amp_backward_manual_loss_scale=1024., phases=phases,
        qa_file_path="next_stage/stage_checks_memory_safe.py",
        qa_file_sha256=before_source["next_stage/stage_checks_memory_safe.py"],
        source_sha256=before_source,
        graph_lifetime_policy="One forward graph per function; only CPU detached parity tensors retained; unused outputs deleted before backward; grads cleared after each phase; only one GPU model",
        backend_settings=dict(matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            cudnn_tf32=bool(torch.backends.cudnn.allow_tf32), cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
            cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
            deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
            deterministic_warn_only=getattr(torch, "is_deterministic_algorithms_warn_only_enabled", lambda: None)(),
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG")),
        checks=["legacy_parameter_and_rng_parity", "constant_logit_exact_parity",
                "alpha0_exact_constant_fallback", "constant_and_alpha0_DS_exact_parity",
                "gdf_shape_finite_nonnegative", "q_r_simplex_and_composition",
                "detached_probability_segmentation_route", "differentiable_probability_segmentation_route",
                "attribute_supervision_head_gradient", "gdf_shared_feature_gradient"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--patch", type=int, default=32)
    parser.add_argument("--batch-size", type=int, choices=(2,), default=2)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.output and Path(args.output).exists():
        raise ValueError("Check output already exists")
    started = time.time()
    result = model_checks(args.device, args.patch, args.batch_size)
    result["seconds"] = time.time() - started
    if args.output:
        with Path(args.output).open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
