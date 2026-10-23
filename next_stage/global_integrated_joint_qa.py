"""One synthetic joint-objective AMP update; default metadata plan, no study data.

This NEW engineering helper does not change any frozen model/trainer/loss.
It is not a formal Fp run, evidence of efficacy, or permission to start one.
"""
import argparse
import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_contract(manifest, recipe):
    require(manifest["status"] == "UNTRAINED_STOPPED_NOT_LAUNCH_AUTHORIZED", "Candidate status")
    require(manifest["formal_training_authorized"] is False, "Formal training prohibited")
    require(manifest["local_integration_implemented"] is False, "No local+GDF claim")
    fixed = dict(conditioning="scale_only", detach_conditions=True, q0=.2, s0=.2,
                 scale_weight=.1, structure_weight=.1, gdf_weight=.1,
                 gdf_normalization="voxel", patch_size=128, batch_size=2,
                 max_steps=2000, alpha_start=0, alpha_ramp=500,
                 lr=.0001, weight_decay=.0001, gradient_clip=12., allow_tf32=False)
    for name, expected in fixed.items():
        require(recipe.get(name) == expected, "Fixed candidate recipe: " + name)
    require(len(manifest["scientific_source_sha256"]) == 33, "Expected frozen scientific33")
    require(manifest["qa"]["successful_optimizer_updates"] == 1, "Exactly one update")
    require(manifest["qa"]["alpha"] == 1., "QA exercises alpha1")


def verify_sources(root, expected):
    actual = {}
    for relative, digest in expected.items():
        path = (root / relative).resolve()
        path.relative_to(root)
        actual[relative] = sha256(path)
        require(actual[relative] == digest, "Frozen source mismatch: " + relative)
    return actual


def execute(args, manifest, recipe):
    root = Path(args.source_root).resolve()
    source_before = verify_sources(root, manifest["scientific_source_sha256"])
    require(args.warmstart, "Synthetic QA requires the named original constant warmstart")
    weight_sha = sha256(args.warmstart)
    require(weight_sha == manifest["candidate_initialization"]["warmstart_sha256"],
            "Only original constant origin permitted; Gp/Kp endpoints are not substitutes")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "Declared CUBLAS profile")
    for directory in (root / "code", root / "next_stage"):
        sys.path.insert(0, str(directory))
    import torch
    modules = {name: importlib.import_module(name) for name in
               ("stage_model", "stage_train", "train", "faithful_model", "losses", "gdf", "common")}
    for name, module in modules.items():
        folder = "next_stage" if name in ("stage_model", "stage_train", "gdf") else "code"
        require(Path(module.__file__).resolve() == root / folder / (name + ".py"),
                "Wrong imported source: " + name)
    for name, module in list(sys.modules.items()):
        if name.startswith("historical") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            path.relative_to(root / "code")
            relative = path.relative_to(root).as_posix()
            require(relative in source_before, "Unbound historical dependency: " + relative)
    require(torch.cuda.is_available(), "CUDA unavailable; no smaller/CPU fallback")
    torch.set_num_threads(4)
    modules["train"].set_seed(recipe["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = modules["stage_model"].StageModel(
        "scale_only", detach_conditions=True, alpha=1., q0=.2, s0=.2,
        gdf_seed=recipe["gdf_seed"])
    loaded = modules["common"].load_trusted_checkpoint(args.warmstart, "cpu")
    state = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
    require(isinstance(state, dict) and state and all(torch.is_tensor(v) for v in state.values()),
            "Warmstart must be tensor-only state")
    report = model.load_state_dict(state, strict=False)
    require(not report.unexpected_keys and set(report.missing_keys) in
            (set(), {"gdf_head.weight", "gdf_head.bias"}), "Legacy/stage state identity")
    del loaded, state
    initial_digest = modules["stage_train"].tensor_digest(model.state_dict())
    model.to("cuda").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"],
                                 weight_decay=recipe["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    patch, batch = 128, 2
    image = torch.linspace(-1., 1., batch * patch ** 3, device="cuda").view(batch, 1, patch, patch, patch)
    axis = torch.arange(patch, device="cuda", dtype=torch.float32) - (patch - 1.) / 2.
    ball = (axis[:, None, None].square() + axis[None, :, None].square() +
            axis[None, None, :].square()) <= (patch / 5.) ** 2
    mask = ball[None, None].expand(batch, 1, patch, patch, patch).float().contiguous()
    arrays = dict(mask=mask,
                  scale=torch.tensor([[0., .7, .1, .1, .1], [0., .1, .2, .3, .4]], device="cuda"),
                  structure=torch.tensor([[0., .8, .2], [0., .3, .7]], device="cuda"),
                  attribute_valid=torch.ones(batch, device="cuda"),
                  gdf_target=.7 * mask, gdf_valid=torch.ones_like(mask, dtype=torch.bool))
    # Arbitrary nonnegative regression fixture; no claim that this is an AVT GDF.
    # Targets are passed only to the frozen objective, never the model forward.
    watched_groups = {
        "segmentation_head": "anatomical_output_fusion.3.",
        "scale_head": "classifier.generation_head.",
        "structure_head": "classifier.structure_head.",
        "gdf_head": "gdf_head.",
        "shared_segmentation_feature": "anatomical_output_fusion.0.",
        "shared_attribute_encoder": "classifier.encoder."}
    parameters = dict(model.named_parameters())
    step_evidence, overflow, forwards = {}, [], []
    original_step = optimizer.step

    def observed_step(*step_args, **step_kwargs):
        require(not step_evidence, "More than one successful optimizer step")
        before, selected = {}, {}
        for group, prefix in watched_groups.items():
            group_parameters = [(n, p) for n, p in parameters.items() if n.startswith(prefix)]
            require(bool(group_parameters), "Missing parameter group: " + group)
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all())
                        for _, p in group_parameters), "Nonfinite group gradient: " + group)
            nonzero = [(n, p) for n, p in group_parameters if p.grad is not None and bool(p.grad.count_nonzero())]
            require(bool(nonzero), "No nonzero joint gradient: " + group)
            name, parameter = nonzero[0]
            selected[group] = name
            before[group] = parameter.detach().cpu().clone()
            step_evidence[group] = dict(parameter=name, finite=True,
                gradient_nonzero_elements=int(parameter.grad.count_nonzero()),
                clipped_gradient_absmax=float(parameter.grad.detach().abs().max()))
        returned = original_step(*step_args, **step_kwargs)
        for group, name in selected.items():
            after = parameters[name].detach().cpu()
            require(bool(torch.isfinite(after).all()), "Nonfinite updated parameter: " + group)
            changes = int((after != before[group]).count_nonzero())
            require(changes > 0, "No parameter change: " + group)
            step_evidence[group]["changed_elements"] = changes
            step_evidence[group]["parameter_delta_absmax"] = float((after - before[group]).abs().max())
        return returned

    optimizer.step = observed_step

    def forward_loss():
        with torch.cuda.amp.autocast(enabled=True):
            output = model(image)  # configured CT-only predictions; no GT override.
            require(output["logits"].dtype == torch.float16, "Actual AMP logits required")
            for name in ("logits", "scale_logits", "structure_logits", "gdf_prediction"):
                require(bool(torch.isfinite(output[name]).all()), "Nonfinite output: " + name)
            trace = model.last_condition_trace
            require(trace["mode"] == "configured" and trace["alpha"] == 1., "Normal alpha1 route")
            require(trace["consumed_probability_detached"] is True and not trace["diagnostic_intervention"],
                    "Detached normal route only")
            require(torch.equal(trace["supplied_scale"], trace["scale_head_probability"]), "Predicted scale consumed")
            require(torch.equal(trace["supplied_structure"], torch.full_like(trace["supplied_structure"], 1. / 3.)),
                    "SAAM remains uniform")
            loss, details = modules["stage_train"].stage_objective(
                output, arrays, .1, .1, .1, "voxel")
            for name in ("segmentation", "scale", "structure", "gdf_mse", "total"):
                require(math.isfinite(details[name]) and details[name] > 0., "Invalid joint component: " + name)
            forwards.append(dict(logits_dtype=str(output["logits"].dtype),
                                 normal_predicted_conditions=True, alpha=1., details=details))
            del output
            return loss, details

    torch.cuda.reset_peak_memory_stats()
    details, update = modules["train"].optimizer_update_with_amp_retries(
        model, optimizer, scaler, forward_loss, gradient_clip=12., max_attempts=8,
        on_overflow=lambda record: overflow.append(record))
    torch.cuda.synchronize()
    require(bool(step_evidence) and len(forwards) == update["amp_attempts"], "One successful update accounting")
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Updated model finite")
    require(source_before == verify_sources(root, manifest["scientific_source_sha256"]), "Source changed")
    return dict(status="passed", scope="synthetic_global_integrated_joint_objective_AMP_one_update",
        study_data_read=False, formal_training=False, efficacy_established=False,
        successful_optimizer_updates=1, checkpoint_saved=False, model_alpha=1.,
        batch_size=2, patch_size=128, precision="CUDA_AMP", normal_CT_only_conditions=True,
        candidate_origin_sha256=weight_sha, initial_state_tensor_digest=initial_digest,
        warmstart_load=dict(missing_keys=report.missing_keys, unexpected_keys=report.unexpected_keys),
        source_sha256=source_before, component_details=details, optimizer_update=update,
        gradient_and_parameter_change=step_evidence, overflow_attempts=overflow,
        forward_attempts=forwards, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        runtime=modules["stage_train"].runtime_identity(torch.device("cuda")),
        fixture="Deterministic synthetic intensity ramp, sphere mask, soft compositions, .7*mask nonnegative regression target; not a physical GDF validation",
        limits=["No AVT arrays, no VAL/TEST, no scientific gain or seed stability",
                "Not formal Fp; failed dynamic/GDF continuation gates unchanged",
                "No local+GDF claim; SAAM constant; one original-constant origin only"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root")
    parser.add_argument("--manifest", default=str(HERE / "global_integrated_candidate_manifest_20261023.json"))
    parser.add_argument("--recipe", default=str(HERE / "global_integrated_candidate_UNTRAINED_20261023.json"))
    parser.add_argument("--warmstart")
    parser.add_argument("--output")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest, recipe = read_json(args.manifest), read_json(args.recipe)
    validate_contract(manifest, recipe)
    require(sha256(args.recipe) == manifest["recipe_sha256"], "Recipe byte binding")
    plan = dict(execute=args.execute, status=manifest["status"], formal_training_authorized=False,
                source_files_pinned=33, study_data_read=False, synthetic_successful_updates=1,
                patch_size=128, batch_size=2, precision="CUDA_AMP", alpha=1.,
                manifest_sha256=sha256(args.manifest), recipe_sha256=sha256(args.recipe),
                helper_sha256=sha256(__file__), candidate_initialization=manifest["candidate_initialization"])
    if not args.execute:
        print(json.dumps(plan, allow_nan=False), flush=True)
        return
    require(args.source_root and args.output, "Execution requires source-root and fresh output directory")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "intent.json", plan)
    started = time.time()
    try:
        result = execute(args, manifest, recipe)
    except Exception as error:
        result = dict(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(), study_data_read=False, formal_training=False)
        result.update(plan=plan, seconds=time.time() - started,
                      utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(output / "result.json", result)
        raise
    result.update(plan=plan, seconds=time.time() - started,
                  utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(output / "result.json", result)
    print(json.dumps(dict(status=result["status"], successful_optimizer_updates=1,
                          seconds=result["seconds"]), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
