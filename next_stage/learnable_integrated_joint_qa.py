"""New learnable interaction + physical Gaussian field, one synthetic AMP step.

Metadata-only by default. This isolated engineering helper does not launch a
formal run or reopen any failed scientific gate. All targets are synthetic and
are passed only to the objective; no AVT arrays or TEST files are opened.
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
    require(manifest["status"] == "STOPPED_UNTRAINED_NOT_LAUNCH_AUTHORIZED", "Candidate status")
    require(manifest["formal_training_authorized"] is False, "Formal training prohibited")
    require(manifest["schema"] == "GTA-AVT-learnable-integrated-candidate-v1", "New named integrated candidate")
    fixed = dict(conditioning="scale_only", detach_conditions=True, q0=.2, s0=.2,
                 scale_weight=.1, structure_weight=.1, gdf_weight=.1,
                 gdf_normalization="voxel", patch_size=128, batch_size=2,
                 max_steps=2000, alpha_start=0, alpha_ramp=500,
                 lr=.0001, weight_decay=.0001, gradient_clip=12., allow_tf32=False)
    for name, expected in fixed.items():
        require(recipe.get(name) == expected, "Fixed candidate recipe: " + name)
    require(len(manifest["scientific_source_sha256"]) == 37, "Expected frozen learnable36 plus isolated integrated trainer")
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


def physical_fixture(geometry, gdf, field_scale):
    """Deterministic synthetic cylinders, using the frozen physical pipeline."""
    import numpy as np
    patch = 128
    z, y, x = np.ogrid[:patch, :patch, :patch]
    values, fields, validity, scales, structures, audits = [], [], [], [], [], []
    for radius in (3., 6.):
        mask = ((z >= 16) & (z < 111) & ((y - 64.)**2 + (x - 64.)**2 <= radius**2))
        geom = geometry.sparse_geometry(mask, np.ones(3))
        require(bool(geom['geometry_valid']) and len(geom['points']) > 0, 'Synthetic physical geometry unsupported')
        spec = gdf.GaussianFieldSpec(field_scale=field_scale)
        field, valid, audit = gdf.gaussian_target(mask, geom['points'], geom['radius'],
                                                spacing_mm=(1., 1., 1.), spec=spec)
        scale, structure = geometry.patch_histograms(geom, (0, 0, 0), (patch, patch, patch))
        require(np.isfinite(field).all() and (field[mask] > 0.).all() and (field[~mask] == 0.).all() and
                valid.all() and scale.sum() > 0 and structure.sum() > 0, 'Physical synthetic field/target invalid')
        values.append(mask[None].astype(np.float32)); fields.append(field[None]); validity.append(valid[None])
        scales.append(scale); structures.append(structure)
        audits.append(dict(synthetic_mask_radius_mm=radius, foreground_voxels=int(mask.sum()),
            centreline_nodes=len(geom['points']), measured_radius_min_mm=float(geom['radius'].min()),
            measured_radius_max_mm=float(geom['radius'].max()), field=audit))
    return dict(arrays=dict(mask=np.stack(values), scale=np.stack(scales), structure=np.stack(structures),
        attribute_valid=np.ones(2, np.float32), gdf_target=np.stack(fields), gdf_valid=np.stack(validity)),
        audit=dict(kind='synthetic_128mm_cylinders_not_AVT', cases=audits, physical_spacing_mm=[1., 1., 1.],
            geometry='Frozen sparse_geometry: physical EDT at Lee skeleton points',
            target='Frozen exact complete-seed cumulative Gaussian, masked foreground, log1p/fixed TRAIN scale',
            fixed_train_field_scale=field_scale, per_case_normalization=False, patient_arrays_opened=False))


def gdf_routing(model, image, arrays, modules, groups, enabled):
    if not enabled:
        return dict(status='not_executed_explicitly_omitted', passed=False,
                    optimizer_updates=0, limitation='Joint gradient evidence does not independently attribute each loss')
    import torch
    buffers = {k:v.detach().clone() for k,v in model.named_buffers()}
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), [v.clone() for v in torch.cuda.get_rng_state_all()]
    selected = {label:next((n,p) for n,p in model.named_parameters() if n.startswith(prefix))
                for label,prefix in groups.items()}
    with torch.cuda.amp.autocast(enabled=True):
        output = model(image)
        loss, details = modules['gdf'].gdf_mse(output['gdf_prediction'], arrays['gdf_target'], arrays['gdf_valid'],
                                             foreground=arrays['mask'], normalization='voxel')
    # Fixed diagnostic scale prevents small whole-volume derivatives being
    # rounded away in AMP; this is not an optimizer/scaler update or new loss.
    diagnostic_scale = 1024.
    gradients = torch.autograd.grad(loss * diagnostic_scale, [p for _,p in selected.values()], allow_unused=True)
    result = {}
    disconnected = {'segmentation_head', 'scale_head', 'structure_head'}
    for (label, (name, parameter)), grad in zip(selected.items(), gradients):
        finite = grad is None or bool(torch.isfinite(grad).all())
        nonzero = 0 if grad is None else int(grad.count_nonzero())
        require(finite, 'Nonfinite GDF-only diagnostic gradient: ' + label)
        if label in disconnected:
            require(grad is None or nonzero == 0, 'GDF bypassed detached probabilities/projection hook: ' + label)
        else:
            require(nonzero > 0, 'Expected GDF-only shared/interaction route absent in fixture: ' + label)
        result[label] = dict(parameter=name, finite=finite, gradient_nonzero_elements=nonzero,
            unscaled_absmax=0. if grad is None else float(grad.detach().float().abs().max()) / diagnostic_scale,
            expected_direct_gradient=label not in disconnected)
    del output, loss, gradients
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
    torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
    require(all(torch.equal(value,buffers[name]) for name,value in model.named_buffers()) and
            torch.equal(torch.get_rng_state(),cpu_rng) and
            all(torch.equal(x,y) for x,y in zip(torch.cuda.get_rng_state_all(),cuda_rng)), 'Routing BN/RNG restore failed')
    model.zero_grad(set_to_none=True)
    return dict(status='passed', passed=True, optimizer_updates=0, diagnostic_gradient_scale=diagnostic_scale,
                no_graph_retained_for_joint_update=True, BN_RNG_restored=True, gradients=result,
                scope='One physical synthetic fixture; routing evidence, not training gradient dominance or efficacy')


def execute(args, manifest, recipe):
    root = Path(args.source_root).resolve()
    source_before = verify_sources(root, manifest["scientific_source_sha256"])
    require(args.warmstart, "Synthetic QA requires the named original constant warmstart")
    weight_sha = sha256(args.warmstart)
    require(weight_sha == manifest["warmstart_sha256"],
            "Only original constant origin permitted; Gp/Kp endpoints are not substitutes")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "Declared CUBLAS profile")
    for directory in (root / "code", root / "next_stage"):
        sys.path.insert(0, str(directory))
    import torch
    modules = {name: importlib.import_module(name) for name in
               ("learnable_stage_model", "learnable_integrated_train", "train", "faithful_model", "losses", "gdf", "common", "geometry")}
    for name, module in modules.items():
        folder = "next_stage" if name in ("learnable_stage_model", "learnable_integrated_train", "gdf") else "code"
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
    model = modules["learnable_stage_model"].LearnableStageModel(
        "scale_only", detach_conditions=True, alpha=1., q0=.2, s0=.2,
        gdf_seed=recipe["gdf_seed"])
    loaded = modules["common"].load_trusted_checkpoint(args.warmstart, "cpu")
    state = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
    require(isinstance(state, dict) and state and all(torch.is_tensor(v) for v in state.values()),
            "Warmstart must be tensor-only state")
    report = model.load_common_state(state)
    require(not bool(model.interaction_weight.count_nonzero()) and not bool(model.interaction_bias.count_nonzero()),
            "Interaction must begin at exact zero")
    require(model.interaction_weight.shape == model.interaction_bias.shape == (128,), "256 interaction scalars")
    del loaded, state
    initial_state = model.state_dict()
    td = modules["learnable_integrated_train"].tensor_digest
    initialization = dict(state_tensor_digest=td(initial_state),
        legacy_tensor_digest=td({k:v for k,v in initial_state.items() if not k.startswith("gdf_head.") and k not in ("interaction_weight","interaction_bias")}),
        gdf_head_digest=td({k:v for k,v in initial_state.items() if k.startswith("gdf_head.")}),
        interaction_digest=td({k:v for k,v in initial_state.items() if k in ("interaction_weight","interaction_bias")}))
    require(initialization == manifest["initialization_four_digests"], "Same original common/interaction/GDF initialization")
    del initial_state
    model.to("cuda").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"],
                                 weight_decay=recipe["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    patch, batch = 128, 2
    image = torch.linspace(-1., 1., batch * patch ** 3, device="cuda").view(batch, 1, patch, patch, patch)
    fixture = physical_fixture(modules["geometry"], modules["gdf"], manifest["fixed_train_field_scale"])
    arrays = {name:torch.as_tensor(value,device="cuda") for name,value in fixture["arrays"].items()}
    watched_groups = {
        "segmentation_head": "anatomical_output_fusion.3.",
        "scale_head": "classifier.generation_head.",
        "structure_head": "classifier.structure_head.",
        "gdf_head": "gdf_head.",
        "shared_segmentation_feature": "anatomical_output_fusion.0.",
        "shared_attribute_encoder": "classifier.encoder.",
        "interaction_weight": "interaction_weight", "interaction_bias": "interaction_bias"}
    parameters = dict(model.named_parameters())
    routing = gdf_routing(model, image, arrays, modules, watched_groups, enabled=not args.skip_gdf_routing)
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
            interaction = model.last_interaction_trace
            require(torch.equal(interaction["consumed_c"], interaction["raw_fine"]) and
                    not interaction["consumed_c"].requires_grad and not interaction["override_supplied"],
                    "Only normal detached CT raw-fine may be consumed")
            loss, details = modules["learnable_integrated_train"].stage_objective(
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
    return dict(schema="GTA-AVT-learnable-integrated-joint-QA-v1", status="passed",
        scope="synthetic_learnable_integrated_joint_objective_AMP_one_update",
        study_data_read=False, TEST_files_read=False, formal_training=False, efficacy_established=False,
        successful_optimizer_updates=1, checkpoint_saved=False, model_alpha=1.,
        batch_size=2, patch_size=128, precision="CUDA_AMP", normal_CT_only_conditions=True,
        warmstart_sha256=weight_sha, initialization=initialization, gdf_weight=.1,
        physical_synthetic_target=True,
        recipe_sha256=sha256(args.recipe), helper_sha256=sha256(__file__), gdf_only_routing=routing,
        warmstart_load=dict(missing_keys=report.missing_keys, unexpected_keys=report.unexpected_keys),
        scientific_source_sha256=source_before, component_details=details, optimizer_update=update,
        gradient_and_parameter_change=step_evidence, overflow_attempts=overflow,
        forward_attempts=forwards, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        runtime=modules["learnable_integrated_train"].runtime_identity(torch.device("cuda")),
        fixture=fixture["audit"],
        limits=["No AVT arrays, no VAL/TEST, no scientific gain or seed stability",
                "Not a formal integrated endpoint; old failed GDF and dynamic gates remain unchanged",
                "No local+GDF claim; SAAM constant; one original-constant origin only"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root")
    parser.add_argument("--manifest", default=str(HERE / "learnable_integrated_candidate_manifest_20261023.json"))
    parser.add_argument("--recipe", default=str(HERE / "learnable_integrated_candidate_UNTRAINED_20261023.json"))
    parser.add_argument("--warmstart")
    parser.add_argument("--output")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-gdf-routing", action="store_true", help="Explicitly omit optional diagnostic VJP; never reported as passed")
    args = parser.parse_args()
    manifest, recipe = read_json(args.manifest), read_json(args.recipe)
    validate_contract(manifest, recipe)
    require(sha256(args.recipe) == manifest["recipe_sha256"], "Recipe byte binding")
    require(sha256(__file__) == manifest["qa_helper_sha256"], "Named joint-QA source binding")
    plan = dict(execute=args.execute, status=manifest["status"], formal_training_authorized=False,
                source_files_pinned=37, study_data_read=False, synthetic_successful_updates=1,
                patch_size=128, batch_size=2, precision="CUDA_AMP", alpha=1.,
                manifest_sha256=sha256(args.manifest), recipe_sha256=sha256(args.recipe),
                helper_sha256=sha256(__file__), candidate_initialization=manifest["initialization_four_digests"],
                physical_synthetic_target=True, optional_GDF_only_routing=not args.skip_gdf_routing)
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
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
