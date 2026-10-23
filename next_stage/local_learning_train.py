"""Matched L0/Lp local-supervision study; metadata plan unless --execute.

The frozen stage objective, scheduler convention and proven AMP retry/optimizer
function are reused. The only added objective is the fixed .1 local soft-CE.
No TEST evaluation or best-checkpoint selection is exposed.
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
for directory in (PROJECT / "code", HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from stage_train import (alpha_at, effective_recipe as parent_recipe,
                         stage_objective, source_inventory, tensor_digest,
                         runtime_identity)


def effective_recipe(config):
    requested = dict(config)
    requested.setdefault("stage", "local_learning_joint")
    requested.setdefault("conditioning", "constant")
    requested.setdefault("alpha_start", 500)
    requested.setdefault("alpha_ramp", 500)
    requested.setdefault("local_weight", .1)
    requested.setdefault("local_seed", 20261023)
    mode = requested["conditioning"]
    if mode not in ("constant", "local_scale"):
        raise ValueError("Only the matched constant/local_scale pair is allowed")
    # Parent's metadata validator does not know the new local_scale name.
    mapped = dict(requested, conditioning="constant")
    recipe = parent_recipe(mapped)
    recipe["conditioning"] = mode
    for key, value in (("local_weight", .1), ("gdf_weight", 0.),
                       ("scale_weight", .1), ("structure_weight", .1),
                       ("lr", .0001), ("q0", .2), ("s0", .2),
                       ("alpha_start", 500), ("alpha_ramp", 500)):
        if recipe[key] != value:
            raise ValueError("The A protocol fixes %s=%s" % (key, value))
    if not isinstance(recipe["local_seed"], int) or isinstance(recipe["local_seed"], bool):
        raise ValueError("local_seed must be an integer")
    if not recipe["detach_conditions"] or recipe["batch_size"] != 2:
        raise ValueError("A fixes detached consumption and batch2")
    if recipe["stage"] == "smoke":
        if recipe["max_steps"] > 8:
            raise ValueError("Smoke is a maximum8-update engineering check")
    elif recipe["max_steps"] != 2000:
        raise ValueError("Formal A uses one fixed2000-update endpoint")
    return recipe


def local_soft_ce(logits, target, valid):
    """Each patch's supported-window mean, followed by the full batch mean.

    A zero-support patch contributes differentiable zero and retains its place
    in the batch denominator. No foreground/class rebalance is introduced.
    """
    import torch
    from torch.nn import functional as F
    if logits.ndim != 5 or logits.shape[1] != 5 or target.shape != logits.shape:
        raise ValueError("Expected matching B,5,D,H,W local logits/targets")
    if valid.ndim == 5 and valid.shape[1] == 1:
        valid = valid[:, 0]
    if valid.shape != logits[:, 0].shape or valid.dtype != torch.bool:
        raise ValueError("local_valid must be B,1,D,H,W or B,D,H,W bool")
    target = target.float()
    if not bool(torch.isfinite(target).all()) or bool((target < 0.).any()):
        raise ValueError("Invalid local soft target")
    if bool(valid.any()) and not bool(torch.allclose(
            target.sum(1)[valid], torch.ones_like(target.sum(1)[valid]), atol=1e-5, rtol=1e-5)):
        raise ValueError("Supported local targets must sum to one")
    values = -(target * F.log_softmax(logits.float(), dim=1)).sum(1)
    weight = valid.float()
    count = weight.flatten(1).sum(1)
    per_patch = (values * weight).flatten(1).sum(1) / count.clamp_min(1.)
    return per_patch.mean(), dict(
        local=float(per_patch.mean().detach()), local_valid_windows=int(count.sum().detach()),
        local_valid_patches=int((count > 0.).sum().detach()),
        local_batch_patches=int(logits.shape[0]))


def objective(output, arrays, recipe):
    total, details = stage_objective(
        output, arrays, recipe["scale_weight"], recipe["structure_weight"],
        recipe["gdf_weight"], recipe["gdf_normalization"])
    local, local_details = local_soft_ce(output["local_logits"], arrays["local_target"],
                                       arrays["local_valid"])
    total = total + recipe["local_weight"] * local
    details.update(local_details)
    details.update(local_weight=recipe["local_weight"], total=float(total.detach()))
    return total, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "manifest", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--gdf-root")
    parser.add_argument("--warmstart")
    parser.add_argument("--warmstart-sha256")
    parser.add_argument("--readout")
    parser.add_argument("--readout-sha256")
    parser.add_argument("--preflight-receipt")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    from common import atomic_json, load_json, sha256, load_trusted_checkpoint
    recipe = effective_recipe(load_json(args.config))
    if not args.execute:
        print(json.dumps(dict(execute=False, recipe=recipe, manifest=args.manifest,
                              output=args.output, warmstart=args.warmstart,
                              common_readout=args.readout, initialization="old_constant_plus_common_readout",
                              selected_step=recipe["max_steps"], selection="fixed_endpoint",
                              allowed_splits=["train", "val"], test_used=False)), flush=True)
        return
    import numpy as np
    import torch
    from local_learning_data import LocalStageData, batch
    from local_learning_model import LocalLearningModel
    from train import optimizer_update_with_amp_retries, set_seed, save_checkpoint
    from local_learning_inference import evaluate_cases
    if not all((args.warmstart, args.warmstart_sha256, args.readout, args.readout_sha256,
                args.gdf_root, args.preflight_receipt)):
        raise ValueError("Execution requires pinned common warmstart/readout, GDF root and preflight")
    if sha256(args.warmstart) != args.warmstart_sha256 or sha256(args.readout) != args.readout_sha256:
        raise ValueError("Initialization byte identity mismatch")
    data = LocalStageData(args.manifest, gdf_root=args.gdf_root)
    if not data.manifest.get("complete"):
        raise ValueError("A complete frozen cache is required")
    splits = data.manifest["splits"]
    train_ids, val_ids = list(splits["train"]), list(splits["val"])
    test_ids = set(splits.get("test", []))
    if (len(train_ids) != 36 or not val_ids or len(set(train_ids)) != len(train_ids)
            or len(set(val_ids)) != len(val_ids) or set(train_ids) & set(val_ids)
            or (set(train_ids) | set(val_ids)) & test_ids):
        raise ValueError("Invalid frozen TRAIN/VAL split")
    run = Path(args.output).resolve()
    if args.resume:
        if not (run / "config.json").is_file() or (run / "complete.json").exists():
            raise ValueError("Resume requires an incomplete owned run")
    else:
        if run.exists() and any(run.iterdir()):
            raise ValueError("Run is immutable; use a new empty output")
        run.mkdir(parents=True, exist_ok=True)
    source = source_inventory()
    gdf_root = Path(args.gdf_root).resolve()
    gdf_metadata = {p.name: sha256(p) for p in sorted(gdf_root.glob("*.json"))}
    if not gdf_metadata:
        raise ValueError("Missing frozen GDF metadata")
    signature = dict(
        schema="GTA-AVT-matched-local-learning-training-v1", recipe=recipe,
        manifest_sha256=sha256(args.manifest), source_sha256=source,
        train_ids=train_ids, validation_ids=val_ids, allowed_splits=["train", "val"],
        gdf_metadata_sha256=gdf_metadata,
        preflight_receipt_sha256=sha256(args.preflight_receipt),
        warmstart_sha256=args.warmstart_sha256, readout_sha256=args.readout_sha256,
        actual_cache_bytes_rehashed_by_trainer=False,
        initialization="old_constant_plus_common_readout_optimizer_reset",
        selected_step=recipe["max_steps"], selection="fixed_endpoint", test_used=False,
        interpretation="Local auxiliary learning is shared; only detached condition consumption differs")
    if args.resume:
        if load_json(run / "config.json") != signature:
            raise ValueError("Resume source/protocol/data changed")
    else:
        atomic_json(run / "config.json", signature)
        shutil.copy2(args.config, run / "requested_config.json")
        for relative in source:
            target = run / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PROJECT / relative, target)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    torch.set_num_threads(recipe["cpu_threads"])
    set_seed(recipe["seed"])
    torch.backends.cuda.matmul.allow_tf32 = bool(recipe["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(recipe["allow_tf32"])
    runtime = runtime_identity(device)
    if args.resume:
        if load_json(run / "runtime.json") != runtime:
            raise ValueError("Runtime/GPU origin changed")
    else:
        atomic_json(run / "runtime.json", runtime)
        atomic_json(run / "process.json", dict(
            pid=os.getpid(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            hostname=platform.node(), command=sys.argv))
    model = LocalLearningModel(recipe["conditioning"], alpha=0., q0=recipe["q0"],
                               s0=recipe["s0"], gdf_seed=recipe["gdf_seed"],
                               local_seed=recipe["local_seed"])
    if not args.resume:
        loaded = load_trusted_checkpoint(args.warmstart, "cpu")
        state = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
        if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
            raise ValueError("Warmstart must contain a tensor state_dict")
        common_load = model.load_common_state(state)
        readout = load_trusted_checkpoint(args.readout, "cpu")
        model.load_readout_state(readout)
        initial = model.state_dict()
        atomic_json(run / "initialization.json", dict(
            mode=signature["initialization"], source_sha256=args.warmstart_sha256,
            readout_sha256=args.readout_sha256, state_tensor_digest=tensor_digest(initial),
            legacy_tensor_digest=tensor_digest({k: v for k, v in initial.items()
                                               if not k.startswith(("gdf_head.", "local_head."))}),
            local_head_digest=tensor_digest({k: v for k, v in initial.items() if k.startswith("local_head.")}),
            gdf_head_digest=tensor_digest({k: v for k, v in initial.items() if k.startswith("gdf_head.")}),
            missing_keys=sorted(common_load.missing_keys), optimizer_reset=True))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    start, elapsed, last_validation = 0, 0., None
    if args.resume:
        checkpoint = load_trusted_checkpoint(run / "latest.pt", device)
        if checkpoint["signature"] != signature:
            raise ValueError("Checkpoint provenance differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, elapsed = checkpoint["step"], checkpoint["training_seconds"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint["cuda_rng"]])
    for step in range(start, recipe["max_steps"]):
        started = time.time()
        model.train()
        alpha = alpha_at(step, recipe["alpha_start"], recipe["alpha_ramp"])
        model.set_alpha(alpha)
        sampler_step = recipe["sampler_step_offset"] + step
        arrays = batch(data, recipe["seed"], sampler_step, recipe["batch_size"],
                       recipe["patch_size"], recipe["foreground_probability"])
        required = ("image", "mask", "scale", "structure", "attribute_valid", "coordinate",
                    "local_target", "local_valid", "input_valid", "gdf_target", "gdf_valid")
        if any(key not in arrays for key in required):
            raise ValueError("LocalStageData batch missing required keys")
        tensors = {key: torch.as_tensor(arrays[key], device=device) for key in required}
        lr = recipe["lr"] * max(0., 1. - step / recipe["max_steps"]) ** .9
        for group in optimizer.param_groups:
            group["lr"] = lr

        def forward_loss():
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = model(tensors["image"], tensors["coordinate"], input_valid=tensors["input_valid"])
                return objective(output, tensors, recipe)

        def on_overflow(record):
            with open(run / "amp_overflow.jsonl", "a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(step=step + 1, **record), allow_nan=False) + "\n")

        details, update = optimizer_update_with_amp_retries(
            model, optimizer, scaler, forward_loss,
            gradient_clip=recipe["gradient_clip"], max_attempts=8, on_overflow=on_overflow)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.time() - started
        elapsed += seconds
        log = dict(step=step + 1, sampler_step=sampler_step, alpha=alpha, lr=lr,
                   seconds=seconds, training_seconds=elapsed, **details, **update)
        with open(run / "training.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(log, allow_nan=False) + "\n")
        if step == start or (step + 1) % recipe["log_every"] == 0:
            atomic_json(run / "status.json", dict(log, state="training", role=recipe["role"], test_used=False))
            print(json.dumps(log, allow_nan=False), flush=True)
        if (step + 1) % 500 == 0 or step + 1 == recipe["max_steps"]:
            save_checkpoint(run / "latest.pt", dict(
                step=step + 1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(), training_seconds=elapsed, signature=signature,
                python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []))
    model.set_alpha(alpha_at(recipe["max_steps"], recipe["alpha_start"], recipe["alpha_ramp"]))
    if not recipe["skip_validation"]:
        model.eval()
        _, last_validation = evaluate_cases(
            model, data, val_ids, device, run / "validation" / ("step_%06d" % recipe["max_steps"]),
            patch=recipe["inference_patch_size"], batch_size=recipe["inference_batch_size"],
            overlap=recipe["inference_overlap"], save_predictions=recipe["save_validation_predictions"],
            condition_mode="predicted", reuse=args.resume)
    complete = dict(
        schema="GTA-AVT-matched-local-learning-complete-v1", state="complete", role=recipe["role"],
        steps=recipe["max_steps"], selected_step=recipe["max_steps"], selection="fixed_endpoint",
        final_alpha=model.alpha, training_seconds=elapsed, checkpoint_file="latest.pt",
        checkpoint_sha256=sha256(run / "latest.pt"), last_validation=last_validation,
        test_used=False, selection_valid=not recipe["skip_validation"], initialization=signature["initialization"])
    atomic_json(run / "complete.json", complete)
    atomic_json(run / "status.json", complete)
    print(json.dumps(complete, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
