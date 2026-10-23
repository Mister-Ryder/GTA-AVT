"""Isolated learnable interaction plus physical GDF auxiliary candidate.

Transparent derivative of the frozen GDF0 learnable trainer. The named recipe
adds only fixed voxel-normalized GDF weight .1. The current candidate is stopped
and untrained; a plan is inert, and execution needs a separately authorized,
source-bound FULL preflight. No old GDF0 receipt authorizes this candidate.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
CODE = PROJECT / "code"
for directory in (CODE, HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


OPERATOR_VARIANT = "learnable_signed_interaction_v1"
CANDIDATE_FILENAME = "learnable_integrated_candidate_manifest_20261023.json"
ORIGINAL_WARMSTART_SHA256 = "7406a3ba7e5bac1a7d8ee24898cb954ff8576f37fcf5fba0e26ef9a2cb5c43e6"


def alpha_at(step, start, ramp):
    """step is zero-based successful-update index, shared by all arms."""
    if step < start:
        return 0.
    if ramp <= 1:
        return 1.
    return min(1., max(0., (step - start) / float(ramp - 1)))


def source_inventory():
    from common import sha256
    files = []
    for root in (CODE, HERE):
        files.extend(p for p in root.rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts
                     and (p.suffix == ".py" or ("historical" in p.parts
                                               and p.suffix.lower() in (".txt", ".json", ".md"))))
    return {p.relative_to(PROJECT).as_posix(): sha256(p) for p in sorted(files)}


def tensor_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str((str(value.dtype), tuple(value.shape))).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def runtime_identity(device):
    import torch
    packages = {}
    for name in ("torch", "numpy", "scipy", "scikit-image", "SimpleITK"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu_rows = None
    if device.type == "cuda":
        gpu_rows = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader"], text=True, timeout=20).strip().splitlines()
    return dict(
        hostname=platform.node(), platform=platform.platform(),
        python=sys.version, executable=sys.executable, packages=packages,
        torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        selected_device=str(device), selected_device_name=(
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"),
        visible_gpu_inventory=gpu_rows,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        backend=dict(cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                     cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                     deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
                     deterministic_warn_only=getattr(
                         torch, "is_deterministic_algorithms_warn_only_enabled", lambda: None)(),
                     matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                     cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
                     cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG")))


def stage_objective(output, arrays, scale_weight=.1, structure_weight=.1,
                    gdf_weight=0., gdf_normalization="voxel"):
    import torch
    from torch.nn import functional as F
    from losses import dice_bce
    segmentation = dice_bce(output["logits"], arrays["mask"])
    for weight, logits in zip((.5, .25), output["aux_logits"]):
        expanded = F.interpolate(logits, size=arrays["mask"].shape[2:],
                                 mode="trilinear", align_corners=False)
        segmentation = segmentation + weight * dice_bce(expanded, arrays["mask"])
    total = segmentation
    details = {"segmentation": float(segmentation.detach())}
    for name, weight in (("scale", scale_weight), ("structure", structure_weight)):
        target = arrays[name].float()
        values = -(target * F.log_softmax(output[name + "_logits"].float(), 1)).sum(1)
        valid = arrays.get("attribute_valid", torch.ones_like(values)).float()
        valid = valid * (target.sum(1) > 0.).to(valid.dtype)
        loss = (values * valid).sum() / valid.sum().clamp_min(1.)
        # Compute both head diagnostics in every arm, including zero-weight U.
        total = total + float(weight) * loss
        details[name] = float(loss.detach())
        details[name + "_valid_cases"] = int((valid > 0.).sum().detach())
        details[name + "_weight"] = float(weight)
    have_gdf = "gdf_target" in arrays and "gdf_valid" in arrays
    if gdf_weight > 0. and not have_gdf:
        raise ValueError("Positive GDF weight requires target and validity")
    if have_gdf:
        from gdf import gdf_mse
        field, field_details = gdf_mse(
            output["gdf_prediction"], arrays["gdf_target"], arrays["gdf_valid"],
            foreground=arrays["mask"], normalization=gdf_normalization)
        total = total + float(gdf_weight) * field
        details.update(field_details)
    details.update(total=float(total.detach()), gdf_weight=float(gdf_weight))
    return total, details


def effective_recipe(config):
    values = dict(config)
    defaults = dict(
        stage="learnable_integrated_stopped_candidate", role="learnable_integrated_UNTRAINED", conditioning="scale_only",
        operator_variant=OPERATOR_VARIANT,
        detach_conditions=True, q0=.2, s0=.2, gdf_seed=20261023,
        seed=20261023, max_steps=2000, batch_size=2, patch_size=128,
        foreground_probability=.5, lr=.0001, weight_decay=.0001,
        gradient_clip=12., scale_weight=.1, structure_weight=.1,
        gdf_weight=.1, gdf_normalization="voxel", alpha_start=0, alpha_ramp=500,
        checkpoint_every=500, log_every=100, cpu_threads=4,
        inference_patch_size=128, inference_batch_size=2, inference_overlap=.5,
        save_validation_predictions=True, skip_validation=False, allow_tf32=False)
    for key, value in defaults.items():
        values.setdefault(key, value)
    # Accept the frozen recipe's common aux key without silently ignoring U=0.
    if "auxiliary_weight" in config:
        if "scale_weight" not in config:
            values["scale_weight"] = config["auxiliary_weight"]
        if "structure_weight" not in config:
            values["structure_weight"] = config["auxiliary_weight"]
    values.setdefault("sampler_step_offset", 0)
    for name in ("lr", "weight_decay", "scale_weight", "structure_weight", "gdf_weight"):
        if not math.isfinite(float(values[name])) or float(values[name]) < 0.:
            raise ValueError("Invalid nonnegative recipe value: " + name)
    if values["lr"] <= 0. or values["max_steps"] <= 0:
        raise ValueError("A new positive learning-rate/update budget is required")
    for name in ("max_steps", "seed", "gdf_seed", "batch_size", "log_every",
                 "alpha_start", "alpha_ramp", "sampler_step_offset"):
        if not isinstance(values[name], int) or isinstance(values[name], bool):
            raise ValueError("Integer recipe value required: " + name)
    if values["batch_size"] <= 0 or values["log_every"] <= 0 or values["gradient_clip"] <= 0:
        raise ValueError("Batch size, log interval and gradient clip must be positive")
    if values["sampler_step_offset"] < 0 or not isinstance(values["detach_conditions"], bool):
        raise ValueError("Invalid sampler offset/detach flag")
    if values["patch_size"] != 128 or values["inference_patch_size"] != 128:
        raise ValueError("The next-stage study fixes train and inference patches at128")
    if values["checkpoint_every"] != 500:
        raise ValueError("The next-stage checkpoint interval is fixed at500")
    if values["skip_validation"] and values["stage"] != "smoke":
        raise ValueError("Only explicit smoke runs may omit VAL")
    if values["conditioning"] not in ("constant", "scale_only"):
        raise ValueError("Unknown conditioning")
    if values["gdf_normalization"] not in ("voxel", "balanced"):
        raise ValueError("Unknown field normalization")
    if values["alpha_start"] < 0 or values["alpha_ramp"] < 0:
        raise ValueError("Invalid stage schedule")
    if values["stage"] != "smoke" and values["conditioning"] == "scale_only" and (
            values["alpha_start"] + values["alpha_ramp"] > values["max_steps"]):
        raise ValueError("The fixed endpoint must finish condition introduction")
    # This is one named new operator, not a generic search or an old-stage resume.
    fixed = dict(operator_variant=OPERATOR_VARIANT, seed=20261023, gdf_seed=20261023,
                 batch_size=2, patch_size=128, inference_patch_size=128,
                 inference_batch_size=2, inference_overlap=.5,
                 lr=.0001, weight_decay=.0001, gradient_clip=12.,
                 foreground_probability=.5, scale_weight=.1, structure_weight=.1,
                 gdf_weight=.1, gdf_normalization="voxel", detach_conditions=True,
                 q0=.2, s0=.2, alpha_start=0, alpha_ramp=500,
                 sampler_step_offset=0, allow_tf32=False)
    fixed["conditioning"] = "scale_only"
    for key, expected in fixed.items():
        if values[key] != expected:
            raise ValueError("The matched signed-interaction protocol fixes %s=%s" % (key, expected))
    if values["stage"] == "smoke":
        if values["max_steps"] > 8:
            raise ValueError("Smoke is at most8 successful engineering updates")
    elif values["max_steps"] != 2000:
        raise ValueError("The signed-interaction study fixes the2000-update endpoint")
    return values


def _require(value, message):
    if not value:
        raise ValueError(message)


def _bound_json(item):
    path = Path(item['path'])
    _require(path.is_file() and not path.is_symlink() and
             hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], 'Full-preflight reference bytes differ')
    return json.loads(path.read_text(encoding='utf-8'))


def validate_candidate_manifest(path, recipe_path, recipe):
    candidate = json.loads(Path(path).read_text(encoding='utf-8'))
    _require(candidate.get('schema') == 'GTA-AVT-learnable-integrated-candidate-v1' and
             candidate.get('operator_variant') == OPERATOR_VARIANT and
             candidate.get('formal_training_authorized') in (False, True), 'Wrong named integrated candidate')
    _require(candidate['status'] == ('FORMAL_AUTHORIZED_UNTRAINED' if candidate['formal_training_authorized'] else
             'STOPPED_UNTRAINED_NOT_LAUNCH_AUTHORIZED'), 'Candidate authorization/status inconsistent')
    _require(candidate['recipe_sha256'] == hashlib.sha256(Path(recipe_path).read_bytes()).hexdigest() and
             candidate['recipe'] == recipe and candidate['warmstart_sha256'] == ORIGINAL_WARMSTART_SHA256 and
             candidate['fixed_train_field_scale'] == 2.8707158419291448 and
             candidate['gdf_target_manifest_sha256'] == '18820f0b0cfdc2316aa2563bdad44a6209fb8cd35d1d9c4d04a4c3943e1fb1e9',
             'Named integrated recipe/origin/physical field identity differs')
    source = candidate['scientific_source_sha256']
    _require(len(source) == 37 and 'next_stage/learnable_integrated_train.py' in source and
             candidate['original_learnable36_source_sha256'] == {k: v for k, v in source.items()
                 if k != 'next_stage/learnable_integrated_train.py'}, 'Expected frozen36 plus isolated integrated trainer')
    return candidate


def validate_full_preflight(args, candidate, recipe, manifest):
    """Verify genuine, bound QA/input/GDF evidence; never accept old GDF0 QA.

    The receipt is produced only after actual jobs. Its status alone is not
    evidence. Actual result/execution files and the current source/data bytes
    are cross-bound here. Historical path relocation is permitted only through
    identical per-case mask/geometry identities, never by rewriting the cache.
    """
    _require(args.preflight_receipt, 'Integrated FULL preflight is mandatory')
    receipt = json.loads(Path(args.preflight_receipt).read_text(encoding='utf-8'))
    _require(receipt.get('schema') == 'GTA-AVT-learnable-integrated-full-preflight-v1' and
             receipt.get('status') == 'passed' and receipt.get('test_used') is False,
             'A GDF0/old-global/engineering-only receipt is not an integrated FULL preflight')
    source = source_inventory()
    from common import sha256
    _require(source == candidate['scientific_source_sha256'] == receipt['source_sha256'] and
             receipt['candidate_manifest_sha256'] == sha256(args.candidate_manifest) and
             receipt['recipe_sha256'] == sha256(args.config) and receipt['manifest_sha256'] == sha256(args.manifest) and
             receipt['warmstart_sha256'] == ORIGINAL_WARMSTART_SHA256, 'FULL source/recipe/input/origin binding differs')
    qa, qa_execution = _bound_json(receipt['joint_qa_result']), _bound_json(receipt['joint_qa_execution'])
    _require(qa_execution.get('exit_code') == 0 and json.loads(qa_execution['stdout']) == qa and
             qa.get('schema') == 'GTA-AVT-learnable-integrated-joint-QA-v1' and qa.get('status') == 'passed' and
             qa.get('scientific_source_sha256') == source and qa.get('recipe_sha256') == receipt['recipe_sha256'] and
             qa.get('helper_sha256') == candidate['qa_helper_sha256'] and
             qa.get('initialization') == candidate['initialization_four_digests'] and
             qa.get('warmstart_sha256') == ORIGINAL_WARMSTART_SHA256 and qa.get('gdf_weight') == .1 and
             qa.get('successful_optimizer_updates') == 1 and qa.get('patch_size') == 128 and qa.get('batch_size') == 2 and
             qa.get('precision') == 'CUDA_AMP' and qa.get('study_data_read') is False and
             qa.get('TEST_files_read') is False and qa.get('normal_CT_only_conditions') is True and
             qa.get('physical_synthetic_target') is True and len(qa.get('gradient_and_parameter_change', {})) == 8 and
             all(v['finite'] and v['gradient_nonzero_elements'] > 0 and v['changed_elements'] > 0
                 for v in qa['gradient_and_parameter_change'].values()), 'Actual NEW integrated joint QA is incomplete')
    import torch
    _require(args.device == 'cuda' and qa['runtime'] == runtime_identity(torch.device(args.device)),
             'Actual integrated QA/training runtime differs')
    audit, audit_execution = _bound_json(receipt['data_audit_result']), _bound_json(receipt['data_audit_execution'])
    _require(audit_execution.get('exit_code') == 0 and
             json.loads(audit_execution['stdout'])['data_audit_sha256'] == receipt['data_audit_result']['sha256'] and
             audit.get('schema') == 'GTA-AVT-learnable-data-audit-v1' and audit.get('status') == 'passed' and
             audit['manifest_sha256'] == receipt['manifest_sha256'] and
             audit['warmstart_sha256'] == ORIGINAL_WARMSTART_SHA256 and
             audit['source_sha256'] in (source, candidate['original_learnable36_source_sha256']) and
             audit['train_ids'] == manifest['splits']['train'] and audit['validation_ids'] == manifest['splits']['val'] and
             audit['TEST_arrays_opened'] is False and audit['full_cache_files_rehashed'] == 124 and
             audit['native_validation_files_rehashed'] == 16, 'Actual complete TRAIN/VAL input audit differs')
    cases = {r['id']: r for r in manifest['cases']}
    expected = {(cid, name): cases[cid]['cache_file_sha256'][name]
                for cid in manifest['splits']['train'] for name in ('image.npy', 'mask.npy', 'geometry.npz')}
    expected.update({(cid, name): cases[cid]['cache_file_sha256'][name]
                     for cid in manifest['splits']['val'] for name in ('image.npy', 'mask.npy')})
    _require({(r['case_id'], r['file']): r['sha256'] for r in audit['cache_files']} == expected and
             len(audit['cache_files']) == 124, 'TRAIN108/VAL16 audited cache coverage differs')
    native = {(r['case_id'], r['kind']): r for r in audit['native_validation_files']}
    _require(len(native) == len(audit['native_validation_files']) == 16 and
             set(native) == {(cid, name) for cid in manifest['splits']['val'] for name in ('image', 'label')},
             'Native VAL16 audit coverage differs')
    for row in audit['cache_files']:
        path = Path(cases[row['case_id']]['cache_dir']) / row['file']
        _require(path.resolve() == Path(row['path']).resolve() and sha256(path) == row['sha256'],
                 'Current trainer cache bytes differ from the actual input audit')
    for (cid, name), row in native.items():
        _require(row['sha256'] == cases[cid][name + '_sha256'] and
                 Path(row['path']).resolve() == Path(cases[cid][name]).resolve() and sha256(row['path']) == row['sha256'],
                 'Current native VAL bytes differ from actual audit')
    gdf, gdf_execution = _bound_json(receipt['gdf_cache_QA_result']), _bound_json(receipt['gdf_cache_QA_execution'])
    _require(gdf_execution.get('observed_QA_exit_code') == 0 and
             gdf_execution['report_sha256'] == receipt['gdf_cache_QA_result']['sha256'] and
             gdf.get('schema') == 'GTA-AVT-GDF-independent-cache-QA-v1' and gdf.get('status') == 'passed' and
             gdf.get('complete') is True and gdf['training_case_ids'] == manifest['splits']['train'] and
             gdf['target_files_fully_rehashed'] == 36 and gdf['input_files_fully_rehashed'] == 72 and
             gdf['heldout_case_files_opened'] is False and gdf['model_forward_or_training_run'] is False and
             gdf['fixed_train_scale'] == candidate['fixed_train_field_scale'], 'Actual full TRAIN GDF QA differs')
    target_root = Path(args.gdf_root)
    _require(sha256(target_root / 'manifest.json') == gdf['target_manifest_sha256'] ==
             candidate['gdf_target_manifest_sha256'] and
             sha256(target_root / 'probe_storage6.json') == gdf['probe_sha256'], 'Frozen physical target metadata differs')
    target_manifest = json.loads((target_root / 'manifest.json').read_text(encoding='utf-8'))
    targets = {r['id']: r for r in target_manifest['cases']}
    _require(set(targets) == set(manifest['splits']['train']) and len(targets) == 36 and target_manifest['complete'],
             'Exactly TRAIN36 physical fields required')
    reports = {r['id']: r for r in gdf['cases']}
    _require(set(reports) == set(targets) and len(gdf['cases']) == 36, 'GDF QA case coverage differs')
    for cid, row in targets.items():
        name = Path(row['target_file'])
        _require(not name.is_absolute() and '..' not in name.parts and
                 sha256(target_root / name) == row['target_file_sha256'] == reports[cid]['target_file_sha256'],
                 'Current GDF full target SHA differs')
        _require(reports[cid]['actual_input_file_sha256'] ==
                 {name: cases[cid]['cache_file_sha256'][name] for name in ('mask.npy', 'geometry.npz')},
                 'Physical field and current TRAIN geometry/mask identities differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-manifest", default=str(HERE / CANDIDATE_FILENAME))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gdf-root")
    parser.add_argument("--warmstart", help="Owned CPU tensor bundle or owned checkpoint")
    parser.add_argument("--warmstart-sha256")
    parser.add_argument("--preflight-receipt")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    from common import atomic_json, load_json, sha256, load_trusted_checkpoint
    recipe = effective_recipe(load_json(args.config))
    candidate = validate_candidate_manifest(args.candidate_manifest, args.config, recipe)
    if not args.execute:
        print(json.dumps(dict(execute=False, recipe=recipe, manifest=args.manifest,
                              output=args.output, initialization="required_original_constant_warmstart_optimizer_reset",
                              required_warmstart_sha256=ORIGINAL_WARMSTART_SHA256,
                              selected_step=recipe["max_steps"], selection="fixed_endpoint",
                              candidate_status=candidate["status"], formal_training_authorized=candidate["formal_training_authorized"],
                              required_full_preflight_schema="GTA-AVT-learnable-integrated-full-preflight-v1",
                              allowed_splits=["train", "val"], test_used=False)), flush=True)
        return
    if candidate["formal_training_authorized"] is not True:
        raise ValueError("STOPPED_UNTRAINED: candidate preparation/QA is not formal-training authorization")
    import numpy as np
    import torch
    from next_data import StageData, batch
    from learnable_stage_model import LearnableStageModel as StageModel
    from train import optimizer_update_with_amp_retries, set_seed, save_checkpoint
    from inference import evaluate_cases
    if Path(sys.modules[StageModel.__module__].__file__).resolve() != (HERE / "learnable_stage_model.py").resolve():
        raise ValueError("The loaded candidate class must come from this frozen source snapshot")
    if not args.warmstart or args.warmstart_sha256 != ORIGINAL_WARMSTART_SHA256:
        raise ValueError("Both arms require the named original constant checkpoint; not Gp/Kp continuation")
    if not args.gdf_root:
        raise ValueError("Named integrated candidate requires its frozen physical TRAIN GDF root")
    if bool(args.warmstart) != bool(args.warmstart_sha256):
        raise ValueError("Warmstart file and explicit SHA256 must be supplied together")
    if recipe["gdf_weight"] > 0. and not args.gdf_root:
        raise ValueError("Positive GDF weight requires an explicit immutable target root")
    if args.warmstart and sha256(args.warmstart) != args.warmstart_sha256:
        raise ValueError("Warmstart byte identity mismatch")
    data = StageData(args.manifest, gdf_root=args.gdf_root)
    if not data.manifest.get("complete"):
        raise ValueError("Only a complete frozen cache is allowed")
    splits = data.manifest["splits"]
    train_ids, val_ids = list(splits["train"]), list(splits["val"])
    test_ids = set(splits.get("test", []))
    if (not train_ids or not val_ids or len(set(train_ids)) != len(train_ids)
            or len(set(val_ids)) != len(val_ids)):
        raise ValueError("Invalid TRAIN/VAL case lists")
    if set(train_ids) & set(val_ids) or (set(train_ids) | set(val_ids)) & test_ids:
        raise ValueError("Split overlap")
    torch.set_num_threads(recipe["cpu_threads"])
    set_seed(recipe["seed"])
    torch.backends.cuda.matmul.allow_tf32 = bool(recipe["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(recipe["allow_tf32"])
    validate_full_preflight(args, candidate, recipe, data.manifest)
    run = Path(args.output).resolve()
    if args.resume:
        if not (run / "config.json").is_file() or (run / "complete.json").exists():
            raise ValueError("Resume requires a partial owned run")
    else:
        if run.exists() and any(run.iterdir()):
            raise ValueError("Run is immutable; choose a new empty output")
        run.mkdir(parents=True, exist_ok=True)
    source = source_inventory()
    gdf_metadata = {}
    if args.gdf_root:
        root = Path(args.gdf_root).resolve()
        gdf_metadata = {str(p.relative_to(root)): sha256(p) for p in sorted(root.glob("*.json"))}
        if not gdf_metadata:
            raise ValueError("GDF root lacks immutable metadata JSON")
    signature = dict(
        schema="GTA-AVT-learnable-integrated-training-v1",
        candidate_manifest_sha256=sha256(args.candidate_manifest),
        operator_variant=OPERATOR_VARIANT, recipe=recipe,
        manifest_sha256=sha256(args.manifest), source_sha256=source,
        train_ids=train_ids, validation_ids=val_ids, allowed_splits=["train", "val"],
        gdf_metadata_sha256=gdf_metadata,
        declared_canonical_data_sha256=recipe.get("canonical_data_sha256"),
        actual_cache_bytes_rehashed_by_trainer=False,
        initialization=("warmstart_optimizer_reset" if args.warmstart else "random"),
        warmstart_sha256=args.warmstart_sha256,
        selected_step=recipe["max_steps"], selection="fixed_endpoint",
        test_used=False,
        interpretation=("Warm-start cannot remove inherited attribute supervision"
                        if args.warmstart else "Paired random initialization; no airway pretraining"))
    if args.preflight_receipt:
        signature["preflight_receipt_sha256"] = sha256(args.preflight_receipt)
    if args.resume:
        previous = load_json(run / "config.json")
        if previous != signature:
            raise ValueError("Resume source, recipe or data identity changed")
    else:
        atomic_json(run / "config.json", signature)
        shutil.copy2(args.config, run / "requested_config.json")
        for relative in source:
            target = run / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PROJECT / relative, target)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback prohibited")
    runtime = runtime_identity(device)
    if args.resume:
        if load_json(run / "runtime.json") != runtime:
            raise ValueError("Training runtime/GPU origin changed on resume")
    else:
        atomic_json(run / "runtime.json", runtime)
        atomic_json(run / "process.json", dict(
            pid=os.getpid(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            hostname=platform.node(), command=sys.argv))
    model = StageModel(recipe["conditioning"], recipe["detach_conditions"],
                       alpha=0., q0=recipe["q0"], s0=recipe["s0"],
                       gdf_seed=recipe["gdf_seed"])
    if tuple(model.interaction_weight.shape) != (128,) or tuple(model.interaction_bias.shape) != (128,):
        raise ValueError("The fixed original GAAM site requires two128-channel interaction vectors")
    interaction_scalar_count = int(model.interaction_weight.numel() + model.interaction_bias.numel())
    load_report = None
    if args.warmstart and not args.resume:
        loaded = load_trusted_checkpoint(args.warmstart, "cpu")
        state = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
        if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
            raise ValueError("Warmstart must contain a tensor state_dict")
        loaded_report = model.load_common_state(state)
        load_report = dict(missing_keys=sorted(loaded_report.missing_keys),
                           unexpected_keys=sorted(loaded_report.unexpected_keys))
        if bool(model.interaction_weight.detach().count_nonzero()) or bool(model.interaction_bias.detach().count_nonzero()):
            raise ValueError("The new interaction scalars must start at exact zero")
        del loaded, state
    if not args.resume:
        initial_state = model.state_dict()
        atomic_json(run / "initialization.json", dict(
            mode=signature["initialization"], source_sha256=args.warmstart_sha256,
            state_tensor_digest=tensor_digest(initial_state),
            legacy_tensor_digest=tensor_digest({k: v for k, v in initial_state.items()
                                               if not k.startswith("gdf_head.") and k not in ("interaction_weight", "interaction_bias")}),
            gdf_head_digest=tensor_digest({k: v for k, v in initial_state.items()
                                          if k.startswith("gdf_head.")}),
            interaction_digest=tensor_digest({k: v for k, v in initial_state.items()
                                             if k in ("interaction_weight", "interaction_bias")}),
            operator_variant=OPERATOR_VARIANT, new_interaction_scalar_count=interaction_scalar_count,
            warmstart_load=load_report, optimizer_reset=True))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"],
                                 weight_decay=recipe["weight_decay"])
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
        required = ("image", "mask", "scale", "structure", "attribute_valid", "coordinate")
        if any(key not in arrays for key in required):
            raise ValueError("Stage batch missing required keys")
        tensor_keys = required + tuple(k for k in ("gdf_target", "gdf_valid") if k in arrays)
        tensors = {key: torch.as_tensor(arrays[key], device=device) for key in tensor_keys}
        lr = recipe["lr"] * max(0., 1. - step / recipe["max_steps"]) ** .9
        for group in optimizer.param_groups:
            group["lr"] = lr

        def forward_loss():
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = model(tensors["image"], tensors["coordinate"])
                total, details = stage_objective(
                    output, tensors, recipe["scale_weight"], recipe["structure_weight"],
                    recipe["gdf_weight"], recipe["gdf_normalization"])
                # Detached scalar diagnostics of THIS successful-forward attempt;
                # no full feature/logit tensor or GPU autograd graph enters logs.
                details.update(model.interaction_diagnostics())
                return total, details

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
        with torch.no_grad():
            parameters_after_update = {
                "interaction_weight_after_update_l2": float(model.interaction_weight.detach().norm()),
                "interaction_weight_after_update_absmax": float(model.interaction_weight.detach().abs().max()),
                "interaction_bias_after_update_l2": float(model.interaction_bias.detach().norm()),
                "interaction_bias_after_update_absmax": float(model.interaction_bias.detach().abs().max())}
        log = dict(step=step + 1, sampler_step=sampler_step, alpha=alpha, lr=lr,
                   seconds=seconds, training_seconds=elapsed, operator_variant=OPERATOR_VARIANT,
                   **details, **update, **parameters_after_update)
        with open(run / "training.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(log, allow_nan=False) + "\n")
        if step == start or (step + 1) % recipe["log_every"] == 0:
            atomic_json(run / "status.json", dict(
                log, state="training", role=recipe["role"], test_used=False))
            print(json.dumps(log, allow_nan=False), flush=True)
        if (step + 1) % 500 == 0 or step + 1 == recipe["max_steps"]:
            checkpoint = dict(
                step=step + 1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(), training_seconds=elapsed, signature=signature,
                python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
            save_checkpoint(run / "latest.pt", checkpoint)
    # No best-checkpoint selection. The final update is the sole evaluation.
    model.set_alpha(alpha_at(recipe["max_steps"], recipe["alpha_start"], recipe["alpha_ramp"]))
    if not recipe["skip_validation"]:
        model.eval()
        _, last_validation = evaluate_cases(
            model, data, val_ids, device, run / "validation" / ("step_%06d" % recipe["max_steps"]),
            patch=recipe["inference_patch_size"], batch_size=recipe["inference_batch_size"],
            overlap=recipe["inference_overlap"],
            save_predictions=recipe["save_validation_predictions"], condition_mode="predicted",
            reuse=args.resume)
    complete = dict(
        schema="GTA-AVT-learnable-integrated-complete-v1", state="complete",
        operator_variant=OPERATOR_VARIANT, new_interaction_scalar_count=interaction_scalar_count,
        role=recipe["role"], steps=recipe["max_steps"], selected_step=recipe["max_steps"],
        selection="fixed_endpoint", final_alpha=model.alpha, training_seconds=elapsed,
        checkpoint_file="latest.pt", checkpoint_sha256=sha256(run / "latest.pt"),
        last_validation=last_validation, test_used=False,
        selection_valid=not recipe["skip_validation"], initialization=signature["initialization"])
    atomic_json(run / "complete.json", complete)
    atomic_json(run / "status.json", complete)
    print(json.dumps(complete, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
