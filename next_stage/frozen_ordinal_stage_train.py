"""Fresh-only frozen-classifier derivative of the original ordinal trainer.

The parent ordinal trainer retains the public source-hash metadata mapping.
The full classifier stays frozen/eval; the ordinal condition,256 interaction
parameters, losses, sampler, AMP, alpha schedule and fixed2000 endpoint persist.
New protocol, source map, full-size QA and preflight evidence are mandatory.
Default remains an inert metadata plan; no old receipt authorizes execution.
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


OPERATOR_VARIANT = "frozen_classifier_ordinal_cumulative_signed_interaction_v1"
CLASSIFIER_POLICY = "all_classifier_parameters_frozen_eval_v1"
PROTOCOL_SCHEMA = "GTA-AVT-strict-frozen-ordinal-protocol-v1"
APPROVED_TRAINING_SEEDS = (20261024, 20261025, 20261026)
PROJECTION = "mean_three_nonempty_ordered_radius_CDF_masses"
BASE_SOURCE_SHA256 = {'code/common.py': '79edd4ce7623632c602b33124d4653a28d7cb41ee9ae7a0ee4e664a0efba5d5a',
 'code/data.py': '9ad4c237e2b0e7e1fdde00391ebba8dc320b3001cc4ee68c04ca23e64e5f62cf',
 'code/evaluate.py': 'b50ced9813fe14f923af4e9773606c5ca7d275583fffbdd86d1dd7ca5284155d',
 'code/faithful_model.py': 'c02a222dc92b231cbaf5a3768217028ee74567d166e09d2df979c9ebdff72337',
 'code/faithful_smoke.py': '5b83ceb0654234e79a0189530e118d2cedf19e8dc376eb9e7f980e1d6ac56129',
 'code/geometry.py': '456f72da13e17dc877ab0a8af8567c4710cd39aca7141066a4b22b68f65e8ac8',
 'code/historical/NaviAirway-MIT.txt': 'a8d637dcbcd086a0072c4394565c715f88d0327f3b927067c0e596579cfcfd1c',
 'code/historical/THIRD_PARTY_NOTICES.md': '511bb82ea8773e30f58c0ea5531d187708cde071fd17c213824f4bcc75f8be3b',
 'code/historical/__init__.py': '275f71602236a43c1b6288874186ef8ec11077124051b2733d3b95b9dc315ecb',
 'code/historical/func/AnatomicalClassifier.py': 'f5288c74ce71db97eca63f9665e29dfb0099aa010326ce7d6748377bc0156222',
 'code/historical/func/AnatomicalFusionModules.py': '7309281bb296eb7d861050f5ead6b563823b8393addf6ff7c18a594225030b01',
 'code/historical/func/EnhancedAirwaySegmentation2gau.py': '13808eeac5c23d084104d18e4b8425c01ba035b84168d1dc6a69cc6a25de28fc',
 'code/historical/func/__init__.py': '8b611c2073b7bec1691a687085deb8707402d6063f6625dc737f3b62599f24b3',
 'code/historical/func/model_arch_model10.py': '55d51ea850efd2b31aa07acd68ad5e24f64fafe6a714d26dc357919f639f3003',
 'code/historical/provenance.json': '44cc0176c30adb08e83b00a0976183f9d66d6df8ff3e9661f72637b3521430b8',
 'code/historical/source_hashes.json': 'f792cc0ed802ab119ef173870390b2c015d8d2727eee94d499614876a4dbdd27',
 'code/inference.py': 'a30b7511b6272f627bd71c9a62c1b068fc3f1c2d3ff0f5e81bd45b15828f6805',
 'code/intensity.py': 'ccb6a79fe9db66244c370608873b55fa2b059ad743d853ba03422057d0f85a7a',
 'code/losses.py': 'c3eafe2d492444d5b909d0597ca04459b2df812c9c8723597c75928298afc87d',
 'code/metrics.py': 'c729a6f4894758407ab3955889f6db60f8f92a724d883b83c9169a81f2fd96ec',
 'code/model.py': '59f4cdd9823e11bdd94d6c2d0ebd4c68fcd237c656040ad9fef365323af4d9ca',
 'code/prepare.py': '1cf125da4837b100d8c669c7896272c81582f3b0455fb5213af271d6e9fda6e0',
 'code/smoke.py': 'c1c26685249638a0cf56408cf067cffffab495ee92395dd3ac42fc88e63f71ee',
 'code/train.py': 'ec867cd4922a2a18ccf5cb7fbd23922c02a6ac8464f36f2f6bc61a843b35d461',
 'next_stage/attribute_audit.py': 'c312327b9105639b238a8766015a076f2cac8cd51a4cb91abcc1890e2a8f9e07',
 'next_stage/gdf.py': '1d12a556b9e638e466e2104aaf34cf99120bdb716ff737ac73eb20cfb9b8c9e9',
 'next_stage/gdf_checks.py': 'd34ff18b8475753990f5896385d034415995eddb922566470df7f5791401ec75',
 'next_stage/mechanism_probe.py': '5d65c5a983358c5055178827326cf3b712286b45436ddfb082eb449d9fa215e7',
 'next_stage/next_data.py': 'd3d1d4f22097925b06f850f606becb9f240a007d86c8a380ec99043e03e961de',
 'next_stage/prepare_gdf.py': '8003067d218ddb9b01e3c80e18f22e8f534c35a08d598979661cbf22de21c591',
 'next_stage/stage_checks.py': '6513292727409dfd000e437c1223b1ec9ffad9ae9cf0af364cfdad811b5ad1c4',
 'next_stage/stage_model.py': 'bba378a8d384ddbe6694491bbabd3ef384cf3796f2e3eb6db647ef02ac965d7b',
 'next_stage/stage_train.py': 'c134b3b61a0c2a0f0ace7868464f2d1ec292f64e42e059d23e6373545043fa79'}
PARENT_ORDINAL_SHA256 = {
    "next_stage/ordinal_stage_model.py":"196932825a4b85a7f6eafb1b9b828c5ce79c3a4bb398b956e4206fa33b2d6ebd",
    "next_stage/ordinal_stage_train.py":"8c4f3026b15b8b9c0d2cedede415dcf28e2d97297c19ea76f181a12417359fcb",
    "next_stage/ordinal_stage_checks.py":"984d461ff49e255e4e51b95be8eb4cdff1f1ddf6235256e0be9cb6a641da0d50"}
FROZEN_SOURCE_PATHS = ("next_stage/frozen_ordinal_stage_model.py", "next_stage/frozen_ordinal_stage_train.py", "next_stage/frozen_ordinal_stage_checks.py")
ORIGINAL_WARMSTART_SHA256 = "7406a3ba7e5bac1a7d8ee24898cb954ff8576f37fcf5fba0e26ef9a2cb5c43e6"


def alpha_at(step, start, ramp):
    """step is zero-based successful-update index, shared by all arms."""
    if step < start:
        return 0.
    if ramp <= 1:
        return 1.
    return min(1., max(0., (step - start) / float(ramp - 1)))


def source_inventory(manifest_path, expected_sha256):
    """Exact original36 plus named frozen3; no recursive private-tree capture."""
    from common import sha256, load_json
    if not manifest_path or not expected_sha256 or sha256(manifest_path) != expected_sha256:
        raise ValueError("Explicit frozen source-map file and its SHA256 are required")
    declared = load_json(manifest_path)
    if declared.get("schema") != "GTA-AVT-frozen-ordinal-scientific-source-map-v1":
        raise ValueError("Wrong ordinal scientific-source map schema")
    values = declared.get("source_sha256", {})
    if set(values) != set(BASE_SOURCE_SHA256) | set(PARENT_ORDINAL_SHA256) | set(FROZEN_SOURCE_PATHS):
        raise ValueError("The isolated ordinal scientific snapshot must contain exactly39 declared files")
    if any(values[name] != digest for name, digest in dict(BASE_SOURCE_SHA256, **PARENT_ORDINAL_SHA256).items()):
        raise ValueError("An original36 scientific source differs from the frozen reference")
    for name, digest in values.items():
        path = PROJECT / name
        if (not path.is_file() or path.is_symlink() or path.resolve() != path or sha256(path) != digest):
            raise ValueError("Actual ordinal scientific source bytes differ: " + name)
    return dict(values)


def validate_preflight(path, source, manifest_sha, source_manifest_sha, recipe, protocol_sha256, runtime=None):
    """Validate a newly acquired ordinal receipt, not a renamed old preflight."""
    from common import sha256, load_json
    if not path:
        raise ValueError("A NEW actual ordinal-pair preflight receipt is mandatory")
    receipt = load_json(path)
    required = dict(schema="GTA-AVT-frozen-ordinal-pair-preflight-v1", status="passed",
        test_used=False, operator_variant=OPERATOR_VARIANT, source_sha256=source,
        classifier_policy=CLASSIFIER_POLICY, protocol_sha256=protocol_sha256,
        manifest_sha256=manifest_sha, source_manifest_sha256=source_manifest_sha,
        warmstart_sha256=ORIGINAL_WARMSTART_SHA256,
        allowed_training_seeds=list(APPROVED_TRAINING_SEEDS),
        condition_projection=PROJECTION, constant_c=.4)
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("Ordinal source/data/definition preflight binding differs")
    if recipe["seed"] not in receipt["allowed_training_seeds"]:
        raise ValueError("Undeclared finetuning/augmentation seed")
    for key in ("protocol_sha256", "gate_definition_sha256", "fixed_bank_sha256"):
        value = receipt.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("A frozen protocol/gate/bank SHA256 binding is required")
    references = {}
    for key in ("model_QA", "data_audit", "fixed_TRAIN_gate"):
        ref = receipt.get(key, {})
        target = Path(ref.get("path", ""))
        if (not target.is_file() or target.is_symlink() or sha256(target) != ref.get("sha256")
                or receipt.get("actual_exit_codes", {}).get(key) != 0):
            raise ValueError("Missing/changed actual ordinal evidence: " + key)
        item = load_json(target)
        if item.get("status") != "passed":
            raise ValueError("Ordinal evidence is not a passed result: " + key)
        references[key] = item
    qa = references["model_QA"]
    qa_required = dict(schema="GTA-AVT-frozen-ordinal-classifier-QA-v1",
        operator_variant=OPERATOR_VARIANT, source_sha256=source,
        classifier_policy=CLASSIFIER_POLICY, classifier_frozen_eval=True,
        classifier_state_unchanged=True, classifier_gradients_absent=True,
        repeat_classifier_outputs_exact=True, optimizer_excludes_classifier=True,
        source_manifest_sha256=source_manifest_sha, patch_size=128, batch_size=2,
        device="cuda", precision="CUDA_AMP", original_warmstart_sha256=ORIGINAL_WARMSTART_SHA256,
        new_parameter_count=256, successful_optimizer_updates=2,
        normal_condition_inputs="CT predictions only; no GT")
    if any(qa.get(key) != value for key, value in qa_required.items()):
        raise ValueError("Actual full-size ordinal QA does not cover this source/warmstart")
    if references["data_audit"].get("manifest_sha256") != manifest_sha:
        raise ValueError("Actual data audit does not bind the trainer-consumed manifest")
    if references["fixed_TRAIN_gate"].get("gate_passed") is not True:
        raise ValueError("The prospective ordinal attribute/measurement prerequisite did not pass")
    if runtime is not None and qa.get("runtime") != runtime:
        raise ValueError("Training runtime does not equal its actual ordinal QA runtime")
    return receipt


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
        stage="frozen_ordinal_interaction_probe", role="unspecified", conditioning="constant",
        classifier_policy=CLASSIFIER_POLICY,
        operator_variant=OPERATOR_VARIANT,
        condition_projection=PROJECTION, constant_c=.4,
        detach_conditions=True, q0=.2, s0=.2, gdf_seed=20261023,
        seed=20261024, max_steps=2000, batch_size=2, patch_size=128,
        foreground_probability=.5, lr=.0001, weight_decay=.0001,
        gradient_clip=12., scale_weight=.1, structure_weight=.1,
        gdf_weight=0., gdf_normalization="voxel", alpha_start=0, alpha_ramp=500,
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
    fixed = dict(operator_variant=OPERATOR_VARIANT, classifier_policy=CLASSIFIER_POLICY, gdf_seed=20261023,
                 condition_projection=PROJECTION, constant_c=.4,
                 batch_size=2, patch_size=128, inference_patch_size=128,
                 inference_batch_size=2, inference_overlap=.5,
                 lr=.0001, weight_decay=.0001, gradient_clip=12.,
                 foreground_probability=.5, scale_weight=.1, structure_weight=.1,
                 gdf_weight=0., gdf_normalization="voxel", detach_conditions=True,
                 q0=.2, s0=.2, alpha_start=0, alpha_ramp=500,
                 sampler_step_offset=0, allow_tf32=False)
    for key, expected in fixed.items():
        if values[key] != expected:
            raise ValueError("The matched ordered-radius protocol fixes %s=%s" % (key, expected))
    if values["stage"] not in ("smoke", "frozen_ordinal_interaction_probe") or values["seed"] not in APPROVED_TRAINING_SEEDS:
        raise ValueError("Only the declared ordinal stage and three frozen finetuning seeds are allowed")
    if values["stage"] == "smoke":
        if values["max_steps"] > 8:
            raise ValueError("Smoke is at most8 successful engineering updates")
    elif values["max_steps"] != 2000:
        raise ValueError("The ordered-radius study fixes the2000-update endpoint")
    if values["role"] not in ("C_fixed", "D_fixed") or values["conditioning"] != ("constant" if values["role"] == "C_fixed" else "scale_only"):
        raise ValueError("Frozen roles must be C_fixed/constant or D_fixed/scale_only")
    return values


def assert_classifier_frozen(model):
    if any(module.training for module in model.classifier.modules()) or any(
            parameter.requires_grad or parameter.grad is not None for parameter in model.classifier.parameters()):
        raise ValueError("The entire classifier must remain eval, nontrainable and gradient-free")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gdf-root")
    parser.add_argument("--warmstart", help="Owned CPU tensor bundle or owned checkpoint")
    parser.add_argument("--warmstart-sha256")
    parser.add_argument("--preflight-receipt")
    parser.add_argument("--protocol")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--source-manifest")
    parser.add_argument("--source-manifest-sha256")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.resume:
        raise ValueError("This fixed-classifier candidate permits fresh original7406 starts only; resume is forbidden")
    from common import atomic_json, load_json, sha256, load_trusted_checkpoint
    recipe = effective_recipe(load_json(args.config))
    if not args.execute:
        print(json.dumps(dict(execute=False, recipe=recipe, manifest=args.manifest,
                              output=args.output, initialization="required_original_constant_warmstart_optimizer_reset",
                              required_warmstart_sha256=ORIGINAL_WARMSTART_SHA256,
                              required_source_files=39, required_preflight_schema="GTA-AVT-frozen-ordinal-pair-preflight-v1",
                              required_protocol_schema=PROTOCOL_SCHEMA, classifier_policy=CLASSIFIER_POLICY, fresh_only=True,
                              approved_training_seeds=list(APPROVED_TRAINING_SEEDS),
                              selected_step=recipe["max_steps"], selection="fixed_endpoint",
                              allowed_splits=["train", "val"], test_used=False)), flush=True)
        return
    if not args.protocol or not args.protocol_sha256 or sha256(args.protocol) != args.protocol_sha256:
        raise ValueError("A NEW actual frozen-classifier protocol and exact file SHA256 are required")
    if load_json(args.protocol).get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("Old ordinal protocols cannot authorize the frozen-classifier candidate")
    import numpy as np
    import torch
    from next_data import StageData, batch
    from frozen_ordinal_stage_model import FrozenOrdinalStageModel as StageModel
    from train import optimizer_update_with_amp_retries, set_seed, save_checkpoint
    from inference import evaluate_cases
    if Path(sys.modules[StageModel.__module__].__file__).resolve() != (HERE / "frozen_ordinal_stage_model.py").resolve():
        raise ValueError("The loaded candidate class must come from this frozen source snapshot")
    if not args.warmstart or args.warmstart_sha256 != ORIGINAL_WARMSTART_SHA256:
        raise ValueError("Both arms require the named original constant checkpoint; not Gp/Kp continuation")
    if args.gdf_root is not None:
        raise ValueError("This isolated interaction study fixes GDF0 and consumes no GDF target root")
    if bool(args.warmstart) != bool(args.warmstart_sha256):
        raise ValueError("Warmstart file and explicit SHA256 must be supplied together")
    if recipe["gdf_weight"] > 0. and not args.gdf_root:
        raise ValueError("Positive GDF weight requires an explicit immutable target root")
    if args.warmstart and sha256(args.warmstart) != args.warmstart_sha256:
        raise ValueError("Warmstart byte identity mismatch")
    source = source_inventory(args.source_manifest, args.source_manifest_sha256)
    validate_preflight(args.preflight_receipt, source, sha256(args.manifest),
                       args.source_manifest_sha256, recipe, args.protocol_sha256)
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
    run = Path(args.output).resolve()
    if args.resume:
        if not (run / "config.json").is_file() or (run / "complete.json").exists():
            raise ValueError("Resume requires a partial owned run")
    else:
        if run.exists() and any(run.iterdir()):
            raise ValueError("Run is immutable; choose a new empty output")
        run.mkdir(parents=True, exist_ok=True)
    gdf_metadata = {}
    if args.gdf_root:
        root = Path(args.gdf_root).resolve()
        gdf_metadata = {str(p.relative_to(root)): sha256(p) for p in sorted(root.glob("*.json"))}
        if not gdf_metadata:
            raise ValueError("GDF root lacks immutable metadata JSON")
    signature = dict(
        schema="GTA-AVT-frozen-ordinal-cumulative-interaction-training-v1",
        classifier_policy=CLASSIFIER_POLICY, fresh_only=True, protocol_sha256=args.protocol_sha256,
        operator_variant=OPERATOR_VARIANT, recipe=recipe,
        condition_projection=PROJECTION, constant_c=.4,
        source_manifest_sha256=args.source_manifest_sha256,
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
    torch.set_num_threads(recipe["cpu_threads"])
    set_seed(recipe["seed"])
    torch.backends.cuda.matmul.allow_tf32 = bool(recipe["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(recipe["allow_tf32"])
    runtime = runtime_identity(device)
    validate_preflight(args.preflight_receipt, source, sha256(args.manifest),
                       args.source_manifest_sha256, recipe, args.protocol_sha256, runtime=runtime)
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
    assert_classifier_frozen(model)
    classifier_initial_digest = tensor_digest(model.classifier.state_dict())
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
            warmstart_load=load_report, optimizer_reset=True,
            classifier_policy=CLASSIFIER_POLICY, classifier_tensor_digest=classifier_initial_digest,
            classifier_frozen_eval=True, optimizer_excludes_classifier=True))
    model.to(device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=recipe["lr"],
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
        assert_classifier_frozen(model)
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
        assert_classifier_frozen(model)
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
            classifier_current_digest = tensor_digest(model.classifier.state_dict())
            if classifier_current_digest != classifier_initial_digest:
                raise ValueError("Frozen classifier parameter/BN/buffer state changed")
            checkpoint = dict(
                classifier_policy=CLASSIFIER_POLICY, classifier_tensor_digest=classifier_current_digest,
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
    assert_classifier_frozen(model)
    classifier_final_digest = tensor_digest(model.classifier.state_dict())
    if classifier_final_digest != classifier_initial_digest:
        raise ValueError("Frozen classifier state changed during the complete run/evaluation")
    complete = dict(
        schema="GTA-AVT-frozen-ordinal-cumulative-interaction-complete-v1", state="complete",
        classifier_policy=CLASSIFIER_POLICY, classifier_frozen_eval=True,
        classifier_initial_digest=classifier_initial_digest, classifier_final_digest=classifier_final_digest,
        classifier_state_unchanged=True, classifier_gradients_absent=True,
        optimizer_excludes_classifier=True, fresh_only=True, protocol_sha256=args.protocol_sha256,
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
