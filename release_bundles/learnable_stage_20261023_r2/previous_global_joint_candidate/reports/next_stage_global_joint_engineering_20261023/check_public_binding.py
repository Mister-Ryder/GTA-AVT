"""Check public joint-QA byte bindings and stdlib plan only; never execute Torch.

Use this after placing the three source-overlay files under next_stage/ in an
isolated copy of the public project. This is not a new CUDA pass or model replay.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    project = Path(args.project).resolve()
    folder = project / "next_stage"
    local = Path(__file__).resolve().parent
    identity = read(local / "source_identity_mapping.json")
    summary = read(local / "engineering_summary.json")
    assert summary["successful_optimizer_updates"] == 1
    assert summary["optimizer_update"]["amp_attempts"] == 2
    assert summary["optimizer_update"]["amp_retries"] == 1
    assert len(summary["gradient_and_parameter_change"]) == 6
    assert all(v["finite"] and v["gradient_nonzero_elements"] > 0 and
               v["changed_elements"] > 0 for v in summary["gradient_and_parameter_change"].values())
    components = summary["losses"]
    summed = components["segmentation"] + .1 * (
        components["scale"] + components["structure"] + components["gdf_mse"])
    assert math.isclose(components["total"], summed, rel_tol=0., abs_tol=5e-7)
    assert summary["source_identity"]["public_manifest_sha256"] == identity["public_manifest_sha256"]
    manifest_path = folder / "global_integrated_candidate_manifest_20261023.json"
    recipe_path = folder / "global_integrated_candidate_UNTRAINED_20261023.json"
    helper_path = folder / "global_integrated_joint_qa.py"
    assert sha(manifest_path) == identity["public_manifest_sha256"]
    assert sha(recipe_path) == identity["recipe_sha256"]
    assert sha(helper_path) == identity["executed_helper_sha256"]
    manifest, recipe = read(manifest_path), read(recipe_path)
    spec = importlib.util.spec_from_file_location("public_joint_qa_binding", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.validate_contract(manifest, recipe)
    actual = helper.verify_sources(project, manifest["scientific_source_sha256"])
    assert actual == identity["public_source_sha256"]
    original = identity["actual_QA_original_source_sha256"]
    differences = sorted(k for k in actual if actual[k] != original[k])
    assert differences == ["next_stage/mechanism_probe.py"]
    # The unadapted actual manifest must reject the public tree. Do not relabel
    # the historical pass as an execution of a different manifest or inventory.
    refused = False
    try:
        helper.verify_sources(project, original)
    except ValueError:
        refused = True
    assert refused
    assert "torch" not in sys.modules
    child = subprocess.run([sys.executable, "-I", "-S", str(helper_path),
                            "--manifest", str(manifest_path), "--recipe", str(recipe_path)],
                           check=True, capture_output=True, text=True)
    plan = json.loads(child.stdout)
    assert plan["execute"] is False and plan["formal_training_authorized"] is False
    assert plan["manifest_sha256"] == identity["public_manifest_sha256"]
    assert plan["recipe_sha256"] == identity["recipe_sha256"]
    assert plan["helper_sha256"] == identity["executed_helper_sha256"]
    assert plan["source_files_pinned"] == 33 and plan["synthetic_successful_updates"] == 1
    print(json.dumps(dict(status="passed", scope="isolated_public_source_hash_binding_and_stdlib_plan_only",
        public_source_files=33, original_exact_files=32, public_metadata_adaptations=1,
        original_manifest_rejected_public_tree=True, default_plan_exit_code=child.returncode,
        reported_joint_loss_FP32_rounding_tolerance=5e-7, six_positive_gradient_change_groups=True,
        Torch_imported=False, CUDA_executed=False, model_updates=0,
        limitation="Historical CUDA result belongs to the original manifest only; this check is not a second GPU pass")))


if __name__ == "__main__":
    main()
