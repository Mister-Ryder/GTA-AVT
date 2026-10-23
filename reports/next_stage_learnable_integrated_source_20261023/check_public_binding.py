"""Check corresponding public source bytes and inert plans, using stdlib only.

This does not execute the synthetic CUDA fixture, import a model, open data or
create a preflight. A source binding pass is not an engineering or efficacy pass.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--qa-helper", required=True, help="Exact public helper, outside the isolated scientific37 tree")
    args = parser.parse_args()
    project = Path(args.project).resolve()
    identity = read(HERE / "source_identity_mapping.json")
    folder = project / "next_stage"
    manifest_path = folder / identity["public_manifest_filename"]
    recipe_path = folder / "learnable_integrated_candidate_UNTRAINED_20261023.json"
    helper_path = Path(args.qa_helper).resolve()
    trainer_path = folder / "learnable_integrated_train.py"
    manifest, recipe = read(manifest_path), read(recipe_path)
    assert sha(manifest_path) == identity["public_manifest_sha256"]
    for name, digest in identity["exact_new_source_and_recipe_sha256"].items():
        path = helper_path if name == "next_stage/learnable_integrated_joint_qa.py" else project / name
        assert sha(path) == digest, name
    expected = identity["corresponding_public37_source_sha256"]
    original = identity["original_private37_source_sha256"]
    assert len(expected) == len(original) == 37
    assert manifest["scientific_source_sha256"] == expected
    assert manifest["original_learnable36_source_sha256"] == {
        k: v for k, v in expected.items() if k != "next_stage/learnable_integrated_train.py"}
    discovered = {p.relative_to(project).as_posix() for root in (project / "code", folder)
                  for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                  and (p.suffix == ".py" or ("historical" in p.parts and p.suffix.lower() in (".txt", ".json", ".md")))}
    assert discovered == set(expected), "Use the isolated scientific37 snapshot; keep the QA helper external"
    differences = sorted(k for k in expected if expected[k] != original[k])
    assert differences == ["next_stage/mechanism_probe.py"]
    restored = copy.deepcopy(manifest)
    for name in ("scientific_source_sha256", "original_learnable36_source_sha256"):
        restored[name][differences[0]] = original[differences[0]]
    restored_bytes = (json.dumps(restored, indent=2) + "\n").encode("utf-8")
    assert hashlib.sha256(restored_bytes).hexdigest() == identity["original_private_manifest_sha256"]
    spec = importlib.util.spec_from_file_location("public_integrated_source_binding", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.validate_contract(manifest, recipe)
    assert helper.verify_sources(project, expected) == expected
    rejected = False
    try:
        helper.verify_sources(project, original)
    except ValueError:
        rejected = True
    assert rejected
    plans = []
    commands = [
        [sys.executable, "-I", "-S", str(helper_path), "--manifest", str(manifest_path),
         "--recipe", str(recipe_path)],
        [sys.executable, "-I", "-S", str(trainer_path), "--candidate-manifest", str(manifest_path),
         "--config", str(recipe_path), "--manifest", "NOT_OPENED_BY_PLAN.json",
         "--output", "NOT_CREATED_BY_PLAN"],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        assert plan["execute"] is False and plan["formal_training_authorized"] is False
        plans.append(plan)
    assert plans[0]["source_files_pinned"] == 37
    assert plans[0]["synthetic_successful_updates"] == 1  # Requirement, not an observed update.
    assert plans[0]["physical_synthetic_target"] is True
    assert plans[1]["recipe"]["gdf_weight"] == .1
    assert plans[1]["candidate_status"] == "STOPPED_UNTRAINED_NOT_LAUNCH_AUTHORIZED"
    assert plans[1]["test_used"] is False
    assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
    print(json.dumps(dict(status="passed", scope="public_byte_bindings_and_two_inert_plans_only",
        corresponding_public_source_files=37, exact_to_original_private=36,
        existing_metadata_adaptations=1, original_manifest_reconstructed_exactly=True,
        original_source_map_rejects_public_tree=True, inert_plans=2,
        reported_actual_joint_CUDA_QA=identity["actual_joint_CUDA_QA"],
        historical_original37_successful_updates=identity["observed_successful_updates_supplied"],
        public_manifest_CUDA_executed=False, formal_training_authorized=False,
        model_updates=0, Torch_imported=False, GPU_executed=False)))


if __name__ == "__main__":
    main()
