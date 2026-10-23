"""Recompute released engineering arithmetic and tables with stdlib only.

No model/GPU/data-array execution. Arithmetic checks do not independently attest
execution, prove efficacy or reproduce the original private CUDA result.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(name):
    with (HERE / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_equal(rows, original):
    assert len(rows) == len(original)
    for row in rows:
        group = row.pop("group")
        expected = original[group]
        assert set(row) == set(expected)
        for name, value in row.items():
            ref = expected[name]
            if isinstance(ref, bool):
                assert value == str(ref)
            elif isinstance(ref, (int, float)):
                assert float(value) == ref
            else:
                assert value == ref


def check():
    summary = read(HERE / "QA_summary.json")
    identity = read(HERE / "source_identity_mapping.json")
    assert summary["status"] == "passed" and summary["observed_exit_code"] == 0
    assert summary["study_data_read"] is False and summary["TEST_files_read"] is False
    assert summary["formal_training"] is False and summary["efficacy_established"] is False
    assert summary["public_manifest_CUDA_executed"] is False
    assert summary["whole_expanded_public_tree_CUDA_executed"] is False
    assert summary["original_private_result_sha256"] == identity["actual_original_private_result_sha256"]
    assert summary["original_private_manifest_sha256"] == identity["original_private_manifest_sha256"]
    assert summary["corresponding_public_manifest_sha256"] == identity["public_manifest_sha256"]
    assert summary["actual_scientific_source_files"] == 37
    assert summary["public_source_exact_count"] == 36
    assert summary["batch_size"] == 2 and summary["patch_size"] == 128
    assert summary["precision"] == "CUDA_AMP" and summary["model_alpha"] == 1.
    assert summary["normal_CT_only_conditions"] and summary["physical_synthetic_target"]
    loss = summary["component_details"]
    for name in ("segmentation", "scale", "structure", "gdf_mse", "total"):
        assert math.isfinite(loss[name]) and loss[name] > 0.
    assert loss["scale_weight"] == loss["structure_weight"] == loss["gdf_weight"] == .1
    reconstructed = loss["segmentation"] + .1 * (loss["scale"] + loss["structure"] + loss["gdf_mse"])
    error = abs(reconstructed - loss["total"])
    assert error <= 5e-7  # FP32 objective arithmetic; not Python bit-exact.
    update = summary["optimizer_update"]
    assert summary["successful_optimizer_updates"] == 1
    assert update["amp_attempts"] == 2 and update["amp_retries"] == 1
    assert update["loss_scale_before"] == 65536. and update["loss_scale_after"] == 32768.
    assert len(summary["forward_attempts"]) == update["amp_attempts"]
    assert all(row["details"] == loss and row["normal_predicted_conditions"] and row["alpha"] == 1.
               for row in summary["forward_attempts"])
    assert len(summary["overflow_attempts"]) == 1
    overflow = summary["overflow_attempts"][0]
    assert overflow["optimizer_step_skipped"] and overflow["buffers_and_rng_restored"]
    groups = summary["gradient_and_parameter_change"]
    assert len(groups) == 8
    assert all(v["finite"] and v["gradient_nonzero_elements"] > 0 and v["changed_elements"] > 0
               and v["parameter_delta_absmax"] > 0. for v in groups.values())
    csv_equal(table("joint_gradient_groups.csv"), groups)
    route = summary["gdf_only_routing"]
    assert route["passed"] and route["status"] == "passed" and route["optimizer_updates"] == 0
    assert route["diagnostic_gradient_scale"] == 1024.
    assert route["BN_RNG_restored"] and route["no_graph_retained_for_joint_update"]
    zero = {"segmentation_head", "scale_head", "structure_head"}
    for name, item in route["gradients"].items():
        assert item["finite"]
        assert item["expected_direct_gradient"] == (name not in zero)
        assert (item["gradient_nonzero_elements"] == 0) == (name in zero)
        assert (item["unscaled_absmax"] == 0.) == (name in zero)
    assert len(route["gradients"]) == 8
    csv_equal(table("GDF_routing_groups.csv"), route["gradients"])
    fixture = summary["fixture"]
    assert fixture["kind"] == "synthetic_128mm_cylinders_not_AVT"
    assert fixture["patient_arrays_opened"] is False and fixture["per_case_normalization"] is False
    scale = fixture["fixed_train_field_scale"]
    assert scale == 2.8707158419291448
    assert fixture["physical_spacing_mm"] == [1., 1., 1.]
    rows = table("physical_fixtures.csv")
    assert len(rows) == len(fixture["cases"]) == 2
    target_error = 0.
    for row, case in zip(rows, fixture["cases"]):
        field = case["field"]
        for name, value in row.items():
            assert float(value) == (case[name] if name in case else field[name])
        assert field["valid_voxels"] == 128 ** 3
        assert field["known_background_voxels"] + case["foreground_voxels"] == 128 ** 3
        assert field["kernel_evaluations"] == case["centreline_nodes"] * case["foreground_voxels"]
        assert field["fixed_train_field_scale"] == scale and field["target_min"] == 0.
        assert field["target_used_as_model_input"] is False and field["kernel_truncation"] is None
        err = abs(math.log1p(field["raw_sum_foreground_max"]) / scale - field["target_max"])
        target_error = max(target_error, err)
        assert err <= 1e-7  # Published target is FP32; Python logarithm is FP64.
    assert loss["gdf_valid_voxels"] == sum(case["field"]["valid_voxels"] for case in fixture["cases"])
    manifest = read(HERE / "report_manifest.json")
    assert set(manifest["files"]) == {p.name for p in HERE.iterdir() if p.is_file() and p.name != "report_manifest.json"}
    for name, value in manifest["files"].items():
        path = HERE / name
        assert path.stat().st_size == value["bytes"] and sha(path) == value["sha256"]
    return dict(status="passed", scope="released_engineering_arithmetic_and_table_bindings_only",
        weighted_total_reconstructed=reconstructed, weighted_total_absolute_error=error,
        weighted_total_FP32_tolerance=5e-7, physical_log1p_max_absolute_error=target_error,
        physical_target_FP32_tolerance=1e-7, joint_nonzero_changed_parameter_groups=8,
        GDF_only_nonzero_groups=5, GDF_only_terminal_zero_groups=3,
        historical_successful_updates=1, actual_original37_engineering_status="passed",
        this_process_model_updates=0, Torch_imported=False, GPU_executed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--project")
    parser.add_argument("--qa-helper")
    args = parser.parse_args()
    result = check()
    if args.project:
        assert args.qa_helper
        command = [sys.executable, "-B", "-I", "-S", str(HERE / "check_public_binding.py"),
                   "--project", args.project, "--qa-helper", args.qa_helper]
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        result["public_binding"] = json.loads(process.stdout)
    assert "torch" not in sys.modules
    print(json.dumps(result))


if __name__ == "__main__":
    main()
