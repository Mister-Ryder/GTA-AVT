"""Stdlib recomputation of released aggregate tables; no private inputs or models.

--check checks derived moments/support/attribute summaries. Optional --project
checks the separate corresponding-public36 identity and inert training plans.
Neither mode imports Torch or opens study images. Floating statistics use1e-11
absolute tolerance; Python arithmetic is not declared bit-exact to Torch.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    if a is None or b is None:
        assert a is None and b is None
    else:
        assert math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=0., abs_tol=1e-11), (a, b)


def aggregate(group):
    group = [r for r in group if r["support"]]
    n = sum(r["support"] for r in group)
    if not n:
        return dict(support=0, GT_mean=None, predicted_mean=None, GT_population_std=None,
                    predicted_population_std=None, predicted_min=None, predicted_max=None,
                    MSE=None, constant_point2_MSE=None, in_bank_GT_mean_MSE=None, pearson=None)
    mx = sum(r["support"] * r["GT_mean"] for r in group) / n
    my = sum(r["support"] * r["predicted_mean"] for r in group) / n
    vx = sum(r["support"] * (r["GT_population_std"] ** 2 + (r["GT_mean"] - mx) ** 2) for r in group) / n
    vy = sum(r["support"] * (r["predicted_population_std"] ** 2 + (r["predicted_mean"] - my) ** 2) for r in group) / n
    covariance = sum(r["support"] * ((r["pearson"] or 0.) * r["GT_population_std"] * r["predicted_population_std"] +
                                    (r["GT_mean"] - mx) * (r["predicted_mean"] - my)) for r in group) / n
    return dict(support=n, GT_mean=mx, predicted_mean=my, GT_population_std=math.sqrt(vx),
                predicted_population_std=math.sqrt(vy), predicted_min=min(r["predicted_min"] for r in group),
                predicted_max=max(r["predicted_max"] for r in group),
                MSE=sum(r["support"] * r["MSE"] for r in group) / n,
                constant_point2_MSE=sum(r["support"] * r["constant_point2_MSE"] for r in group) / n,
                in_bank_GT_mean_MSE=vx, pearson=covariance / math.sqrt(vx * vy) if vx > 0 and vy > 0 else None)


def compare(observed, expected):
    assert observed["support"] == expected["support"]
    for key in observed:
        if key != "support":
            close(observed[key], expected[key])


def check_numbers():
    bank, attributes, qa = [load(HERE / name) for name in ("bank_summary.json", "attribute_summary.json", "QA_summary.json")]
    group = rows(HERE / "bank_groupwise.csv")
    strings = {"case_id", "source", "stratum", "representation"}
    for r in group:
        for key in r:
            if key not in strings:
                r[key] = None if r[key] == "" else (int(r[key]) if key == "support" else float(r[key]))
    assert len(group) == 48
    case_table = rows(HERE / "bank_casewise_support.csv")
    assert len(case_table) == 12 and sum(int(r["fixed_patches"]) for r in case_table) == bank["complete_patches"] == 96
    raw = aggregate([r for r in group if r["representation"] == "raw"])
    conditional = aggregate([r for r in group if r["representation"] == "conditional"])
    compare(raw, bank["known_nonempty_raw_fine"])
    compare(conditional, bank["known_nonempty_conditional_fine"])
    assert raw["support"] == bank["known_nonempty_patches"] == 74
    close(raw["predicted_population_std"], bank["global_nonempty_raw_c_population_std"])
    for source, value in bank["source_groups"].items():
        for rep, name in (("raw", "raw_fine"), ("conditional", "conditional_fine")):
            compare(aggregate([r for r in group if r["source"] == source and r["representation"] == rep]), value[name])
    compare(aggregate([r for r in group if r["stratum"] == "GT_centerline_plus_jitter" and r["representation"] == "raw"]), bank["primary_GT_jitter_raw_fine"])
    compare(aggregate([r for r in group if r["stratum"] == "CT_uniform" and r["representation"] == "raw"]), bank["CT_uniform_nonempty_raw_fine"])
    informative = []
    for row in case_table:
        value = aggregate([r for r in group if r["case_id"] == row["case_id"] and r["representation"] == "raw"])
        assert int(row["known_nonempty_patches"]) == value["support"]
        close(float(row["raw_c_population_std"]), value["predicted_population_std"])
        close(float(row["GT_fine_population_variance"]), value["GT_population_std"] ** 2)
        active = value["support"] >= 2 and value["predicted_population_std"] > .0001 and value["GT_population_std"] ** 2 >= .0001
        assert active == (row["informative"] == "True")
        if active:
            informative.append(row["case_id"])
    assert len(informative) == bank["informative_case_count"] == 11 and set(informative) == set(bank["informative_cases"])
    passed = len(case_table) == 12 and raw["predicted_population_std"] > .0001 and len(informative) >= 8
    assert passed == bank["gate_passed"] is True
    attr = rows(HERE / "attribute_casewise.csv")
    assert len(attr) == 36 and sum(int(r["nodes"]) for r in attr) == attributes["all_train"]["nodes"] == 55294
    for column, key in (("fine_fraction", "fine_fraction_case_stats"),
                        ("near2_within025_fraction", "near_2mm_within_025_case_stats"),
                        ("near2_within05_fraction", "near_2mm_within_05_case_stats"),
                        ("unknown_structure_fraction", "unknown_structure_case_stats"),
                        ("local_mixed_fraction", "local_mixed_case_stats")):
        close(statistics.mean(float(r[column]) for r in attr), attributes["all_train"][key]["mean"])
    unknown = sum(round(int(r["nodes"]) * float(r["unknown_structure_fraction"])) for r in attr)
    assert unknown == attributes["unknown_structure_nodes"] == 6665
    close(unknown / 55294, attributes["unknown_structure_node_weighted_fraction"])
    assert sum(attributes["all_train"]["scale_bin_node_counts"]) == 55294
    for value in qa["platforms"].values():
        assert value["observed_exit_code"] == 0 and value["new_parameter_count"] == 256
        assert value["successful_optimizer_updates"] == 2 and value["train_mode_zero_init_forward_buffers_RNG_exact"] is True
        background = value["baseline_own_repeat_common_gradient_max_abs_error"]
        for result in value["zero_init_checks"]:
            assert result["forward_exact"] and result["weight_bias_gradients_nonzero"]
            assert result["preclip_common_gradient_max_abs_error"] <= max(1e-6, 2 * background)
        for result in value["optimizer_update"].values():
            assert result["successful_steps"] == 1 and result["amp_attempts"] == result["amp_retries"] + 1
            assert result["fresh_optimizer_and_scaler"] and result["initial_RNG_restored"] and result["initial_state_restored_including_BN"]
    return dict(known_nonempty_patches=74, informative_cases=11, support_gate=passed,
                attribute_nodes=55294, attribute_unknown_nodes=unknown,
                comparison_tolerance=1e-11, raw_MSE=raw["MSE"], conditional_MSE=conditional["MSE"])


def check_report_files():
    manifest = load(HERE / "report_manifest.json")
    expected = manifest["files"]
    actual = {p.name for p in HERE.iterdir() if p.is_file() and p.name != "report_manifest.json"}
    assert actual == set(expected)
    for name, entry in expected.items():
        path = HERE / name
        assert path.name == name and path.stat().st_size == entry["bytes"]
        assert sha(path) == entry["sha256"], name
    return len(expected)


def check_project(project):
    project = Path(project).resolve()
    identity = load(project / "next_stage/learnable_source_identity_20261023.json")
    expected = identity["corresponding_public36_source_sha256"]
    assert len(expected) == 36
    for name, digest in expected.items():
        assert sha(project / name) == digest, name
    different = [name for name in expected if expected[name] != identity["actual_QA36_source_sha256"][name]]
    assert different == ["next_stage/mechanism_probe.py"]
    assert sha(project / "docs/learnable_global_prospective_protocol_20261023.txt") == identity["public_metadata_sanitized_protocol_sha256"]
    assert sha(project / "next_stage/source_identity_20261023.json") == identity["previous_published_identity_file_sha256"]
    plans = []
    for name, digest in identity["requested_recipe_files_exact"].items():
        assert sha(project / name) == digest
        process = subprocess.run([sys.executable, "-I", "-S", str(project / "next_stage/learnable_stage_train.py"),
            "--config", str(project / name), "--manifest", "NOT_OPENED_BY_PLAN.json", "--output", "NOT_CREATED_BY_PLAN"],
            check=True, capture_output=True, text=True)
        plan = json.loads(process.stdout)
        assert plan["execute"] is False and plan["test_used"] is False
        plans.append(plan["recipe"])
    assert {k for k in plans[0] if plans[0][k] != plans[1][k]} == {"conditioning", "role"}
    return dict(corresponding_public_sources=36, exact_to_actual=35, existing_metadata_adaptations=1,
                inert_plans=2, Torch_imported=False, GPU_executed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--project")
    args = parser.parse_args()
    result = check_numbers()
    result["report_files_verified"] = check_report_files()
    if args.project:
        result["public_binding"] = check_project(args.project)
    assert "torch" not in sys.modules
    print(json.dumps(dict(status="passed", scope="Released aggregate arithmetic and optional public byte bindings only", **result)))


if __name__ == "__main__":
    main()
