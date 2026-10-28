"""Aggregate the completed r3 native VAL8 pair and unchanged F, without inference.

Only the fixed three endpoints F, AR (full-S no GDF) and AR_GDF (full-S GDF)
are accepted. A complete, identity-checked final receipt is mandatory. Pending
or mismatched runs are rejected before any public output directory is created.
This local tool never connects to a server or reads images, labels, or weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import v12_core_objective_readout as core

ENDPOINTS = ("F", "AR", "AR_GDF")
ARMS = ENDPOINTS[1:]
CONTRASTS = {
    "AR-F": {"AR": 1, "F": -1},
    "AR_GDF-F": {"AR_GDF": 1, "F": -1},
    "AR_GDF-AR": {"AR_GDF": 1, "AR": -1},
}
SEED = 20261032
SOURCE_COMMIT = "bc305e5a449a5f1031e5b85100630b6dd543ef7e"
SOURCE_MANIFEST = "b3935b438946a0e16e06f9ff3ceec2244fc1081cfd3b84266c4c4e11c16eb2e1"
PLAN = "ab6eba3d7d3fab473a506dd47fc3b9978f32a8312824907b60022a6cd6bd8434"
SHARED = "2fb4f3eddc0d786a44cabd5cad80e17670b4cfe43c3d10ed1d88182947ec2b87"
CONDITION = "9abdec01ac53b08b21d9cf9894e174dd8b88c4ed630a3887688e2e9de4aa0e66"
BASE = "0ab4f06977be0241f37f640c9730dd29d6c1ed5239d9e8050541cae4cc76b630"
PREDICTOR = "00452c51b77d6e97cd4ebf5187219c997307d0123049d681efcaf84faf8b07e3"
CALIBRATION = "3e17915b01d41d8e6f7189aea24aa8536d56a559493890549140812cc1dd63cb"
CALIBRATION_CANONICAL = "64818dbbeb09060c5e503e9db11db89c1c41ef4e779febf578da20bc4279a554"
PHASES = {"geometry": 20000, "refine": 10000}
STAGES = ("bounded_target_overlay", "cuda_qa", "shared_head_and_calibration", "AR", "AR_GDF")
NATIVE_DEFINITIONS = {
    "centerline": "Lee skeleton voxel overlap on each native grid; not physical branch length",
    "fine_centerline": "GT native Lee skeleton sites with native physical EDT radius <=2mm; report supported voxel count; no sites yields null recall",
    "dispersion": "population SD across defined cases; worst20 means ceil(20% cases)",
    "empty": "Both masks empty Dice/clDice=1; undefined precision or recall is null",
}


def require(test, message):
    if not test:
        raise ValueError(message)


def canonical_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def equal(actual, expected, label):
    require(type(actual) is type(expected) and actual == expected, label + " differs")


def validate_calibration(prepared, calibration):
    require(isinstance(prepared, dict) and isinstance(calibration, dict),
            "Prepared and calibration records must be objects")
    for key, value in {
        "schema": "GTA-AVT-gdf-joint-prepared-1", "plan_sha256": PLAN,
        "shared_checkpoint_sha256": SHARED, "condition_checkpoint_sha256": CONDITION,
        "selected_base_sha256": BASE, "common_head_successful_updates": 2000,
        "segmentation_updates_in_head_warmup": 0, "calibration_sha256": CALIBRATION,
    }.items():
        equal(prepared.get(key), value, "Prepared/" + key)
    equal(calibration.get("schema"), "GTA-AVT-gdf-joint-calibration-1", "Calibration/schema")
    equal(calibration.get("checkpoint_sha256"), SHARED, "Calibration/checkpoint")
    equal(calibration.get("optimizer_updates"), 0, "Calibration/optimizer_updates")
    equal(calibration.get("microbatch_size"), 1, "Calibration/microbatch_size")
    equal(calibration.get("logical_batch_size"), 2, "Calibration/logical_batch_size")
    batches = calibration.get("batches")
    require(isinstance(batches, list) and len(batches) == 8, "Exactly eight TRAIN calibration batches required")
    require([b.get("batch_index") for b in batches] == list(range(8)), "Calibration batch indices differ")
    loss_ratios, gradient_ratios = [], []
    for batch in batches:
        ids = batch.get("case_ids")
        require(isinstance(ids, list) and len(ids) == 2 and all(isinstance(x, str) for x in ids),
                "Calibration requires two TRAIN case identities per batch")
        require(not set(ids).intersection(core.CASE_IDS), "Calibration must not contain fixed VAL cases")
        values = {k: core.number(batch.get(k), "Calibration/" + k) for k in
                  ("seg_loss", "gdf_loss", "seg_s_grad_norm", "gdf_s_grad_norm")}
        require(all(v > 0 for v in values.values()), "Calibration losses and gradient norms must be positive")
        loss_ratios.append(values["seg_loss"] / values["gdf_loss"])
        gradient_ratios.append(values["seg_s_grad_norm"] / values["gdf_s_grad_norm"])
    loss_candidate = statistics.median(loss_ratios)
    gradient_candidate = statistics.median(gradient_ratios)
    peak = min(max(loss_candidate, .5 * gradient_candidate), 2 * gradient_candidate)
    for key, expected in (
        ("loss_matching_candidate", loss_candidate),
        ("shared_gradient_matching_candidate", gradient_candidate),
        ("lambda_peak", peak),
        ("predicted_median_weighted_shared_gradient_ratio", peak / gradient_candidate),
    ):
        core.close(core.number(calibration.get(key), "Calibration/" + key), expected, "Calibration/" + key)
    core.close(core.number(prepared.get("lambda_peak"), "Prepared/lambda_peak"), peak, "Prepared/lambda_peak")
    # The shared preparation has already happened. This canonical content identity
    # pins its actual eight-batch record, separately from the saved-file byte hash.
    equal(canonical_sha(calibration), CALIBRATION_CANONICAL, "Calibration/canonical content")
    return {"common_head_successful_updates": 2000, "segmentation_updates_in_head_warmup": 0,
            "calibration_batches": 8, "calibration_optimizer_updates": 0,
            "calibration_file_sha256": CALIBRATION,
            "calibration_canonical_JSON_sha256": CALIBRATION_CANONICAL,
            "lambda_peak": peak, "loss_matching_candidate": loss_candidate,
            "shared_gradient_matching_candidate": gradient_candidate,
            "predicted_median_weighted_shared_gradient_ratio": peak / gradient_candidate}


def validate_phase(summary, phase, arm, parent_sha):
    require(isinstance(summary, dict), arm + "/" + phase + " summary missing")
    equal(summary.get("kind"), "train", arm + "/" + phase + "/kind")
    equal(summary.get("stage"), "full_joint", arm + "/" + phase + "/stage")
    equal(summary.get("phase"), phase, arm + "/" + phase + "/phase")
    equal(summary.get("successful_updates"), PHASES[phase], arm + "/" + phase + "/successful_updates")
    attempts = core.count(summary.get("attempts"), arm + "/" + phase + "/attempts")
    require(attempts >= PHASES[phase], arm + "/" + phase + " attempts below successes")
    elapsed = core.number(summary.get("elapsed_seconds"), arm + "/" + phase + "/elapsed_seconds")
    require(elapsed >= 0, arm + "/" + phase + " negative elapsed time")
    path = summary.get("last_checkpoint")
    require(isinstance(path, str) and path.replace("\\", "/").rsplit("/", 1)[-1] == "last.pt",
            arm + "/" + phase + " must use LAST")
    lineage = summary.get("lineage", {})
    p_init, s_init = lineage.get("predictor_init", {}), lineage.get("segmentation_init", {})
    equal(p_init.get("predictor_state_sha256"), PREDICTOR, arm + "/" + phase + "/P")
    for key, value in {
        "parent_checkpoint_sha256": parent_sha, "common_checkpoint_sha256": parent_sha,
        "common_stage": "gdf_head" if phase == "geometry" else "full_joint",
        "common_successful_updates": 2000 if phase == "geometry" else 20000,
        "selected_base_sha256": BASE, "predictor_state_sha256": PREDICTOR,
        "optimizer_reset": True,
    }.items():
        equal(s_init.get(key), value, arm + "/" + phase + "/lineage/" + key)
    equal(s_init.get("parent_phase"), "gdf_head" if phase == "geometry" else "geometry",
          arm + "/" + phase + "/parent_phase")
    return {"successful_updates": PHASES[phase], "attempts": attempts, "elapsed_seconds": elapsed,
            "stage": "full_joint", "evaluation_checkpoint": "last",
            "parent_checkpoint_sha256": parent_sha,
            "phase_lineage_common_checkpoint_scope": "Direct preceding phase checkpoint; the shared-head anchor is recorded separately at arm level."}


def validate_completion(receipt):
    require(isinstance(receipt, dict), "Final receipt must be an object")
    equal(receipt.get("schema"), "GTA-AVT-gdf-joint-final-receipt-1", "Final receipt/schema")
    equal(receipt.get("complete"), True, "Final completion")
    equal(receipt.get("source_commit"), SOURCE_COMMIT, "Source commit")
    equal(receipt.get("source_manifest_sha256"), SOURCE_MANIFEST, "Source manifest")
    equal(receipt.get("source_files_verified"), True, "Source-file verification")
    supervisor = receipt.get("supervisor", {})
    equal(supervisor.get("schema"), "GTA-AVT-gdf-joint-handoff-1", "Supervisor/schema")
    equal(supervisor.get("status"), "complete", "Supervisor/status")
    stages = supervisor.get("stages")
    require(isinstance(stages, list) and len(stages) == len(STAGES), "Exactly the five authorized stages required")
    require(sorted(s.get("name", "") for s in stages) == sorted(STAGES), "Authorized supervisor stages differ")
    for stage in stages:
        equal(stage.get("status"), "complete", stage["name"] + "/status")
        equal(stage.get("exit_code"), 0, stage["name"] + "/exit_code")
    common = validate_calibration(receipt.get("prepared"), receipt.get("calibration"))
    arms = receipt.get("arms")
    require(isinstance(arms, dict) and set(arms) == set(ARMS), "Exactly the two revised arms required")
    reports, branch_provenance = {}, {}
    for arm in ARMS:
        row = arms[arm]
        equal(row.get("job_exit_code"), 0, arm + "/job_exit_code")
        equal(row.get("process_identity_alive"), False, arm + "/process_identity_alive")
        state = row.get("joint_state", {})
        for key, value in {
            "schema": "GTA-AVT-gdf-joint-arm-1", "status": "complete", "arm": arm,
            "plan_sha256": PLAN, "shared_checkpoint_sha256": SHARED,
            "condition_checkpoint_sha256": CONDITION, "selected_base_sha256": BASE, "seed": SEED,
            "planned_segmentation_updates": 30000, "planned_lineage_updates": 32000,
        }.items():
            equal(state.get(key), value, arm + "/" + key)
        require(state.get("current") is None, arm + " has an unfinished current stage")
        geo_sha = core.digest(row.get("geometry_checkpoint_sha256"), arm + "/geometry checkpoint")
        final_sha = core.digest(row.get("refine_checkpoint_sha256"), arm + "/refine checkpoint")
        equal(state.get("final_checkpoint_sha256"), final_sha, arm + "/final LAST checkpoint")
        completed = state.get("completed", {})
        require(set(completed) == {"geometry", "refine", "native_VAL"}, arm + " completion stages differ")
        phase_proof = {
            "geometry": validate_phase(completed["geometry"], "geometry", arm, SHARED),
            "refine": validate_phase(completed["refine"], "refine", arm, geo_sha),
        }
        metrics = row.get("metrics")
        require(isinstance(metrics, dict) and completed["native_VAL"] == metrics,
                arm + " saved metrics differ from completed native report")
        equal(metrics.get("scope"), "native whole-case evaluation", arm + "/native scope")
        core.validate_report(metrics, arm)
        inferred = row.get("inference_receipts")
        require(isinstance(inferred, list) and len(inferred) == 8, arm + " requires all eight inference receipts")
        require(sorted(x.get("case_id", "") for x in inferred) == sorted(core.CASE_IDS),
                arm + " inference IDs differ from VAL8")
        for case in inferred:
            equal(case.get("checkpoint_sha256"), final_sha, arm + "/" + case["case_id"] + "/inference checkpoint")
        reports[arm] = metrics
        branch_provenance[arm] = {
            "seed": SEED, "plan_sha256": PLAN, "shared_checkpoint_sha256": SHARED,
            "condition_checkpoint_sha256": CONDITION, "selected_base_sha256": BASE,
            "predictor_state_sha256": PREDICTOR, "geometry_checkpoint_sha256": geo_sha,
            "final_checkpoint_sha256": final_sha, "job_exit_code": 0, "process_identity_alive": False,
            "evaluation_endpoint": "last", "native_inference_receipts": 8,
            "segmentation_successful_updates": 30000, "head_including_lineage_updates": 32000,
            "phases": phase_proof, "native_report_canonical_JSON_sha256": canonical_sha(metrics),
        }
    return reports, {
        "schema": "GTA-AVT-gdf-joint-native-provenance-1", "source_commit": SOURCE_COMMIT,
        "source_manifest_sha256": SOURCE_MANIFEST, "source_files_verified_in_supplied_receipt": True,
        "seed": SEED, "seed_count": 1, "seed_stability_assessed": False,
        "plan_sha256": PLAN, "shared_checkpoint_sha256": SHARED,
        "condition_checkpoint_sha256": CONDITION, "selected_base_sha256": BASE,
        "predictor_state_sha256": PREDICTOR, "planned_total_new_training_compute": 62000,
        **common, "arms": branch_provenance,
    }


def validate_reported_summary(report, rows, endpoint):
    supplied = report.get("primary_metrics") if endpoint == "F" else report.get("summary")
    require(isinstance(supplied, dict), endpoint + " reported summary missing")
    for metric in core.METRICS:
        actual = supplied.get(metric)
        require(isinstance(actual, dict), endpoint + "/" + metric + " reported summary missing")
        expected = core.summarize_values({case: rows[case][metric] for case in core.CASE_IDS})
        for source, output in (("mean", "mean_pct"), ("std_population", "sd_population_pp"),
                               ("minimum", "minimum_pct"), ("worst20_mean", "worst_ceil20_mean_pct")):
            value = core.number(actual.get(source), endpoint + "/" + metric + "/summary/" + source,
                                nullable=True)
            normalized = expected[output]
            core.close(value, None if normalized is None else normalized / 100.,
                       endpoint + "/" + metric + "/summary/" + source)


def summarize(receipt, f):
    require(isinstance(f, dict), "F native report must be an object")
    equal(f.get("schema"), "GTA-AVT-v1.2-neutral-base-native-readout-1", "F/schema")
    equal(f.get("checkpoint_sha256"), BASE, "F/checkpoint")
    equal(f.get("checkpoint_unchanged"), True, "F/checkpoint_unchanged")
    equal(f.get("new_training_updates"), 0, "F/new_training_updates")
    equal(f.get("selected_checkpoint_joint_P_S_updates"), 47000, "F/selected S lineage")
    equal(f.get("completed_prefix_training_budgets"), {
        "attribute_pretraining": 30000, "joint_P_S_base_warm": 20000, "joint_P_S_base": 30000,
    }, "F/completed prefix budgets")
    reports, evidence = validate_completion(receipt)
    reports = {"F": f, **reports}
    rows = {name: core.validate_report(report, name) for name, report in reports.items()}
    definitions = NATIVE_DEFINITIONS
    reference = {case: core.gt_support(rows["F"][case]) for case in core.CASE_IDS}
    for endpoint in ENDPOINTS:
        equal(reports[endpoint].get("definitions"), definitions, endpoint + "/metric definitions")
        validate_reported_summary(reports[endpoint], rows[endpoint], endpoint)
        for case in core.CASE_IDS:
            support = core.gt_support(rows[endpoint][case])
            for key, expected in reference[case].items():
                if key == "target_volume_ml":
                    core.close(support[key], expected, endpoint + "/" + case + "/GT support")
                else:
                    equal(support[key], expected, endpoint + "/" + case + "/GT support/" + key)
    result = {
        "schema": "GTA-AVT-gdf-joint-native-objective-readout-1", "split": "VAL",
        "case_ids": list(core.CASE_IDS), "available_endpoints": list(ENDPOINTS), "missing_endpoints": [],
        "source_revision": "20261028-gdf-joint-r3",
        "endpoint_labels": {"F": "Unchanged selected pre-condition P/S fusion base",
                            "AR": "r3 full-S no-GDF matched control",
                            "AR_GDF": "r3 full-S GDF matched arm"},
        "definitions": {
            "units": "Input fractions; endpoint scores are percentages; paired differences and population SDs are percentage points.",
            "aggregation": "Equal case weight; undefined metrics excluded and case IDs listed; worst mean uses ceil(20% of defined cases).",
            "reporting_priority": "First report each revised endpoint minus F as the system-level net change; then AR_GDF-AR as the matched Gaussian-supervision contrast.",
            "F_to_final": "Additional condition and suffix training versus selected F; not a budget-matched isolated component ablation.",
            "GDF_contrast": "Two full-S arms start from one shared head checkpoint and use equal planned update budgets; only the declared GDF loss schedule differs.",
            "scope": "Fixed single seed native VAL8 only; no TEST, seed-stability or causal phase-localization claim.",
            "native_GT_support": "Reported native GT voxel/volume/skeleton/fine supports agree; voxelwise label-file hashes are not independently checked by this local aggregator.",
            "centerline": definitions["centerline"],
            "fine_centerline": definitions["fine_centerline"],
            "patch_comparison": "Patch VAL64, CPU VAL16 and native whole-case metrics are distinct; GDF MSE reduction alone is not connectivity evidence.",
        },
        "validation": {"fixed_unique_VAL8_ids": True, "native_grid_certificates_accepted": True,
                       "segmentation_count_arithmetic_checked": True, "reported_summaries_recomputed": True,
                       "reported_native_GT_support_equal": True, "metric_definitions_equal": True,
                       "GT_label_file_hashes_checked": False},
        "native_GT_support": [{"case_id": c, **reference[c]} for c in core.CASE_IDS],
        "provenance": evidence, "endpoints": {}, "paired_contrasts": {},
    }
    for endpoint in ENDPOINTS:
        result["endpoints"][endpoint] = {
            "status": "available",
            "summary": {metric: core.summarize_values({c: rows[endpoint][c][metric] for c in core.CASE_IDS})
                        for metric in core.METRICS},
            "per_case": [{"case_id": c,
                **{metric + "_pct": None if rows[endpoint][c][metric] is None else 100. * rows[endpoint][c][metric]
                   for metric in core.METRICS},
                **{key: rows[endpoint][c][key] for key in core.COUNTS + core.VOLUMES + ("grid_certification",)}}
                for c in core.CASE_IDS],
        }
    for name, coefficients in CONTRASTS.items():
        delta = {c: {metric: None if any(rows[e][c][metric] is None for e in coefficients)
                     else math.fsum(w * rows[e][c][metric] for e, w in coefficients.items())
                     for metric in core.METRICS} for c in core.CASE_IDS}
        result["paired_contrasts"][name] = {
            "status": "available", "coefficients": coefficients,
            "comparison_scope": "system_net_change_with_additional_training" if "F" in coefficients
                                else "fixed_seed_fullS_matched_GDF_contrast",
            "summary": {metric: core.summarize_values({c: delta[c][metric] for c in core.CASE_IDS}, delta=True)
                        for metric in core.METRICS},
            "per_case": [{"case_id": c, **{metric + "_delta_pp": None if delta[c][metric] is None
                                        else 100. * delta[c][metric] for metric in core.METRICS}}
                         for c in core.CASE_IDS],
        }
    return result, evidence


def readme(result):
    lines = ["# GTA-AVT gdf-joint-r3: completed fixed-seed native VAL8 readout", "",
             "This directory contains only the unchanged F base and the two revised full-S endpoints. Old local-adaptation endpoints are not included. All eight fixed VAL cases have equal weight; no TEST cases are used.", "",
             "## Native endpoint scores and system-level net change", "",
             "| Endpoint | Dice % | Case SD (pp) | Dice minus F (pp) | clDice % | Fine centerline recall % |",
             "|---|---:|---:|---:|---:|---:|"]
    for endpoint in ENDPOINTS:
        s = result["endpoints"][endpoint]["summary"]
        delta = 0. if endpoint == "F" else result["paired_contrasts"][endpoint + "-F"]["summary"]["dice"]["delta_mean_pp"]
        values = [s["dice"]["mean_pct"], s["dice"]["sd_population_pp"], delta,
                  s["cldice"]["mean_pct"], s["fine_centerline_recall"]["mean_pct"]]
        lines.append("| " + endpoint + " | " + " | ".join("NA" if v is None else f"{v:.6f}" for v in values) + " |")
    lines += ["", "F is the unchanged selected base before condition adaptation, with 47,000 segmentation-weight lineage updates. The revised pair inherits the clean AR condition checkpoint (57,000 S updates), shares one 2,000-update head-only warmup (zero S updates), and each executes geometry 20,000 plus refine 10,000 full-S updates. Each final S lineage is 87,000; total new shared-head-plus-two-arms training compute is 62,000. Both final native evaluations use the prescribed LAST checkpoint.", "",
              "F-to-final is the observed system-level net change after additional training, not a budget-matched isolated component ablation. If the GDF endpoint exceeds the matched control but remains below F, that comparison alone does not establish improvement beyond the base.", "",
              "## Case-paired contrasts", "",
              "System-level net changes are listed first, followed by the matched Gaussian-supervision contrast. Differences are calculated within each case before aggregation.", "",
              "| Contrast | Mean Dice difference (pp) | Case SD (pp) | Positive cases | Negative cases | Equal cases |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, entry in result["paired_contrasts"].items():
        s = entry["summary"]["dice"]
        lines.append(f"| {name} | {s['delta_mean_pp']:.6f} | {s['delta_sd_population_pp']:.6f} | {s['positive_cases']} | {s['negative_cases']} | {s['zero_cases']} |")
    lines += ["", "## Files and measurement scope", "",
              "- endpoint_per_case.csv and endpoint_summary.csv contain all three endpoints, all five metrics, native support counts, equal-case means, population SDs, minima and worst-ceil(20%) case means.",
              "- paired_per_case.csv and paired_summary.csv contain the three revised comparisons, including fine centerline recall and the Dice signs across cases.",
              "- objective_readout.json records definitions and validation; provenance.json contains allowlisted source, plan, checkpoint and input identities. No machine paths, credentials, images, labels or weights are copied.",
              "", "Scores are percentages; SDs and paired differences are percentage points. Population SD is across cases within one seed, not across seeds. Fine centerline recall uses native GT Lee-skeleton sites with physical EDT radius at most 2 mm; clDice measures native skeleton voxel overlap, not physical branch length. Undefined supported metrics remain null and are listed rather than filled with zero.",
              "", "This local aggregation checks reported GT support and metric arithmetic; it does not independently compare voxelwise label-file hashes. Fixed VAL64 training blocks and auxiliary CPU VAL16 metrics are separate from native whole-case results. GDF error reduction is not itself evidence of connectivity improvement, and these final endpoints do not identify which intermediate stage caused any change.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", default="results/v12_gdf_revision_20261028/new_pair_final_complete.json")
    parser.add_argument("--f", default="results/v12_core_native_20261028/F/readout.json")
    parser.add_argument("--output", default="results/v12_gdf_joint_native_20261029/final_objective")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        receipt, receipt_sha = core.load_json(args.completion, "Revised-pair final receipt")
        f, f_sha = core.load_json(args.f, "F native readout")
        result, evidence = summarize(receipt, f)
        evidence["completion_receipt_JSON_sha256"] = receipt_sha
        evidence["F_native_JSON_sha256"] = f_sha
        result["input_JSON_sha256"] = {"F": f_sha, "completion_receipt": receipt_sha}
        result["input_hash_scope"] = "Saved F and enclosing completion-receipt byte hashes; embedded native reports use canonical sorted-key compact JSON hashes in provenance."
        if not args.check_only:
            core.export(result, args.output)
            output = Path(args.output)
            (output / "README.md").write_text(readme(result), encoding="utf-8", newline="\n")
            (output / "provenance.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
            for path in output.iterdir():
                if path.suffix in (".csv", ".json", ".md"):
                    content = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
                    path.write_text(content, encoding="utf-8", newline="\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, "Revised native readout not written: " + str(exc) + "\n")
    print(json.dumps({"status": "validated" if args.check_only else "written",
                      "available_endpoints": list(ENDPOINTS),
                      "dice_pct": {n: result["endpoints"][n]["summary"]["dice"]["mean_pct"] for n in ENDPOINTS}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
