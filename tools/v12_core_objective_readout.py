"""Summarize fixed single-seed native VAL8 JSON, without inference or training.

Inputs are frozen evaluate() metrics.json or the neutral F readout.json. Missing
optional endpoints remain missing. Optional --receipt ENDPOINT=JSON accepts a
paired plan, a {"plan": ...} wrapper, or a normalized receipt containing seed
and base_checkpoint_sha256 (checkpoint_sha256 for F). No input paths or arbitrary
receipt text are copied to the fresh JSON/CSV output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ENDPOINTS = ("F", "C0", "AR", "C0_GDF", "AR_GDF")
CASE_IDS = ("Dongyang_D2", "Dongyang_D5", "KiTS_K13", "KiTS_K15",
            "KiTS_K18", "KiTS_K8", "Rider_R10", "Rider_R9")
METRICS = ("dice", "precision", "recall", "cldice", "fine_centerline_recall")
CERTIFICATES = {"direct_grid_match", "certified_CT_float32_header"}
COUNTS = ("tp_voxels", "fp_voxels", "fn_voxels", "predicted_skeleton_voxels",
          "target_skeleton_voxels", "fine_centerline_supported_voxels")
VOLUMES = ("predicted_volume_ml", "target_volume_ml", "fp_volume_ml", "fn_volume_ml")
CONTRASTS = {
    "AR-C0": {"AR": 1, "C0": -1},
    "AR_GDF-C0_GDF": {"AR_GDF": 1, "C0_GDF": -1},
    "C0_GDF-C0": {"C0_GDF": 1, "C0": -1},
    "AR_GDF-AR": {"AR_GDF": 1, "AR": -1},
    "interaction": {"AR_GDF": 1, "C0_GDF": -1, "AR": -1, "C0": 1},
    **{name + "-F": {name: 1, "F": -1} for name in ENDPOINTS[1:]},
}


def number(value, label, *, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(label + " must be a finite number")
    return float(value)


def count(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(label + " must be a nonnegative integer")
    return value


def close(actual, expected, label):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(label + " has inconsistent undefined support")
    elif not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(label + " is inconsistent with reported counts")


def validate_report(report, endpoint):
    if not isinstance(report, dict) or str(report.get("split", "")).upper() != "VAL":
        raise ValueError(endpoint + " requires an explicit VAL report")
    raw = report.get("per_case")
    if not isinstance(raw, list) or len(raw) != 8:
        raise ValueError(endpoint + " requires exactly eight per_case rows")
    if any(not isinstance(row, dict) for row in raw):
        raise ValueError(endpoint + " per_case rows must be objects")
    if any(not isinstance(row.get("case_id"), str) for row in raw) or sorted(row.get("case_id", "") for row in raw) != sorted(CASE_IDS):
        raise ValueError(endpoint + " case IDs must equal the fixed unique VAL8 IDs")
    rows = {}
    for original in raw:
        case = original["case_id"]
        label = endpoint + "/" + case
        row = {"case_id": case, "grid_certification": original.get("grid_certification")}
        if row["grid_certification"] not in CERTIFICATES:
            raise ValueError(label + " lacks an accepted native-grid certificate")
        for key in COUNTS:
            row[key] = count(original.get(key), label + "/" + key)
        for key in VOLUMES:
            row[key] = number(original.get(key), label + "/" + key)
            if row[key] < 0:
                raise ValueError(label + "/" + key + " must be nonnegative")
        for key in METRICS + ("centerline_precision", "centerline_recall"):
            row[key] = number(original.get(key), label + "/" + key,
                              nullable=key not in ("dice", "cldice"))
            if row[key] is not None and not 0 <= row[key] <= 1:
                raise ValueError(label + "/" + key + " must be a fraction in [0,1]")
        row["fine_radius_threshold_mm"] = number(original.get("fine_radius_threshold_mm"), label + "/fine_radius_threshold_mm")
        if row["fine_radius_threshold_mm"] != 2.0:
            raise ValueError(label + " requires the frozen 2 mm fine radius threshold")
        tp, fp, fn = (row[key] for key in COUNTS[:3])
        predicted, target = tp + fp, tp + fn
        close(row["dice"], 2 * tp / (predicted + target) if predicted + target else 1., label + "/dice")
        close(row["precision"], tp / predicted if predicted else None, label + "/precision")
        close(row["recall"], tp / target if target else None, label + "/recall")
        for metric, support in (("centerline_precision", "predicted_skeleton_voxels"),
                                ("centerline_recall", "target_skeleton_voxels"),
                                ("fine_centerline_recall", "fine_centerline_supported_voxels")):
            if (row[metric] is None) != (row[support] == 0):
                raise ValueError(label + "/" + metric + " disagrees with its support")
        if row["fine_centerline_supported_voxels"] > row["target_skeleton_voxels"]:
            raise ValueError(label + " fine support exceeds GT skeleton support")
        cp, cr = row["centerline_precision"], row["centerline_recall"]
        cl = 2 * cp * cr / (cp + cr) if cp is not None and cr is not None and cp + cr else (1. if not predicted and not target else 0.)
        close(row["cldice"], cl, label + "/cldice")
        if target:
            voxel_ml = row["target_volume_ml"] / target
            if voxel_ml <= 0:
                raise ValueError(label + " positive GT support requires positive volume")
            for key, voxels in (("predicted_volume_ml", predicted), ("fp_volume_ml", fp), ("fn_volume_ml", fn)):
                close(row[key], voxels * voxel_ml, label + "/" + key)
        rows[case] = row
    return rows


def gt_support(row):
    return {"target_voxels": row["tp_voxels"] + row["fn_voxels"],
            **{key: row[key] for key in ("target_volume_ml", "target_skeleton_voxels",
               "fine_centerline_supported_voxels", "fine_radius_threshold_mm")}}


def summarize_values(values, *, delta=False):
    defined = sorted((float(value) * 100., case) for case, value in values.items() if value is not None)
    prefix = "delta_" if delta else ""
    unit = "pp" if delta else "pct"
    result = {"case_count": len(values), "defined_cases": len(defined),
              "undefined_case_ids": [case for case, value in values.items() if value is None]}
    scores = [value for value, _ in defined]
    worst_n = math.ceil(.2 * len(scores))
    result.update({prefix + "mean_" + unit: statistics.fmean(scores) if scores else None,
                   prefix + "sd_population_pp": statistics.pstdev(scores) if scores else None,
                   prefix + "minimum_" + unit: min(scores) if scores else None,
                   prefix + "worst_ceil20_mean_" + unit: statistics.fmean(scores[:worst_n]) if scores else None,
                   "worst_ceil20_count": worst_n,
                   "minimum_case_ids": [case for value, case in defined if value == scores[0]] if scores else [],
                   "worst_ceil20_case_ids": [case for _, case in defined[:worst_n]]})
    if delta:
        result.update(positive_cases=sum(value > 0 for value in scores),
                      negative_cases=sum(value < 0 for value in scores),
                      zero_cases=sum(value == 0 for value in scores))
    return result


def digest(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
        raise ValueError(label + " requires a SHA256 hex digest")
    return value.lower()


def provenance(reports, receipts, expected_seed):
    """Validate supplied claims; absence remains unverified, never inferred."""
    observed = {}
    for endpoint, receipt in receipts.items():
        if endpoint not in ENDPOINTS or endpoint not in reports:
            raise ValueError("Receipt endpoint requires its available native metrics")
        receipt = receipt.get("plan", receipt) if isinstance(receipt, dict) else receipt
        if not isinstance(receipt, dict):
            raise ValueError(endpoint + " receipt must be an object")
        common = receipt.get("common", {})
        if not isinstance(common, dict):
            raise ValueError(endpoint + " receipt common must be an object")
        seed = receipt.get("seed", common.get("seed"))
        if type(seed) is not int or seed != expected_seed or common.get("seed", seed) != seed:
            raise ValueError(endpoint + " receipt does not match the fixed single seed")
        identity = receipt.get("variant", receipt.get("endpoint", endpoint))
        if (endpoint == "F" and "variant" not in receipt and
                receipt.get("schema") == "GTA-AVT-v1.2-neutral-base-native-readout-1"):
            identity = "F"  # The frozen F plan's endpoint field is descriptive prose.
        if identity != endpoint:
            raise ValueError(endpoint + " receipt endpoint identity differs")
        parent = receipt.get("base_checkpoint_sha256", receipt.get("parent_sha256"))
        if endpoint == "F":
            parent = receipt.get("checkpoint_sha256", parent)
        row = {"seed": seed, "base_checkpoint_sha256": digest(parent, endpoint + " parent")}
        for key in ("source_manifest_sha256", "predictor_state_sha256", "prefix_plan_sha256"):
            if key in receipt:
                row[key] = digest(receipt[key], endpoint + " " + key)
        if "evaluation_endpoint" in receipt and receipt["evaluation_endpoint"] != "last":
            raise ValueError(endpoint + " receipt must use the final last endpoint")
        if "tail_successful_updates" in receipt and receipt["tail_successful_updates"] != 42000:
            raise ValueError(endpoint + " receipt must retain the 42000-update tail")
        observed[endpoint] = row
    f_hash = reports["F"].get("checkpoint_sha256")
    if f_hash is not None:
        f_hash = digest(f_hash, "F checkpoint")
    if observed and f_hash is None and "F" not in observed:
        raise ValueError("Supplied tail receipts require an F checkpoint SHA256 anchor")
    parent_hashes = {row["base_checkpoint_sha256"] for row in observed.values()}
    if f_hash is not None:
        parent_hashes.add(f_hash)
    if len(parent_hashes) > 1:
        raise ValueError("Supplied receipts and F do not share one parent checkpoint")
    for key in ("source_manifest_sha256", "predictor_state_sha256", "prefix_plan_sha256"):
        if len({row[key] for row in observed.values() if key in row}) > 1:
            raise ValueError("Supplied receipts disagree on " + key)
    if reports["F"].get("checkpoint_unchanged") is False or reports["F"].get("new_training_updates", 0) != 0:
        raise ValueError("F readout must preserve its parent and execute zero training updates")
    return {"planned_seed": expected_seed, "observed_seed_count": 1 if observed else 0,
            "seed_stability_assessed": False, "base_checkpoint_sha256": next(iter(parent_hashes), None),
            "receipt_endpoints": list(observed), "available_endpoints_without_receipts": [name for name in reports if name not in observed],
            "supplied_receipt_seed_and_parent_matching": "validated" if observed else "not_supplied",
            "all_final_endpoint_receipts_supplied": all(name in observed for name in ENDPOINTS[1:]),
            "receipts": observed}


def summarize(reports, receipts=None, *, expected_seed=20261032):
    if "F" not in reports or set(reports) - set(ENDPOINTS):
        raise ValueError("F is required and only fixed core endpoints are accepted")
    rows = {name: validate_report(report, name) for name, report in reports.items()}
    reference = {case: gt_support(rows["F"][case]) for case in CASE_IDS}
    for endpoint, cases in rows.items():
        for case in CASE_IDS:
            support = gt_support(cases[case])
            for key, value in reference[case].items():
                actual = support[key]
                if key == "target_volume_ml":
                    close(actual, value, endpoint + "/" + case + "/GT support")
                elif actual != value:
                    raise ValueError(endpoint + "/" + case + " GT support differs from F")
    report = {"schema": "GTA-AVT-v1.2-core-objective-readout-1", "split": "VAL", "case_ids": list(CASE_IDS),
              "available_endpoints": [name for name in ENDPOINTS if name in rows],
              "missing_endpoints": [name for name in ENDPOINTS if name not in rows],
              "provenance": provenance(reports, receipts or {}, expected_seed),
              "validation": {"fixed_unique_VAL8_ids": True, "accepted_native_grid_certificates": True,
                             "reported_native_GT_support_equal": True, "GT_label_file_hashes_checked": False,
                             "segmentation_metric_count_arithmetic_checked": True},
              "definitions": {
                  "scores": "Input fractions; endpoint output percentages; all SDs and paired differences in percentage points.",
                  "aggregation": "Each defined case has equal weight; population SD uses divisor n; nulls are excluded and listed.",
                  "worst_ceil20": "Mean of the lowest ceil(0.2*n_defined) scores; selected independently per metric or paired-delta vector.",
                  "paired_delta": "Per-case linear contrast first, then case-equal summary; all terms must be defined for that case.",
                  "interaction": "(AR_GDF-C0_GDF)-(AR-C0), computed per case.",
                  "native_GT_support": "Reported GT voxel count, volume, skeleton count, fine support and radius agree; this is not voxelwise label-hash verification.",
                  "centerline": "Lee skeleton voxel overlap on each native grid; not physical branch length.",
                  "fine_centerline_recall": "Recall on GT native Lee skeleton sites with native physical EDT radius <=2 mm; zero support yields null.",
                  "F_to_final": "Subsequent training pipeline difference from selected F, not a budget-matched isolated component ablation.",
                  "dispersion_scope": "Across-case dispersion within the fixed seed; seed stability is not measured.",
                  "missing": "No metric rows or deltas are generated for an absent endpoint.",
              }, "native_GT_support": [{"case_id": case, **reference[case]} for case in CASE_IDS],
              "endpoints": {}, "paired_contrasts": {}}
    for name in ENDPOINTS:
        entry = {"status": "available" if name in rows else "missing"}
        if name in rows:
            entry["summary"] = {metric: summarize_values({case: rows[name][case][metric] for case in CASE_IDS}) for metric in METRICS}
            entry["per_case"] = [{"case_id": case,
                **{metric + "_pct": None if rows[name][case][metric] is None else rows[name][case][metric] * 100. for metric in METRICS},
                **{key: rows[name][case][key] for key in COUNTS + VOLUMES + ("grid_certification",)}} for case in CASE_IDS]
        report["endpoints"][name] = entry
    for name, coefficients in CONTRASTS.items():
        missing = [endpoint for endpoint in coefficients if endpoint not in rows]
        entry = {"status": "missing" if missing else "available", "coefficients": coefficients,
                 "comparison_scope": "subsequent_training_pipeline_difference" if "F" in coefficients else "fixed_seed_2x2_contrast",
                 "missing_endpoints": missing}
        if not missing:
            deltas = {case: {metric: None if any(rows[endpoint][case][metric] is None for endpoint in coefficients)
                      else math.fsum(weight * rows[endpoint][case][metric] for endpoint, weight in coefficients.items())
                      for metric in METRICS} for case in CASE_IDS}
            entry["summary"] = {metric: summarize_values({case: deltas[case][metric] for case in CASE_IDS}, delta=True) for metric in METRICS}
            entry["per_case"] = [{"case_id": case, **{metric + "_delta_pp": None if deltas[case][metric] is None else deltas[case][metric] * 100.
                                  for metric in METRICS}} for case in CASE_IDS]
        report["paired_contrasts"][name] = entry
    return report


def write_csv(path, records, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in record.items()})


def export(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "objective_readout.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for group, id_key, summary_file, case_file, delta in (
        ("endpoints", "endpoint", "endpoint_summary.csv", "endpoint_per_case.csv", False),
        ("paired_contrasts", "contrast", "paired_summary.csv", "paired_per_case.csv", True)):
        summaries, cases = [], []
        for name, entry in report[group].items():
            if entry["status"] == "missing":
                summaries.append({id_key: name, "status": "missing"})
            else:
                summaries.extend({id_key: name, "status": "available", "metric": metric, **values} for metric, values in entry["summary"].items())
                cases.extend({id_key: name, **row} for row in entry["per_case"])
        fields = [id_key, "status", "metric", "case_count", "defined_cases", "undefined_case_ids"]
        fields += (["delta_mean_pp", "delta_sd_population_pp", "delta_minimum_pp", "delta_worst_ceil20_mean_pp"] if delta else
                   ["mean_pct", "sd_population_pp", "minimum_pct", "worst_ceil20_mean_pct"])
        fields += ["worst_ceil20_count", "minimum_case_ids", "worst_ceil20_case_ids"]
        if delta:
            fields += ["positive_cases", "negative_cases", "zero_cases"]
        write_csv(output / summary_file, summaries, fields)
        case_fields = [id_key, "case_id"] + [metric + ("_delta_pp" if delta else "_pct") for metric in METRICS]
        if not delta:
            case_fields += list(COUNTS + VOLUMES + ("grid_certification",))
        write_csv(output / case_file, cases, case_fields)


def load_json(path, label):
    try:
        data = Path(path).read_bytes()
        return json.loads(data.decode("utf-8-sig")), hashlib.sha256(data).hexdigest()
    except (OSError, ValueError) as exc:
        raise ValueError(label + " could not be read as JSON") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ENDPOINTS:
        parser.add_argument("--" + name.lower().replace("_", "-"), dest=name, required=name == "F")
    parser.add_argument("--receipt", action="append", default=[], metavar="ENDPOINT=JSON")
    parser.add_argument("--expected-seed", type=int, default=20261032)
    parser.add_argument("--output", required=True, help="Fresh output directory")
    args = parser.parse_args()
    try:
        reports, hashes, receipts, input_status = {}, {}, {}, {}
        for name in ENDPOINTS:
            path = getattr(args, name)
            input_status[name] = "not_supplied" if path is None else "input_file_not_found" if not Path(path).is_file() else "read"
            if path is not None and (name == "F" or Path(path).is_file()):
                reports[name], hashes[name] = load_json(path, name)
        for value in args.receipt:
            name, separator, path = value.partition("=")
            if not separator or name not in ENDPOINTS or name in receipts:
                raise ValueError("Receipts require unique ENDPOINT=JSON entries")
            receipts[name], _ = load_json(path, name + " receipt")
        result = summarize(reports, receipts, expected_seed=args.expected_seed)
        result["input_JSON_sha256"] = hashes
        result["endpoint_input_status"] = input_status
        export(result, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, "error: " + (str(exc) if isinstance(exc, ValueError) else "fresh output directory could not be created or written") + "\n")
    print(json.dumps({"available_endpoints": result["available_endpoints"], "missing_endpoints": result["missing_endpoints"]}))


if __name__ == "__main__":
    main()
