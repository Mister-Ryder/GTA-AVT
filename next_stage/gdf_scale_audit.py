"""TRAIN-only, centerline-node audit of a frozen cumulative GDF target.

Read-only NumPy/stdlib diagnostic: no target generation, fitting, nearest-node
volume assignment, prediction access or performance claim. Default is plan.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

FIXED_SCALE = 2.8707158419291448
BIN_NAMES = ("radius_le_2mm", "radius_2_to_4mm", "radius_4_to_8mm", "radius_gt_8mm")
QUANTILES = (0., .1, .25, .5, .75, .9, .99, 1.)
DEFINITION = "mask*log1p(sum_all_centerline_Gaussians_sigma_native_radius)/fixed_TRAIN_RMS"


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(value, message):
    if not value:
        raise ValueError(message)


def source_name(case_id):
    for name in ("Dongyang", "KiTS", "Rider"):
        if case_id.startswith(name + "_"):
            return name
    return "unknown"


def weighted_quantiles(values, weights):
    """Left inverse weighted empirical CDF; no interpolated fictitious node."""
    if not len(values):
        return None
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights, dtype=np.float64)
    indices = np.searchsorted(cumulative, np.asarray(QUANTILES) * cumulative[-1], side="left")
    indices = np.minimum(indices, len(values) - 1)
    return {str(q): float(values[index]) for q, index in zip(QUANTILES, indices)}


def weighted_stats(radius, target, weights):
    radius, target, weights = (np.asarray(x, np.float64) for x in (radius, target, weights))
    count = len(target)
    if not count:
        return dict(node_support=0, radius_mean_mm=None, target_mean=None,
                    target_mean_square=None, target_sd=None, target_quantiles=None,
                    radius_target_Pearson=None, Pearson_null_reason="no_supported_nodes")
    require(radius.shape == target.shape == weights.shape and np.isfinite(radius).all()
            and np.isfinite(target).all() and np.isfinite(weights).all()
            and (weights > 0.).all(), "Invalid node vectors/weights")
    weight = weights / weights.sum()
    rmean, tmean = float(np.dot(weight, radius)), float(np.dot(weight, target))
    rs, ts = radius - rmean, target - tmean
    rvar, tvar = float(np.dot(weight, rs * rs)), float(np.dot(weight, ts * ts))
    if count < 2:
        correlation, reason = None, "fewer_than_two_nodes"
    elif np.ptp(radius) == 0. or np.ptp(target) == 0. or rvar <= 0. or tvar <= 0.:
        correlation, reason = None, "zero_radius_or_target_variance"
    else:
        correlation = float(np.clip(np.dot(weight, rs * ts) / math.sqrt(rvar * tvar), -1., 1.))
        reason = None
    return dict(node_support=count, radius_mean_mm=rmean, target_mean=tmean,
                target_mean_square=float(np.dot(weight, target * target)),
                target_sd=math.sqrt(max(0., tvar)), target_quantiles=weighted_quantiles(target, weights),
                radius_target_Pearson=correlation, Pearson_null_reason=reason)


def summarize(vectors, bin_index=None):
    """Both measures condition on cases having nodes in the selected bin."""
    groups = []
    for case_id, radius, target in vectors:
        selected = np.ones(len(radius), bool) if bin_index is None else np.searchsorted(
            np.asarray([2., 4., 8.]), radius, side="left") == bin_index
        if selected.any():
            groups.append((case_id, radius[selected], target[selected]))
    if groups:
        radius = np.concatenate([r for _, r, t in groups])
        target = np.concatenate([t for _, r, t in groups])
        case_weight = np.concatenate([np.full(len(t), 1. / len(t)) for _, r, t in groups])
    else:
        radius, target, case_weight = np.empty(0), np.empty(0), np.empty(0)
    return dict(cases_in_scope=len(vectors), cases_with_support=len(groups),
                cases_without_support=len(vectors) - len(groups),
                case_equal=weighted_stats(radius, target, case_weight),
                node_weighted=weighted_stats(radius, target, np.ones(len(target))),
                case_equal_definition="each supported case has total weight1; within its selected bin nodes are equal",
                node_weighted_definition="each selected centerline node has weight1; not vessel length weighting")


def metadata(args):
    root, cache = Path(args.gdf_root).resolve(), Path(args.cache_root).resolve()
    gdf_manifest = Path(args.gdf_manifest).resolve() if args.gdf_manifest else root / "manifest.json"
    manifest = Path(args.manifest).resolve() if args.manifest else cache / "manifest.json"
    producer, inputs = read_json(gdf_manifest), read_json(manifest)
    require(inputs.get("complete") is True, "A complete frozen input manifest is required")
    train = inputs["splits"]["train"]
    heldout = set(inputs["splits"].get("val", []) + inputs["splits"].get("test", []))
    require(len(train) == 36 and len(set(train)) == 36 and not set(train) & heldout,
            "Expected disjoint frozen TRAIN36")
    require(producer.get("complete") is True and producer.get("case_count") == 36
            and producer.get("completed_case_count") == 36
            and producer.get("status") == "TRAIN_target_cache_complete", "Incomplete producer cache")
    require(producer["training_case_ids"] == train and isinstance(producer["cases"], list),
            "Producer TRAIN identities/order differ")
    rows = {row["id"]: row for row in producer["cases"]}
    require(len(rows) == len(producer["cases"]) == 36 and set(rows) == set(train),
            "Producer rows contain missing/duplicate/held-out case")
    require(producer.get("definition") == DEFINITION and producer.get("dtype") == "float16",
            "Unexpected target definition/dtype")
    scale = float(producer["fixed_train_field_scale"])
    require(math.isfinite(scale) and math.isclose(scale, FIXED_SCALE, rel_tol=0., abs_tol=1e-12),
            "Frozen TRAIN field scale differs")
    for case_id in train:
        require(Path(case_id).name == case_id and case_id not in (".", ".."), "Unsafe case ID")
        row = rows[case_id]
        require(row.get("complete") is True and row.get("dtype") == "float16"
                and row.get("fixed_train_field_scale") == scale
                and row.get("probe_sha256") == producer.get("probe_sha256"), "Unbound case target: " + case_id)
        require(Path(row["target_file"]).name == "gdf.npy", "Unexpected target filename")
        require(len(row["input_file_sha256"].get("geometry.npz", "")) == 64,
                "Missing producer geometry SHA")
    return root, cache, gdf_manifest, manifest, producer, inputs, train, rows


def audit(args):
    started = time.time()
    root, cache, gdf_manifest, manifest, producer, inputs, train, rows = metadata(args)
    output = Path(args.output).resolve()
    for protected in (root, cache):
        require(output != protected and protected not in output.parents,
                "Output must stay outside immutable GDF/input-cache directories")
    require(not output.exists(), "Immutable output already exists")
    vectors, cases, geometry_bytes = [], [], 0
    for ordinal, case_id in enumerate(train):
        row = rows[case_id]
        geometry_file = cache / case_id / "geometry.npz"
        actual_sha = sha256(geometry_file)
        require(actual_sha == row["input_file_sha256"]["geometry.npz"], case_id + ": producer geometry SHA differs")
        geometry_bytes += geometry_file.stat().st_size
        with np.load(geometry_file, allow_pickle=False) as geometry:
            points = np.asarray(geometry["points"])
            radius = np.asarray(geometry["radius"], np.float64)
            geometry_valid = bool(np.asarray(geometry.get("geometry_valid", True)).item())
        require(points.ndim == 2 and points.shape[1] == 3 and radius.shape == (len(points),),
                case_id + ": bad centerline/radius shape")
        require(np.isfinite(points).all() and np.equal(points, np.round(points)).all()
                and np.isfinite(radius).all() and (radius > 0.).all(), case_id + ": invalid geometry values")
        points = points.astype(np.int64)
        require(len(np.unique(points, axis=0)) == len(points), case_id + ": duplicate centerline nodes")
        target_file = root / case_id / "gdf.npy"
        target_file.resolve().relative_to(root)
        field = np.load(target_file, mmap_mode="r", allow_pickle=False)
        mask = np.load(cache / case_id / "mask.npy", mmap_mode="r", allow_pickle=False)
        shape = tuple(row["shape_zyx"])
        require(field.dtype == np.float16 and field.shape == shape and mask.shape == shape
                and mask.dtype == np.uint8 and len(shape) == 3, case_id + ": target/mask header differs")
        require(target_file.stat().st_size == row["target_file_bytes"], case_id + ": target file size differs")
        require(len(points) == row["centerline_points"], case_id + ": node count differs from producer")
        require(not len(points) or ((points >= 0) & (points < np.asarray(shape))).all(),
                case_id + ": centerline point outside target grid")
        foreground_valid = bool(geometry_valid and len(points))
        require(foreground_valid == bool(row["foreground_geometry_valid"]), case_id + ": producer geometry validity differs")
        if len(points):
            require((mask[tuple(points.T)] > 0).all(), case_id + ": centerline node outside foreground")
        values = np.asarray(field[tuple(points.T)], np.float64) if foreground_valid else np.empty(0)
        require(np.isfinite(values).all() and (values >= 0.).all(), case_id + ": selected target invalid")
        # No full mask scan, target rehash, Gaussian recomputation, or nearest
        # point assignment occurs. Only complete geometry bytes are rehashed.
        selected_radius = radius if foreground_valid else np.empty(0)
        vectors.append((case_id, selected_radius, values))
        case = dict(id=case_id, source=source_name(case_id), geometry_sha256=actual_sha,
                    geometry_node_count=len(points), geometry_valid=geometry_valid,
                    selected_node_count=len(values), excluded_reason=(None if foreground_valid else
                        ("no_centerline_nodes" if not len(points) else "geometry_marked_unknown")),
                    shape_zyx=list(shape), target_file_sha256_declared=row["target_file_sha256"],
                    target_fully_rehashed=False,
                    all_nodes=weighted_stats(selected_radius, values, np.ones(len(values))),
                    bins={})
        for index, name in enumerate(BIN_NAMES):
            selected = np.searchsorted(np.asarray([2., 4., 8.]), selected_radius, side="left") == index
            case["bins"][name] = weighted_stats(selected_radius[selected], values[selected], np.ones(int(selected.sum())))
        cases.append(case)
        if not args.quiet:
            print(json.dumps(dict(event="TRAIN_centerline_GDF_scale_audited", case=case_id,
                                  done=ordinal + 1, total=36, nodes=len(values))), flush=True)
        for array in (field, mask):
            array._mmap.close()
        del field, mask
    groups = {"all_TRAIN": vectors}
    for source in sorted({case["source"] for case in cases}):
        groups[source] = [item for item in vectors if source_name(item[0]) == source]
    summary = {name: dict(all_nodes=summarize(items), bins={label: summarize(items, i)
                           for i, label in enumerate(BIN_NAMES)}) for name, items in groups.items()}
    result = dict(
        schema="GTA-AVT-frozen-GDF-centerline-scale-audit-v1", status="complete", TRAIN_case_count=36,
        test_used=False, heldout_case_files_opened=False, parameters_fitted=False,
        field_modified=False, training_recipe_modified=False,
        source_sha256=sha256(__file__), gdf_manifest_sha256=sha256(gdf_manifest),
        input_manifest_sha256_actual=sha256(manifest), input_manifest_sha256_producer=producer["input_manifest_sha256"],
        geometry_binding="actual complete geometry-file SHA matches producer for every TRAIN case; explicit cache-root relocation",
        geometry_files_fully_rehashed=36, geometry_bytes_rehashed=geometry_bytes,
        target_integrity_scope="producer SHA declared only; actual header, file size, selected node values checked; no full target rehash",
        mask_read_scope="mmap header plus centerline point values; no full mask read/hash",
        logical_selected_target_values=sum(case["selected_node_count"] for case in cases),
        physical_IO_bytes_measured=False, fixed_train_field_scale=producer["fixed_train_field_scale"],
        target_definition=DEFINITION, radius_bins_mm=[2., 4., 8.], bin_boundary_rule="upper edge included (side=left)",
        quantile_definition="left inverse weighted empirical CDF; minimum atq0, maximum atq1",
        squared_target_interpretation="E[target^2] at centerline nodes only; not loss or whole-volume energy",
        straight_axis_reference="Uniform infinite straight centerline, constant radius r=sigma, spacing h, h<<r: S~sqrt(2*pi)*r/h; far from endpoints, no other branches. This is a density/radius dependence approximation, not a vessel-tree identity.",
        inference_limits=["node samples are not independent patients", "node weighting is not physical vessel-length weighting",
                          "radius association may reflect seed density, adjacent anatomy and branching; not a causal test",
                          "centerline-only statistics cannot quantify whole-voxel loss contribution or GDF segmentation benefit"],
        cases=cases, summaries=summary, seconds=time.time() - started)
    payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(payload)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdf-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--gdf-manifest")
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    _, _, gdf_manifest, manifest, producer, inputs, train, _ = metadata(args)
    if not args.execute:
        print(json.dumps(dict(execute=False, gdf_root=args.gdf_root, cache_root=args.cache_root,
                              gdf_manifest=str(gdf_manifest), input_manifest=str(manifest), TRAIN_case_count=len(train),
                              TRAIN_ids=train, output=args.output, arrays_opened=False,
                              fixed_train_field_scale=producer["fixed_train_field_scale"], test_used=False)), flush=True)
        return
    if not args.output:
        parser.error("--execute requires a new immutable --output JSON outside both cache roots")
    result = audit(args)
    print(json.dumps(dict(status=result["status"], TRAIN_case_count=36,
                          selected_nodes=result["logical_selected_target_values"], output=args.output,
                          heldout_case_files_opened=False, field_modified=False)), flush=True)


if __name__ == "__main__":
    main()
