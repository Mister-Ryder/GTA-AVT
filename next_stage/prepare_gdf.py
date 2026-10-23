"""Two-phase, TRAIN-only preparation of exact cumulative Gaussian targets.

Probe freezes source bytes, sample coordinates and equal-case TRAIN RMS scale.
Execute requires that completed probe and never reads VAL/TEST image/label data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from gdf import CumulativeRadiusGeometry, GaussianFieldSpec


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".tmp")
    part.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    part.replace(path)


def emit(args, value):
    if not args.quiet:
        print(json.dumps(value, allow_nan=False), flush=True)


def source_map():
    root = Path(__file__).resolve().parent
    return {name: sha256(root / name) for name in ("prepare_gdf.py", "gdf.py")}


def load_protocol(args):
    manifest = read_json(args.manifest)
    train = manifest["splits"]["train"]
    heldout = manifest["splits"]["val"] + manifest["splits"]["test"]
    if len(train) != 36 or len(set(train)) != 36 or set(train) & set(heldout):
        raise ValueError("Require the fixed 36-case disjoint TRAIN partition")
    rows = {r["id"]: r for r in manifest["cases"]}
    if len(rows) != len(manifest["cases"]):
        raise ValueError("Duplicate manifest case ID")
    # Inspect only manifest metadata for held-out cases; never resolve/open them.
    return manifest, train, rows


def spacing(record, manifest):
    value = np.asarray(record.get("spacing_mm", manifest["spacing_mm"]),
                       dtype=np.float64)
    if value.ndim == 0:
        value = np.repeat(value, 3)
    if value.shape != (3,) or not np.isfinite(value).all() or (value <= 0).any():
        raise ValueError("Invalid physical cache spacing")
    return value


def inputs(case_id, record, manifest, cache_root, frozen=None):
    directory = Path(cache_root) / case_id
    paths = {name: directory / name for name in ("mask.npy", "geometry.npz")}
    actual = {name: sha256(path) for name, path in paths.items()}
    expected = record.get("cache_file_sha256", {})
    for name in paths:
        if expected.get(name) != actual[name]:
            raise ValueError("Original manifest SHA mismatch: " + case_id + "/" + name)
    if frozen is not None and actual != frozen["input_file_sha256"]:
        raise ValueError("Frozen probe input changed: " + case_id)
    mask = np.load(paths["mask.npy"], mmap_mode="r", allow_pickle=False)
    if mask.ndim != 3 or mask.dtype != np.uint8:
        raise ValueError("Expected the frozen uint8 3D mask: " + case_id)
    if list(mask.shape) != list(record["shape_zyx"]):
        raise ValueError("Mask header shape differs from source manifest")
    with np.load(paths["geometry.npz"], allow_pickle=False) as stream:
        valid = bool(np.asarray(stream.get("geometry_valid", True)).item())
        geometry = CumulativeRadiusGeometry(stream["points"], stream["radius"],
                                             spacing(record, manifest), valid)
    flat = np.flatnonzero(mask)
    if int(record.get("foreground_voxels", len(flat))) != len(flat):
        raise ValueError("Foreground metadata disagrees with actual mask")
    info = {
        "id": case_id, "shape_zyx": list(mask.shape), "mask_dtype": str(mask.dtype),
        "input_files": {name: str(path.resolve()) for name, path in paths.items()},
        "input_file_sha256": actual,
        "input_file_bytes": {name: path.stat().st_size for name, path in paths.items()},
        "spacing_zyx_mm": geometry.spacing_mm.tolist(),
        "foreground_voxels": int(len(flat)),
        "centerline_points": len(geometry.points_zyx),
        "foreground_geometry_valid": bool(valid and len(geometry.points_zyx)),
        "float16_payload_bytes": int(mask.size * 2),
        "estimated_npy_file_bytes": int(mask.size * 2 + 256),
        "kernel_evaluations_full_foreground": int(len(flat) * len(geometry.points_zyx)),
    }
    return mask, flat, geometry, info


def setup_backend(requested, allow_fallback):
    if requested == "cpu":
        return None, {"requested": "cpu", "actual": "cpu", "dtype": "float64"}
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available is false")
        device = torch.device("cuda:0")
        torch.ones(1, dtype=torch.float64, device=device).sum().item()
        props = torch.cuda.get_device_properties(device)
        return torch, {
            "requested": "cuda", "actual": "cuda", "dtype": "float64",
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "device_index": 0, "device_name": props.name,
            "total_memory_bytes": props.total_memory,
            "capability": [props.major, props.minor],
        }
    except Exception as error:
        if not allow_fallback:
            raise RuntimeError("CUDA probe failed; explicit CPU fallback required") from error
        return None, {
            "requested": "cuda", "actual": "cpu", "dtype": "float64",
            "cpu_fallback_explicitly_allowed": True,
            "cuda_error": type(error).__name__ + ": " + str(error),
        }


def cuda_sum(torch, query_zyx, geometry, query_chunk=2048, center_chunk=1024):
    """Exact all-centre float64 CUDA sum; no nearest/max/truncation."""
    device = torch.device("cuda:0")
    query = torch.as_tensor(np.asarray(query_zyx), dtype=torch.float64, device=device)
    centers = torch.as_tensor(geometry.points_zyx, dtype=torch.float64, device=device)
    scale = torch.as_tensor(geometry.spacing_mm, dtype=torch.float64, device=device)
    sigma2 = torch.as_tensor(geometry.sigma_mm ** 2,
                             dtype=torch.float64, device=device)
    result = torch.zeros(len(query), dtype=torch.float64, device=device)
    with torch.no_grad():
        for qb in range(0, len(query), query_chunk):
            qe = min(qb + query_chunk, len(query))
            value = torch.zeros(qe - qb, dtype=torch.float64, device=device)
            for cb in range(0, len(centers), center_chunk):
                ce = min(cb + center_chunk, len(centers))
                delta = (query[qb:qe, None, :] - centers[None, cb:ce, :]) * scale
                squared = (delta * delta).sum(-1)
                value += torch.exp(-.5 * squared / sigma2[None, cb:ce]).sum(1)
            result[qb:qe] = value
    return result.cpu().numpy()


def probe(args):
    manifest, train, records = load_protocol(args)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    probe_file = Path(args.probe_file)
    if probe_file.exists():
        raise FileExistsError("A probe is immutable; select a fresh probe file")
    torch, backend = setup_backend(args.backend, args.allow_cpu_fallback)
    rows, means = [], []
    started = time.perf_counter()
    for index, case_id in enumerate(train):
        mask, flat, geometry, info = inputs(
            case_id, records[case_id], manifest, args.cache_root)
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, index]))
        sample = np.sort(rng.choice(flat, min(args.sample_per_case, len(flat)),
                                    replace=False)) if len(flat) else flat
        info["sample_foreground_flat_indices"] = sample.tolist()
        if len(sample) and info["foreground_geometry_valid"]:
            coords = np.stack(np.unravel_index(sample, mask.shape), axis=1)
            t0 = time.perf_counter()
            raw = geometry.evaluate_sum(coords)
            cpu_seconds = time.perf_counter() - t0
            logfield = np.log1p(raw)
            info["sample_log1p_raw_sum"] = logfield.tolist()
            info["sample_mean_square_logfield"] = float(np.mean(logfield ** 2))
            means.append(info["sample_mean_square_logfield"])
            info["sample_cpu64_seconds"] = cpu_seconds
            info["sample_kernel_pairs"] = int(len(sample) * len(geometry.points_zyx))
            if torch is not None:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                gpu = cuda_sum(torch, coords, geometry, args.query_chunk,
                               args.center_chunk)
                torch.cuda.synchronize()
                info["sample_cuda64_seconds"] = time.perf_counter() - t0
                info["cpu_cuda_max_abs_error"] = float(np.max(np.abs(raw - gpu)))
                info["cpu_cuda_max_relative_error"] = float(np.max(
                    np.abs(raw - gpu) / np.maximum(np.abs(raw), 1e-12)))
                np.testing.assert_allclose(gpu, raw, rtol=1e-10, atol=1e-10)
        else:
            info["sample_log1p_raw_sum"] = []
            info["sample_mean_square_logfield"] = None
        rows.append(info)
        emit(args, {"event": "probe_case_verified", "case": case_id,
                    "done": index + 1, "total": len(train),
                    "foreground_voxels": len(flat),
                    "centerline_points": len(geometry.points_zyx)})
        del mask, flat, geometry
    if not means:
        raise ValueError("No supported nonempty TRAIN case for normalization")
    field_scale = float(np.sqrt(np.mean(means)))
    if not np.isfinite(field_scale) or field_scale <= 0:
        raise ValueError("Degenerate TRAIN field scale")
    storage = sum(r["estimated_npy_file_bytes"] for r in rows)
    result = {
        "schema": "GTA-AVT-GDF-TRAIN-probe-v1", "status": "probe_complete",
        "complete": True, "mode": "probe_only_no_dense_target_written",
        "training_case_ids": train, "case_count": len(train),
        "heldout_case_files_opened": False,
        "source_code_sha256": source_map(),
        "input_manifest": str(Path(args.manifest).resolve()),
        "input_manifest_sha256": sha256(args.manifest),
        "cache_root_relocation": str(Path(args.cache_root).resolve()),
        "output_dir": str(out.resolve()),
        "definition": "mask*log1p(sum_all_centerline_Gaussians_sigma_native_radius)/fixed_TRAIN_RMS",
        "sigma_scale": 1., "sigma_floor_mm": .001, "kernel_truncation": None,
        "fixed_train_field_scale": field_scale,
        "normalization": {
            "kind": "sqrt_equal_case_mean_of_sample_mean_square_log1p_sum",
            "supported_cases": len(means), "sample_per_case_max": args.sample_per_case,
            "seed": args.seed, "targets_capped_at_one": False,
        },
        "backend": backend, "cuda_query_chunk": args.query_chunk,
        "cuda_center_chunk": args.center_chunk,
        "estimated_float16_npy_bytes": storage,
        "max_output_bytes": args.max_output_bytes,
        "storage_within_requested_budget": storage <= args.max_output_bytes,
        "free_output_filesystem_bytes_at_probe": shutil.disk_usage(out).free,
        "elapsed_seconds": time.perf_counter() - started, "cases": rows,
    }
    write_json(probe_file, result)
    emit(args, {"event": "probe_complete", "probe_file": str(probe_file),
                "field_scale": field_scale, "estimated_bytes": storage,
                "storage_within_budget": result["storage_within_requested_budget"]})
    return result


def execute(args):
    manifest, train, records = load_protocol(args)
    frozen = read_json(args.probe_file)
    if frozen.get("status") != "probe_complete" or frozen.get("complete") is not True:
        raise ValueError("An actual completed probe is required")
    if frozen["training_case_ids"] != train:
        raise ValueError("TRAIN identity changed after probe")
    if frozen["source_code_sha256"] != source_map():
        raise ValueError("Preparation source changed after probe")
    if frozen["input_manifest_sha256"] != sha256(args.manifest):
        raise ValueError("Input manifest bytes changed after probe")
    if frozen["cache_root_relocation"] != str(Path(args.cache_root).resolve()):
        raise ValueError("Cache relocation changed after probe")
    if not frozen["storage_within_requested_budget"]:
        raise ValueError("Probe estimated dense output beyond its fixed storage budget")
    out = Path(args.output_dir).resolve()
    if frozen["output_dir"] != str(out):
        raise ValueError("Output directory changed after probe")
    if (out / "manifest.json").exists() and not args.resume:
        raise FileExistsError("Completed target manifest exists; explicit resume required")
    requested = frozen["backend"]["actual"]
    torch, actual_backend = setup_backend(requested, False)
    probe_sha = sha256(args.probe_file)
    rows, frozen_rows = [], {r["id"]: r for r in frozen["cases"]}
    start_time = time.perf_counter()
    partial_manifest = {
        "schema": "GTA-AVT-GDF-TRAIN-cache-v1", "complete": False,
        "training_case_ids": train, "case_count": len(train), "cases": rows,
        "probe_sha256": probe_sha, "source_code_sha256": source_map(),
        "input_manifest_sha256": frozen["input_manifest_sha256"],
        "fixed_train_field_scale": frozen["fixed_train_field_scale"],
        "definition": frozen["definition"], "backend": actual_backend,
        "heldout_case_files_opened": False,
        "validity_rule": "background_known; foreground_valid_iff_foreground_geometry_valid; exclude_input_padding",
        "dtype": "float16", "normalization": frozen["normalization"],
    }
    for index, case_id in enumerate(train):
        mask, flat, geometry, binding = inputs(
            case_id, records[case_id], manifest, args.cache_root, frozen_rows[case_id])
        case_dir = out / case_id
        case_dir.mkdir(exist_ok=True)
        target_file, audit_file = case_dir / "gdf.npy", case_dir / "audit.json"
        if audit_file.exists():
            if not args.resume:
                raise FileExistsError("Case output exists; explicit resume required")
            row = read_json(audit_file)
            if (row.get("complete") is not True or row["probe_sha256"] != probe_sha
                    or row["input_file_sha256"] != binding["input_file_sha256"]
                    or row["target_file_sha256"] != sha256(target_file)):
                raise ValueError("Resume case is not byte/source/probe consistent")
            existing = np.load(target_file, mmap_mode="r", allow_pickle=False)
            if existing.shape != mask.shape or existing.dtype != np.float16:
                raise ValueError("Resume target header mismatch")
            del existing
        else:
            if target_file.exists():
                raise FileExistsError("Orphan final target without complete case audit")
            temporary = case_dir / "gdf.partial.npy"
            if temporary.exists() and not args.resume:
                raise FileExistsError("Partial file exists; explicit resume required")
            if shutil.disk_usage(out).free < binding["estimated_npy_file_bytes"] + args.min_free_reserve_bytes:
                raise OSError("Insufficient output space plus fixed reserve")
            t0 = time.perf_counter()
            result = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16,
                                               shape=mask.shape)
            result[:] = 0.
            raw_min, raw_max, value_sum, error_square, max_error = None, None, 0., 0., 0.
            if binding["foreground_geometry_valid"]:
                for begin in range(0, len(flat), frozen["cuda_query_chunk"]):
                    positions = flat[begin:begin + frozen["cuda_query_chunk"]]
                    coords = np.stack(np.unravel_index(positions, mask.shape), axis=1)
                    raw = (cuda_sum(torch, coords, geometry,
                                    frozen["cuda_query_chunk"], frozen["cuda_center_chunk"])
                           if torch is not None else geometry.evaluate_sum(coords))
                    values = np.log1p(raw) / frozen["fixed_train_field_scale"]
                    compressed = values.astype(np.float16)
                    if not np.isfinite(compressed).all():
                        raise FloatingPointError("Nonfinite float16 target; no clipping applied")
                    result.reshape(-1)[positions] = compressed
                    raw_min = float(raw.min()) if raw_min is None else min(raw_min, float(raw.min()))
                    raw_max = float(raw.max()) if raw_max is None else max(raw_max, float(raw.max()))
                    value_sum += float(values.sum())
                    error = compressed.astype(np.float64) - values
                    error_square += float(np.dot(error, error))
                    max_error = max(max_error, float(np.abs(error).max()))
            result.flush()
            del result
            temporary.replace(target_file)
            row = dict(binding)
            row.update(
                complete=True, probe_sha256=probe_sha,
                target_file=str(target_file), target_file_bytes=target_file.stat().st_size,
                target_file_sha256=sha256(target_file), dtype="float16",
                fixed_train_field_scale=frozen["fixed_train_field_scale"],
                raw_sum_foreground_min=raw_min, raw_sum_foreground_max=raw_max,
                foreground_normalized_target_mean=(value_sum / len(flat)
                    if len(flat) and binding["foreground_geometry_valid"] else None),
                float16_max_abs_error=max_error,
                float16_foreground_RMSE=(float(np.sqrt(error_square / len(flat)))
                    if len(flat) and binding["foreground_geometry_valid"] else None),
                preparation_seconds=time.perf_counter() - t0,
            )
            write_json(audit_file, row)
        rows.append(row)
        partial_manifest["cases"] = rows
        partial_manifest["completed_case_count"] = len(rows)
        write_json(out / "manifest.partial.json", partial_manifest)
        emit(args, {"event": "target_case_complete", "case": case_id,
                    "done": index + 1, "total": len(train),
                    "target_file_sha256": row["target_file_sha256"]})
        del mask, flat, geometry
    final = dict(partial_manifest)
    final.update(complete=True, status="TRAIN_target_cache_complete",
                 completed_case_count=len(rows),
                 target_file_bytes_total=sum(r["target_file_bytes"] for r in rows),
                 execution_elapsed_seconds=time.perf_counter() - start_time)
    write_json(out / "manifest.json", final)
    emit(args, {"event": "TRAIN_GDF_cache_complete", "manifest": str(out / "manifest.json"),
                "cases": len(rows), "bytes": final["target_file_bytes_total"]})
    return final


def self_check():
    """Real two-phase tiny 36-case fixture with inaccessible held-out paths."""
    with tempfile.TemporaryDirectory(prefix="avt_gdf_prepare_fixture_") as temp:
        root = Path(temp)
        cache, out = root / "cache", root / "out"
        cache.mkdir()
        train = ["TRAIN_%02d" % i for i in range(36)]
        cases = []
        for case_id in train:
            folder = cache / case_id
            folder.mkdir()
            mask = np.zeros((5, 5, 5), np.uint8)
            mask[:, 2, 2] = 1
            np.save(folder / "mask.npy", mask)
            np.savez(folder / "geometry.npz", points=np.argwhere(mask),
                     radius=np.ones(5, np.float32), geometry_valid=np.uint8(1))
            cases.append({"id": case_id, "shape_zyx": list(mask.shape),
                          "foreground_voxels": int(mask.sum()), "spacing_mm": 1.,
                          "cache_dir": "STALE_PATH_MUST_NOT_BE_USED",
                          "cache_file_sha256": {name: sha256(folder / name)
                              for name in ("mask.npy", "geometry.npz")}})
        source = cache / "manifest.json"
        write_json(source, {"spacing_mm": 1., "splits": {
            "train": train, "val": ["VAL_DO_NOT_OPEN"], "test": ["TEST_DO_NOT_OPEN"]},
            "cases": cases + [{"id": name, "cache_dir": "FORBIDDEN_NONEXISTENT"}
                             for name in ("VAL_DO_NOT_OPEN", "TEST_DO_NOT_OPEN")]})
        args = argparse.Namespace(manifest=str(source), cache_root=str(cache),
            output_dir=str(out), probe_file=str(out / "probe.json"),
            backend="cpu", allow_cpu_fallback=False, sample_per_case=3,
            seed=20261023, query_chunk=16, center_chunk=16,
            max_output_bytes=5 * 1024 ** 3, min_free_reserve_bytes=0,
            quiet=True, resume=False)
        p = probe(args)
        assert p["complete"] and p["case_count"] == 36
        assert not list(out.rglob("gdf.npy")), "Probe wrote dense fields"
        target = execute(args)
        assert target["complete"] and target["case_count"] == 36
        args.resume = True
        resumed = execute(args)
        assert resumed["target_file_bytes_total"] == target["target_file_bytes_total"]
        for row in target["cases"]:
            field = np.load(row["target_file"])
            assert field.dtype == np.float16 and np.count_nonzero(field) == 5
        print(json.dumps({"status": "passed",
              "checks": ["probe_writes_no_dense_target", "36_train_only",
                         "source_sha_and_stale_cache_relocation",
                         "fixed_equal_case_RMS", "execute_complete_manifest",
                         "float16_background_zero", "resume_rehashes_outputs",
                         "inaccessible_VAL_TEST_never_opened"]}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest")
    p.add_argument("--cache-root")
    p.add_argument("--output", "--output-dir", dest="output_dir")
    p.add_argument("--probe-file")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--probe-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--self-check", action="store_true")
    p.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--allow-cpu-fallback", action="store_true")
    p.add_argument("--sample-per-case", type=int, default=256)
    p.add_argument("--seed", type=int, default=20261023)
    p.add_argument("--query-chunk", type=int, default=2048)
    p.add_argument("--center-chunk", type=int, default=1024)
    p.add_argument("--max-output-bytes", type=int, default=5 * 1024 ** 3)
    p.add_argument("--min-free-reserve-bytes", type=int, default=512 * 1024 ** 2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    if args.self_check:
        self_check()
        return
    if not (args.probe_only or args.execute):
        p.error("Select --probe-only or --execute")
    if not args.manifest or not args.cache_root or not args.output_dir:
        p.error("--manifest, --cache-root and --output are required")
    if not 1 <= args.sample_per_case <= 256:
        p.error("--sample-per-case must be in [1,256]")
    if args.query_chunk <= 0 or args.center_chunk <= 0:
        p.error("Chunk sizes must be positive")
    args.probe_file = args.probe_file or str(Path(args.output_dir) / "probe.json")
    (probe if args.probe_only else execute)(args)


if __name__ == "__main__":
    main()
