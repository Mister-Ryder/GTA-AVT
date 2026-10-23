"""Independent sampler parity QA using only synthetic cases and frozen data.py.

No Torch, study CT, study GT, optimizer or GPU execution is used. The new sampler
is compared byte-for-byte with the old sampler, not with a rewritten sampler.
This QA can live outside the frozen training packet; pass --project explicitly.
"""
import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import time


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same_array(left, right, name):
    import numpy as np
    left, right = np.asarray(left), np.asarray(right)
    if (left.shape != right.shape or left.dtype != right.dtype
            or left.tobytes(order="C") != right.tobytes(order="C")):
        raise AssertionError("Frozen/new sampler byte mismatch: " + name)


def _close_data(item):
    for values in item.opened.values():
        for name in ("image", "mask"):
            array = values.get(name)
            handle = getattr(array, "_mmap", None)
            if handle is not None:
                handle.close()
    item.opened.clear()
    for array in getattr(item, "gdf_opened", {}).values():
        handle = getattr(array, "_mmap", None)
        if handle is not None:
            handle.close()
    if hasattr(item, "gdf_opened"):
        item.gdf_opened.clear()


def _fixture(root, case_id, shape, geometry_valid=True):
    import numpy as np
    directory = root / case_id
    cache, ones_cache, field = directory / "cache", directory / "ones", directory / "field"
    for path in (cache, ones_cache, field):
        path.mkdir(parents=True)
    # Unique direction-dependent intensity and disconnected non-symmetric mask.
    z, y, x = np.indices(shape, dtype=np.int16)
    image = (z * 2 + y * 3 + x * 5 - 120).astype(np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[7:18, 11:19, 13:29] = 1
    mask[shape[0] - 22:shape[0] - 12, 37:53, shape[2] - 31:shape[2] - 19] = 1
    mask[31:44, shape[1] - 24:shape[1] - 15, 41:47] = 1
    points = np.asarray([[12, 14, 18], [14, 15, 21],
                         [shape[0] - 17, 43, shape[2] - 25],
                         [36, shape[1] - 20, 44]], dtype=np.int32)
    geometry = dict(points=points, radius=np.asarray([1.5, 2., 5., 9.], np.float32),
                    structure=np.asarray([1, 2, 255, 1], np.uint8),
                    geometry_valid=np.asarray(geometry_valid))
    for destination, source_mask in ((cache, mask), (ones_cache, np.ones(shape, np.uint8))):
        np.save(destination / "image.npy", image)
        np.save(destination / "mask.npy", source_mask)
        np.savez(destination / "geometry.npz", **geometry)
    np.save(field / (case_id + ".npy"), mask.astype(np.float32))
    record = dict(id=case_id, cache_dir=str(cache), crop_start_zyx=[3, 5, 7],
                  full_shape_zyx=(np.asarray(shape) + [15, 25, 35]).tolist())
    forbidden = root / "TEST_MUST_NOT_OPEN"
    heldout = dict(id="TEST_SENTINEL", cache_dir=str(forbidden),
                   crop_start_zyx=[0, 0, 0], full_shape_zyx=[128, 128, 128])
    val = dict(heldout, id="VAL_METADATA_ONLY")
    manifest = dict(complete=True, cases=[record, val, heldout],
                    splits=dict(train=[case_id], val=[val["id"]], test=[heldout["id"]]))
    real_manifest = directory / "manifest.json"
    _write_json(real_manifest, manifest)
    ones_manifest_value = json.loads(json.dumps(manifest))
    ones_manifest_value["cases"][0]["cache_dir"] = str(ones_cache)
    ones_manifest = directory / "ones_manifest.json"
    _write_json(ones_manifest, ones_manifest_value)
    _write_json(field / "manifest.json", dict(
        complete=True, cases=[dict(id=case_id, target_file=case_id + ".npy",
                                  foreground_geometry_valid=geometry_valid)]))
    return real_manifest, ones_manifest, field, forbidden


def run_checks(project):
    import numpy as np
    project = Path(project).resolve()
    code, next_dir = project / "code", project / "next_stage"
    for path in (code, next_dir):
        sys.path.insert(0, str(path))
    frozen = importlib.import_module("data")
    candidate = importlib.import_module("next_data")
    if Path(frozen.__file__).resolve() != (code / "data.py").resolve():
        raise ValueError("Unexpected old sampler import")
    if Path(candidate.__file__).resolve() != (next_dir / "next_data.py").resolve():
        raise ValueError("Unexpected candidate sampler import")
    tracker = dict(active=False, forbidden=None, forbidden_attempts=0, file_open_events=0)

    def audit(event, args):
        if event != "open" or not tracker["active"] or not args:
            return
        name = args[0]
        if not isinstance(name, (str, bytes)):
            return
        if isinstance(name, bytes):
            name = name.decode(sys.getfilesystemencoding(), errors="replace")
        resolved = str(Path(name).resolve())
        tracker["file_open_events"] += 1
        if resolved.startswith(tracker["forbidden"]):
            tracker["forbidden_attempts"] += 1
            raise AssertionError("Attempt to open a synthetic TEST sentinel file")

    # Python audit hooks are process-wide; deactivate before returning.
    sys.addaudithook(audit)
    counts = dict(samples=0, padded_samples=0, foreground_unknown_samples=0)
    variants = set()
    fixture_rows = []
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="gta_avt_sampler_qa_") as temporary:
        root = Path(temporary)
        fixtures = [
            ("ASYMMETRIC_PAD", (93, 137, 117), True),
            ("ASYMMETRIC_FULL", (145, 139, 141), True),
            ("FOREGROUND_UNKNOWN", (91, 135, 119), False)]
        for case_id, shape, geometry_valid in fixtures:
            real_path, ones_path, field, forbidden = _fixture(root, case_id, shape, geometry_valid)
            old = frozen.CachedData(real_path)
            old_support = frozen.CachedData(ones_path)
            new = candidate.StageData(real_path, gdf_root=field)
            no_field = candidate.StageData(real_path, gdf_root=None)
            tracker.update(active=True, forbidden=str(forbidden.resolve()))
            try:
                if not isinstance(new.gdf_record["cases"], list):
                    raise AssertionError("The tested field manifest is not a cases list")
                try:
                    new.gdf_case("TEST_SENTINEL")
                except ValueError:
                    pass
                else:
                    raise AssertionError("TEST GDF access was accepted")
                for seed in (17, 20261023):
                    for step in (0, 1, 4, 11):
                        reference = old.sample(seed, step, 0, 128, .5)
                        support = old_support.sample(seed, step, 0, 128, .5)["mask"]
                        actual = new.sample(seed, step, 0, 128, .5)
                        without_field = no_field.sample(seed, step, 0, 128, .5)
                        for name in ("image", "mask", "scale", "structure",
                                     "attribute_valid", "coordinate"):
                            _same_array(reference[name], actual[name], name)
                            _same_array(reference[name], without_field[name], "without_field_" + name)
                        if actual["case_id"] != reference["case_id"] or actual["start"] != reference["start"]:
                            raise AssertionError("Sample identity changed")
                        _same_array(actual["gdf_target"], reference["mask"], "target_flip_rotation_alignment")
                        expected_valid = support if geometry_valid else support * (1. - reference["mask"])
                        _same_array(actual["gdf_valid"], expected_valid, "padding_and_geometry_validity")
                        if np.any(actual["gdf_valid"][support == 0.] != 0.):
                            raise AssertionError("Input padding was supervised")
                        if not geometry_valid:
                            if np.any(actual["gdf_valid"][reference["mask"] > 0.] != 0.):
                                raise AssertionError("Unknown geometry foreground was supervised")
                            if not np.all(actual["gdf_valid"][(support > 0.) & (reference["mask"] == 0.)] == 1.):
                                raise AssertionError("Known background incorrectly masked")
                            counts["foreground_unknown_samples"] += 1
                        if np.any(without_field["gdf_target"]) or np.any(without_field["gdf_valid"]):
                            raise AssertionError("No-field placeholders are not zero/invalid")
                        variants.add(hashlib.sha256(actual["mask"].tobytes()).hexdigest())
                        counts["samples"] += 1
                        counts["padded_samples"] += int(np.any(support == 0.))
                # Batch consumes precisely the old sampler's sequence as well.
                expected_batch = frozen.batch(old, 31, 7, 2, 128, .5)
                actual_batch = candidate.batch(new, 31, 7, 2, 128, .5)
                for name in expected_batch:
                    _same_array(expected_batch[name], actual_batch[name], "batch_" + name)
                _same_array(actual_batch["gdf_target"], actual_batch["mask"], "batch_target_alignment")
                fixture_rows.append(dict(case_id=case_id, shape=list(shape),
                                         geometry_valid=geometry_valid, sample_parity=True))
            finally:
                tracker["active"] = False
                for item in (old, old_support, new, no_field):
                    _close_data(item)
                gc.collect()
        # Schema failures must be rejected before any training sample is drawn.
        real_path, _, field, _ = _fixture(root, "SCHEMA", (91, 133, 115), True)
        field_manifest = field / "manifest.json"
        original = json.loads(field_manifest.read_text(encoding="utf-8"))
        schema_rejections = 0
        for malformed in (
                dict(original, cases={original["cases"][0]["id"]: original["cases"][0]}),
                dict(original, cases=original["cases"] * 2),
                dict(original, cases=[]),
                dict(original, complete=False)):
            _write_json(field_manifest, malformed)
            try:
                candidate.StageData(real_path, field)
            except ValueError:
                schema_rejections += 1
            else:
                raise AssertionError("Malformed/incomplete field manifest was accepted")
    if tracker["forbidden_attempts"] or counts["samples"] != 24 or len(variants) < 6:
        raise AssertionError("Coverage/TEST sentinel contract failed")
    return dict(
        status="passed", schema="GTA-AVT-stage-sampler-QA-v1",
        seconds=time.time() - started, synthetic_only=True, patch_size=128,
        frozen_reference="code/data.py CachedData.sample and batch",
        source_sha256={"code/data.py": _sha(code / "data.py"),
                       "next_stage/next_data.py": _sha(next_dir / "next_data.py")},
        counts=counts, distinct_augmented_mask_hashes=len(variants),
        fixture_cases=fixture_rows, field_manifest_schema_rejections=schema_rejections,
        test_file_open_attempts=tracker["forbidden_attempts"],
        audited_open_events=tracker["file_open_events"], torch_used=False,
        checks=["frozen_new_common_fields_byte_parity", "batch_sequence_byte_parity",
                "asymmetric_mask_field_flip_rotation_alignment", "padding_valid_zero",
                "foreground_unknown_only_masking", "known_background_valid",
                "no_field_zero_placeholders", "list_case_manifest",
                "malformed_manifest_rejection", "no_TEST_file_open"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run_checks(args.project)
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise ValueError("QA output is immutable")
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_json(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
