"""Analytic geometry/validity fixtures for the cumulative AVT GDF pilot.

Run locally without torch for target checks. Torch loss/gradient checks execute
when torch is present and are explicitly reported skipped otherwise.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from gdf import CumulativeRadiusGeometry, GaussianFieldSpec, gdf_mse


def check_targets():
    checks = {}
    seed = np.asarray([[4., 4., 4.]])
    single = CumulativeRadiusGeometry(seed, np.array([2.]))
    query = np.asarray([[4., 4., 4.], [4., 4., 6.], [4., 4., 8.]])
    np.testing.assert_allclose(single.evaluate_sum(query),
                               [1., np.exp(-.5), np.exp(-2.)],
                               rtol=1e-14, atol=1e-14)
    checks["single_seed_analytic_profile"] = "passed"

    # Two kernels must sum, not choose the nearest or maximum response.
    double = CumulativeRadiusGeometry([[0., 0., -1.], [0., 0., 1.]],
                                      np.ones(2))
    value = double.evaluate_sum([[0., 0., 0.]])[0]
    np.testing.assert_allclose(value, 2. * np.exp(-.5), rtol=1e-14)
    assert value > 1.
    checks["two_seed_additivity_unbounded_raw_sum"] = "passed"

    anisotropic = CumulativeRadiusGeometry([[0., 0., 0.]], [2.],
                                           spacing_mm=[2., 1., .5])
    values = anisotropic.evaluate_sum([[1., 0., 0.], [0., 2., 0.],
                                       [0., 0., 4.]])
    np.testing.assert_allclose(values, [np.exp(-.5)] * 3, rtol=1e-14)
    checks["anisotropic_equal_physical_distance"] = "passed"

    broad = CumulativeRadiusGeometry(seed, np.array([4.]))
    assert broad.evaluate_sum(query[1:])[0] > single.evaluate_sum(query[1:])[0]
    checks["larger_native_radius_broadens_kernel"] = "passed"

    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[2:7, 2:7, 2:7] = True
    target, valid, audit = single.target(mask)
    assert valid.all() and np.all(target[~mask] == 0.)
    np.testing.assert_allclose(target[4, 4, 4], np.log(2.), rtol=1e-7)
    assert audit["target_used_as_model_input"] is False
    checks["mask_restriction_log_transform"] = "passed"

    empty_geometry = CumulativeRadiusGeometry(np.empty((0, 3)), np.empty(0))
    zero, valid_empty, empty_audit = empty_geometry.target(np.zeros_like(mask))
    assert valid_empty.all() and np.count_nonzero(zero) == 0
    assert empty_audit["foreground_unknown_voxels"] == 0
    _, valid_unknown, unknown_audit = empty_geometry.target(mask)
    assert not valid_unknown[mask].any() and valid_unknown[~mask].all()
    assert unknown_audit["foreground_unknown_voxels"] == int(mask.sum())
    checks["true_empty_distinct_from_unknown_foreground"] = "passed"

    input_valid = np.ones(mask.shape, dtype=bool)
    input_valid[0] = False
    _, padding_valid, _ = single.target(mask, input_valid=input_valid)
    assert not padding_valid[0].any() and padding_valid[1:].all()
    checks["padding_has_no_supervision"] = "passed"

    # The centreline seed lies outside this crop; it must still contribute.
    crop = (slice(2, 4), slice(2, 7), slice(2, 7))
    cropped, _, _ = single.target(mask[crop], start_zyx=[2., 2., 2.])
    np.testing.assert_array_equal(cropped, target[crop])
    assert cropped.max() > 0.
    checks["complete_case_seed_crop_consistency"] = "passed"

    shift = np.asarray([10., 20., 30.])
    translated = CumulativeRadiusGeometry(seed + shift, np.array([2.]))
    shifted, _, _ = translated.target(mask, start_zyx=shift)
    np.testing.assert_array_equal(shifted, target)
    checks["physical_translation_equivariance"] = "passed"

    rescaled = CumulativeRadiusGeometry(seed, np.array([2.]),
                                        spec=GaussianFieldSpec(field_scale=2.))
    half, _, _ = rescaled.target(mask)
    np.testing.assert_array_equal(half, target / 2.)
    checks["one_fixed_scale_not_case_normalization"] = "passed"

    rng = np.random.default_rng(20261023)
    random_seed = rng.uniform(0., 10., (37, 3))
    random_radius = rng.uniform(.3, 5., 37)
    random_query = rng.uniform(0., 10., (53, 3))
    small = CumulativeRadiusGeometry(
        random_seed, random_radius,
        spec=GaussianFieldSpec(query_chunk=7, center_chunk=5))
    large = CumulativeRadiusGeometry(
        random_seed, random_radius,
        spec=GaussianFieldSpec(query_chunk=512, center_chunk=1024))
    np.testing.assert_allclose(small.evaluate_sum(random_query),
                               large.evaluate_sum(random_query),
                               rtol=5e-15, atol=5e-15)
    checks["two_axis_chunking_preserves_exact_sum"] = "passed"

    for points, radius in (([[0., 0., 0.]], [0.]),
                           ([[0., 0., 0.], [0., 0., 0.]], [1., 1.])):
        try:
            CumulativeRadiusGeometry(points, radius)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid radius/duplicate seed accepted")
    checks["invalid_radius_duplicate_seed_rejected"] = "passed"
    return checks


def check_torch():
    try:
        import torch
    except ImportError:
        return {"status": "skipped", "reason": "torch_not_installed"}
    prediction = torch.tensor([1., 2., 1., 1.], dtype=torch.float16).reshape(
        2, 1, 1, 1, 2).requires_grad_()
    target = torch.tensor([3., 0., float("nan"), float("nan")]).reshape_as(
        prediction)
    valid = torch.tensor([1., 1., 0., 0.]).reshape_as(prediction)
    foreground = torch.tensor([1., 0., 1., 1.]).reshape_as(prediction)
    loss, details = gdf_mse(prediction, target, valid, foreground)
    assert loss.dtype == torch.float32 and float(loss.detach()) == 4.
    assert details["gdf_valid_cases"] == 1
    loss.backward()
    expected = torch.tensor([-2., 2., 0., 0.], dtype=torch.float16)
    assert torch.equal(prediction.grad.flatten(), expected)

    blank_prediction = torch.full((1, 1, 1, 1, 4), 1.,
                                  requires_grad=True)
    all_unknown = torch.zeros_like(blank_prediction)
    zero, _ = gdf_mse(blank_prediction, torch.full_like(blank_prediction,
                                                      float("nan")),
                      all_unknown)
    assert float(zero.detach()) == 0.
    zero.backward()
    assert torch.count_nonzero(blank_prediction.grad) == 0

    p = torch.zeros((1, 1, 1, 1, 4), requires_grad=True)
    t = torch.tensor([2., 0., 0., 0.]).reshape_as(p)
    v = torch.ones_like(p)
    fg = torch.tensor([1., 0., 0., 0.]).reshape_as(p)
    plain, _ = gdf_mse(p, t, v, fg, "voxel")
    balanced, _ = gdf_mse(p, t, v, fg, "balanced")
    assert float(plain.detach()) == 1.
    assert float(balanced.detach()) == 2.
    return {"status": "passed", "torch": torch.__version__,
            "checks": ["fp16_input_fp32_mse_known_gradient",
                       "unknown_nan_target_excluded",
                       "all_unknown_graph_connected_zero",
                       "explicit_voxel_vs_balanced_normalization"]}


def benchmark():
    rng = np.random.default_rng(20261023)
    points = rng.uniform(0., 128., (512, 3))
    radius = rng.uniform(.5, 20., 512)
    queries = rng.uniform(0., 128., (4096, 3))
    geometry = CumulativeRadiusGeometry(points, radius)
    began = time.perf_counter()
    result = geometry.evaluate_sum(queries)
    seconds = time.perf_counter() - began
    pairs = len(points) * len(queries)
    assert np.isfinite(result).all()
    return {
        "kind": "local_CPU_synthetic_kernel_benchmark",
        "query_points": len(queries), "centerline_points": len(points),
        "kernel_evaluations": pairs, "elapsed_seconds": seconds,
        "kernel_pairs_per_second": pairs / max(seconds, 1e-9),
        "not_a_full_AVT_preparation_time_estimate": True,
        "workspace_default_pair_matrix": 512 * 1024,
        "distance_delta_workspace_bytes": 512 * 1024 * 3 * 8,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    result = {"schema": "GTA-AVT-cumulative-GDF-checks-v1",
              "target_checks": check_targets(), "loss_checks": check_torch()}
    if args.benchmark:
        result["benchmark"] = benchmark()
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
