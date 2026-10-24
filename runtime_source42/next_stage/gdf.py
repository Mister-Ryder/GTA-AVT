"""Physical cumulative Gaussian-field supervision for the next AVT pilot.

This preserves the legacy sum-of-centreline-kernels target family. The raw sum
is NOT bounded by one. We explicitly supervise log1p(sum) / fixed_train_scale
with a positive softplus prediction. This is a declared AVT adaptation, not a
byte-identical reproduction of the legacy Gaussian target or output head.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianFieldSpec:
    sigma_scale: float = 1.0
    # Numerical protection, not a claimed resolvable or clinical vessel size.
    sigma_floor_mm: float = 1e-3
    # One scalar fixed from TRAIN, never a per-case/patch maximum.
    field_scale: float = 1.0
    query_chunk: int = 512
    center_chunk: int = 1024

    def __post_init__(self):
        for name in ("sigma_scale", "sigma_floor_mm", "field_scale"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if self.query_chunk <= 0 or self.center_chunk <= 0:
            raise ValueError("Chunk sizes must be positive")


def _triple(value, name, positive=False):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(name + " must have three finite z,y,x values")
    if positive and (result <= 0).any():
        raise ValueError(name + " must be positive")
    return result


class CumulativeRadiusGeometry:
    """Complete-case centreline kernels; reuse this object for many patches."""

    def __init__(self, points_zyx, radius_mm, spacing_mm=(1., 1., 1.),
                 geometry_valid=True, spec=None):
        self.spec = GaussianFieldSpec() if spec is None else spec
        self.spacing_mm = _triple(spacing_mm, "spacing_mm", positive=True)
        points = np.asarray(points_zyx, dtype=np.float64)
        radius = np.asarray(radius_mm, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_zyx must have shape [N,3]")
        if radius.shape != (len(points),):
            raise ValueError("radius_mm must have one value per centreline point")
        if not np.isfinite(points).all() or not np.isfinite(radius).all():
            raise ValueError("Geometry must be finite")
        if len(radius) and (radius <= 0).any():
            raise ValueError("Physical centreline radii must be strictly positive")
        if len(points) and len(np.unique(points, axis=0)) != len(points):
            raise ValueError("Centreline seeds must be unique; duplicates alter sum")
        self.points_zyx = points
        self.radius_mm = radius
        self.sigma_mm = np.maximum(radius * self.spec.sigma_scale,
                                   self.spec.sigma_floor_mm)
        self.geometry_valid = bool(geometry_valid)

    def evaluate_sum(self, query_zyx):
        """Exact all-seed sum at query coordinates, with bounded CPU workspace.

        No distance truncation, nearest-seed assignment or max approximation.
        Complexity is N_query * N_center kernels; only foreground queries are
        needed by target(). All distance/kernel arithmetic is float64 here.
        """
        query = np.asarray(query_zyx, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 3:
            raise ValueError("query_zyx must have shape [N,3]")
        if not np.isfinite(query).all():
            raise ValueError("Queries must be finite")
        if not self.geometry_valid:
            raise ValueError("Cannot evaluate an invalid geometry")
        output = np.zeros(len(query), dtype=np.float64)
        for begin in range(0, len(query), self.spec.query_chunk):
            stop = min(begin + self.spec.query_chunk, len(query))
            query_part = query[begin:stop]
            total = np.zeros(len(query_part), dtype=np.float64)
            for cb in range(0, len(self.points_zyx), self.spec.center_chunk):
                ce = min(cb + self.spec.center_chunk, len(self.points_zyx))
                delta = (query_part[:, None, :] -
                         self.points_zyx[None, cb:ce, :]) * self.spacing_mm
                distance_squared = np.einsum("qcd,qcd->qc", delta, delta)
                variance = self.sigma_mm[None, cb:ce] ** 2
                total += np.exp(-.5 * distance_squared / variance).sum(axis=1)
            output[begin:stop] = total
        return output

    def target(self, mask, start_zyx=(0, 0, 0), input_valid=None):
        """Return float32 transformed field, boolean validity and target audit.

        mask is an UN-AUGMENTED 3D patch in the complete-case geometry grid.
        start_zyx locates it in the full case. Build/crop before augmentation;
        then apply the same flips/rotations to mask, target and validity.
        Complete-case seeds outside the crop remain available, so merely
        cropping does not alter a physical voxel's supervised field.

        True empty background has zero field. Foreground with missing geometry
        is unknown and excluded; background remains known zero. input_valid
        independently removes CT padding. Targets are never network inputs.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 3 or any(n <= 0 for n in mask.shape):
            raise ValueError("mask must be a nonempty-shape 3D array")
        start = _triple(start_zyx, "start_zyx")
        if input_valid is None:
            valid = np.ones(mask.shape, dtype=bool)
        else:
            valid = np.asarray(input_valid, dtype=bool)
            if valid.shape != mask.shape:
                raise ValueError("input_valid shape must match mask")
            valid = valid.copy()
        target = np.zeros(mask.shape, dtype=np.float32)
        foreground = mask & valid
        flat = np.flatnonzero(foreground)
        audit = {
            "target_definition": "physical-cumulative-log1p-Gaussian-v1",
            "sigma_scale": float(self.spec.sigma_scale),
            "sigma_floor_mm": float(self.spec.sigma_floor_mm),
            "fixed_train_field_scale": float(self.spec.field_scale),
            "spacing_zyx_mm": self.spacing_mm.tolist(),
            "start_zyx": start.tolist(),
            "complete_case_centerline_points": len(self.points_zyx),
            "foreground_voxels_in_input": int(len(flat)),
            "kernel_evaluations": int(len(flat) * len(self.points_zyx)),
            "geometry_valid": self.geometry_valid,
            "target_used_as_model_input": False,
            "foreground_unknown_voxels": 0,
            "kernel_truncation": None,
        }
        raw_sum_min, raw_sum_max = None, None
        if len(flat) and (not self.geometry_valid or not len(self.points_zyx)):
            valid[foreground] = False
            audit["foreground_unknown_voxels"] = int(len(flat))
            audit["status"] = "foreground_geometry_unknown"
        elif len(flat):
            for begin in range(0, len(flat), self.spec.query_chunk):
                indices = flat[begin:begin + self.spec.query_chunk]
                coords = np.stack(np.unravel_index(indices, mask.shape), axis=1)
                raw = self.evaluate_sum(coords + start)
                values = np.log1p(raw) / self.spec.field_scale
                target.ravel()[indices] = values.astype(np.float32)
                raw_sum_min = (float(raw.min()) if raw_sum_min is None
                               else min(raw_sum_min, float(raw.min())))
                raw_sum_max = (float(raw.max()) if raw_sum_max is None
                               else max(raw_sum_max, float(raw.max())))
            audit["status"] = "supported_foreground"
        else:
            audit["status"] = "true_empty_or_no_valid_foreground"
        audit.update(
            valid_voxels=int(valid.sum()),
            known_background_voxels=int((valid & ~mask).sum()),
            target_min=float(target.min()), target_max=float(target.max()),
            raw_sum_foreground_min=raw_sum_min,
            raw_sum_foreground_max=raw_sum_max,
            foreground_target_mean=(float(target[mask & valid].mean())
                                    if (mask & valid).any() else None),
        )
        return target, valid, audit


def gaussian_target(mask, points_zyx, radius_mm, spacing_mm=(1., 1., 1.),
                    start_zyx=(0, 0, 0), input_valid=None,
                    geometry_valid=True, spec=None):
    """Convenience wrapper; reuse CumulativeRadiusGeometry for many patches."""
    geometry = CumulativeRadiusGeometry(points_zyx, radius_mm, spacing_mm,
                                        geometry_valid, spec)
    return geometry.target(mask, start_zyx, input_valid)


def gdf_mse(prediction, target, valid, foreground=None, normalization="voxel"):
    """FP32 MSE of an already-softplus prediction, per valid case.

    Inputs have identical [B,1,D,H,W] shapes. "voxel" is the first candidate,
    closest to historical whole-volume MSE. "balanced" explicitly averages
    foreground/background means when each exists and must be predeclared.
    Unknown samples are excluded from the case mean; an all-unknown batch has
    graph-connected zero loss. Positive predictions and targets may exceed one.
    """
    import torch
    if prediction.ndim != 5 or prediction.shape[1] != 1:
        raise ValueError("Expected B,1,D,H,W prediction")
    if target.shape != prediction.shape or valid.shape != prediction.shape:
        raise ValueError("GDF prediction, target and validity shapes must match")
    if normalization not in ("voxel", "balanced"):
        raise ValueError("Unknown GDF MSE normalization")
    if foreground is not None and foreground.shape != prediction.shape:
        raise ValueError("foreground shape must match prediction")
    if normalization == "balanced" and foreground is None:
        raise ValueError("Balanced GDF MSE requires a foreground mask")
    weights = valid.float()
    safe_target = torch.where(weights > 0, target.float(),
                              torch.zeros_like(target, dtype=torch.float32))
    squared = (prediction.float() - safe_target).square()
    axes = (1, 2, 3, 4)
    counts = weights.sum(axes)
    per_case = (squared * weights).sum(axes) / counts.clamp_min(1.)
    valid_case = (counts > 0).to(squared.dtype)
    details = {
        "gdf_normalization": normalization,
        "gdf_prediction_transform": "already_softplus_in_model",
        "gdf_valid_cases": int(valid_case.sum().detach()),
        "gdf_valid_voxels": int(counts.sum().detach()),
    }
    if foreground is not None:
        fg = weights * foreground.float()
        bg = weights * (1. - foreground.float())
        fg_count, bg_count = fg.sum(axes), bg.sum(axes)
        fg_mean = (squared * fg).sum(axes) / fg_count.clamp_min(1.)
        bg_mean = (squared * bg).sum(axes) / bg_count.clamp_min(1.)
        fg_present = (fg_count > 0).to(squared.dtype)
        bg_present = (bg_count > 0).to(squared.dtype)
        details.update(
            gdf_foreground_mse=float(
                ((fg_mean * fg_present).sum() /
                 fg_present.sum().clamp_min(1.)).detach()),
            gdf_background_mse=float(
                ((bg_mean * bg_present).sum() /
                 bg_present.sum().clamp_min(1.)).detach()),
            gdf_foreground_cases=int(fg_present.sum().detach()),
            gdf_background_cases=int(bg_present.sum().detach()),
        )
        if normalization == "balanced":
            per_case = ((fg_mean * fg_present + bg_mean * bg_present) /
                        (fg_present + bg_present).clamp_min(1.))
    loss = (per_case * valid_case).sum() / valid_case.sum().clamp_min(1.)
    details["gdf_mse"] = float(loss.detach())
    return loss, details
