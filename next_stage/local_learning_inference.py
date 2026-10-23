"""Frozen inference formulas plus explicit CT-only padding validity.

Tiling, Gaussian blending, threshold, native restoration, and metrics use the
historical evaluation definitions. No GT geometry is supplied to the model.
"""
import itertools
from pathlib import Path
import time

import numpy as np
import torch
import SimpleITK as sitk
from common import atomic_json
from data import normalize
from metrics import binary_metrics, aggregate
from inference import tile_starts, native_images


def sliding_probability(model, image, record, device, patch=128, batch_size=2,
                        overlap=.5, condition_mode="predicted"):
    shape = np.asarray(image.shape)
    padded_shape = np.maximum(shape, patch)
    probabilities = np.zeros(tuple(padded_shape), np.float32)
    weights = np.zeros_like(probabilities)
    coordinates = np.arange(patch, dtype=np.float32) - (patch - 1.) / 2.
    gaussian = np.exp(-.5 * (coordinates / (patch / 8.)) ** 2)
    weight = np.maximum(gaussian[:, None, None] * gaussian[None, :, None] *
                        gaussian[None, None, :], 1e-5)
    starts = list(itertools.product(*(tile_starts(int(n), patch, overlap) for n in padded_shape)))
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            chosen = starts[offset:offset + batch_size]
            patches, locations, validities = [], [], []
            for start in chosen:
                slices = tuple(slice(s, min(s + patch, int(n))) for s, n in zip(start, shape))
                sample = np.asarray(image[slices], np.float32)
                pad = [(0, patch - n) for n in sample.shape]
                # The mask comes from CT extent alone, before any padding.
                validities.append(np.pad(np.ones(sample.shape, bool), pad)[None])
                sample = np.pad(sample, pad, constant_values=-1024)
                patches.append(normalize(sample)[None])
                centre = np.asarray(record["crop_start_zyx"]) + np.asarray(start) + patch / 2.
                locations.append(2. * centre / np.maximum(np.asarray(record["full_shape_zyx"]) - 1., 1.) - 1.)
            tensor = torch.from_numpy(np.stack(patches)).to(device)
            location = torch.as_tensor(np.stack(locations), dtype=torch.float32, device=device)
            valid = torch.from_numpy(np.stack(validities)).to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                prediction = model(tensor, location, condition_mode=condition_mode,
                                   input_valid=valid)["logits"].sigmoid().float().cpu().numpy()[:, 0]
            for start, value in zip(chosen, prediction):
                slices = tuple(slice(s, s + patch) for s in start)
                probabilities[slices] += value * weight
                weights[slices] += weight
    probabilities /= np.maximum(weights, 1e-8)
    return probabilities[tuple(slice(0, int(n)) for n in shape)], len(starts)


def evaluate_cases(model, data, case_ids, device, output, patch=128, batch_size=2,
                   overlap=.5, save_predictions=True, condition_mode="predicted", reuse=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id in case_ids:
        case_output = output / case_id
        case_output.mkdir(exist_ok=True)
        metric_path = case_output / "metrics.json"
        if reuse and metric_path.exists():
            from common import load_json
            rows.append(load_json(metric_path))
            continue
        started = time.time()
        record = data.cases[case_id]
        probability, tiles = sliding_probability(
            model, data.case(case_id)["image"], record, device,
            patch, batch_size, overlap, condition_mode)
        prediction, truth = native_images(probability, record)
        metrics = binary_metrics(sitk.GetArrayFromImage(prediction), sitk.GetArrayFromImage(truth),
                                 np.array(prediction.GetSpacing())[::-1])
        metrics.update(case_id=case_id, cohort=record["cohort"], seconds=time.time() - started,
                       inference_tiles=tiles, fixed_threshold=.5, condition_mode=condition_mode,
                       evaluation_grid="original CT", postprocessing="none")
        if save_predictions:
            sitk.WriteImage(sitk.Cast(prediction, sitk.sitkUInt8),
                            str(case_output / "prediction.nii.gz"), True)
        atomic_json(metric_path, metrics)
        rows.append(metrics)
        print({"evaluated": case_id, "dice": metrics["dice"], "fp_ml": metrics["fp_ml"],
               "fn_ml": metrics["fn_ml"], "seconds": metrics["seconds"]}, flush=True)
        atomic_json(output / "summary.json", aggregate(rows))
    return rows, aggregate(rows)

