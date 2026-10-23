"""Paired sampler with optional precomputed, case-consistent GDF targets.

Image/mask sampling and augmentation reproduce the frozen sampler exactly.
This module does not construct any held-out geometry or read TEST for training.
"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from data import CachedData, normalize
from geometry import patch_histograms


class StageData(CachedData):
    def __init__(self, manifest, gdf_root=None):
        super().__init__(manifest)
        self.gdf_root = Path(gdf_root) if gdf_root else None
        self.gdf_opened = {}
        self.gdf_record = None
        if self.gdf_root is not None:
            self.gdf_record = json.loads((self.gdf_root/'manifest.json').read_text())
            rows = self.gdf_record['cases']
            if not isinstance(rows, list) or len({x['id'] for x in rows}) != len(rows):
                raise ValueError('Invalid/duplicate GDF cases')
            self.gdf_cases = {x['id']: x for x in rows}
            if set(self.gdf_cases) != set(self.manifest['splits']['train']):
                raise ValueError('GDF cache must cover exactly frozen TRAIN IDs')
            if not self.gdf_record.get('complete'):
                raise ValueError('Incomplete GDF cache')

    def gdf_case(self, case_id):
        if case_id not in self.manifest['splits']['train']:
            raise ValueError('Only TRAIN GDF can be consumed by sampler')
        if case_id not in self.gdf_opened:
            record = self.gdf_cases[case_id]
            self.gdf_opened[case_id] = np.load(self.gdf_root/record['target_file'], mmap_mode='r')
        return self.gdf_opened[case_id]

    def sample(self, seed, step, index, patch, foreground_probability=.5):
        # Keep RNG calls/order byte-for-byte equivalent to data.CachedData.sample.
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, index]))
        ids = self.manifest['splits']['train']
        case_id = ids[rng.integers(len(ids))]
        item = self.case(case_id, geometry=True)
        shape = np.asarray(item['image'].shape)
        points = item['geometry']['points']
        if len(points) and rng.random() < foreground_probability:
            centre = points[rng.integers(len(points))] + rng.integers(-patch // 4, patch // 4 + 1, size=3)
            start = np.clip(centre - patch // 2, 0, np.maximum(shape - patch, 0))
        else:
            start = np.array([rng.integers(max(1, int(n) - patch + 1)) for n in shape])
        slices = tuple(slice(int(x), min(int(x) + patch, int(n))) for x, n in zip(start, shape))
        image = np.asarray(item['image'][slices], dtype=np.float32)
        mask = np.asarray(item['mask'][slices], dtype=np.float32)
        if self.gdf_root is not None:
            target = np.asarray(self.gdf_case(case_id)[slices], dtype=np.float32)
            if tuple(self.gdf_case(case_id).shape) != tuple(shape):
                raise ValueError('GDF and input grid mismatch: '+case_id)
            gdf_valid = np.ones_like(mask, dtype=np.float32)
            if not self.gdf_cases[case_id].get('foreground_geometry_valid', True):
                gdf_valid[mask > 0] = 0.
        else:
            target = np.zeros_like(mask, dtype=np.float32)
            gdf_valid = np.zeros_like(mask, dtype=np.float32)
        pad = [(0, max(0, patch - n)) for n in image.shape]
        image = np.pad(image, pad, constant_values=-1024)
        mask = np.pad(mask, pad)
        target, gdf_valid = np.pad(target, pad), np.pad(gdf_valid, pad)
        scale, structure = patch_histograms(item['geometry'], start, [patch] * 3)
        attribute_valid = float(not (mask.any() and scale[0] == 1.))
        for axis in range(3):
            if rng.random() < .5:
                image = np.flip(image, axis)
                mask = np.flip(mask, axis)
                target = np.flip(target, axis)
                gdf_valid = np.flip(gdf_valid, axis)
        rotation = int(rng.integers(4))
        axes = ((0, 1), (0, 2), (1, 2))[int(rng.integers(3))]
        image, mask = np.rot90(image, rotation, axes), np.rot90(mask, rotation, axes)
        target, gdf_valid = np.rot90(target, rotation, axes), np.rot90(gdf_valid, rotation, axes)
        image = normalize(image)
        image = image * rng.uniform(.9, 1.1) + rng.uniform(-.1, .1)
        if rng.random() < .15:
            image += rng.normal(0., .03, image.shape).astype(np.float32)
        record = self.cases[case_id]
        centre = np.asarray(record['crop_start_zyx']) + start + patch / 2.
        coordinate = 2. * centre / np.maximum(np.asarray(record['full_shape_zyx']) - 1., 1.) - 1.
        return {'image': np.ascontiguousarray(image[None]), 'mask': np.ascontiguousarray(mask[None]),
                'scale': scale, 'structure': structure, 'attribute_valid': np.float32(attribute_valid),
                'coordinate': coordinate.astype(np.float32), 'case_id': case_id, 'start': start.tolist(),
                'gdf_target': np.ascontiguousarray(target[None]),
                'gdf_valid': np.ascontiguousarray(gdf_valid[None])}


def batch(data, seed, step, batch_size, patch, foreground_probability):
    examples = [data.sample(seed, step, index, patch, foreground_probability) for index in range(batch_size)]
    return {name: np.stack([x[name] for x in examples])
            for name in ('image', 'mask', 'scale', 'structure', 'attribute_valid', 'coordinate',
                         'gdf_target', 'gdf_valid')}
