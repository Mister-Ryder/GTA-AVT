"""Stateless paired patch sampling makes model comparisons and resumes exact."""
from pathlib import Path
import numpy as np
from common import load_json
from geometry import patch_histograms


class CachedData:
    def __init__(self, manifest):
        self.path = str(Path(manifest).resolve())
        self.manifest = load_json(manifest)
        self.cases = {x['id']: x for x in self.manifest['cases']}
        self.opened = {}

    def case(self, case_id, geometry=False):
        if case_id not in self.opened:
            directory = Path(self.cases[case_id]['cache_dir'])
            item = {'image': np.load(directory / 'image.npy', mmap_mode='r'),
                    'mask': np.load(directory / 'mask.npy', mmap_mode='r')}
            if (directory / 'geometry.npz').exists():
                with np.load(directory / 'geometry.npz') as stream:
                    item['geometry'] = {x: stream[x] for x in stream.files}
            self.opened[case_id] = item
        item = self.opened[case_id]
        if geometry and 'geometry' not in item:
            raise ValueError('Train-only geometry cache missing: ' + case_id)
        return item

    def sample(self, seed, step, index, patch, foreground_probability=.5):
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
        pad = [(0, max(0, patch - n)) for n in image.shape]
        image = np.pad(image, pad, constant_values=-1024)
        mask = np.pad(mask, pad)
        scale, structure = patch_histograms(item['geometry'], start, [patch] * 3)
        # A tangent/wall patch may contain vessel but no centreline. It is unknown,
        # not background, for attribute supervision.
        attribute_valid = float(not (mask.any() and scale[0] == 1.))
        # Rotations/flips leave scalar radius and structure mixture labels invariant.
        for axis in range(3):
            if rng.random() < .5:
                image = np.flip(image, axis)
                mask = np.flip(mask, axis)
        rotation = int(rng.integers(4))
        axes = ((0, 1), (0, 2), (1, 2))[int(rng.integers(3))]
        image, mask = np.rot90(image, rotation, axes), np.rot90(mask, rotation, axes)
        image = normalize(image)
        image = image * rng.uniform(.9, 1.1) + rng.uniform(-.1, .1)
        if rng.random() < .15:
            image += rng.normal(0., .03, image.shape).astype(np.float32)
        record = self.cases[case_id]
        centre = np.asarray(record['crop_start_zyx']) + start + patch / 2.
        coordinate = 2. * centre / np.maximum(np.asarray(record['full_shape_zyx']) - 1., 1.) - 1.
        return {'image': np.ascontiguousarray(image[None]), 'mask': np.ascontiguousarray(mask[None]),
                'scale': scale, 'structure': structure, 'attribute_valid': np.float32(attribute_valid),
                'coordinate': coordinate.astype(np.float32),
                'case_id': case_id, 'start': start.tolist()}


def normalize(image):
    return (np.clip(image, -200., 1000.) - 200.) / 400.


def batch(data, seed, step, batch_size, patch, foreground_probability):
    examples = [data.sample(seed, step, index, patch, foreground_probability) for index in range(batch_size)]
    return {name: np.stack([x[name] for x in examples])
            for name in ('image', 'mask', 'scale', 'structure', 'attribute_valid', 'coordinate')}
