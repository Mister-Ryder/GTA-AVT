"""Native-header certification and CT-only inference input contracts."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from gta_avt.inference import certify_native_grid, infer, model_from_checkpoint
from gta_avt.models import AttributePredictor, GTAAVT, MODEL_CONFIG
from gta_avt.engine import SCHEMA


class Grid:
    def __init__(self, origin=(0., 0., -1019.4798999999998), size=(32, 32, 32), rounded=False):
        cast = lambda x: tuple(np.asarray(x, dtype=np.float32).astype(np.float64)) if rounded else tuple(x)
        self.origin = cast(origin)
        self.spacing = cast((.775390625, .775390625, 1.2499999999999987))
        self.direction = cast((1., 0., 0., 0., 1., 0., 0., 0., 1.))
        self.size = size
    def GetDimension(self): return 3
    def GetNumberOfComponentsPerPixel(self): return 1
    def GetSize(self): return self.size
    def GetSpacing(self): return self.spacing
    def GetOrigin(self): return self.origin
    def GetDirection(self): return self.direction


class InferenceContracts(unittest.TestCase):
    def test_real_model_checkpoint_roundtrip_and_revision_rejection(self):
        torch.set_num_threads(2)
        stats = dict(mu=1., sigma=.5, pi_support=.2)
        model = GTAAVT(AttributePredictor(stats), arm='AR', use_gdf=True).eval()
        x = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad(): expected = model(x)['logits']
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pt'
            value = dict(schema=SCHEMA, kind='train', model_config=dict(MODEL_CONFIG, arm='AR', use_gdf=True),
                         statistics=stats, model=model.state_dict())
            torch.save(value, path)
            restored, _ = model_from_checkpoint(path)
            with torch.no_grad(): actual = restored(x)['logits']
            self.assertTrue(torch.equal(actual, expected))
            value['model_config']['revision'] = 'old'
            torch.save(value, path)
            with self.assertRaisesRegex(ValueError, 'different method revision'):
                model_from_checkpoint(path)

    def test_exact_grid_passes_without_reference(self):
        self.assertEqual(certify_native_grid(Grid(), Grid()), 'direct_grid_match')

    def test_nifti_rounding_needs_certified_original_ct(self):
        with self.assertRaises(ValueError):
            certify_native_grid(Grid(rounded=True), Grid())
        self.assertEqual(certify_native_grid(Grid(rounded=True), Grid(), Grid()),
                         'certified_CT_float32_header')

    def test_real_shift_and_wrong_ct_are_rejected(self):
        with self.assertRaises(ValueError):
            certify_native_grid(Grid(origin=(0., 0., -1019.4)), Grid(), Grid())
        with self.assertRaises(ValueError):
            certify_native_grid(Grid(rounded=True), Grid(), Grid(origin=(0., 0., -1019.5)))
        with self.assertRaises(ValueError):
            certify_native_grid(Grid(size=(31, 32, 32)), Grid(), Grid())

    def test_inference_rejects_label_and_geometry_inputs_before_loading(self):
        for name in ('label', 'mask', 'geometry', 'roi'):
            with self.assertRaisesRegex(ValueError, 'does not accept'):
                infer(dict(checkpoint='not-read.pt', image='not-read.nrrd', **{name: 'forbidden'}))


if __name__ == '__main__':
    unittest.main()
