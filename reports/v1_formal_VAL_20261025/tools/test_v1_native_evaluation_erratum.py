"""Focused CPU-only tests for the external evaluation engineering erratum.

Run: python -B ops/test_v1_native_evaluation_erratum.py
No server connection, GPU, model initialization or training is used.
The parity test executes the unmodified original evaluate function extracted
from its AST, avoiding its unrelated Torch imports on a NumPy/SimpleITK host.
"""
import ast
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import v1_native_evaluation_erratum as correction

ROOT = Path(__file__).resolve().parents[1]
CAPSULE = ROOT / 'deliverables' / 'GTA-AVT-v1'


def grid():
    return dict(size=[512, 666, 176], spacing=[.6660156249999999, .6660156249999999, 2.4999988571428577],
                origin=[-194.36999999999998, -51.78320312500001, -1019.4798999999998],
                direction=[1., 0., 0., 0., 1., 0., 0., 0., .9999999999999999],
                dimension=3, components=1)


def quantized(original):
    result = copy.deepcopy(original)
    for key in correction.FIELDS:
        result[key] = np.asarray(original[key], np.float32).astype(np.float64).tolist()
    return result


class HeaderTests(unittest.TestCase):
    def test_d5_exact_float32_rounding_explains_original_failure(self):
        original = grid(); prediction = quantized(original)
        self.assertFalse(np.allclose(original['origin'], prediction['origin'], rtol=0, atol=1e-5))
        certificate = correction.certify_prediction(original, prediction)
        self.assertTrue(certificate['float32_ctgrid_equal'])
        self.assertAlmostEqual(certificate['max_abs_header_difference']['origin'], 1.9433593934081728e-5)
        self.assertLess(certificate['corner_mm'], .00003)

    def test_exact_ct_grid_is_also_accepted(self):
        self.assertTrue(correction.certify_prediction(grid(), grid())['ct_grid_exact_equal'])

    def test_shift_size_and_direction_errors_fail(self):
        for field, index, amount in (('origin', 2, .001), ('spacing', 2, .001), ('direction', 0, -.001)):
            bad = quantized(grid()); bad[field][index] += amount
            with self.subTest(field=field), self.assertRaises(ValueError):
                correction.certify_prediction(grid(), bad)
        bad = quantized(grid()); bad['size'][2] -= 1
        with self.assertRaises(ValueError):
            correction.certify_prediction(grid(), bad)

    def test_original_label_rounding_and_misalignment(self):
        original = grid(); label = copy.deepcopy(original)
        label['origin'][0] += 8e-14
        self.assertEqual(correction.certify_source_pair(original, label)['status'], 'certified')
        label['origin'][0] += 1e-8
        with self.assertRaises(ValueError):
            correction.certify_source_pair(original, label)

    def test_nonfinite_nonscalar_and_wrong_dimensions_rejected(self):
        for key, value in (('dimension', 4), ('components', 2), ('origin', [0., 0., float('nan')])):
            bad = grid(); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                correction.certify_prediction(grid(), bad)

    def test_inference_identity_is_strict(self):
        path = Path('prediction.nii.gz').resolve()
        report = dict(case_id='Dongyang_D5', checkpoint_sha256='c' * 64,
                      checkpoint_successful_updates=5000, arm='AR', condition_mode='predicted',
                      threshold=.5, patch_size=128, overlap=.5, prediction=str(path), probability=None, tiles=9)
        kwargs = dict(case_id='Dongyang_D5', checkpoint_sha='c' * 64, arm='AR_GDF', prediction_path=path)
        correction.verify_inference_report(report, **kwargs)
        for key, value in (('arm', 'C0'), ('checkpoint_successful_updates', 4999), ('patch_size', 96),
                           ('condition_mode', 'neutral'), ('checkpoint_sha256', 'd' * 64), ('threshold', .4)):
            bad = dict(report); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                correction.verify_inference_report(bad, **kwargs)


class SimpleITKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import SimpleITK as sitk
        except ImportError:
            raise unittest.SkipTest('SimpleITK unavailable: Root CPU parity check required')
        cls.sitk = sitk
        spec = importlib.util.spec_from_file_location('original_v1_metrics', CAPSULE / 'gta_avt' / 'metrics.py')
        metrics = importlib.util.module_from_spec(spec); spec.loader.exec_module(metrics)
        cls.metrics = metrics

    def test_written_nifti_has_certified_d5_quantization(self):
        image = self.sitk.GetImageFromArray(np.zeros((3, 4, 5), np.uint8))
        image.SetOrigin(grid()['origin']); image.SetSpacing(grid()['spacing'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prediction.nii.gz'
            self.sitk.WriteImage(image, str(path))
            actual = correction.read_header(path)
            certificate = correction.certify_prediction(correction.image_header(image), actual)
            self.assertTrue(certificate['float32_ctgrid_equal'])

    def test_original_evaluate_exact_grid_full_metric_parity(self):
        # Compile exactly the saved original evaluation functions without changing
        # their bodies; all geometry and metric work still uses real SimpleITK.
        source = CAPSULE / 'gta_avt' / 'inference.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in ('_evaluation_cases', 'evaluate')]
        namespace = dict(Path=Path, json=json, np=np, csv=csv, _json=correction.write,
                         binary_metrics=self.metrics.binary_metrics,
                         summarize_cases=self.metrics.summarize_cases)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
        target = np.zeros((13, 15, 17), np.uint8); target[2:11, 6:9, 7:10] = 1
        predicted = target.copy(); predicted[2:4, 6, 7] = 0; predicted[6, 9, 8] = 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cases = []
            for name, values in (('label', target), ('prediction', predicted)):
                image = self.sitk.GetImageFromArray(values)
                image.SetOrigin((8., -4., -512.)); image.SetSpacing((.75, .75, 2.))
                self.sitk.WriteImage(image, str(root / (name + '.nii.gz')))
            case = dict(id='synthetic', label=str(root / 'label.nii.gz'), prediction=str(root / 'prediction.nii.gz'))
            original = namespace['evaluate'](dict(cases=[case], output_dir=str(root / 'original'), centerline=True))
            label = self.sitk.ReadImage(case['label']); prediction = self.sitk.ReadImage(case['prediction'])
            correction.certify_source_pair(correction.image_header(label), correction.image_header(label))
            correction.certify_prediction(correction.image_header(label), correction.image_header(prediction))
            row, digest = correction.metric_row('synthetic', prediction, label, self.metrics.binary_metrics)
            revised = correction.write_metrics(root / 'revised', [row], self.metrics.summarize_cases)
            self.assertEqual(original, revised)
            self.assertEqual(len(digest), 64)

    def test_nonbinary_prediction_is_rejected(self):
        label = self.sitk.GetImageFromArray(np.zeros((4, 5, 6), np.uint8))
        values = np.zeros((4, 5, 6), np.float32); values[1, 1, 1] = .6
        prediction = self.sitk.GetImageFromArray(values)
        with self.assertRaises(ValueError):
            correction.metric_row('synthetic', prediction, label, self.metrics.binary_metrics)


if __name__ == '__main__':
    unittest.main(verbosity=2)
