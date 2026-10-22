"""NumPy-only synthetic schema/gate fixtures, not model/GPU validation."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import numpy as np
from reference_weights_support import (
    PROJECT, SPATIAL_FIELDS, FROZEN_SPLIT_SHA256, atomic_json, file_hash, source_hashes,
    frozen_patient_ids, canonical_cache_identity, assert_same_data, evaluation_cases,
    require_reference_runtime)


def must_reject(call, label):
    try:
        call()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('Expected rejection: ' + label)


class ArrayTensor:
    """Tiny NumPy adapter only for executing the probability formula fixture."""
    def __init__(self, value):
        self.value = np.asarray(value, np.float32)
    def float(self):
        return self
    def softmax(self, dim):
        value = np.exp(self.value - self.value.max(axis=dim, keepdims=True))
        return ArrayTensor(value / value.sum(axis=dim, keepdims=True))
    def __getitem__(self, item):
        return ArrayTensor(self.value[item])
    def __rsub__(self, other):
        return ArrayTensor(other - self.value)
    def __truediv__(self, value):
        return ArrayTensor(self.value / value)
    def expand(self, *shape):
        return ArrayTensor(np.broadcast_to(self.value, tuple(self.value.shape[i] if n == -1 else n for i, n in enumerate(shape))))


def main():
    before_sources = source_hashes()
    scripts = ['reference_weights_support.py', 'export_reference_weights.py', 'evaluate_reference_weights.py',
               'reference_input_bank.py', 'reference_interventions.py', 'verify_reference_schema.py']
    for name in scripts:
        ast.parse((Path(__file__).parent / name).read_text(), feature_version=(3, 8))
    frozen = frozen_patient_ids()
    with tempfile.TemporaryDirectory(prefix='gta_reference_schema_') as temporary:
        root = Path(temporary)
        manifests = []
        for host in ['training_location', 'reference_location']:
            location = root / host
            cases = []
            for ident in sum(frozen.values(), []):
                directory = location / 'cache' / ident
                directory.mkdir(parents=True)
                raw = location / 'raw'
                raw.mkdir(exist_ok=True)
                image, label = raw / (ident + '.image.fixture'), raw / (ident + '.label.fixture')
                image.write_bytes(('synthetic-CT:' + ident).encode())
                label.write_bytes(('synthetic-mask:' + ident).encode())
                train = ident in frozen['train']
                names = ['image.npy', 'mask.npy'] + (['geometry.npz'] if train else [])
                for name in names:
                    (directory / name).write_bytes(('synthetic-schema-bytes:' + ident + ':' + name).encode())
                record = dict(id=ident, cohort=ident.split('_')[0], source_case=ident.split('_')[1],
                    cache_dir=str(directory), image=str(image), label=str(label),
                    image_sha256=file_hash(image), label_sha256=file_hash(label),
                    geometry_training_only=train,
                    preprocessing_source_sha256={k: before_sources[k] for k in ('prepare.py', 'geometry.py', 'common.py', 'intensity.py')},
                    cache_file_sha256={name: file_hash(directory / name) for name in names},
                    preprocessing_seconds=1. if host=='training_location' else 9.)
                record.update({key: [1, 1, 1] for key in SPATIAL_FIELDS})
                cases.append(record)
            manifest = location / 'manifest.json'
            atomic_json(manifest, dict(complete=True, splits=frozen, split_sha256=FROZEN_SPLIT_SHA256,
                                      protocol='synthetic_schema_fixture_only', spacing_mm=1., cases=cases))
            manifests.append(manifest)
        a = canonical_cache_identity(manifests[0], True)
        b = canonical_cache_identity(manifests[1], True)
        assert a['canonical_sha256'] == b['canonical_sha256'] and a['canonical'] == b['canonical']
        assert a['verification']['file_count'] == 260
        copied = json.loads(manifests[1].read_text())
        path = Path(copied['cases'][0]['cache_dir']) / 'mask.npy'
        content = path.read_bytes()
        path.write_bytes(content + b'changed')
        must_reject(lambda: canonical_cache_identity(manifests[1], True), 'cache-byte corruption')
        path.write_bytes(content)
        image = Path(copied['cases'][0]['image'])
        content = image.read_bytes()
        image.write_bytes(content + b'changed')
        must_reject(lambda: canonical_cache_identity(manifests[1], True), 'raw-source corruption')
        image.write_bytes(content)
        copied['cases'][0]['original_spacing'][0] = 2
        atomic_json(manifests[1], copied)
        altered = canonical_cache_identity(manifests[1], True)
        must_reject(lambda: assert_same_data({'canonical_data_identity': a}, altered), 'semantic spatial mismatch')
        copied['splits']['test'][0] = copied['splits']['train'][0]
        atomic_json(manifests[1], copied)
        must_reject(lambda: canonical_cache_identity(manifests[1], True), 'patient split changed')
    pilot = dict(frozen_patient_ids=frozen, stage='pilot', final_test_eligible=False,
                 original_run_record={'validation_ids': frozen['val'][:3]})
    assert evaluation_cases(pilot, 'val', False) == frozen['val'][:3]
    must_reject(lambda: evaluation_cases(pilot, 'test', False), 'pilot test')
    must_reject(lambda: evaluation_cases(pilot, 'test', True), 'pilot test with explicit flag')
    formal = dict(frozen_patient_ids=frozen, stage='formal', final_test_eligible=True,
                  original_run_record={'validation_ids': frozen['val']})
    must_reject(lambda: evaluation_cases(formal, 'test', False), 'missing explicit formal test flag')
    assert evaluation_cases(formal, 'test', True) == frozen['test']
    must_reject(lambda: evaluation_cases(formal, 'test', True, 'availability_only'), 'test intervention')
    must_reject(lambda: evaluation_cases(formal, 'val', True), 'test flag on val')
    import reference_weights_support as support
    old_info = sys.version_info
    old_torch = sys.modules.get('torch')
    try:
        sys.version_info = (3, 8, 0)
        sys.modules['torch'] = types.SimpleNamespace(__version__='1.11.0+cu113')
        require_reference_runtime()
        sys.version_info = (3, 11, 0)
        sys.modules['torch'] = types.SimpleNamespace(__version__='2.7.1+cu128')
        must_reject(require_reference_runtime, 'wrong reference runtime')
        sys.modules['torch'] = types.SimpleNamespace(cat=lambda values, dim: ArrayTensor(np.concatenate([v.value for v in values], axis=dim)))
        from reference_interventions import _availability_only_conditions, attach_availability_only
        rng = np.random.default_rng(20261022)
        logits = dict(generation=ArrayTensor(rng.normal(size=(7, 5))), structure=ArrayTensor(rng.normal(size=(7, 3))))
        scale, structure = _availability_only_conditions(None, logits, 'predicted')
        for result, key in [(scale.value, 'generation'), (structure.value, 'structure')]:
            expected = logits[key].softmax(1).value
            assert np.allclose(result[:, 0], expected[:, 0]) and np.allclose(result.sum(1), 1.)
            assert np.allclose(result[:, 1:], result[:, 1:2])
        must_reject(lambda: _availability_only_conditions(None, logits, 'constant'), 'combined intervention')
        must_reject(lambda: attach_availability_only(types.SimpleNamespace(conditions=None), 'faithful_soft', 'test'), 'heldout intervention')
        must_reject(lambda: attach_availability_only(types.SimpleNamespace(conditions=None), 'faithful_joint', 'val'), 'wrong method intervention')
    finally:
        sys.version_info = old_info
        if old_torch is None:
            sys.modules.pop('torch', None)
        else:
            sys.modules['torch'] = old_torch
    assert source_hashes() == before_sources
    result = dict(passed=True, synthetic_schema_fixtures_only=True, torch_or_gpu_tested=False,
        python38_syntax_checked=scripts, canonical_rebase_identity=True, all56_fixture_file_checks=260,
        cache_corruption_rejected=True, raw_source_corruption_rejected=True, spatial_metadata_change_rejected=True,
        split_change_rejected=True, pilot_test_rejected_even_with_flag=True,
        formal_test_needs_explicit_flag=True, intervention_test_rejected=True,
        reference_runtime_guard_checked_with_synthetic_identifiers=True,
        availability_formula_numpy_adapter_checked=True, frozen_code_unchanged=True)
    path = PROJECT / 'results' / 'reference_schema_QA.json'
    atomic_json(path, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
