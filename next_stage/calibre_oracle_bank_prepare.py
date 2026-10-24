"""CPU-only fixed TRAIN sparse-radius oracle-bank preparation; default inert.

No model is loaded. Only image headers and sparse geometry are interpreted;
existing cache bytes, including masks, are read only for integrity hashing.
Unknown support consumes neutral0 but remains explicitly UNKNOWN in state.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault('OMP_NUM_THREADS', '4')
HELPER_SHA = '016157cf52f7e9d9b7593435d5557dc55166881e94c59ad70bc68857186c8604'
ORACLE_SHA = '00dbec1b9230f2613a3fa90236eed9842228bcd6f6fb96f6c40f9d29eb4f86ed'
GRID = (64, 64, 64)
RECIPE = dict(patch_shape=[128, 128, 128], grid_shape=list(GRID), spacing_zyx_mm=[1., 1., 1.],
    query_sites='align_corners=False nominal cache voxel centers .5+2u, not actual receptive-field centers',
    observed_domain='Patch actual CT/cache intersection voxel outer faces [start-.5,min(start+128,cache_shape)-.5]; oracle additionally intersects conservative cache-center bounds',
    geometry_only=True, GT_mask_background_certificate=False,
    A='measurable_support uint8; unavailable including unknown consumes neutral0, not a vessel-negative target',
    R='float32 log(radius_mm/1mm) only where A1; other entries neutral0, never interpreted as radius0',
    unknown_state=255, no_bins_or_window_histograms=True,
    normalization='Not fit in shards; report stored-F32 supported R sums/squared sums/counts per case for ROOT case-equal FIT24 merge',
    radius_intervention='Within each patch ownA1 flattened order, circular shift floor(N/2); no cross-patch imputation or donor filtering',
    oracle_only=True, predicted_attribute_or_GTA_efficacy_claim=False)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            value.update(part)
    return value.hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value))


def load(path, expected, name):
    require(sha(path) == expected, 'Pinned helper changed: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def store_arrays(np, result):
    active = np.asarray(result['measurable_support'], bool)
    state = np.asarray(result['state'], np.uint8)
    radius = np.zeros(active.shape, np.float32)
    radius[active] = result['rawR'][active].astype(np.float32)
    require(active.shape == GRID and np.array_equal(active, state == 1) and
        bool(np.isfinite(radius).all()) and bool((radius[~active] == 0.).all()) and
        bool(np.isin(state, [1, 255]).all()), 'Oracle finite/unknown/support policy failed')
    return dict(A=active.astype(np.uint8), R=radius, state=state, reason=np.asarray(result['reason'], np.uint16))


def case_statistics(np, arrays):
    active = arrays['A'].astype(bool)
    values = arrays['R'][active].astype(np.float64)
    return dict(case_sum_R=math.fsum(values.tolist()),
        case_sum_R2=math.fsum(float(v) * float(v) for v in values), case_supported_count=int(len(values)),
        supported_patches=int(active.reshape(len(active), -1).any(1).sum()), total_cells=int(active.size),
        unknown_cells=int((arrays['state'] == 255).sum()))


def copy_exact(source, destination):
    with Path(source).open('rb') as src, Path(destination).open('xb') as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 ** 2)
    require(sha(source) == sha(destination) and Path(source).stat().st_size == Path(destination).stat().st_size,
        'Shared publication bytes differ')


def execute(args):
    started = time.monotonic()
    helper = load(args.helper, HELPER_SHA, 'calibre_bank_source_helper')
    oracle = load(args.oracle_helper, ORACLE_SHA, 'calibre_bank_geometry_oracle')
    request = helper.read(args.request)
    root, source, manifest, bank, records = helper.metadata(request)
    selected = request['case_ids'][:1] if args.mode == 'qa' else request['case_ids']
    active_request = dict(request, case_ids=selected)
    before = helper.cached_inputs(active_request, records)
    output, shared = Path(args.output), Path(args.shared_output)
    require(not output.exists() and not shared.exists() and output.resolve() != shared.resolve(), 'Fresh separate local and shared shard directories required')
    import numpy as np
    output.mkdir(parents=True); (output / 'cases').mkdir(); shared.mkdir(parents=True)
    write(output / 'intent.json', dict(schema='GTA-AVT-calibre-oracle-bank-shard-intent-v1', mode=args.mode,
        recipe=RECIPE, worker_sha256=sha(__file__), producer_sha256=sha(__file__), oracle_helper_sha256=ORACLE_SHA,
        source_helper_sha256=HELPER_SHA, request_sha256=sha(args.request), model_loaded=False, no_VAL_or_TEST_arrays=True))
    entries, patches = {}, 0
    for case in selected:
        record = records[case]; home = Path(record['cache_dir'])
        image = np.load(home / 'image.npy', mmap_mode='r', allow_pickle=False)
        shape = np.asarray(image.shape, np.int64)
        require(image.ndim == 3 and list(image.shape) == bank['headers'][case]['shape_zyx'] and
            float(record['spacing_mm']) == 1., 'Original1mm cache image header required')
        del image  # no image intensity is accessed or copied
        with np.load(home / 'geometry.npz', allow_pickle=False) as stream:
            geometry = {key: stream[key] for key in stream.files}
        specs = [spec for spec in bank['patches'] if spec['case_id'] == case]
        if args.mode == 'qa':
            specs = specs[:2]
        arrays_by_patch, patch_records = [], []
        for spec in specs:
            start = np.asarray(spec['start_zyx'], np.int64)
            require((start >= 0).all() and (start < shape).all(), 'Fixed patch start outside real cache')
            actual_upper = np.minimum(start + 128, shape)
            bounds = np.stack((start.astype(np.float64) - .5, actual_upper.astype(np.float64) - .5))
            result = oracle.patch_oracle(geometry, start, grid_shape=GRID, cache_shape_zyx=shape,
                spacing_zyx_mm=(1., 1., 1.), observed_bounds_mm=bounds, known_background=None)
            arrays = store_arrays(np, result); arrays_by_patch.append(arrays)
            active = arrays['A'].astype(bool)
            patch_records.append(dict(slot=spec['slot'], stratum=spec['stratum'],
                real_CT_shape_zyx=(actual_upper - start).tolist(), supplied_observed_bounds_mm=bounds.tolist(),
                effective_observed_bounds_mm=result['domain_bounds_mm'].tolist(),
                supported_count=int(active.sum()), unknown_count=int((arrays['state'] == 255).sum()),
                reason_counts={name: int(((arrays['reason'] & bit) != 0).sum()) for name, bit in oracle.REASONS.items()},
                radius_value_variance=float(arrays['R'][active].astype(np.float64).var()) if active.any() else None))
            patches += 1
            print(json.dumps(dict(case_id=case, slot=spec['slot'], patches=patches, supported_count=int(active.sum()))), flush=True)
        arrays = {key: np.stack([x[key] for x in arrays_by_patch]) for key in ('A', 'R', 'state', 'reason')}
        arrays['slot'] = np.asarray([x['slot'] for x in patch_records], np.int64)
        npz = output / 'cases' / (case + '.oracle.npz')
        with npz.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        statistics = case_statistics(np, arrays)
        receipt = dict(schema='GTA-AVT-calibre-oracle-case-v1', mode=args.mode, case_id=case,
            source=case.split('_')[0], producer_sha256=sha(__file__), oracle_helper_sha256=ORACLE_SHA,
            source_manifest_sha256=helper.MAP_SHA, bank_sha256=helper.BANK_SHA,
            geometry_sha256=record['cache_file_sha256']['geometry.npz'], grid_shape=list(GRID),
            slots=arrays['slot'].tolist(), sha256=sha(npz), bytes=npz.stat().st_size,
            **statistics, patches=patch_records, recipe=RECIPE,
            neutral_R0_is_not_a_radius_label=True, geometry_only=True, no_GT_mask_background_certificate=True)
        receipt_path = output / 'cases' / (case + '.oracle.json'); write(receipt_path, receipt)
        shared_npz = shared / npz.name; shared_receipt = shared / receipt_path.name
        copy_exact(npz, shared_npz); copy_exact(receipt_path, shared_receipt)
        entries[case] = dict(path=str(shared_npz), sha256=sha(shared_npz), receipt_path=str(shared_receipt),
            receipt_sha256=sha(shared_receipt), **statistics)
        del arrays, arrays_by_patch, geometry
    after = helper.cached_inputs(active_request, records)
    require(before == after and sha(args.oracle_helper) == ORACLE_SHA and sha(args.helper) == HELPER_SHA,
        'Frozen input/helper changed during oracle preparation')
    for name, digest in source.items():
        require(sha(root / name) == digest, 'Source42 changed')
    shared_manifest = dict(schema='GTA-AVT-calibre-oracle-bank-shard-v1', status='verified_complete', mode=args.mode,
        producer_sha256=sha(__file__), oracle_helper_sha256=ORACLE_SHA, source_manifest_sha256=helper.MAP_SHA,
        bank_sha256=helper.BANK_SHA, case_ids=selected, fit_cases=bank['fit_cases'], probe_cases=bank['probe_cases'],
        cases=entries, patches=patches, grid_shape=list(GRID), recipe=RECIPE,
        disposable_engineering_only=args.mode == 'qa', normalization_fitted=False,
        model_loaded=False, no_VAL_or_TEST_arrays=True)
    write(shared / 'manifest.json', shared_manifest); write(output / 'shared_manifest.json', shared_manifest)
    write(output / 'inputs.json', dict(schema='GTA-AVT-calibre-oracle-bank-inputs-v1', mode=args.mode,
        source_manifest_sha256=helper.MAP_SHA, bank_sha256=helper.BANK_SHA, common_weights_sha256=helper.WEIGHT_SHA,
        request_sha256=sha(args.request), producer_sha256=sha(__file__), oracle_helper_sha256=ORACLE_SHA,
        source_sha256=source, cache_before=before, cache_after=after,
        case_ids=selected, no_CT_intensities_interpreted=True, no_GT_mask_values_interpreted=True,
        model_loaded=False, no_VAL_or_TEST_arrays=True))
    write(output / 'qa.json', dict(schema='GTA-AVT-calibre-oracle-preparation-engineering-v1',
        engineering_qa_passed=True, mode=args.mode, patches=patches,
        cache_before_equals_after=before == after, unknown_state_preserved=True,
        neutral_R0_is_not_a_radius_label=True, model_loaded=False,
        scientific_utility_established=False))
    files = {p.relative_to(output).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
        for p in sorted(output.rglob('*')) if p.is_file()}
    summary = dict(schema='GTA-AVT-calibre-oracle-bank-shard-summary-v1', status='verified_complete', mode=args.mode,
        engineering_qa_passed=True, case_ids=selected, cases=entries, patches=patches, files=files,
        producer_sha256=sha(__file__), oracle_helper_sha256=ORACLE_SHA, bank_sha256=helper.BANK_SHA,
        source_manifest_sha256=helper.MAP_SHA, shared_manifest_path=str(shared / 'manifest.json'),
        shared_manifest_sha256=sha(shared / 'manifest.json'), grid_shape=list(GRID),
        normalization_fitted=False, model_loaded=False, no_VAL_or_TEST_arrays=True, elapsed_seconds=time.monotonic() - started)
    write(output / 'summary.json', summary)
    return dict(status='verified_complete', mode=args.mode, engineering_qa_passed=True, patches=patches,
        summary_sha256=sha(output / 'summary.json'), shared_manifest_sha256=sha(shared / 'manifest.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('request', 'output', 'shared-output'):
        parser.add_argument('--' + name)
    parser.add_argument('--helper', default=str(Path(__file__).with_name('boundary_attribute_probe.py')))
    parser.add_argument('--oracle-helper', default=str(Path(__file__).with_name('local_radius_oracle.py')))
    parser.add_argument('--mode', choices=('qa', 'run'), default='qa'); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='inert', recipe=RECIPE, files_read=False, model_loaded=False, outputs_written=False))); return
    require(args.request and args.output and args.shared_output, 'Fixed request and fresh local/shared outputs required')
    print(json.dumps(execute(args)))


if __name__ == '__main__':
    main()
