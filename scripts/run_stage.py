"""Run the packaged, unchanged v0.1 worker with explicit external inputs.

This wrapper writes only a local request and then invokes the original worker.
It does not download data, construct an oracle bank, or obtain common weights.
The default is inert; --execute is required for GPU work.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BANK_SHA = '9be42890017341e4cc32fb0f17688d859a13e84643f95cd21e347462043419d6'
ORACLE_SHA = '98c639be40fea8df7d72e9fc97835e8147c87c47fe1ce57e317f232794c2cde5'
MODEL_SHA = '9acd1c9d8b04fdef3e5a7e4cefe9d8f0d234978093eaf09d7d2a03a64288f58e'
HELPER_SHA = '00dbec1b9230f2613a3fa90236eed9842228bcd6f6fb96f6c40f9d29eb4f86ed'
SOURCE_SHA = 'f311c9fbe6ea55be70e72ed220ab0562da08b3ee267863fc0d78509a30c3654e'
WEIGHT_SHA = '7406a3ba7e5bac1a7d8ee24898cb954ff8576f37fcf5fba0e26ef9a2cb5c43e6'
SEED_PORT = {20261030: 34153, 20261031: 17552, 20261032: 38773}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', help='External 1mm prepared cache manifest')
    parser.add_argument('--bank', help='External original fixed TRAIN36 576-patch bank')
    parser.add_argument('--oracle-bank', help='External complete TRAIN36 radius-oracle bank')
    parser.add_argument('--weights', help='External common7406 tensor-only weights')
    parser.add_argument('--seed', type=int, choices=tuple(SEED_PORT))
    parser.add_argument('--output', help='Fresh worker output folder; sibling request must also be absent')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='inert', inputs_read=False, GPU=False)))
        return
    if not all((args.manifest, args.bank, args.oracle_bank, args.weights, args.seed, args.output)):
        parser.error('All external inputs, seed and fresh output are required with --execute')
    manifest, bank, oracle, weights = [Path(x).resolve() for x in
        (args.manifest, args.bank, args.oracle_bank, args.weights)]
    for path, expected in ((bank, BANK_SHA), (oracle, ORACLE_SHA), (weights, WEIGHT_SHA),
            (ROOT / 'runtime_source42/source_manifest.json', SOURCE_SHA),
            (ROOT / 'next_stage/calibre_routing_stage_model.py', MODEL_SHA),
            (ROOT / 'next_stage/local_radius_oracle.py', HELPER_SHA)):
        if sha(path) != expected:
            raise ValueError('Required input/source identity differs: ' + str(path))
    bank_value = json.loads(bank.read_bytes())
    output = Path(args.output).resolve()
    request_path = output.with_name(output.name + '_request.json')
    if output.exists() or request_path.exists():
        raise ValueError('Output and sibling request must be fresh')
    output.parent.mkdir(parents=True, exist_ok=True)
    request = dict(schema='GTA-AVT-boundary-attribute-probe-request-v1',
        port=SEED_PORT[args.seed], runtime=sys.executable,
        source_root=str(ROOT / 'runtime_source42'),
        source_manifest=str(ROOT / 'runtime_source42/source_manifest.json'),
        source_manifest_sha256=SOURCE_SHA, manifest=str(manifest),
        manifest_sha256=sha(manifest), bank=str(bank), bank_sha256=BANK_SHA,
        weights=str(weights), weights_sha256=WEIGHT_SHA,
        case_ids=bank_value['probe_cases'], settings=dict(patch_size=128, cell_size=16,
            shell_mm=2., halo_voxels=2, FOV_censor_voxels=2, minimum_shell_voxels=16,
            HU_clip=[-200, 1000], batch_size=2, AMP=True, probability_threshold=.5,
            spacing_mm=1., conditioning='constant', alpha=0., q0=.2, s0=.2,
            augmentation=False))
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    command = [sys.executable, '-B', str(ROOT / 'next_stage/calibre_routing_pilot.py'),
        '--execute', '--mode', 'run', '--request', str(request_path),
        '--request-sha256', sha(request_path), '--bank', str(bank), '--bank-sha256', BANK_SHA,
        '--oracle-bank', str(oracle), '--oracle-bank-sha256', ORACLE_SHA,
        '--oracle-helper', str(ROOT / 'next_stage/local_radius_oracle.py'),
        '--oracle-helper-sha256', HELPER_SHA, '--model',
        str(ROOT / 'next_stage/calibre_routing_stage_model.py'), '--model-sha256', MODEL_SHA,
        '--helper', str(ROOT / 'next_stage/boundary_attribute_probe.py'),
        '--old-pilot', str(ROOT / 'next_stage/boundary_field_pilot.py'),
        '--seed', str(args.seed), '--port', str(SEED_PORT[args.seed]), '--output', str(output)]
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
