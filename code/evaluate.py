"""Explicit frozen-run evaluation. Training never invokes the final test split."""
import argparse
from pathlib import Path
import torch
from common import atomic_json, load_json, sha256, source_hashes, load_trusted_checkpoint
from data import CachedData
from model import GTAAVT
from inference import evaluate_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', default='best.pt')
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--condition-mode', choices=('predicted', 'constant', 'permuted'), default='predicted')
    parser.add_argument('--allow-diagnostic-test', action='store_true')
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    run = Path(args.run)
    record = load_json(run / 'config.json')
    config = record['recipe']
    if source_hashes() != record['source_sha256']:
        raise ValueError('Evaluation source differs from frozen training source; use the saved run/source snapshot')
    if args.split == 'test' and not record['selection_valid'] and not args.allow_diagnostic_test:
        raise ValueError('A pilot/partial-validation model is not eligible for final-test results')
    data = CachedData(record['manifest_path'])
    if sha256(record['manifest_path']) != record['manifest_sha256']:
        raise ValueError('Manifest changed after training')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(config.get('cpu_threads', 4))
    model = GTAAVT(config['method'], config['width']).to(device)
    checkpoint_path = run / args.checkpoint
    checkpoint = load_trusted_checkpoint(checkpoint_path, device)
    if checkpoint['signature'] != record:
        raise ValueError('Checkpoint and frozen run configuration disagree')
    model.load_state_dict(checkpoint['model'])
    evaluation_signature = {'run': str(run.resolve()), 'split': args.split,
                'checkpoint_sha256': sha256(checkpoint_path), 'step': checkpoint['step'],
                'condition_mode': args.condition_mode, 'threshold': .5, 'postprocessing': 'none',
                'final_test_eligible': record['selection_valid'], 'case_ids': data.manifest['splits'][args.split],
                'manifest_sha256': record['manifest_sha256'], 'source_sha256': source_hashes()}
    protocol_path = Path(args.output) / 'evaluation_protocol.json'
    if protocol_path.exists():
        if load_json(protocol_path) != evaluation_signature:
            raise ValueError('Existing evaluation output belongs to a different checkpoint/condition/source')
        if not args.reuse:
            raise ValueError('Existing evaluation output is immutable; use --reuse for an exact interrupted evaluation')
    atomic_json(protocol_path, evaluation_signature)
    evaluate_cases(model, data, data.manifest['splits'][args.split], device, args.output,
                   patch=config['inference_patch_size'], batch_size=config['inference_batch_size'],
                   overlap=config['inference_overlap'], condition_mode=args.condition_mode, reuse=args.reuse)


if __name__ == '__main__':
    main()
