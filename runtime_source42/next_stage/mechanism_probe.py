"""Frozen-weight TRAIN-only q/r and geometric correspondence diagnostic."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'code'))
sys.path.insert(0, str(HERE))
import numpy as np
import torch
from common import sha256
from data import CachedData
from stage_model import StageModel


def scores(logits, mask):
    predicted = logits.sigmoid() >= .5
    truth = mask > .5
    axes = (1, 2, 3, 4)
    intersection = (predicted & truth).sum(axes).float()
    denominator = predicted.sum(axes).float()+truth.sum(axes).float()
    return dict(
        dice=((2*intersection+1e-6)/(denominator+1e-6)).cpu().tolist(),
        fp_voxels=(predicted & ~truth).sum(axes).cpu().tolist(),
        fn_voxels=(~predicted & truth).sum(axes).cpu().tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('Immutable diagnostic output exists')
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(20261023)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    data = CachedData(args.manifest)
    model = StageModel('scale_only', alpha=1.).cuda().eval()
    weights = torch.load(args.weights, map_location='cpu')
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if set(missing) != {'gdf_head.weight', 'gdf_head.bias'} or unexpected:
        raise ValueError('Unexpected checkpoint structure')
    examples, positive, empty = [], [], []
    for step in range(2048):
        sample = data.sample(20261023, step, 0, 128, .5)
        if sample['mask'].any():
            if (len(positive)<8 and sample['attribute_valid'] and
                    sample['scale'][0] == 0 and sample['structure'].sum()>0):
                positive.append(sample)
        elif len(empty)<8:
            empty.append(sample)
        if len(positive)==8 and len(empty)==8:
            break
    if len(positive)!=8 or len(empty)!=8:
        raise ValueError('Could not fill fixed enriched diagnostic bank')
    examples = positive+empty
    ids = [x['case_id'] for x in examples]
    if not set(ids).issubset(data.manifest['splits']['train']):
        raise ValueError('Non-TRAIN sample in probe')
    inputs = np.stack([x['image'] for x in examples])
    masks = np.stack([x['mask'] for x in examples])
    scale_gt = np.stack([x['scale'] for x in examples])
    structure_gt = np.stack([x['structure'] for x in examples])
    with torch.no_grad():
        probabilities = []
        for start in range(0,16,2):
            image = torch.from_numpy(inputs[start:start+2]).cuda()
            with torch.cuda.amp.autocast():
                attr = model.classifier(image)
            probabilities.append((attr['generation'].float().softmax(1).cpu(),
                                  attr['structure'].float().softmax(1).cpu()))
        scale = torch.cat([x[0] for x in probabilities]).numpy()
        structure = torch.cat([x[1] for x in probabilities]).numpy()
    correct_s, correct_t = scale.copy(), structure.copy()
    wrong_s, wrong_t = scale.copy(), structure.copy()
    # q is always that patch's predicted q. Only nonempty composition is paired.
    mapping = list(range(1,8))+[0]
    for i in range(8):
        correct_s[i,1:] = (1-scale[i,0])*scale_gt[i,1:]/scale_gt[i,1:].sum()
        correct_t[i,1:] = (1-structure[i,0])*structure_gt[i,1:]/structure_gt[i,1:].sum()
        j = mapping[i]
        wrong_s[i,1:] = (1-scale[i,0])*scale_gt[j,1:]/scale_gt[j,1:].sum()
        wrong_t[i,1:] = (1-structure[i,0])*structure_gt[j,1:]/structure_gt[j,1:].sum()
    uniform_structure = np.full_like(structure,1./3.)
    modes = [('qpred_rpred', 'qpred_rpred', None),
             ('qpred_runiform', 'qpred_runiform', None),
             ('qfixed_rpred', 'qfixed_rpred', None),
             ('qfixed_runiform', 'qfixed_runiform', None),
             ('correct_GT_r', 'qpred_rpred', (correct_s,correct_t)),
             ('repaired_GT_r', 'qpred_rpred', (wrong_s,wrong_t)),
             ('configured_constant', 'constant', None),
             ('configured_predicted_scale', 'configured', None),
             ('configured_correct_GT_scale', 'configured', (correct_s,uniform_structure)),
             ('configured_repaired_GT_scale', 'configured', (wrong_s,uniform_structure))]
    result = {}
    baseline = None
    configured_constant = None
    began = time.time()
    with torch.no_grad():
        for label, mode, override in modes:
            pieces, values = [], []
            for start in range(0,16,2):
                image = torch.from_numpy(inputs[start:start+2]).cuda()
                mask = torch.from_numpy(masks[start:start+2]).cuda()
                supplied = None if override is None else {
                    'scale': override[0][start:start+2],
                    'structure': override[1][start:start+2]}
                with torch.cuda.amp.autocast():
                    logits = model(image, condition_mode=mode, condition_override=supplied)['logits']
                values.append(scores(logits,mask))
                pieces.append(logits.float().cpu())
            logits = torch.cat(pieces)
            if baseline is None:
                baseline = logits
            if label=='configured_constant':
                configured_constant=logits
            result[label] = {
                k:[v for chunk in values for v in chunk[k]] for k in values[0]}
            result[label]['mean_abs_logit_delta_vs_predicted'] = (
                (logits-baseline).abs().mean((1,2,3,4)).tolist())
            if configured_constant is not None:
                result[label]['mean_abs_logit_delta_vs_configured_constant'] = (
                    (logits-configured_constant).abs().mean((1,2,3,4)).tolist())
            print(json.dumps({'mode':label,
                              'enriched_patch_mean_dice':float(np.mean(result[label]['dice']))}),flush=True)
    correct = np.asarray(result['configured_correct_GT_scale']['dice'][:8])
    repaired = np.asarray(result['configured_repaired_GT_scale']['dice'][:8])
    mean_dice_pp = float(np.mean(correct-repaired)*100.)
    wins = int(np.sum(correct-repaired>1e-12))
    mean_logit = float(np.mean(result['configured_correct_GT_scale'][
        'mean_abs_logit_delta_vs_configured_constant'][:8]))
    gate = dict(mean_correct_minus_repaired_foreground_patch_Dice_pp=mean_dice_pp,
                positive_foreground_patches=wins,foreground_patch_count=8,
                mean_abs_logit_response_vs_configured_constant=mean_logit,
                predeclared_thresholds=dict(mean_Dice_pp=.1,positive_patches=5,mean_abs_logit=.001),
                passed=bool(mean_dice_pp>=.1 and wins>=5 and mean_logit>=.001),
                scope='Exploratory TRAIN-enriched computational gate; not a significance test')
    report = dict(schema='GTA-AVT-next-stage-frozen-TRAIN-condition-probe-v2',
        weights_sha256=sha256(args.weights), manifest_sha256=sha256(args.manifest),
        device=torch.cuda.get_device_name(0), gpu_origin='17552',
        parameters_updated=False, models_trained=False, test_used=False,
        positive_patches=8, true_empty_patches=8,
        warning='GT-enriched TRAIN patch sensitivity, not independent case performance or deployed GT inputs',
        oracle_warning='Frozen model GT substitution changes condition distribution; failure cannot reject the framework',
        geometric_repair_mapping=mapping,
        bank=[dict(case_id=x['case_id'],start=x['start'],scale=x['scale'].tolist(),
                   structure=x['structure'].tolist()) for x in examples],
        head_scale=scale.tolist(),head_structure=structure.tolist(),
        outcomes=result,scale_only_continuation_gate=gate,elapsed_seconds=time.time()-began,
        code_sha256={p.name:sha256(p) for p in (Path(__file__),HERE/'stage_model.py')})
    (output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'complete':True,'elapsed_seconds':report['elapsed_seconds']}))


if __name__=='__main__':
    main()
