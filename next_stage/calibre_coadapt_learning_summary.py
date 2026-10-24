"""Explain completed fixed training logs after the final readout; default inert.

No model/data/endpoint evaluation, convergence criterion, selection or rescue.
Fields/losses are forward-time; gate norms are after the recorded update.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

READER_SHA = '478a41fd87b5d4def485c2a010f0c76e69ee29f089ab7185b9a5a94cd573871d'
WORKER_SHA = '231b99fefb5b17e9e71d7b24bf9a4ba74f8cb032fd05b6ab62b8fa11ecee63f8'
MODEL_SHA = '7741821580cf6fded92b2d5033aa4649bdbab51a8f218378525c28533763c942'
SEEDS = (20261030, 20261031, 20261032)
ARMS = ('C0', 'A_only', 'AR')
PARAMETERS = ('structure_guidance.morph_fusion.weight', 'structure_guidance.morph_fusion.bias',
    'anatomical_output_fusion.3.weight', 'anatomical_output_fusion.3.bias',
    'calibre_bias', 'calibre_availability_weight', 'calibre_radius_weight')
LOSSES = ('loss', 'BCE', 'Dice', 'bce_loss', 'dice_loss')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def stats(values):
    values = list(values); present = [v for v in values if finite(v)]
    mean = math.fsum(present) / len(present) if present else None
    return dict(records=len(values), finite_count=len(present), missing_count=sum(v is None for v in values),
        nonfinite_or_nonnumeric_count=sum(v is not None and not finite(v) for v in values),
        mean=mean, population_SD=math.sqrt(math.fsum((v - mean) ** 2 for v in present) / len(present)) if present else None,
        minimum=min(present) if present else None, maximum=max(present) if present else None,
        first=values[0] if values and finite(values[0]) else None,
        last=values[-1] if values and finite(values[-1]) else None)


def gradient_stats(values):
    values = list(values); result = stats(values)
    result.update(exact_zero_count=sum(finite(v) and v == 0. for v in values),
        positive_maxabs_count=sum(finite(v) and v > 0. for v in values),
        invalid_negative_maxabs_count=sum(finite(v) and v < 0. for v in values))
    return result


def nested(row, path):
    value = row
    for key in path.split('.'):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def summarize_arm(seed, arm, rows):
    require([r['step'] for r in rows] == list(range(1, 2001)), 'Completed2000 successful updates required')
    require(all(finite(r.get('loss')) and r.get('alpha') == min(1., (i + 1) / 250.) for i, r in enumerate(rows)),
        'Formal loss/alpha identity differs')
    require(all(set(r.get('gradient_maxabs', {})) == set(PARAMETERS) for r in rows), 'Actual parameter gradient inventory differs')
    fields = ('D_RMS', 'gain_absmean', 'gain_maxabs', 'zA.mean', 'zA.SD', 'zA.maxabs', 'zR.mean', 'zR.SD', 'zR.maxabs')
    blocks = []
    for begin in range(0, 2000, 250):
        selected = rows[begin:begin + 250]
        norms = {key: stats(r.get('gate_norms', {}).get(key) for r in selected) for key in ('bias', 'availability', 'radius')}
        for key in norms:
            previous = 0. if begin == 0 else rows[begin - 1].get('gate_norms', {}).get(key)
            norms[key]['previous_block_last'] = previous if finite(previous) else None
            norms[key]['last_minus_previous_block_last'] = norms[key]['last'] - previous if finite(previous) and norms[key]['last'] is not None else None
        blocks.append(dict(seed=seed, arm=arm, block=len(blocks) + 1, first_step=begin + 1, last_step=begin + 250,
            successful_updates=250, losses={key: stats(r.get(key) for r in selected) for key in LOSSES},
            alpha=stats(r['alpha'] for r in selected), gate_norms_post_update=norms,
            fields_pre_update={key: stats(nested(r.get('fields', {}), key) for r in selected) for key in fields},
            parameter_gradient_maxabs={key: gradient_stats(r['gradient_maxabs'].get(key) for r in selected) for key in PARAMETERS},
            AMP_retries=stats(r.get('amp_retries') for r in selected)))
    return dict(blocks=blocks, last_state=dict(step=2000,
        gate_norms_post_update={key: rows[-1].get('gate_norms', {}).get(key) for key in ('bias', 'availability', 'radius')},
        fields_pre_last_update={key: nested(rows[-1].get('fields', {}), key) for key in fields},
        parameter_gradient_maxabs={key: rows[-1]['gradient_maxabs'].get(key) for key in PARAMETERS}),
        all2000_parameter_gradients={key: gradient_stats(r['gradient_maxabs'].get(key) for r in rows) for key in PARAMETERS})


def paired_blocks(seed, rows):
    output = []
    for control in ('C0', 'A_only'):
        for begin in range(0, 2000, 250):
            pairs = list(zip(rows['AR'][begin:begin + 250], rows[control][begin:begin + 250]))
            values = [a['loss'] - b['loss'] for a, b in pairs]
            output.append(dict(seed=seed, comparison='AR_minus_' + control, block=begin // 250 + 1,
                first_step=begin + 1, last_step=begin + 250,
                paired_same_step_loss=stats(values),
                negative_AR_lower_loss_steps=sum(v < 0. for v in values),
                positive_AR_higher_loss_steps=sum(v > 0. for v in values), exact_ties=sum(v == 0. for v in values)))
    return output


def analyze(roots, final_analysis):
    require(len(roots) == 3 and len({str(Path(p).resolve()) for p in roots}) == 3, 'Three completed distinct run roots required')
    authority = read(final_analysis)
    require(authority.get('schema') == 'GTA-AVT-calibre-coadapt-final-readout-v1' and
        authority.get('readout_sha256') == READER_SHA and tuple(x['seed'] for x in authority['per_seed']) == SEEDS and
        len(authority['input_hashes']) == 3, 'Completed exact478 final readout must precede learning explanation')
    seeds, hashes = [], {}
    for root, seed, proof in zip(roots, SEEDS, authority['input_hashes']):
        home = Path(root)
        require(sha(home / 'summary.json') == proof['summary.json'] and sha(home / 'inputs.json') == proof['inputs.json'],
            'Run roots do not match final-reader inputs')
        summary = read(home / 'summary.json')
        require(summary.get('status') == 'verified_complete' and summary.get('mode') == 'run' and
            summary.get('seed') == seed and summary.get('worker_sha256') == WORKER_SHA and summary.get('model_sha256') == MODEL_SHA,
            'Completed exact worker/model/seed required')
        logs = {}; hashes[str(seed)] = {}
        for arm in ARMS:
            name = arm + '/training.jsonl'; path = home / name
            require(sha(path) == proof[name] == summary['files'][name]['sha256'], 'Training log differs from final-reader verified bytes')
            logs[arm] = [json.loads(line) for line in path.read_bytes().splitlines()]
            hashes[str(seed)][arm] = sha(path)
        arms = {arm: summarize_arm(seed, arm, logs[arm]) for arm in ARMS}
        seeds.append(dict(seed=seed, arms=arms, paired_loss_blocks=paired_blocks(seed, logs)))
    return dict(schema='GTA-AVT-calibre-coadapt-learning-explanation-v1', source_sha256=sha(__file__),
        final_readout_source_sha256=READER_SHA, final_analysis_sha256=sha(final_analysis),
        worker_sha256=WORKER_SHA, model_sha256=MODEL_SHA, training_log_hashes=hashes, per_seed=seeds,
        method=dict(blocks='Exactly8 consecutive250-successful-update blocks per arm, all three seeds retained',
            missing='BCE/Dice and other unrecorded components remain null; total loss is not decomposed retrospectively',
            timing='Loss/fields describe accepted pre-update forward; gradient maxabs after backward, before optimizer step; gate norms after update',
            gradients='Per-parameter-TENSOR maxabs only; None, exactzero and positive values remain distinct; not channel-wise gradient vectors',
            paired='AR-control losses compared at identical sample-plan steps; this describes whole learned trajectories, not instantaneous attribute causality',
            interpretation='Random patch composition changes between blocks; slopes and late loss do not certify convergence or justify extending training'),
        no_new_success_or_convergence_criterion=True, no_selection_or_rescue=True, scientific_verdict_modified=False,
        new_training_authorized=False, GTA_goal_completed=False, oracle_not_CT_only=True)


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def write_csv(path, rows):
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(rows)


def execute(args):
    output = Path(args.output); require(not output.exists(), 'Fresh explanation output required')
    result = analyze(args.inputs, args.final_analysis)
    blocks, gradients, pairs, texts = [], [], [], ['完成后的固定训练轨迹说明；不建立新成功/收敛标准，不救回实验结论。']
    for seed in result['per_seed']:
        for arm, record in seed['arms'].items():
            for block in record['blocks']:
                row = {key: block[key] for key in ('seed', 'arm', 'block', 'first_step', 'last_step', 'successful_updates')}
                for key, value in block['losses'].items():
                    for stat in ('finite_count', 'mean', 'population_SD', 'first', 'last'):
                        row[key + '_' + stat] = value[stat]
                for key, value in block['gate_norms_post_update'].items():
                    for stat in ('mean', 'population_SD', 'first', 'last', 'last_minus_previous_block_last'):
                        row['post_' + key + '_norm_' + stat] = value[stat]
                for key in ('D_RMS', 'gain_absmean', 'gain_maxabs'):
                    for stat in ('mean', 'population_SD', 'minimum', 'maximum', 'first', 'last'):
                        row['pre_' + key + '_' + stat] = block['fields_pre_update'][key][stat]
                blocks.append(row)
                for parameter, value in block['parameter_gradient_maxabs'].items():
                    gradients.append(dict(seed=seed['seed'], arm=arm, block=block['block'], parameter=parameter, **value))
            radius = record['all2000_parameter_gradients']['calibre_radius_weight']
            texts.append('seed %s / %s：半径参数张量梯度有限=%s，精确0=%s，正maxabs=%s，None=%s；末次gate norms=%s。' % (
                seed['seed'], arm, radius['finite_count'], radius['exact_zero_count'], radius['positive_maxabs_count'],
                radius['missing_count'], json.dumps(record['last_state']['gate_norms_post_update'], ensure_ascii=False)))
        for pair in seed['paired_loss_blocks']:
            values = pair['paired_same_step_loss']
            row = {key: value for key, value in pair.items() if key != 'paired_same_step_loss'}
            row.update({'loss_difference_' + key: value for key, value in values.items()})
            pairs.append(row)
    texts.extend(['BCE/Dice若原日志未分别记录，保持null，不由总loss反推。',
        '第1个250步块含alpha爬坡；字段/loss/梯度与更新后的参数norm时间不同。',
        '非零梯度或非零gate只说明该路径被优化器触及；不等于条件对应、泛化或分割收益。',
        '随机区块构成影响窗口loss；同样本配对差也包含全部学习轨迹差异，不能证明收敛、梯度竞争或允许追加训练。'])
    output.mkdir(parents=True)
    (output / 'analysis.json').write_bytes(encoded(result)); (output / 'readout.txt').write_bytes(('\n'.join(texts) + '\n').encode('utf-8'))
    write_csv(output / 'blocks.csv', blocks); write_csv(output / 'parameter_gradients.csv', gradients); write_csv(output / 'paired_loss.csv', pairs)
    return dict(status='complete', analysis_sha256=sha(output / 'analysis.json'), block_rows=len(blocks),
        gradient_rows=len(gradients), paired_rows=len(pairs), scientific_verdict_modified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs=3); parser.add_argument('--final-analysis'); parser.add_argument('--output')
    parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='inert', inputs_read=False, outputs_written=False))); return
    require(args.inputs and args.final_analysis and args.output, 'Completed final readout, same three runs and fresh output required')
    print(json.dumps(execute(args)))


if __name__ == '__main__':
    main()
