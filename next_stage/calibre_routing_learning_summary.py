"""Explain completed three-axis routing training logs after final readout; default inert.

No model/data/endpoint evaluation, convergence criterion, selection or rescue.
Fields/losses are forward-time; gate norms are after the recorded update.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

READER_SHA = 'b83544ee7a9c7664a1786764ef78be7580b6c979e4ce3794d5ede21c020f9fe1'
WORKER_SHA = 'a9320edc72b1dcde9fe4486355566bf9c491b78d109bf5315f578e08db32c9ef'
MODEL_SHA = '9acd1c9d8b04fdef3e5a7e4cefe9d8f0d234978093eaf09d7d2a03a64288f58e'
SEEDS = (20261030, 20261031, 20261032)
ARMS = ('C0', 'A_only', 'AR')
PARAMETERS = ('structure_guidance.morph_fusion.weight', 'structure_guidance.morph_fusion.bias',
    'anatomical_output_fusion.3.weight', 'anatomical_output_fusion.3.bias') + tuple(
    'structure_guidance.morphology_convs.%d.%s' % (i, field)
    for i in range(4) for field in ('weight', 'bias')) + (
    'calibre_bias', 'calibre_availability_weight', 'calibre_radius_weight')
LOSSES = ('loss',)
FIELD_PATHS = ('legacy_axis1_only.D_RMS', 'legacy_axis1_only.gain_absmean', 'legacy_axis1_only.gain_maxabs',
    'zA.mean', 'zA.SD', 'zA.maxabs', 'zR.mean', 'zR.SD', 'zR.maxabs',
    'route_coeff_min', 'route_coeff_max', 'route_coeff_sum_residual_maxabs') + tuple(
        'routing_axes.%d.%s' % (index, key) for index in range(3)
        for key in ('gain_RMS', 'gain_absmean', 'gain_maxabs', 'direction_RMS'))


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
        if isinstance(value, list) and key.isdigit() and int(key) < len(value):
            value = value[int(key)]
        elif isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def summarize_arm(seed, arm, rows):
    require([r['step'] for r in rows] == list(range(1, 2001)), 'Completed2000 successful updates required')
    require(all(finite(r.get('loss')) and r.get('alpha') == min(1., (i + 1) / 250.) for i, r in enumerate(rows)),
        'Formal loss/alpha identity differs')
    require(all(set(r.get('gradient_maxabs', {})) == set(PARAMETERS) for r in rows), 'Actual parameter gradient inventory differs')
    require(all(isinstance(r.get('coefficient_row_gradient_maxabs'), dict) and
        set(r['coefficient_row_gradient_maxabs']) == {'calibre_bias', 'calibre_availability_weight', 'calibre_radius_weight'} and
        all(isinstance(value, list) and len(value) == 3 for value in r['coefficient_row_gradient_maxabs'].values()) for r in rows),
        'Three-axis coefficient row-gradient records required')
    require(all(isinstance(r.get('fields', {}).get('routing_axes'), list) and len(r['fields']['routing_axes']) == 3 and
        [axis['axis'] for axis in r['fields']['routing_axes']] == [1, 2, 3] for r in rows), 'Three ordered routing-axis traces required')
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
            fields_pre_update={key: stats(nested(r.get('fields', {}), key) for r in selected) for key in FIELD_PATHS},
            parameter_gradient_maxabs={key: gradient_stats(r['gradient_maxabs'].get(key) for r in selected) for key in PARAMETERS},
            radius_axis_gradient_maxabs={str(index + 1): gradient_stats(
                r['coefficient_row_gradient_maxabs']['calibre_radius_weight'][index] for r in selected) for index in range(3)},
            AMP_retries=stats(r.get('amp_retries') for r in selected)))
    return dict(blocks=blocks, last_state=dict(step=2000,
        gate_norms_post_update={key: rows[-1].get('gate_norms', {}).get(key) for key in ('bias', 'availability', 'radius')},
        fields_pre_last_update={key: nested(rows[-1].get('fields', {}), key) for key in FIELD_PATHS},
        parameter_gradient_maxabs={key: rows[-1]['gradient_maxabs'].get(key) for key in PARAMETERS},
        radius_axis_gradient_maxabs=rows[-1]['coefficient_row_gradient_maxabs']['calibre_radius_weight']),
        all2000_parameter_gradients={key: gradient_stats(r['gradient_maxabs'].get(key) for r in rows) for key in PARAMETERS},
        all2000_radius_axis_gradients={str(index + 1): gradient_stats(
            r['coefficient_row_gradient_maxabs']['calibre_radius_weight'][index] for r in rows) for index in range(3)})


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


def analyze(roots, final_analysis, final_analysis_sha256):
    require(len(roots) == 3 and len({str(Path(p).resolve()) for p in roots}) == 3, 'Three completed distinct run roots required')
    require(sha(final_analysis) == final_analysis_sha256, 'Explicit completed final-analysis SHA differs')
    authority = read(final_analysis)
    require(authority.get('schema') == 'GTA-AVT-calibre-routing-coadapt-final-readout-v1' and
        authority.get('readout_sha256') == READER_SHA and tuple(x['seed'] for x in authority['per_seed']) == SEEDS and
        len(authority['input_hashes']) == 3, 'Completed exactb83544 final readout must precede learning explanation')
    seeds, hashes, plan_hashes = [], {}, {}
    for root, seed, proof in zip(roots, SEEDS, authority['input_hashes']):
        home = Path(root)
        require(sha(home / 'summary.json') == proof['summary.json'] and sha(home / 'inputs.json') == proof['inputs.json'],
            'Run roots do not match final-reader inputs')
        summary = read(home / 'summary.json')
        require(summary.get('status') == 'verified_complete' and summary.get('mode') == 'run' and
            summary.get('seed') == seed and summary.get('worker_sha256') == WORKER_SHA and summary.get('model_sha256') == MODEL_SHA,
            'Completed exact worker/model/seed required')
        inputs = read(home / 'inputs.json')
        require(set(inputs['trainable_parameter_names']) == set(PARAMETERS), 'Exact15 parameter-tensor inventory required')
        plan_path = home / 'training_plan.json'
        require(sha(plan_path) == proof['training_plan.json'] == summary['files']['training_plan.json']['sha256'],
            'Paired training-plan bytes differ from final-reader verified source')
        plan = read(plan_path)
        require(plan['common_to_all_three_arms'] is True and len(plan['plan']) == 2000 and
            all(len(batch) == 2 for batch in plan['plan']) and plan['sha256'] == inputs['training_plan_sha256'],
            'Paired same-plan claim requires all2000 original B2 steps')
        plan_hashes[str(seed)] = sha(plan_path)
        logs = {}; hashes[str(seed)] = {}
        for arm in ARMS:
            name = arm + '/training.jsonl'; path = home / name
            require(sha(path) == proof[name] == summary['files'][name]['sha256'], 'Training log differs from final-reader verified bytes')
            logs[arm] = [json.loads(line) for line in path.read_bytes().splitlines()]
            hashes[str(seed)][arm] = sha(path)
        arms = {arm: summarize_arm(seed, arm, logs[arm]) for arm in ARMS}
        seeds.append(dict(seed=seed, arms=arms, paired_loss_blocks=paired_blocks(seed, logs)))
    return dict(schema='GTA-AVT-calibre-routing-coadapt-learning-explanation-v1', source_sha256=sha(__file__),
        final_readout_source_sha256=READER_SHA, final_analysis_sha256=sha(final_analysis),
        worker_sha256=WORKER_SHA, model_sha256=MODEL_SHA, training_log_hashes=hashes, training_plan_hashes=plan_hashes,
        parameter_tensors=15, trainable_parameters=115458, per_seed=seeds,
        final_scientific_verdict={key: authority.get(key) for key in ('primary_screen_passed', 'all3_screens_passed',
            'all3_original_primary_screens_passed', 'all3_CT_uniform_transfer_safety_passed', 'all3_SOURCE_safety_passed',
            'capacity_continue', 'GTA_goal_completed')},
        method=dict(blocks='Exactly8 consecutive250-successful-update blocks per arm, all three seeds retained',
            missing='BCE/Dice and other unrecorded components remain null; total loss is not decomposed retrospectively',
            timing='Loss/fields describe accepted pre-update forward; gradient maxabs after backward, before optimizer step; gate norms after update',
            gradients='Per-parameter-TENSOR maxabs only; None, exactzero and positive values remain distinct; not channel-wise gradient vectors',
            radius_axis_gradients='Three recorded R-row maxabs scalars summarize64 entries each; finite/None/zero/nonzero counts, not full row gradients or per-row weight norms',
            routing='Each axis g_RMS/absmean/maxabs and D_RMS reported separately. Legacy g/D fields refer only to axis1, never total routing gain or energy',
            coefficient_bounds='Observed q min/max/sum residual are forward scalars; not feature-energy conservation or a new success gate',
            branch_observations='Eight extra branch weight/bias maxabs scalars are recorded; no branch weight norms, parameter changes or gradient vectors are reconstructed. These gradients combine ordinary F0 and conditional D paths',
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
    result = analyze(args.inputs, args.final_analysis, args.final_analysis_sha256)
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
                for key in FIELD_PATHS:
                    for stat in ('mean', 'population_SD', 'minimum', 'maximum', 'first', 'last'):
                        row['pre_' + key + '_' + stat] = block['fields_pre_update'][key][stat]
                for stat in ('mean', 'first', 'last'):
                    row['alpha_' + stat] = block['alpha'][stat]
                row['AMP_retries_mean'] = block['AMP_retries']['mean']
                row['AMP_retries_maximum'] = block['AMP_retries']['maximum']
                blocks.append(row)
                for parameter, value in block['parameter_gradient_maxabs'].items():
                    gradients.append(dict(seed=seed['seed'], arm=arm, block=block['block'],
                        kind='parameter_tensor', parameter=parameter, axis=None, **value))
                for axis, value in block['radius_axis_gradient_maxabs'].items():
                    gradients.append(dict(seed=seed['seed'], arm=arm, block=block['block'],
                        kind='radius_axis_row', parameter='calibre_radius_weight', axis=int(axis), **value))
            radius = record['all2000_parameter_gradients']['calibre_radius_weight']
            texts.append('seed %s / %s：半径参数张量梯度有限=%s，精确0=%s，正maxabs=%s，None=%s；末次gate norms=%s。' % (
                seed['seed'], arm, radius['finite_count'], radius['exact_zero_count'], radius['positive_maxabs_count'],
                radius['missing_count'], json.dumps(record['last_state']['gate_norms_post_update'], ensure_ascii=False)))
            for axis, values in record['all2000_radius_axis_gradients'].items():
                texts.append('  R axis%s：有限=%s，精确0=%s，正maxabs=%s，None=%s，非法负maxabs=%s。' % (
                    axis, values['finite_count'], values['exact_zero_count'], values['positive_maxabs_count'],
                    values['missing_count'], values['invalid_negative_maxabs_count']))
            texts.append('  末次forward三轴/系数标量=%s。' % json.dumps(
                {key: value for key, value in record['last_state']['fields_pre_last_update'].items()
                    if key.startswith('routing_axes.') or key.startswith('route_coeff_')}, ensure_ascii=False))
        for pair in seed['paired_loss_blocks']:
            values = pair['paired_same_step_loss']
            row = {key: value for key, value in pair.items() if key != 'paired_same_step_loss'}
            row.update({'loss_difference_' + key: value for key, value in values.items()})
            pairs.append(row)
    texts.extend(['本worker只记录main loss；不从总loss反推未记录的BCE/Dice分项。',
        '第1个250步块含alpha爬坡；字段/loss/梯度与更新后的参数norm时间不同。',
        '15个参数张量共115458参数；新增8项是原四分支的weight/bias梯度maxabs观测。原日志没有分支权重norm或完整梯度向量，不重建权重变化，也不把其梯度单独归因条件D路径。',
        '三轴R各行只记录64项梯度的maxabs；总3×64系数norm不能拆成各行weight norm。',
        'legacy gain/D_RMS仅axis1，不能称总gain或总响应能量；三轴g/D RMS分别报告。q界限与sum残差是本forward观测，不是特征能量守恒。',
        '非零梯度或非零gate只说明该路径被优化器触及；不等于条件对应、泛化或分割收益。',
        '随机区块构成影响窗口loss；同样本配对差也包含全部学习轨迹差异，不能证明收敛、梯度竞争或允许追加训练。'])
    output.mkdir(parents=True)
    (output / 'analysis.json').write_bytes(encoded(result)); (output / 'readout.txt').write_bytes(('\n'.join(texts) + '\n').encode('utf-8'))
    write_csv(output / 'blocks.csv', blocks); write_csv(output / 'gradients.csv', gradients); write_csv(output / 'paired_loss.csv', pairs)
    return dict(status='complete', analysis_sha256=sha(output / 'analysis.json'), block_rows=len(blocks),
        gradient_rows=len(gradients), paired_rows=len(pairs), scientific_verdict_modified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs=3); parser.add_argument('--final-analysis'); parser.add_argument('--final-analysis-sha256'); parser.add_argument('--output')
    parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='inert', inputs_read=False, outputs_written=False))); return
    require(args.inputs and args.final_analysis and args.final_analysis_sha256 and args.output,
        'Completed final readout+SHA, same three runs and fresh output required')
    print(json.dumps(execute(args)))


if __name__ == '__main__':
    main()
