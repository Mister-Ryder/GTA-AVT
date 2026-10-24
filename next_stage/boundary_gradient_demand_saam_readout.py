"""CPU-only SAAM64 fixed-moment gradient-demand diagnostic; default inert.

Every original TRAIN patch and every G0 spatial site, including padding, is
retained. Positive r·delta supports a negative-delta gain step locally; it is
not measured segmentation improvement and never authorizes training.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

BANK_SHA = '9be42890017341e4cc32fb0f17688d859a13e84643f95cd21e347462043419d6'
MODEL_SHA = '949a1efd1713a661c73762d7cef387be01205a93e425fba60d2515af4e798cf2'
HEAD_SHA = 'b61885b06fd6413efc0cf86bb664f69aa1a4e9f7fb2a26a32c373a71ce5e11bb'
WORKER_SHA = '0f425a88fa554a0ff8e71e71a77233886ffe5d84a5f88388efabe7c4763c4b0a'
PROBE_SITE = 'SAAM64_second_decoder_soft_structure_output'
SOURCE_SHA = 'f311c9fbe6ea55be70e72ed220ab0562da08b3ee267863fc0d78509a30c3654e'
WEIGHT_SHA = '7406a3ba7e5bac1a7d8ee24898cb954ff8576f37fcf5fba0e26ef9a2cb5c43e6'
NORMALIZATION_SHA = 'b1282feb482ec89fd048d556527299404cc4d1e6a0f1b206b7001ee6edc88842'
MODELS = dict(constant=(0,), A_only=(0, 1), product=(0, 2), separable=(0, 1, 2))
STRATA = dict(all16=tuple(range(16)), GT_jitter=tuple(range(8)), CT_uniform=tuple(range(8, 16)))
SOURCES = ('Dongyang', 'KiTS', 'Rider')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def model_moments(np, rows, beta):
    if beta is None:
        return None
    values = []
    for row in rows:
        energy = float(row['rr'].mean())
        prediction = float(np.einsum('ic,ij,jc->', beta, row['gram'], beta) / 64.)
        dot = float(np.sum(beta * row['cross']) / 64.)
        values.append((energy, prediction, dot, energy - 2. * dot + prediction))
    means = [math.fsum(v[i] for v in values) / len(values) for i in range(4)]
    energy, prediction, dot, sse = means
    return dict(patches=len(rows), r_energy=energy, prediction_energy=prediction,
        r_prediction_inner_product=dot, SSE=sse,
        SSE_over_r_energy=sse / energy if energy > 0. else None,
        direction_cosine=dot / math.sqrt(energy * prediction) if energy > 0. and prediction > 0. else None,
        negative_SSE_due_to_moment_arithmetic=sse < 0.)


def fit_model(np, fit_rows, columns):
    gram = np.mean(np.stack([r['gram'] for r in fit_rows]), axis=0)
    cross = np.mean(np.stack([r['cross'] for r in fit_rows]), axis=0)
    selected = gram[np.ix_(columns, columns)]
    solution, residuals, rank, singular = np.linalg.lstsq(selected, cross[list(columns)], rcond=None)
    full_rank = int(rank) == len(columns)
    beta = np.zeros((3, 64), np.float64) if full_rank else None
    if beta is not None:
        beta[list(columns)] = solution
    condition = float(singular[0] / singular[-1]) if full_rank and singular[-1] > 0. else None
    return beta, dict(columns=list(columns), rank=int(rank), expected_rank=len(columns),
        qualified_full_rank=full_rank, Gram_condition_number=condition,
        implied_design_condition_number=math.sqrt(condition) if condition is not None else None,
        singular_values=singular.tolist(), beta=None if beta is None else beta.tolist(),
        method='Unregularized FP64 OLS on fixed FIT24 case-equal/16-patch-equal spatial-mean moments; no tuning')


def aggregate_models(np, rows_by_case, case_ids, models):
    output = {}
    for name, beta in models.items():
        by_stratum = {}
        for label, slots in STRATA.items():
            per_case = {case: model_moments(np, [rows_by_case[case][s] for s in slots], beta) for case in case_ids}
            all_rows = [rows_by_case[case][s] for case in case_ids for s in slots]
            overall = model_moments(np, all_rows, beta)
            if overall is not None:
                ratios = [v['SSE_over_r_energy'] for v in per_case.values() if v['SSE_over_r_energy'] is not None]
                overall['case_equal_mean_SSE_ratio'] = math.fsum(ratios) / len(ratios) if ratios else None
                overall['ratio_supported_cases'] = len(ratios)
            by_stratum[label] = dict(case_ids=case_ids, overall=overall, per_case=per_case,
                source_groups={source: model_moments(np, [rows_by_case[c][s] for c in case_ids
                    if c.startswith(source + '_') for s in slots], beta) for source in SOURCES})
        output[name] = by_stratum
    return output


def delta_record(np, rows, delta, fit_RMS):
    if delta is None or fit_RMS is None or fit_RMS <= 0.:
        return None
    energy_r = math.fsum(float(r['rr'].mean()) for r in rows) / len(rows)
    energy_delta = math.fsum(float(np.einsum('ic,ij,jc->', delta, r['gram'], delta) / 64.) for r in rows) / len(rows)
    dots = [float(np.sum(delta * r['cross']) / 64.) for r in rows]
    dot = math.fsum(dots) / len(dots)
    # r is the derivative elasticity of the actual per-patch loss. Undo both
    # spatial-mean moments and channel averaging for its true first variation.
    budget_signals = [64. * r['M'] * value / fit_RMS for r, value in zip(rows, dots)]
    signal = math.fsum(budget_signals) / len(budget_signals)
    return dict(patches=len(rows), r_energy=energy_r, delta_energy=energy_delta,
        delta_RMS=math.sqrt(energy_delta) if energy_delta > 0. else None,
        r_delta_inner_product=dot, r_delta_inner_product_per_fit_RMS=dot / fit_RMS,
        direction_cosine=dot / math.sqrt(energy_r * energy_delta) if energy_r > 0. and energy_delta > 0. else None,
        positive_unit_budget_descent_signal=signal if energy_r > 0. and energy_delta > 0. else None,
        predicted_first_order_loss_change_for_negative_unit_RMS_step=-signal if energy_r > 0. and energy_delta > 0. else None)


def direction_check(per_case, source_groups):
    values = [v['positive_unit_budget_descent_signal'] if v is not None else None for v in per_case.values()]
    supported = [v for v in values if v is not None]
    source_values = [v['positive_unit_budget_descent_signal'] if v is not None else None for v in source_groups.values()]
    complete = len(supported) == 12 and all(v is not None for v in source_values)
    mean = math.fsum(supported) / len(supported) if supported else None
    parts = dict(all12_supported=len(supported) == 12,
        mean_positive=None if mean is None else mean > 0.,
        at_least8_positive=sum(v > 0. for v in supported) >= 8,
        all3_source_means_positive=None if any(v is None for v in source_values) else all(v > 0. for v in source_values))
    return dict(supported_cases=len(supported), positive_cases=sum(v > 0. for v in supported),
        mean_descent_signal=mean, parts=parts, passed=all(parts.values()) if complete else None,
        interpretation='Positive signal supports only a fixed small counterfactual feature-step check; not training or GTA efficacy')


def SSE_increment(baseline, separable):
    if baseline is None or separable is None:
        return None
    difference = baseline['SSE'] - separable['SSE']
    return dict(SSE_baseline_minus_separable=difference,
        relative_to_baseline_SSE=difference / baseline['SSE'] if baseline['SSE'] > 0. else None,
        baseline_SSE_over_r_energy=baseline['SSE_over_r_energy'],
        separable_SSE_over_r_energy=separable['SSE_over_r_energy'])


def analyze_arrays(np, bank, rows_by_case):
    fit, probe = bank['fit_cases'], bank['probe_cases']
    fit_rows = [rows_by_case[c][s] for c in fit for s in range(16)]
    models, fits = {}, {}
    for name, columns in MODELS.items():
        models[name], fits[name] = fit_model(np, fit_rows, columns)
    output = dict(schema='GTA-AVT-boundary-gradient-demand-saam-readout-v1', fits=fits,
        partitions={label: aggregate_models(np, rows_by_case, ids, models)
            for label, ids in (('fit24', fit), ('probe12', probe))}, increments={})
    for baseline in ('A_only', 'constant'):
        name = 'separable_minus_' + baseline
        delta = models['separable'] - models[baseline] if models['separable'] is not None and models[baseline] is not None else None
        energy = math.fsum(float(np.einsum('ic,ij,jc->', delta, r['gram'], delta) / 64.)
            for r in fit_rows) / len(fit_rows) if delta is not None else None
        scale = math.sqrt(energy) if energy is not None and energy > 0. else None
        result = dict(fit_delta_RMS=scale, normalization='One common FIT24 delta RMS; no probe or case-specific fitting', partitions={})
        for partition, ids in (('fit24', fit), ('probe12', probe)):
            result['partitions'][partition] = {}
            for stratum, slots in STRATA.items():
                cases = {c: delta_record(np, [rows_by_case[c][s] for s in slots], delta, scale) for c in ids}
                sources = {source: delta_record(np, [rows_by_case[c][s] for c in ids
                    if c.startswith(source + '_') for s in slots], delta, scale) for source in SOURCES}
                base_values = output['partitions'][partition][baseline][stratum]
                sep_values = output['partitions'][partition]['separable'][stratum]
                result['partitions'][partition][stratum] = dict(per_case=cases, source_groups=sources,
                    overall=delta_record(np, [rows_by_case[c][s] for c in ids for s in slots], delta, scale),
                    SSE_increment=dict(overall=SSE_increment(base_values['overall'], sep_values['overall']),
                        per_case={c: SSE_increment(base_values['per_case'][c], sep_values['per_case'][c]) for c in ids},
                        source_groups={s: SSE_increment(base_values['source_groups'][s], sep_values['source_groups'][s]) for s in SOURCES}))
        primary = result['partitions']['probe12']['all16']
        result['probe_direction_check'] = direction_check(primary['per_case'], primary['source_groups'])
        output['increments'][name] = result
    check = output['increments']['separable_minus_A_only']['probe_direction_check']
    output.update(primary_B_increment_check=check,
        next_direction='stop_current_B_to_gain_route' if check['passed'] is False else
            'only_supports_one_fixed_small_counterfactual_feature_step' if check['passed'] is True else 'undefined_support_no_conclusion',
        no_training_authorized=True, GTA_goal_completed=False, native_validation=False,
        scope='Posthoc TRAIN gradient diagnostic, all16 patches including empty/unknown and all actual G0 sites including padding',
        sign='r=G0*d(per-patch loss)/dG0; positive r·delta means gain step -delta has negative first-order loss variation',
        units=dict(moments='Spatial means retain64 separate channels; readout energies and dots also average channels',
            gain_budget='Delta gain field divided by one common FIT24 spatial-and-channel RMS',
            first_variation='-M*64*mean_channel_spatial(r*delta)/FIT_delta_RMS; per-patch derivative already restored, no further factor2',
            derivative='The original B2 loss averages per-sample losses; measurement restores each patch derivative at the fixed SAAM forward, not the original Adam trajectory'),
        limitations=['Linear tangent demand at fixed unadapted G0, not achieved Dice/clDice improvement.',
            'Case-equal spatial-mean OLS predicts r; it is not a new fitted segmentation network or an optimizer comparison.',
            'All four fixed models and both strata are retained; no threshold, ridge, feature or support search.',
            'Zero norms and insufficient ranks are undefined, never imputed zero. Overlapping patches are not independent cases.'])
    return output


def analyze_saam_arrays(np, bank, rows):
    result = analyze_arrays(np, bank, rows)
    flags = {name: result['increments'][name]['probe_direction_check']['passed']
        for name in ('separable_minus_A_only', 'separable_minus_constant')}
    passed = False if any(v is False for v in flags.values()) else None if any(v is None for v in flags.values()) else True
    result['primary_dual_direction_check'] = dict(parts=flags, passed=passed,
        primary_demand='Multiplicative r=F*dL/dF at the unchanged SAAM64 output',
        rule='Both increments require12 supported, mean>0, at least8 positive and all3 source means>0')
    result['next_direction'] = ('stop_current_attributes_to_stage_gain_route' if passed is False else
        'only_supports_one_predeclared_small_counterfactual_feature_step' if passed is True else 'undefined_support_no_conclusion')
    additive_rows = {case: [dict(row, cross=row['cross_additive'], rr=row['grad_rr']) for row in values]
        for case, values in rows.items()}
    additive = analyze_arrays(np, bank, additive_rows)
    for value in additive['increments'].values():
        value.pop('probe_direction_check')
    additive.pop('primary_B_increment_check'); additive.pop('next_direction')
    additive.update(descriptive_only=True, can_authorize_or_rescue=False,
        demand='Additive demand dL/dF, with cross_additive and grad_rr at the SAME SAAM64 grid',
        additive_loss_step_units='Feature units, not dimensionless relative-gain units; not a matched multiplicative perturbation budget')
    additive['sign'] = 'For additive demand g=dL/dF, positive g·delta means additive feature step -delta is a local descent direction'
    additive['units'].pop('gain_budget')
    additive['units']['feature_budget'] = 'Additive feature field divided by one common FIT24 spatial-and-channel RMS; units are feature units, not relative gain'
    additive['units']['first_variation'] = '-M*64*mean_channel_spatial(g*delta)/FIT_delta_RMS; no further factor2'
    result.update(probe_site=PROBE_SITE, channels=64, primary_operator='multiplicative',
        additive_demand_descriptive=additive, additive_cannot_replace_primary_or_rescue=True,
        all_gates_predeclared=True, no_training_authorized=True)
    return result


def load_inputs(np, roots, bank):
    require(len(roots) == 3 and len({str(Path(p).resolve()) for p in roots}) == 3, 'Three distinct completed worker outputs required')
    rows, provenance, common_state = {}, [], None
    expected = set(bank['case_ids'])
    for root in roots:
        home = Path(root); summary = read(home / 'summary.json'); inputs = read(home / 'inputs.json')
        require(summary.get('schema') == 'GTA-AVT-boundary-gradient-demand-saam-probe-v1' and
            summary.get('status') == 'verified_complete' and summary.get('mode') == 'run' and
            summary.get('patches') == 192 and summary.get('probe_site') == PROBE_SITE and summary.get('channels') == 64 and summary.get('engineering_qa_passed') is True and
            summary.get('all_model_state_unchanged') is True and summary.get('no_optimization') is True,
            'Completed formal moment worker required')
        for name, item in summary['files'].items():
            path = home / name
            try:
                path.resolve().relative_to(home.resolve())
            except ValueError:
                raise ValueError('Unsafe worker output member')
            require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Worker output bytes differ: ' + name)
        bindings = dict(source_manifest_sha256=SOURCE_SHA, bank_sha256=BANK_SHA, common_weights_sha256=WEIGHT_SHA,
            heads_sha256=HEAD_SHA, field_model_sha256=MODEL_SHA, worker_sha256=WORKER_SHA,
            normalization_sha256=NORMALIZATION_SHA)
        require(inputs.get('probe_site') == PROBE_SITE and inputs.get('channels') == 64 and
            all(inputs.get(k) == summary.get(k) == v for k, v in bindings.items()) and
            inputs['fit_cases'] == bank['fit_cases'] and inputs['probe_cases'] == bank['probe_cases'] and
            inputs['cache_before'] == inputs['cache_after'] and inputs['model_state_before'] == inputs['model_state_after'] and
            inputs['no_optimization'] is True and inputs['no_VAL_or_TEST_arrays'] is True and inputs['RNG_unchanged'] is True,
            'Fixed source, bank, independent heads, original model or unchanged state differs')
        require(sha(home / 'normalization.json') == NORMALIZATION_SHA, 'Actual normalization bytes differ')
        require(common_state is None or common_state == inputs['model_state_before'], 'Shard initial model states differ')
        common_state = inputs['model_state_before']
        members = [name for name in summary['files'] if name.startswith('cases/') and name.endswith('.moments.npz')]
        require(len(members) == 12 and len(summary['case_ids']) == 12 and
            set(summary['case_ids']) == set(inputs['case_ids']) ==
                {Path(name).name[:-len('.moments.npz')] for name in members}, 'Exactly12 cases per formal shard required')
        for name in members:
            case = Path(name).name[:-len('.moments.npz')]
            require(case in expected and case not in rows, 'Duplicate/non-TRAIN case')
            with np.load(home / name, allow_pickle=False) as data:
                require(data['slot'].dtype == np.int64 and data['M'].dtype == np.int64 and
                    data['slot'].tolist() == list(range(16)) and data['M'].shape == (16,) and bool((data['M'] > 0).all()), 'All16 fixed slots required')
                shapes = dict(gram=(16, 3, 3), cross=(16, 3, 64), cross_additive=(16, 3, 64), rr=(16, 64), grad_rr=(16, 64),
                    exactzeroG0_grad_rr=(16, 64), G0_rr=(16, 64))
                for key, shape in shapes.items():
                    require(data[key].shape == shape and data[key].dtype == np.float64 and np.isfinite(data[key]).all(), 'Invalid FP64 moment: ' + key)
                require(np.all(data['gram'][:, 0, 0] == 1.) and
                    np.allclose(data['gram'], data['gram'].transpose(0, 2, 1), rtol=0., atol=1e-12) and
                    all(bool((data[key] >= 0.).all()) for key in ('rr', 'grad_rr', 'exactzeroG0_grad_rr', 'G0_rr')),
                    'Invalid constant coordinate, symmetry or squared moment')
                metadata = read(home / 'cases' / (case + '.json'))
                records = metadata['records']
                require(metadata['case_id'] == case and metadata.get('probe_site') == PROBE_SITE and metadata.get('channels') == 64 and metadata['axes'] == ['1', 'zA', 'zAB'] and
                    metadata['moments_are_spatial_means'] is True and metadata['r_spatially_pooled_before_moments'] is False and
                    [r['slot'] for r in records] == list(range(16)) and
                    all(r['case_id'] == case and r['source'] == case.split('_')[0] and
                        r['stratum'] == ('GT_centerline_plus_jitter' if s < 8 else 'CT_uniform') and
                        r['G0_shape'][0] == 64 and int(np.prod(r['G0_shape'][1:])) == int(data['M'][s]) == r['M'] and
                        r['leaf_forward_bitwise_equal'] is True for s, r in enumerate(records)),
                    'Moment grid, population or fixed-forward identity differs')
                rows[case] = [dict(slot=s, M=int(data['M'][s]), **{key: data[key][s].copy() for key in shapes}) for s in range(16)]
        provenance.append(dict(summary_sha256=sha(home / 'summary.json'), inputs_sha256=sha(home / 'inputs.json'),
            worker_sha256=summary.get('worker_sha256'), files=summary['files']))
    require(set(rows) == expected and len(rows) == 36, 'All576 original TRAIN patches required')
    return rows, provenance


def execute(args):
    require(sha(args.bank) == BANK_SHA, 'Original fixed bank required')
    bank = read(args.bank)
    require(len(bank['fit_cases']) == 24 and len(bank['probe_cases']) == 12 and
        set(bank['fit_cases']).isdisjoint(bank['probe_cases']) and set(bank['case_ids']) == set(bank['fit_cases'] + bank['probe_cases']), 'Original24/12 partition required')
    output = Path(args.output); require(not output.exists(), 'Fresh output required')
    import numpy as np
    rows, provenance = load_inputs(np, args.inputs, bank)
    result = analyze_saam_arrays(np, bank, rows)
    result.update(bank_sha256=BANK_SHA, readout_sha256=sha(__file__), input_provenance=provenance)
    result['operating_space_descriptive'] = {}
    for partition, cases in (('fit24', bank['fit_cases']), ('probe12', bank['probe_cases'])):
        selected = [rows[c][s] for c in cases for s in range(16)]
        grad = math.fsum(float(r['grad_rr'].mean()) for r in selected) / len(selected)
        exact = math.fsum(float(r['exactzeroG0_grad_rr'].mean()) for r in selected) / len(selected)
        result['operating_space_descriptive'][partition] = dict(gradient_energy=grad,
            exact_zero_G0_gradient_energy=exact, exact_zero_G0_gradient_energy_share=exact / grad if grad > 0. else None,
            G0_energy=math.fsum(float(r['G0_rr'].mean()) for r in selected) / len(selected), not_an_unfreezing_gate=True)
    output.mkdir(parents=True)
    (output / 'analysis.json').write_bytes(encoded(result))
    text = ['固定SAAM64输出的梯度需求诊断；所有16槽与实际格点均保留，未进行分割网络训练。',
        '主判断仅为乘性r的两项probe12方向同时通过：%s。' % result['primary_dual_direction_check'],
        '方向支持不等于Dice收益；不授权解冻、延长训练或GTA成功。']
    (output / 'readout.txt').write_bytes(('\n'.join(text) + '\n').encode('utf-8'))
    return dict(status='complete', analysis_sha256=sha(output / 'analysis.json'), next_direction=result['next_direction'], training_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs=3); parser.add_argument('--bank'); parser.add_argument('--output'); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='inert', files_read=False, writes=False))); return
    require(args.inputs and args.bank and args.output, 'Three completed workers, original bank and fresh output required')
    print(json.dumps(execute(args)))


if __name__ == '__main__':
    main()
