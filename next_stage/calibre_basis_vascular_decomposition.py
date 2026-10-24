"""Posthoc CPU decomposition of completed same-weight radius effects.

No new scientific gate, fitting, data/model evaluation or original verdict change.
Default inert. Each case pools fine nodes across its selected patches before
case-equal summaries. Overlapping/repeated nodes are not independent subjects.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

READER_SHA = '1f7fe250df11b96f831850c9eccafa5a4ba3e51d0b3dd224cea9953d95542d44'
WORKER_SHA = '68840bce80d3576c353fe6c43c49c5ea7bfc7db44e772e9bd687283af1198f7c'
MODEL_SHA = '6f5e6fcf84600c737fc8844c0fdce6839f980d19d734512686f374cce1a0c294'
SEEDS = (20261030, 20261031, 20261032)
FRACTIONS = ('dice', 'cldice', 'centerline_recall', 'centerline_precision')
STRATA = ('primary', 'all16', 'GT_jitter_all8', 'CT_uniform_all8', 'uniform_known_nonempty')
EFFECTS = tuple(k + '_pp' for k in FRACTIONS) + ('fine_recall_pp', 'BCE_correct_minus_permuted', 'FP', 'FN', 'TP')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def selected(row, stratum):
    return (row['primary'] if stratum == 'primary' else True if stratum == 'all16' else
        row['slot'] < 8 if stratum == 'GT_jitter_all8' else
        row['slot'] >= 8 if stratum == 'CT_uniform_all8' else row['slot'] >= 8 and row['known_nonempty'])


def case_metrics(rows):
    output = dict(patches=len(rows), slots=[r['slot'] for r in rows], effects={}, levels={}, support={})
    for metric in FRACTIONS + ('BCE', 'FP', 'FN', 'TP'):
        own = [r['slot'] for r in rows if r['normal'][metric] is not None]
        wrong = [r['slot'] for r in rows if r['permuted'][metric] is not None]
        paired = [r for r in rows if r['slot'] in own and r['slot'] in wrong]
        delta = mean((100. if metric in FRACTIONS else 1.) * (r['normal'][metric] - r['permuted'][metric]) for r in paired)
        key = metric + '_pp' if metric in FRACTIONS else 'BCE_correct_minus_permuted' if metric == 'BCE' else metric
        output['effects'][key] = delta
        output['levels'][metric] = dict(correct=mean(r['normal'][metric] for r in rows if r['normal'][metric] is not None),
            permuted=mean(r['permuted'][metric] for r in rows if r['permuted'][metric] is not None))
        output['support'][metric] = dict(correct_slots=own, permuted_slots=wrong,
            paired_slots=[r['slot'] for r in paired], supports_identical=own == wrong)
    nodes = sum(r['normal']['fine_nodes'] for r in rows)
    require(nodes == sum(r['permuted']['fine_nodes'] for r in rows), 'Fine GT support changed')
    correct = sum(r['normal']['fine_covered'] for r in rows)
    wrong = sum(r['permuted']['fine_covered'] for r in rows)
    output['fine_nodes'] = nodes; output['fine_covered_correct'] = correct; output['fine_covered_permuted'] = wrong
    output['levels']['fine_recall'] = dict(correct=correct / nodes if nodes else None, permuted=wrong / nodes if nodes else None)
    output['effects']['fine_recall_pp'] = 100. * (correct / nodes - wrong / nodes) if nodes else None
    output['radius_sites'] = sum(r['radius_moments']['count'] for r in rows)
    output['mean_abs_logit_response'] = mean(r['mean_abs_logit_response'] for r in rows)
    return output


def group_metrics(cases, ids):
    return dict(case_ids=ids, expected_cases=len(ids), metrics={key: dict(
        mean=mean(cases[c]['effects'][key] for c in ids if cases[c]['effects'][key] is not None),
        supported_case_ids=[c for c in ids if cases[c]['effects'][key] is not None],
        positive_cases=sum(cases[c]['effects'][key] is not None and cases[c]['effects'][key] > 0 for c in ids),
        negative_cases=sum(cases[c]['effects'][key] is not None and cases[c]['effects'][key] < 0 for c in ids),
        ties=sum(cases[c]['effects'][key] == 0 for c in ids)) for key in EFFECTS})


def analyze(inputs, final_path, final_sha):
    require(sha(final_path) == final_sha, 'Explicit completed final-analysis SHA differs')
    final = read(final_path)
    require(final['schema'] == 'GTA-AVT-calibre-basis-coadapt-final-readout-v1' and final['readout_sha256'] == READER_SHA and
        tuple(x['seed'] for x in final['per_seed']) == SEEDS and len(inputs) == 3, 'Exact completed basis final readout required')
    ids = final['probe_cases']; output = []; hashes = []
    for root, seed, proof, verdict in zip(inputs, SEEDS, final['input_hashes'], final['per_seed']):
        root = Path(root); summary = read(root / 'summary.json')
        require(sha(root / 'summary.json') == proof['summary.json'] and summary['status'] == 'verified_complete' and
            summary['mode'] == 'run' and summary['seed'] == seed and summary['worker_sha256'] == WORKER_SHA and
            summary['model_sha256'] == MODEL_SHA, 'Completed matched run identity differs')
        files = ('AR/mechanism.jsonl', 'AR/patches.jsonl')
        for name in files:
            require(sha(root / name) == proof[name] == summary['files'][name]['sha256'], 'Final-bound rows changed')
        normal = [json.loads(r) for r in (root / files[1]).read_bytes().splitlines()]
        rows = [json.loads(r) for r in (root / files[0]).read_bytes().splitlines()]
        require([(r['case_id'], r['slot']) for r in rows] == [(c, s) for c in ids for s in range(16)] ==
            [(r['case_id'], r['slot']) for r in normal], 'All192 ordered rows required')
        for row, own in zip(rows, normal):
            require(row['normal'] == own['metrics'] and row['primary'] == own['primary'] and
                row['permutation']['own_A_and_support_preserved'] is True and
                row['permutation']['correct_radius_multiset_sha256'] == row['permutation']['permuted_radius_multiset_sha256'],
                'Same-weight normal/intervention binding differs')
            row['known_nonempty'] = own['known_nonempty']
        strata = {}
        for stratum in STRATA:
            cases = {c: case_metrics([r for r in rows if r['case_id'] == c and selected(r, stratum)]) for c in ids}
            strata[stratum] = dict(per_case=cases, all_cases=group_metrics(cases, ids),
                sources={s: group_metrics(cases, [c for c in ids if c.startswith(s + '_')]) for s in ('Dongyang', 'KiTS', 'Rider')})
        primary = strata['primary']['per_case']
        require(mean(primary[c]['effects']['dice_pp'] for c in ids) ==
            verdict['mechanisms']['own_A_half_ring_radius']['mean_correct_minus_permuted_Dice_pp'], 'Original Dice mechanism arithmetic changed')
        worst = sorted(ids, key=lambda c: (primary[c]['effects']['dice_pp'], c))[:2]
        fixed = verdict['arms']['AR']['own_worst2']['case_ids']
        r15 = [dict(slot=r['slot'], stratum=r['stratum'], primary=r['primary'], known_nonempty=r['known_nonempty'],
            **case_metrics([r])) for r in rows if r['case_id'] == 'Rider_R15']
        output.append(dict(seed=seed, original_screen=verdict['screen'], original_primary_screen=verdict['original_primary_screen'],
            original_CT_uniform_transfer_safety=verdict['CT_uniform_transfer_safety'], strata=strata,
            mechanism_effect_worst2=group_metrics(primary, worst),
            original_AR_normal_Dice_worst2=group_metrics(primary, fixed), Rider_R15_patchwise=r15))
        hashes.append({name: sha(root / name) for name in files})
    return dict(schema='GTA-AVT-calibre-basis-vascular-decomposition-v1', source_sha256=sha(__file__),
        final_analysis_sha256=final_sha, final_readout_sha256=READER_SHA, per_seed=output, input_hashes=hashes,
        method=dict(design='Posthoc decomposition after completed original basis result; no new gate or selection',
            effects='Correct minus same-weight half-ring permuted; fraction differences multiplied100pp; BCE/FP/FN/TP keep original units',
            BCE='Negative correct-minus-permuted BCE is beneficial; do not interpret BCE differences as percentage points',
            fine='Within each case sum covered/sum nodes across the selected patches, then equal defined cases; no patch-mean substitution',
            support='Paired defined patch support shown for every metric; no null-to-zero imputation',
            tail='Worst2 mechanism effects and original final AR normal-Dice worst2 are separate descriptive fixed selections',
            dependence='Fine nodes/voxels can repeat across overlapping patches;27 R15 primary fine-node observations are not27 independent anatomical branches or patients'),
        original_all3_screens_passed=final['all3_screens_passed'], original_capacity_continue=final['capacity_continue'],
        scientific_verdict_modified=False, new_gate_created=False, next_experiment_authorized=False,
        oracle_not_CT_only=True, GTA_goal_completed=False)


def render(result):
    out = ['已完成basis同权半径错配的血管/体素分解（事后描述，不增改门槛）。',
        '正值表示正确−错配；Dice/clDice/召回为pp，BCE为原loss单位（负值更好），FP/FN为每采样区块计数差。']
    for seed in result['per_seed']:
        metrics = seed['strata']['primary']['all_cases']['metrics']
        out.append('seed %s：%s。' % (seed['seed'], '；'.join('%s均值=%s，正%s/支持%s' %
            (k, metrics[k]['mean'], metrics[k]['positive_cases'], len(metrics[k]['supported_case_ids']))
            for k in ('dice_pp','cldice_pp','centerline_recall_pp','fine_recall_pp','BCE_correct_minus_permuted'))))
        r = seed['strata']['primary']['per_case']['Rider_R15']
        out.append(' R15 primary：%s；fine支持%s，correct/permuted covered=%s/%s。' %
            (json.dumps(r['effects'], ensure_ascii=False), r['fine_nodes'], r['fine_covered_correct'], r['fine_covered_permuted']))
        for source, group in seed['strata']['primary']['sources'].items():
            out.append(' %s：%s。' % (source, '；'.join('%s=%s' % (k, group['metrics'][k]['mean'])
                for k in ('dice_pp','cldice_pp','centerline_recall_pp','fine_recall_pp'))))
    out += ['正确半径可以提高骨架/细支覆盖，同时带来体素Dice或BCE代价；两者必须共同报告，不能用后加指标救回原判定。',
        '节点覆盖与体素FN/FP衡量不同几何量；不能从其不一致直接定位为粗管、细支或测量错误。',
        'R15的27个primary fine节点计数可在重叠区块中重复，不是独立分支样本；病例、来源和种子均不合池。',
        '原最终判定、候选停止和CT_uniform保护完全不变；GT oracle不是CT部署、native确认或GTA目标完成。']
    return '\n'.join(out) + '\n'


def execute(args):
    output = Path(args.output); require(not output.exists(), 'Fresh output required')
    result = analyze(args.inputs, args.final_analysis, args.final_analysis_sha256)
    cases, sources, r15 = [], [], []
    for seed in result['per_seed']:
        for stratum, value in seed['strata'].items():
            for cid, record in value['per_case'].items():
                cases.append(dict(seed=seed['seed'], stratum=stratum, case_id=cid, source=cid.split('_')[0],
                    patches=record['patches'], fine_nodes=record['fine_nodes'],
                    fine_covered_correct=record['fine_covered_correct'], fine_covered_permuted=record['fine_covered_permuted'], **record['effects']))
            for source, group in [('all12', value['all_cases'])] + list(value['sources'].items()):
                for metric, record in group['metrics'].items():
                    sources.append(dict(seed=seed['seed'], stratum=stratum, source=source, metric=metric,
                        mean=record['mean'], positive_cases=record['positive_cases'], negative_cases=record['negative_cases'],
                        ties=record['ties'], supported_cases=len(record['supported_case_ids'])))
        for record in seed['Rider_R15_patchwise']:
            r15.append(dict(seed=seed['seed'], slot=record['slot'], stratum=record['stratum'], primary=record['primary'],
                known_nonempty=record['known_nonempty'], fine_nodes=record['fine_nodes'],
                fine_covered_correct=record['fine_covered_correct'], fine_covered_permuted=record['fine_covered_permuted'],
                radius_sites=record['radius_sites'], mean_abs_logit_response=record['mean_abs_logit_response'], **record['effects']))
    output.mkdir(parents=True)
    (output / 'analysis.json').write_bytes((json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
    (output / 'readout.txt').write_bytes(render(result).encode('utf-8'))
    for name, rows in [('casewise.csv', cases), ('source_strata.csv', sources), ('R15_patchwise.csv', r15)]:
        with (output / name).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    return dict(status='complete', analysis_sha256=sha(output / 'analysis.json'), case_stratum_rows=len(cases),
        source_metric_rows=len(sources), R15_patch_rows=len(r15), scientific_verdict_modified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs=3); parser.add_argument('--final-analysis')
    parser.add_argument('--final-analysis-sha256'); parser.add_argument('--output'); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        require(args.inputs and args.final_analysis and args.final_analysis_sha256 and args.output, 'Explicit completed inputs/final SHA/fresh output required')
        print(json.dumps(execute(args)))
    else:
        print(json.dumps(dict(status='inert', files_read=False, writes=False, new_gate_created=False)))
