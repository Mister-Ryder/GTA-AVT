"""Small local readout wrapper for new AB arms with reused completed controls.

All original pilot arithmetic and screening gates come from the unchanged
boundary_field_readout.py. New-versus-old AB differences are descriptive only.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


READER_SHA = '7658a400266cd805f761722ddc2e6183e38834bdbbba1b20460d7f0214738158'
PREVIOUS_SHA = 'a2cab09340b7bebea123c8a1f1a426c7890ce726eae377259a46c2b192f07c7e'
SEEDS = (20261030, 20261031, 20261032)
METRICS = ('dice', 'cldice', 'centerline_recall', 'fine_recall')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')


def load_reader():
    path = Path(__file__).with_name('boundary_field_readout.py')
    require(sha(path) == READER_SHA, 'Original pilot readout changed')
    spec = importlib.util.spec_from_file_location('_unchanged_boundary_field_readout', path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    return reader


def descriptive_comparison(reader, new, old, case_ids):
    """Compare two already aggregated AB endpoints without defining any gate."""
    per_case = {}
    for cid in case_ids:
        n, o = new['per_case'][cid]['primary'], old['per_case'][cid]['primary']
        require(new['per_case'][cid]['source'] == old['per_case'][cid]['source'],
                'AB case/source identity differs')
        require(all(n[key] == o[key] for key in ('patches', 'fine_nodes', 'GT_nodes')),
                'New and old AB use different primary truth supports')
        per_case[cid] = dict(source=new['per_case'][cid]['source'], metrics={})
        for key in METRICS:
            delta = None if n[key] is None or o[key] is None else 100. * (n[key] - o[key])
            per_case[cid]['metrics'][key] = dict(old= None if o[key] is None else 100. * o[key],
                new=None if n[key] is None else 100. * n[key], new_minus_old_pp=delta)

    def summarize(ids):
        result = {}
        for key in METRICS:
            old_ids = [cid for cid in ids if old['per_case'][cid]['primary'][key] is not None]
            new_ids = [cid for cid in ids if new['per_case'][cid]['primary'][key] is not None]
            common = [cid for cid in ids if cid in old_ids and cid in new_ids]
            deltas = [per_case[cid]['metrics'][key]['new_minus_old_pp'] for cid in common]
            result[key] = dict(case_ids=common, supports_identical=old_ids == new_ids,
                old_supported_cases=len(old_ids), new_supported_cases=len(new_ids),
                mean_new_minus_old_pp=reader.mean(deltas),
                positive_cases=sum(value > 0. for value in deltas),
                negative_cases=sum(value < 0. for value in deltas), ties=sum(value == 0. for value in deltas))
        return result

    sd_old, sd_new = old['Dice_population_SD'], new['Dice_population_SD']
    tail_old, tail_new = old['own_worst2'], new['own_worst2']
    tails = {}
    for label, value in (('old_selected_cases', tail_old), ('new_selected_cases', tail_new)):
        ids = value['case_ids']
        before = [old['per_case'][cid]['primary']['dice'] for cid in ids]
        after = [new['per_case'][cid]['primary']['dice'] for cid in ids]
        valid = len(ids) == 2 and all(v is not None for v in before + after)
        tails[label] = dict(case_ids=ids,
            old_Dice_percent=100. * reader.mean(before) if valid else None,
            new_Dice_percent=100. * reader.mean(after) if valid else None,
            new_minus_old_pp=100. * (reader.mean(after) - reader.mean(before)) if valid else None)
    return dict(metrics=summarize(case_ids), per_case=per_case,
        source_groups={s: summarize([cid for cid in case_ids if per_case[cid]['source'] == s])
            for s in ('Dongyang', 'KiTS', 'Rider')},
        Dice_population_SD=dict(old_pp=None if sd_old is None else 100. * sd_old,
            new_pp=None if sd_new is None else 100. * sd_new,
            new_minus_old_pp=None if sd_old is None or sd_new is None else 100. * (sd_new - sd_old),
            interpretation='Cross-case population dispersion only, not run-to-run or inference variance'),
        own_worst2=dict(old_case_ids=tail_old['case_ids'], new_case_ids=tail_new['case_ids'],
            old_Dice_percent=None if tail_old['Dice_mean'] is None else 100. * tail_old['Dice_mean'],
            new_Dice_percent=None if tail_new['Dice_mean'] is None else 100. * tail_new['Dice_mean'],
            new_minus_old_pp=None if tail_old['Dice_mean'] is None or tail_new['Dice_mean'] is None else
                100. * (tail_new['Dice_mean'] - tail_old['Dice_mean'])), matched_tails=tails,
        descriptive_only=True, new_gate_defined=False)


def augment(reader, current, previous):
    require(previous['schema'] == current['schema'] == 'GTA-AVT-boundary-field-pilot-readout-v1' and
            previous['readout_sha256'] == current['readout_sha256'] == READER_SHA and
            previous['bank_sha256'] == current['bank_sha256'] == reader.BANK_SHA and
            previous['method'] == current['method'] == reader.METHOD and
            previous['probe_cases'] == current['probe_cases'], 'Original readout method/population differs')
    require(tuple(x['seed'] for x in previous['per_seed']) ==
            tuple(x['seed'] for x in current['per_seed']) == SEEDS, 'Predeclared matched seed order required')
    comparisons = []
    for index, (new, old) in enumerate(zip(current['per_seed'], previous['per_seed'])):
        for arm in ('C0', 'A_only'):
            name = arm + '/patches.jsonl'
            require(current['input_hashes'][index][name] == previous['input_hashes'][index][name],
                    'Reused control patch bytes differ: ' + arm)
        comparisons.append(dict(seed=new['seed'],
            **descriptive_comparison(reader, new['arms']['AB'], old['arms']['AB'], current['probe_cases'])))
    return dict(schema='GTA-AVT-boundary-separable-field-wrapper-readout-v1',
        unchanged_pilot_readout=current, new_AB_minus_previous_product_AB=comparisons,
        original_pilot_readout_sha256=READER_SHA, previous_analysis_sha256=PREVIOUS_SHA,
        wrapper_sha256=sha(__file__), original_gate_arithmetic_unchanged=True,
        old_AB_trainable_parameters=256, new_AB_trainable_parameters=384,
        controls_reused_byte_exact=True, actual_new_training_arms_per_seed=1,
        seeds_pooled=False, new_gate_defined=False, native_validation=False,
        GTA_goal_completed=False, next_experiment_authorized=False,
        limitations=[
            'New AB adds128 v parameters for zA, increasing the adapter from256 to384 parameters.',
            'Completed same-seed C0 and A_only patch metadata are reused byte-for-byte; controls are not retrained.',
            'Only three new AB arms are trained; this is not nine new arm runs.',
            'New-minus-old AB comparisons are exploratory descriptions; they do not introduce a continuation gate.',
            'Reused TRAIN/backbone, fit24/probe12 and short250-update budget; no blind cohort or native confirmation.',
            'Cross-case SD is descriptive dispersion. Different own-worst2 case sets are retained and matched tails also reported.',
            'Original field-pilot gates remain unchanged; no best seed, pooling, automatic successor or goal update.'])


def analyze(inputs, previous_analysis, bank):
    reader = load_reader()
    require(sha(previous_analysis) == PREVIOUS_SHA, 'Exact completed previous product-AB analysis required')
    previous = reader.read(previous_analysis)
    current = reader.analyze(inputs, bank)
    for root, seed in zip(inputs, SEEDS):
        summary = reader.read(Path(root) / 'summary.json')
        require(summary.get('schema') == 'GTA-AVT-boundary-separable-field-pilot-v1' and
                summary.get('seed') == seed and summary.get('controls_reused') is True and
                summary.get('actual_training_arms') == 1,
                'Completed new AB output with explicit reused controls required')
    return augment(reader, current, previous)


def render(reader, result):
    lines = [reader.render(result['unchanged_pilot_readout']).rstrip(),
        '以下仅描述同seed新AB−旧product-AB；旧256参数，新384参数，C0/A_only逐字复用，未新增门槛。']
    for block in result['new_AB_minus_previous_product_AB']:
        values = ', '.join('%s=%s pp' % (key, block['metrics'][key]['mean_new_minus_old_pp']) for key in METRICS)
        lines.append('Seed %s：%s；SD新−旧=%s pp；own-worst2新−旧=%s pp。' %
            (block['seed'], values, block['Dice_population_SD']['new_minus_old_pp'], block['own_worst2']['new_minus_old_pp']))
    lines.append('探索性复用TRAIN，不是盲验证或native确认；不选最好seed，不自动启动后继或更新目标。')
    return '\n'.join(lines) + '\n'


def execute(args):
    output = Path(args.output)
    require(not output.exists(), 'Fresh output required')
    result = analyze(args.inputs, args.previous_analysis, args.bank)
    reader = load_reader()
    rows = []
    for block in result['new_AB_minus_previous_product_AB']:
        for cid, value in block['per_case'].items():
            row = dict(seed=block['seed'], case_id=cid, source=value['source'])
            for key, metric in value['metrics'].items():
                row.update({key + '_old_percent': metric['old'], key + '_new_percent': metric['new'],
                            key + '_new_minus_old_pp': metric['new_minus_old_pp']})
            rows.append(row)
    output.mkdir(parents=True)
    (output / 'analysis.json').write_bytes(encoded(result))
    (output / 'readout.txt').write_bytes(render(reader, result).encode('utf-8'))
    with (output / 'casewise.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    return dict(status='complete', analysis_sha256=sha(output / 'analysis.json'),
                original_gates_unchanged=True, next_experiment_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs=3)
    parser.add_argument('--previous-analysis'); parser.add_argument('--bank'); parser.add_argument('--output')
    parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='inert', files_read=False, writes=False,
            original_readout_sha256=READER_SHA, new_gate_defined=False))); return
    require(args.inputs and args.previous_analysis and args.bank and args.output,
            'Three completed new outputs, previous analysis, original bank and fresh output required')
    print(json.dumps(execute(args)))


if __name__ == '__main__':
    main()
