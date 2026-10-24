"""Small fixed TRAIN-only readout of the three-arm boundary-field pilot.

Default is inert. A short capacity screen does not replace native validation,
prove GTA efficacy or authorize any successor. Seeds are retained separately.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ARMS = ('C0', 'A_only', 'AB')
INTERVENTIONS = ('whole_field_donor', 'own_A_donor_B')
BANK_SHA = '9be42890017341e4cc32fb0f17688d859a13e84643f95cd21e347462043419d6'
METRICS = ('dice', 'cldice', 'centerline_recall', 'centerline_precision', 'fine_recall', 'BCE', 'FP', 'FN', 'TP')
METHOD = dict(primary='Known-nonempty GT-jitter slots0..7; patch means within case, then equal probe12 cases',
    fine='Sum fine_covered/sum fine_nodes within each case primary patches; then equal supported cases; retain patch-mean separately',
    utility='AB mean Dice strictly greater than both controls; do not transfer the mechanism effect threshold to between-training comparisons',
    safety='AB clDice, CLrecall, fine<=2 and each-arm own-worst2 Dice nonworse than BOTH controls on identical defined case supports',
    stability='Population SD and each-arm own-worst2 plus matched case tails descriptive; no stability-gain claim',
    mechanism='Each intervention: all12 cases supported, mean correct-minus-donor Dice>=.1pp, at least8 strictly positive cases',
    mechanism_support='At least2 known-nonempty primary slots, positive raw8 field population variance, at least1 primary actual z change, finite Dice/BCE/logit measurements',
    donor='Whole8^3 fields; cyclic+1 slots inside the same CASE x STRATUM; own_A_donor_B preserves recipient A',
    missing='Undefined support is inconclusive, never imputed0 or treated as a technical scientific negative',
    seeds='First supplied seed PRIMARY, remaining two predeclared replicas; no best-of3 or rescue; full confirmation requires all3',
    scope='Exploratory frozen-G0 short-budget capacity screening on reused TRAIN cases; not native validation, a blind cohort, goal completion or training authorization')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def lines(path):
    return [json.loads(row) for row in Path(path).read_bytes().splitlines()]


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def variance(values):
    values = list(values)
    m = mean(values)
    return mean((v - m) ** 2 for v in values) if values else None


def validate_metrics(metrics):
    expected = {'dice', 'cldice', 'centerline_recall', 'centerline_precision', 'fine_recall', 'fine_nodes',
        'fine_covered', 'GT_nodes', 'GT_covered', 'FP', 'FN', 'TP', 'BCE', 'real_CT_voxels', 'truth_foreground'}
    require(expected <= set(metrics), 'Missing pilot metrics')
    for key in expected:
        value = metrics[key]
        require(value is None or finite(value), 'Nonfinite pilot metric: ' + key)
    for key in ('dice', 'cldice', 'centerline_recall', 'centerline_precision', 'fine_recall'):
        require(metrics[key] is None or 0. <= metrics[key] <= 1., 'Invalid metric fraction: ' + key)
    for key in ('fine_nodes', 'fine_covered', 'GT_nodes', 'GT_covered', 'FP', 'FN', 'TP', 'real_CT_voxels', 'truth_foreground'):
        require(type(metrics[key]) is int and metrics[key] >= 0, 'Invalid count: ' + key)
    require(metrics['fine_covered'] <= metrics['fine_nodes'] and metrics['GT_covered'] <= metrics['GT_nodes'], 'Coverage exceeds support')


def aggregate(rows):
    result = dict(patches=len(rows))
    for name in METRICS:
        values = [r['metrics'][name] for r in rows if r['metrics'][name] is not None]
        result[name] = mean(values)
        result[name + '_defined_patches'] = len(values)
    result['fine_patch_mean_recall'] = result['fine_recall']
    result['fine_nodes'] = sum(r['metrics']['fine_nodes'] for r in rows)
    result['fine_covered'] = sum(r['metrics']['fine_covered'] for r in rows)
    result['fine_recall'] = result['fine_covered'] / result['fine_nodes'] if result['fine_nodes'] else None
    result['GT_nodes'] = sum(r['metrics']['GT_nodes'] for r in rows)
    result['GT_covered'] = sum(r['metrics']['GT_covered'] for r in rows)
    result['pooled_patch_GT_node_recall_descriptive'] = result['GT_covered'] / result['GT_nodes'] if result['GT_nodes'] else None
    return result


def case_arms(rows, case_ids):
    output = {}
    for arm in ARMS:
        per_case = {}
        for cid in case_ids:
            group = [r for r in rows[arm] if r['case_id'] == cid]
            primary = [r for r in group if r['primary']]
            per_case[cid] = dict(source=group[0]['source'], primary=aggregate(primary),
                uniform_known_nonempty=aggregate([r for r in group if r['slot'] >= 8 and r['known_nonempty']]),
                uniform_empty_patches=sum(r['slot'] >= 8 and r['valid_empty'] for r in group),
                uniform_empty_FP_voxels=sum(r['metrics']['FP'] for r in group if r['slot'] >= 8 and r['valid_empty']),
                unknown_slots=[r['slot'] for r in group if not r['attribute_valid']])
        summary = dict(per_case=per_case, expected_cases=len(case_ids))
        for name in METRICS:
            values = [v['primary'][name] for v in per_case.values() if v['primary'][name] is not None]
            summary[name] = mean(values); summary[name + '_supported_cases'] = len(values)
        dice = [per_case[c]['primary']['dice'] for c in case_ids if per_case[c]['primary']['dice'] is not None]
        summary['Dice_population_SD'] = math.sqrt(variance(dice)) if dice else None
        defined = sorted((c for c in case_ids if per_case[c]['primary']['dice'] is not None),
            key=lambda c: (per_case[c]['primary']['dice'], c))
        worst = defined[:2]
        summary['own_worst2'] = dict(case_ids=worst, Dice_mean=mean(per_case[c]['primary']['dice'] for c in worst) if len(worst) == 2 else None)
        summary['source_groups'] = {}
        for source in ('Dongyang', 'KiTS', 'Rider'):
            members = [c for c in case_ids if per_case[c]['source'] == source]
            summary['source_groups'][source] = dict(expected_cases=len(members),
                metrics={key: dict(mean=mean(per_case[c]['primary'][key] for c in members if per_case[c]['primary'][key] is not None),
                    supported_cases=sum(per_case[c]['primary'][key] is not None for c in members)) for key in METRICS})
        output[arm] = summary
    for reference in ARMS:
        ids = output[reference]['own_worst2']['case_ids']
        output[reference]['matched_worst2'] = dict(reference_arm=reference, case_ids=ids,
            Dice_means={arm: mean(output[arm]['per_case'][c]['primary']['dice'] for c in ids)
                if len(ids) == 2 and all(output[arm]['per_case'][c]['primary']['dice'] is not None for c in ids) else None for arm in ARMS})
    return output


def compare(arms, case_ids, control):
    result = dict(control=control, per_case={}, metrics={})
    for cid in case_ids:
        a, c = arms['AB']['per_case'][cid]['primary'], arms[control]['per_case'][cid]['primary']
        result['per_case'][cid] = {k: None if a[k] is None or c[k] is None else a[k] - c[k] for k in METRICS}
    for key in METRICS:
        ac = [cid for cid in case_ids if arms['AB']['per_case'][cid]['primary'][key] is not None]
        cc = [cid for cid in case_ids if arms[control]['per_case'][cid]['primary'][key] is not None]
        common = [cid for cid in ac if cid in cc]; delta = [result['per_case'][cid][key] for cid in common]
        result['metrics'][key] = dict(case_ids=common, AB_supported_cases=ac, control_supported_cases=cc,
            supports_identical=ac == cc, mean_AB_minus_control=mean(delta),
            mean_AB_minus_control_pp=None if not delta or key in ('FP', 'FN', 'TP', 'BCE') else 100. * mean(delta),
            positive_cases=sum(v > 0 for v in delta), negative_cases=sum(v < 0 for v in delta), ties=sum(v == 0 for v in delta))
    dice = result['metrics']['dice']
    result['mean_Dice_positive'] = (dice['mean_AB_minus_control'] > 0 if len(dice['case_ids']) == len(case_ids) else None)
    safety = {}
    for key in ('cldice', 'centerline_recall', 'fine_recall'):
        m = result['metrics'][key]
        safety[key + '_nonworse'] = None if not m['supports_identical'] or not m['case_ids'] else m['mean_AB_minus_control'] >= 0.
    at, ct = arms['AB']['own_worst2']['Dice_mean'], arms[control]['own_worst2']['Dice_mean']
    safety['own_worst2_Dice_nonworse'] = None if at is None or ct is None else at >= ct
    result['no_vascular_or_tail_harm'] = safety
    return result


def mechanism(rows, arms, case_ids):
    lookup = {(r['case_id'], r['slot']): r for r in arms}
    out = {}
    for name in INTERVENTIONS:
        per_case = {}
        for cid in case_ids:
            group = [r for r in rows if r['case_id'] == cid and lookup[(cid, r['slot'])]['primary']]
            field = [lookup[(cid, r['slot'])]['field'] for r in group]
            raw_var = None if not field else mean(v['raw_std'] ** 2 for v in field) + variance(v['raw_mean'] for v in field)
            changed = sum(bool(r['interventions'][name]['z_changed']) for r in group)
            reasons = []
            if len(group) < 2:
                reasons.append('fewer_than_two_primary_known_nonempty_slots')
            if raw_var is None or raw_var <= 0:
                reasons.append('raw8_field_population_variance_not_positive')
            if changed < 1:
                reasons.append('no_primary_actual_z_change')
            finite_metrics = all(finite(r['normal']['dice']) and finite(r['normal']['BCE']) and
                finite(r['interventions'][name]['metrics']['dice']) and finite(r['interventions'][name]['metrics']['BCE']) and
                finite(r['interventions'][name]['mean_abs_logit_response']) for r in group)
            if not finite_metrics:
                reasons.append('required_metric_undefined')
            delta = mean(100. * (r['normal']['dice'] - r['interventions'][name]['metrics']['dice']) for r in group) if group and finite_metrics else None
            per_case[cid] = dict(source=cid.split('_')[0], supported=not reasons, null_reasons=reasons,
                primary_patches=len(group), raw8_field_population_variance=raw_var, actual_z_changed_patches=changed,
                raw8_changed_patches=sum(bool(r['interventions'][name]['raw8_changed']) for r in group),
                correct_minus_donor_Dice_pp=delta if not reasons else None,
                descriptive_Dice_pp=delta,
                donor_minus_correct_BCE=mean(r['interventions'][name]['metrics']['BCE'] - r['normal']['BCE'] for r in group) if group and finite_metrics else None,
                correct_minus_donor_FP=mean(r['normal']['FP'] - r['interventions'][name]['metrics']['FP'] for r in group),
                correct_minus_donor_FN=mean(r['normal']['FN'] - r['interventions'][name]['metrics']['FN'] for r in group),
                mean_abs_logit_response=mean(r['interventions'][name]['mean_abs_logit_response'] for r in group),
                binary_changed_voxels=sum(r['interventions'][name]['binary_changed_voxels'] for r in group))
        supported = [cid for cid in case_ids if per_case[cid]['supported']]
        effects = [per_case[cid]['correct_minus_donor_Dice_pp'] for cid in supported]
        effect = mean(effects)
        parts = dict(all12_supported=len(supported) == 12, mean_effect_at_least_point1pp=None if effect is None else effect >= .1,
            at_least8_positive=sum(v > 0 for v in effects) >= 8)
        out[name] = dict(per_case=per_case, supported_case_ids=supported,
            mean_correct_minus_donor_Dice_pp=effect, positive_cases=sum(v > 0 for v in effects),
            parts=parts, passed=None if len(supported) != 12 else all(parts.values()),
            source_groups={s: dict(expected_cases=sum(c.startswith(s + '_') for c in case_ids),
                supported_cases=sum(c.startswith(s + '_') for c in supported),
                mean_Dice_effect_pp=mean(per_case[c]['correct_minus_donor_Dice_pp'] for c in supported if c.startswith(s + '_')))
                for s in ('Dongyang', 'KiTS', 'Rider')})
    return out


def load_seed(root, case_ids):
    root = Path(root); summary = read(root / 'summary.json')
    require(summary.get('status') == 'verified_complete' and summary.get('mode') == 'run', 'Completed formal pilot required, not QA')
    require(type(summary.get('seed')) is int, 'Pilot seed required')
    rows, hashes = {}, {'summary.json': sha(root / 'summary.json')}
    expected = [(cid, slot) for cid in case_ids for slot in range(16)]
    for arm in ARMS:
        path = root / arm / 'patches.jsonl'; data = lines(path)
        require([(r['case_id'], r['slot']) for r in data] == expected, 'Complete original192 patch order required')
        for row in data:
            require(row['source'] == row['case_id'].split('_')[0] and row['stratum'] == ('GT_centerline_plus_jitter' if row['slot'] < 8 else 'CT_uniform') and
                type(row['known_nonempty']) is bool and type(row['attribute_valid']) is bool and type(row['valid_empty']) is bool and
                row['primary'] == (row['slot'] < 8 and row['known_nonempty']), 'Patch population metadata differs')
            validate_metrics(row['metrics'])
            require(all(finite(row['field'][k]) for k in ('raw_mean', 'raw_std', 'z_mean', 'z_std', 'gain_absmean', 'gain_maxabs')) and
                all(row['field'][k] >= 0 for k in ('raw_std', 'z_std', 'gain_absmean', 'gain_maxabs')), 'Undefined field diagnostic')
        rows[arm] = data; hashes[arm + '/patches.jsonl'] = sha(path)
    for arm in ('A_only', 'AB'):
        for base, other in zip(rows['C0'], rows[arm]):
            require(all(base[k] == other[k] for k in ('case_id', 'slot', 'stratum', 'known_nonempty', 'attribute_valid', 'valid_empty', 'primary')) and
                all(base['metrics'][k] == other['metrics'][k] for k in ('fine_nodes', 'GT_nodes', 'truth_foreground', 'real_CT_voxels')), 'Arms use different truth supports')
    path = root / 'AB' / 'mechanism.jsonl'; interventions = lines(path)
    require([(r['case_id'], r['slot']) for r in interventions] == expected, 'Complete mechanism192 patch order required')
    for row, normal in zip(interventions, rows['AB']):
        require(row['normal'] == normal['metrics'] and set(row['interventions']) == set(INTERVENTIONS), 'Mechanism normal or intervention names differ')
        for item in row['interventions'].values():
            donor = (row['slot'] // 8) * 8 + (row['slot'] + 1) % 8
            require(item['donor_slot'] == donor and type(item['z_changed']) in (bool, int) and type(item['raw8_changed']) in (bool, int) and
                item['z_changed'] >= 0 and item['raw8_changed'] >= 0 and type(item['binary_changed_voxels']) is int and item['binary_changed_voxels'] >= 0 and
                finite(item['mean_abs_logit_response']) and item['mean_abs_logit_response'] >= 0, 'Invalid donor/actual response metadata')
            validate_metrics(item['metrics'])
            require(all(item['metrics'][k] == normal['metrics'][k] for k in ('fine_nodes', 'GT_nodes', 'truth_foreground', 'real_CT_voxels')), 'Mechanism truth supports differ')
    hashes['AB/mechanism.jsonl'] = sha(path)
    return summary, rows, interventions, hashes


def analyze_seed(summary, rows, interventions, case_ids):
    arms = case_arms(rows, case_ids)
    comparisons = {control: compare(arms, case_ids, control) for control in ('C0', 'A_only')}
    mechanisms = mechanism(interventions, rows['AB'], case_ids)
    tests = {name + '_AB_mean_Dice_positive': value['mean_Dice_positive'] for name, value in comparisons.items()}
    for name, value in comparisons.items():
        tests.update({name + '_' + k: v for k, v in value['no_vascular_or_tail_harm'].items()})
    tests.update({name + '_sameweight_gate': value['passed'] for name, value in mechanisms.items()})
    passed = None if any(v is None for v in tests.values()) else all(tests.values())
    return dict(seed=summary['seed'], arms=arms, AB_vs_controls=comparisons, mechanisms=mechanisms,
        screen=dict(parts=tests, passed=passed, status='inconclusive_support' if passed is None else 'positive_capacity_screen' if passed else 'screen_not_passed'),
        native_validation=False, GTA_goal_completed=False, training_authorized=False)


def analyze(roots, bank_path):
    require(len(roots) == 3 and len({str(Path(p).resolve()) for p in roots}) == 3, 'Three predeclared distinct seed outputs required')
    require(sha(bank_path) == BANK_SHA, 'Original frozen TRAIN partition required')
    bank = read(bank_path); ids = bank['probe_cases']
    require(len(ids) == 12 and len(set(ids)) == 12 and len(bank['fit_cases']) == 24 and set(ids).isdisjoint(bank['fit_cases']), 'Original24/12 partition required')
    seeds, provenance = [], []
    for root in roots:
        summary, rows, interventions, hashes = load_seed(root, ids)
        seeds.append(analyze_seed(summary, rows, interventions, ids)); provenance.append(hashes)
    require(len({r['seed'] for r in seeds}) == 3, 'Distinct predeclared seeds required')
    primary = seeds[0]['screen']['passed']; flags = [r['screen']['passed'] for r in seeds]
    whole = None if any(v is None for v in flags) else all(flags)
    return dict(schema='GTA-AVT-boundary-field-pilot-readout-v1', method=METHOD, bank_sha256=BANK_SHA,
        readout_sha256=sha(__file__), probe_cases=ids, per_seed=seeds, input_hashes=provenance,
        primary_seed=seeds[0]['seed'], primary_screen_passed=primary, all3_screens_passed=whole,
        primary_failure_cannot_be_rescued=primary is not True, seeds_pooled=False,
        next_experiment_authorized=False, native_confirmation_required=True, GTA_goal_completed=False)


def render(result):
    lines = ['固定G0的短程TRAIN局部场容量筛查；未进行native确认。']
    for seed in result['per_seed']:
        lines.append('Seed %s：screen=%s。' % (seed['seed'], seed['screen']['status']))
        for control, value in seed['AB_vs_controls'].items():
            dice = value['metrics']['dice']
            lines.append(' AB−%s Dice=%s pp；正病例%s/%s。' % (control, dice['mean_AB_minus_control_pp'], dice['positive_cases'], len(dice['case_ids'])))
        for name, value in seed['mechanisms'].items():
            lines.append(' %s：correct−donor=%s pp；positive=%s，supported=%s/12，gate=%s。' % (name,
                value['mean_correct_minus_donor_Dice_pp'], value['positive_cases'], len(value['supported_case_ids']), value['passed']))
    lines.append('三个seed单独保留；不互救、不自动启动后续，不把短程阳性写成GTA有效验证。')
    return '\n'.join(lines) + '\n'


def execute(args):
    output = Path(args.output); require(not output.exists(), 'Fresh output required')
    result = analyze(args.inputs, args.bank); payload = encoded(result)
    output.mkdir(parents=True); (output / 'analysis.json').write_bytes(payload)
    (output / 'readout.txt').write_bytes(render(result).encode('utf-8'))
    case_rows = []
    for seed in result['per_seed']:
        for arm, value in seed['arms'].items():
            for cid, case in value['per_case'].items():
                case_rows.append(dict(seed=seed['seed'], arm=arm, case_id=cid, source=case['source'], **case['primary']))
    with (output / 'casewise.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0])); writer.writeheader(); writer.writerows(case_rows)
    return dict(status='complete', analysis_sha256=sha(output / 'analysis.json'), next_experiment_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--inputs', nargs=3)
    parser.add_argument('--bank'); parser.add_argument('--output'); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='inert', method=METHOD, files_read=False, writes=False))); return
    require(args.inputs and args.bank and args.output, 'Explicit completed seed outputs, original bank and fresh output required')
    print(json.dumps(execute(args)))


if __name__ == '__main__':
    main()
