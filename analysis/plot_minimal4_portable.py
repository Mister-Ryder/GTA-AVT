"""Plot existing declared four-model files; no model or bootstrap is run.

The default prints a stdlib-only plan and opens no input. Published declarations
and metric/summary FILE hashes do not attest an author's server, weight contents
or pre-TEST selection time. An author-private freeze marker is never consumed.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

READER_SHA = 'b936e21ba5a0e0169e106cb65fda91de75c8dac8a337dcd319f48ab5a2f7ae6b'
OLD_PLOT_SHA = '9ce120612a4434d31a3b47fc3f7f04773e627f2fba7bfb4f361364cddb5e0c39'
MATH_SHA = '31f355f735f6744589e464bbfb50cef276bc4db650f5edc70c5f6938d834f3c4'
ORIGINAL_PLOT_SHA = '504c2233fdd55858bb1682ae80a5504d02382b14a99124180a15c949cf56bc4c'
SCHEMA = 'GTA-AVT-minimal4-paired-case-analysis-v1'
SCOPE = 'bounded_20261022_v2_minimal4'
SLOTS = ('main_C_concat', 'main_A_soft', 'ablations_A_soft_platform_bridge', 'ablations_A_soft_constant')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def equivalent(actual, expected, label):
    """Reconcile displayed values, retaining exact keys, nulls and integer types."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': fields differ')
        for key, value in expected.items():
            equivalent(actual[key], value, label + '/' + str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list differs')
        for i, value in enumerate(expected):
            equivalent(actual[i], value, label + '/' + str(i))
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), label + ': value differs')
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': identity/null differs')


def load_reader(directory):
    path = Path(directory).resolve() / 'analyze_minimal4_portable.py'
    require(sha(path) == READER_SHA, 'Pinned portable declaration reader differs')
    spec = importlib.util.spec_from_file_location('gta_avt_portable_plot_reader', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def dependencies(repository):
    folder = Path(repository).resolve() / 'analysis'
    for name, expected in (('plot_key_ablations.py', OLD_PLOT_SHA), ('analyze_key_ablations.py', MATH_SHA)):
        require(sha(folder / name) == expected, 'Released style/math dependency differs: ' + name)
    for name in ('plot_key_ablations', 'analyze_key_ablations'):
        loaded = sys.modules.get(name)
        require(loaded is None or Path(loaded.__file__).resolve() == folder / (name + '.py'), 'Different dependency checkout already imported')
    sys.path.insert(0, str(folder))
    import plot_key_ablations as old
    return old


def check_pair(record, values_a, values_b, direction, label):
    """Reconcile every paired statistic except the artifact's stored interval."""
    valid = [(a, b) for a, b in zip(values_a, values_b) if a is not None and b is not None]
    delta = [a - b for a, b in valid]
    expected = dict(total_cases=len(values_a), paired_cases=len(valid),
        missing_a=sum(x is None for x in values_a), missing_b=sum(x is None for x in values_b),
        excluded_cases=len(values_a) - len(valid), a_mean_paired=None, b_mean_paired=None,
        delta_a_minus_b=None, wins=0, ties=0, losses=0,
        estimand='unweighted case mean of paired A minus B; complete pairs only',
        interval='exploratory 95% percentile paired case bootstrap; 20000 draws; seed20261022; patient independence unverified',
        missing_policy='no imputation; undefined metrics excluded pairwise with support reported')
    if delta:
        expected.update(a_mean_paired=math.fsum(a for a, _ in valid) / len(valid),
            b_mean_paired=math.fsum(b for _, b in valid) / len(valid), delta_a_minus_b=math.fsum(delta) / len(delta),
            wins=sum(d * direction > 1e-12 for d in delta), ties=sum(abs(d) <= 1e-12 for d in delta),
            losses=sum(d * direction < -1e-12 for d in delta))
        if len(delta) < 2:
            expected['support_warning'] = 'Fewer than two paired cases; degenerate interval is not uncertainty evidence'
    require(set(record) == set(expected) | {'ci95_lower', 'ci95_upper'}, label + ': paired fields differ')
    equivalent({k: record[k] for k in expected}, expected, label)
    lower, upper = record['ci95_lower'], record['ci95_upper']
    if not delta:
        require(lower is None and upper is None, label + ': unsupported intervals must be null')
    else:
        require(all(type(x) in (int, float) and math.isfinite(x) for x in (lower, upper))
            and lower <= upper and lower >= min(delta) - 1e-9 and upper <= max(delta) + 1e-9,
            label + ': stored interval invalid')
        if len(delta) == 1:
            equivalent(float(lower), float(delta[0]), label + '/single lower')
            equivalent(float(upper), float(delta[0]), label + '/single upper')


def validate_artifact(result, inputs, reader):
    """No bootstrap: match case points, changes and support to admitted raw files."""
    numeric, raw, functions = inputs['numeric_index'], inputs['raw_rows'], inputs['functions']
    math4 = functions.math4; ids = numeric['case_ids']
    require(result.get('schema') == SCHEMA and result.get('scope') == SCOPE
        and result.get('stage') == numeric['stage'], 'Fixed paired artifact schema/stage required')
    equivalent(result.get('case_ids'), ids, 'Exact all8/12 stage cases')
    require(not result.get('synthetic_fixture'), 'Private synthetic artifact cannot be relabelled as a declared result')
    for key, value in inputs['result_provenance'].items():
        equivalent(result.get(key), value, 'Declaration provenance/' + key)
    equivalent(result.get('source_verification_level'), reader.SOURCE_LEVEL, 'Declared source level')
    equivalent(result.get('common_reference_runtime'), numeric['common_reference_runtime'], 'Declared reference metadata')
    equivalent(result.get('source_marker_FILE_sha256'), None, 'Author-private marker never consumed')
    equivalent(result.get('selection_payload_sha256'), numeric['selection_payload_sha256'], 'Declared selection identity')
    equivalent(result.get('availability_only_VAL_evidence'), None, 'Declared normal-model artifact has no private intervention evidence')
    equivalent(result.get('final_readout_rules_sha256'), reader.RULES_SHA, 'Frozen readout rules')
    equivalent(result.get('statistical_math'), dict(source_file_sha256=MATH_SHA, seed=20261022, bootstrap_draws=20000,
        interval='exploratory95%percentile paired-case; no seed/selection/multiplicity correction'), 'Declared statistical method')
    expected_cases = []; summaries = {}; common_support = {}
    for slot in SLOTS:
        require(set(raw[slot]) == set(ids), 'All cases in every fixed slot required')
        for cid in ids:
            value = raw[slot][cid]
            expected_cases.append(dict(slot=slot, case_id=cid, cohort=math4.cohort(cid),
                **{k: functions.scaled(value, k) for k in math4.METRICS},
                **{k + '_missing_reason': math4.missing_reason(value, k) for k in math4.METRICS},
                FP_plus_FN_ml=value['fp_ml'] + value['fn_ml']))
            truth = value.get('truth_skeleton_support')
            if truth is not None:
                require(type(truth) is int and truth >= 0, 'GT skeleton support must be a nonnegative integer, never bool')
                require(common_support.setdefault((cid, 'truth_skeleton_support'), truth) == truth, 'Explicit GT skeleton support differs across four slots')
            for group in GROUPS:
                nodes, _, _ = math4.group_record(value, group)
                if nodes is not None:
                    require(common_support.setdefault((cid, group), nodes) == nodes, 'Explicit GT supports differ across four slots')
        summaries[slot] = {scope: {key: functions.distribution([functions.scaled(raw[slot][c], key) for c in selected])
            for key in math4.METRICS} for scope, selected in math4.scopes(ids)}
    equivalent(result.get('casewise'), expected_cases, 'All plotted case values/missing reasons')
    equivalent(result.get('slot_summaries'), summaries, 'All slot summaries')
    require(set(result.get('comparisons', {})) == {x[0] for x in COMPARISONS}, 'Exactly two distinct within-role comparisons required')
    require(set(result.get('figure_selection', {})) == {x[0] for x in COMPARISONS}, 'Exactly two fixed figure selections required')
    for name, after, before, _, _, _ in COMPARISONS:
        row = result['comparisons'][name]; ra, rb = raw[after], raw[before]
        equivalent([row.get(k) for k in ('role', 'after_slot', 'before_slot')], [after.split('_', 1)[0], after, before], name + '/identity')
        cases = [dict(case_id=cid, cohort=math4.cohort(cid),
            after={k: functions.scaled(ra[cid], k) for k in math4.METRICS},
            before={k: functions.scaled(rb[cid], k) for k in math4.METRICS},
            delta_after_minus_before={k: None if ra[cid][k] is None or rb[cid][k] is None else functions.scaled(ra[cid], k) - functions.scaled(rb[cid], k) for k in math4.METRICS},
            FP_plus_FN_after_ml=ra[cid]['fp_ml'] + ra[cid]['fn_ml'], FP_plus_FN_before_ml=rb[cid]['fp_ml'] + rb[cid]['fn_ml']) for cid in ids]
        equivalent(row.get('cases'), cases, name + '/all case changes')
        require(set(row.get('scopes', {})) == {s for s, _ in math4.scopes(ids)}, name + '/scopes differ')
        for scope, selected in math4.scopes(ids):
            measured = row['scopes'][scope]
            require(set(measured) == {'metrics', 'geometry', 'cohort_scope_is_descriptive_only'}, 'Scope fields differ')
            equivalent(measured['cohort_scope_is_descriptive_only'], scope != 'all', 'Descriptive cohort label')
            require(set(measured['metrics']) == set(METRICS) and set(measured['geometry']) == set(GROUPS), 'All ten metrics and six groups required')
            for key, (unit, _, direction) in math4.METRICS.items():
                metric = measured['metrics'][key]
                equivalent(metric.get('difference_unit'), unit, 'Metric unit')
                check_pair({k: v for k, v in metric.items() if k != 'difference_unit'},
                    [functions.scaled(ra[c], key) for c in selected], [functions.scaled(rb[c], key) for c in selected], direction, name + '/' + scope + '/' + key)
            for group in GROUPS:
                ga = [math4.group_record(ra[c], group) for c in selected]; gb = [math4.group_record(rb[c], group) for c in selected]
                g = measured['geometry'][group]
                extra = dict(no_GT_support_cases=sum(x[0] == 0 and y[0] == 0 for x, y in zip(ga, gb)),
                    unrecorded_cases=sum(x[0] is None or y[0] is None for x, y in zip(ga, gb)),
                    case_support_and_missing=[dict(case_id=c, after_GT_nodes=x[0], before_GT_nodes=y[0], after_recall=x[1], before_recall=y[1], missing_after=x[2], missing_before=y[2]) for c, x, y in zip(selected, ga, gb)],
                    precision_scope='GT-group recall only; no group precision; combine global FP/CLprecision',
                    aggregation='equal supported case pairs, never pooled nodes')
                for key, value in extra.items():
                    equivalent(g.get(key), value, name + '/' + group + '/' + key)
                check_pair({k: v for k, v in g.items() if k not in extra}, [None if x[1] is None else 100 * x[1] for x in ga],
                    [None if x[1] is None else 100 * x[1] for x in gb], 1, name + '/' + scope + '/' + group)
        dice = [c['delta_after_minus_before']['dice'] for c in cases]
        leaveout = {cid: statistics.mean(v for c, v in zip(ids, dice) if c != cid) for cid in ids}
        for key, expected in dict(worst_Dice_difference_pp=min(dice), worst_Dice_cases=[c for c, v in zip(ids, dice) if v == min(dice)],
            leave_one_case_out_Dice_mean_difference_pp=leaveout, leave_one_out_range_pp=[min(leaveout.values()), max(leaveout.values())],
            leave_one_out_is_CI=False, FP_plus_FN_difference_ml=functions.distribution([c['FP_plus_FN_after_ml'] - c['FP_plus_FN_before_ml'] for c in cases]),
            FP_plus_FN_is_derived_descriptive_cost=True).items():
            equivalent(row.get(key), expected, name + '/' + key)
        equivalent(result['figure_selection'][name], dict(case_selection=functions.figure_cases(ids, ra, rb), native_FOV=True,
            GT_volume_axial_quantiles=[.25, .5, .75], shared_CT_window_and_colors=True, maximum_distinct_cases=3,
            no_replacement_or_filling=True, predictions_rendered_by_this_analysis=False), name + '/fixed figure selection')
    reader.recheck_files(inputs['bound_file_sha256'])
    return {slot: {c['case_id']: c for c in expected_cases if c['slot'] == slot} for slot in SLOTS}


def actual_inputs(analysis_path, analysis_sha, declaration_path, declaration_sha, repository, reader_directory,
                  selection_path=None, selection_sha=None):
    reader = load_reader(reader_directory)
    require(reader.valid_sha(analysis_sha) and sha(analysis_path) == analysis_sha, 'Explicit paired artifact FILE SHA required')
    result = read(analysis_path)
    inputs = reader.load_verified_inputs(declaration_path, declaration_sha, repository, selection_path, selection_sha)
    cases = validate_artifact(result, inputs, reader)
    bound = dict(inputs['bound_file_sha256']); bound[str(Path(analysis_path).resolve())] = analysis_sha
    bound[str(Path(reader.__file__).resolve())] = READER_SHA
    bound[str(Path(inputs['functions'].__file__).resolve())] = reader.STATISTICS_SHA
    binding = dict(analysis_FILE_sha256=analysis_sha, evaluation_declaration_FILE_sha256=declaration_sha,
        declared_selection_FILE_sha256=selection_sha, declared_selection_identity_sha256=result.get('declared_selection_identity_sha256'),
        author_private_marker_FILE_sha256=None, portable_reader_FILE_sha256=READER_SHA,
        source_level=reader.SOURCE_LEVEL, intervals='Stored paired artifact intervals; not independently recalculated by plotting')
    return result, cases, binding, reader, bound


# The six plotting functions and four constants below are copied byte-for-byte
# from the frozen four-model plot source. They have no source-admission role.


COMPARISONS = (
    ('MAIN_A_minus_C', 'main_A_soft', 'main_C_concat', 'MAIN: A - C', '#24679B', 'o'),
    ('SECOND_A_minus_constant', 'ablations_A_soft_platform_bridge', 'ablations_A_soft_constant', 'SECOND: A - constant', '#A86D16', 's'),
)



METRICS = {'dice': ('pp', 'Dice'), 'precision': ('pp', 'Precision'), 'recall': ('pp', 'Recall'),
    'fp_ml': ('ml', 'FP volume'), 'fn_ml': ('ml', 'FN volume'), 'assd_mm': ('mm', 'ASSD'),
    'hd95_mm': ('mm', 'HD95'), 'centerline_recall': ('pp', 'CL recall'),
    'centerline_precision': ('pp', 'CL precision'), 'cldice': ('pp', 'clDice')}



GROUPS = ('radius_le_2mm', 'radius_2_to_4mm', 'radius_4_to_8mm', 'radius_gt_8mm', 'nonjunction', 'junction_neighbourhood')



GROUP_LABELS = ('Radius <=2 mm', 'Radius 2-4 mm', 'Radius 4-8 mm', 'Radius >8 mm', 'Nonjunction proxy', 'Junction neighbourhood')



def caption(result):
    return '%s | %d released cases | one training seed | roles not pooled' % ('Held-out TEST' if result['stage'] == 'test' else 'Development VAL', len(result['case_ids']))



def record(sources, figure, panel, pointer, reviewed, **values):
    sources.append(dict(figure=figure, panel=panel, analysis_JSON_pointer=pointer,
                        reviewed_source_record=reviewed, **values))



def case_figure(result, cases, old, sources, synthetic):
    import matplotlib.pyplot as plt
    import numpy as np
    name = '01_case_Dice_by_training_role'
    ids = result['case_ids']; height = max(4.6, .28 * len(ids) + 2.25)
    fig, axes = plt.subplots(1, 2, figsize=(9.3, height), sharex=True, sharey=True)
    fig.subplots_adjust(left=.14, right=.98, bottom=.19, top=.77, wspace=.27)
    row_numbers = {(r['slot'], r['case_id']): i for i, r in enumerate(result['casewise'])}
    for ax, (comparison, after, before, title, color, _) in zip(axes, COMPARISONS):
        for i, cid in enumerate(ids):
            ax.plot([cases[before][cid]['dice'], cases[after][cid]['dice']], [i - .075, i + .075], color='#B9BFC4', linewidth=.8, zorder=1)
        for slot, label, marker, tone, offset in ((before, 'C base fusion' if before == 'main_C_concat' else 'A uniform constant', 's', '#565D63', -.075),
                                                (after, 'A conditioned' if after == 'main_A_soft' else 'A conditioned bridge', 'D', color, .075)):
            values = [cases[slot][cid]['dice'] for cid in ids]
            ax.scatter(values, np.arange(len(ids)) + offset, marker=marker, s=29, edgecolors=tone,
                       facecolors='white' if marker == 's' else tone, linewidths=1., label=label, zorder=3)
            for cid, value in zip(ids, values):
                record(sources, name, title, '/casewise/%d/dice' % row_numbers[(slot, cid)], cases[slot][cid],
                    role=slot.split('_', 1)[0], slot=slot, case_id=cid, metric='dice', value=value, unit='percent')
        for i in range(1, len(ids)):
            if old.analysis.cohort(ids[i]) != old.analysis.cohort(ids[i - 1]):
                ax.axhline(i - .5, color='#D9DEE1', linewidth=.7)
        ax.set(yticks=np.arange(len(ids)), yticklabels=[c.replace('_', ' ') for c in ids], ylim=(len(ids) - .5, -.5),
               xlim=(-1, 101), xticks=range(0, 101, 20), xlabel='Case Dice (%)')
        ax.set_title(title, pad=36)
        ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=2, frameon=False, fontsize=8)
        ax.grid(axis='x', color='#E5E8EA', linewidth=.5)
    fig.suptitle('All cases, paired within each training role', y=.98, fontsize=12)
    fig.text(.14, .915, caption(result), fontsize=8.5)
    fig.text(.14, .055, 'Connectors pair the same case within one role. Source cases are not verified independent patients.\nMAIN A-C changes operations + conditioning; SECOND A-constant includes empty mass and gradient paths.', fontsize=7.7)
    old.watermark(fig, synthetic)
    return name, fig



def draw_effect(ax, y, row, old, color, marker, support):
    old.interval(ax, y, row, color, marker)
    text = ('NA; ' if row['delta_a_minus_b'] is None else '') + support
    ax.text(1.025, y, text, transform=ax.get_yaxis_transform(), va='center', fontsize=7.7)



def metric_figure(result, old, sources, synthetic):
    import matplotlib.pyplot as plt
    name = '02_all_metric_paired_changes'
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.5))
    fig.subplots_adjust(left=.12, right=.91, top=.80, bottom=.17, hspace=.90, wspace=.94)
    for column, unit in enumerate(('pp', 'ml', 'mm')):
        metrics = [key for key in METRICS if METRICS[key][0] == unit]
        all_rows = [result['comparisons'][c[0]]['scopes']['all']['metrics'][key] for c in COMPARISONS for key in metrics]
        domain = old.limits(all_rows)
        for row_index, (comparison, _, _, title, color, marker) in enumerate(COMPARISONS):
            ax = axes[row_index, column]
            for y, metric in enumerate(metrics):
                row = result['comparisons'][comparison]['scopes']['all']['metrics'][metric]
                draw_effect(ax, y, row, old, color, marker, 'n=%d/%d' % (row['paired_cases'], row['total_cases']))
                record(sources, name, title + ' / ' + unit, '/comparisons/%s/scopes/all/metrics/%s' % (comparison, metric), row,
                    comparison=comparison, metric=metric, unit=unit, **{k: row[k] for k in ('delta_a_minus_b', 'ci95_lower', 'ci95_upper', 'paired_cases', 'total_cases')})
            ax.set(yticks=range(len(metrics)), yticklabels=[METRICS[key][1] for key in metrics], ylim=(len(metrics) - .5, -.5),
                   xlim=domain, xlabel='Mean change (%s)' % unit)
            ax.set_title(title + '\n' + {'pp': 'Fraction metrics (percentage points)', 'ml': 'Error volumes (ml)', 'mm': 'Surface distances (mm)'}[unit], fontsize=9, pad=11)
            old.zero_axis(ax)
    fig.suptitle('Paired mean changes with exploratory 95% case-bootstrap intervals', y=.97, fontsize=12)
    fig.text(.12, .91, caption(result), fontsize=8.5)
    fig.text(.12, .045, 'A minus its within-role comparator; effects are never pooled across roles. Larger fraction metrics are better; larger ml/mm errors are worse.\nIntervals are read from the reviewed artifact, not recomputed. NA is undefined, not zero; n<2 intervals are degenerate. Case independence is unverified.', fontsize=7.5)
    old.watermark(fig, synthetic)
    return name, fig



def geometry_figure(result, old, sources, synthetic):
    import matplotlib.pyplot as plt
    name = '03_GT_geometry_recall_changes'
    rows = [result['comparisons'][c[0]]['scopes']['all']['geometry'][g] for c in COMPARISONS for g in GROUPS]
    domain = old.limits(rows)
    fig, axes = plt.subplots(2, 1, figsize=(8.9, 7.1), sharex=True)
    fig.subplots_adjust(left=.27, right=.69, top=.81, bottom=.18, hspace=.72)
    for ax, (comparison, _, _, title, color, marker) in zip(axes, COMPARISONS):
        for y, group in enumerate(GROUPS):
            row = result['comparisons'][comparison]['scopes']['all']['geometry'][group]
            support = 'n=%d/%d; noGT=%d; U=%d' % (row['paired_cases'], row['total_cases'], row['no_GT_support_cases'], row['unrecorded_cases'])
            draw_effect(ax, y, row, old, color, marker, support)
            record(sources, name, title, '/comparisons/%s/scopes/all/geometry/%s' % (comparison, group), row,
                comparison=comparison, group=group, unit='pp', **{k: row[k] for k in ('delta_a_minus_b', 'ci95_lower', 'ci95_upper', 'paired_cases', 'total_cases', 'no_GT_support_cases', 'unrecorded_cases')})
        ax.set(yticks=range(6), yticklabels=GROUP_LABELS, ylim=(5.5, -.5), xlim=domain)
        ax.set_title(title, loc='left', pad=10)
        old.zero_axis(ax)
    axes[-1].set_xlabel('Mean change in GT-group centreline recall (pp)')
    fig.suptitle('Operational GT geometry: recall changes and case support', y=.97, fontsize=12)
    fig.text(.27, .915, caption(result), fontsize=8.1)
    fig.text(.08, .045, 'n = valid paired cases; noGT = cases without GT-group support; U = unrecorded groups. NA remains undefined.\nLee centreline voxel coverage: not branch detection, physical length, group precision or anatomical causality.\nExploratory paired-case intervals; overlapping skeleton nodes are not independent cases. Roles are not pooled.', fontsize=7.6)
    old.watermark(fig, synthetic)
    return name, fig


def render(result, cases, output, binding, repository, reader, bound, watermark_fixture=False):
    """Rendering never changes admission; a watermark only labels temporary QA."""
    old = dependencies(repository)
    import matplotlib
    import matplotlib.pyplot as plt
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink() and output.resolve() == output, 'Fresh ordinary output directory required')
    reader.recheck_files(bound)
    output.mkdir(parents=True, exist_ok=False); old.style()
    sources, saved = [], []
    figures = [case_figure(result, cases, old, sources, watermark_fixture), metric_figure(result, old, sources, watermark_fixture),
               geometry_figure(result, old, sources, watermark_fixture)]
    for name, fig in figures:
        try:
            fig.text(.99, .007, 'Published declaration/file hashes only; no author server or selection-timing attestation',
                     ha='right', fontsize=6.2, color='#565D63')
            fig.canvas.draw()
            for extension in ('png', 'pdf'):
                target = output / (name + '.' + extension)
                fig.savefig(target, dpi=300, metadata={'Creator': 'GTA-AVT portable declared-file plot; no rebootstrap'})
                saved.append(dict(file=target.name, sha256=sha(target), bytes=target.stat().st_size,
                    width_inches=fig.get_figwidth(), height_inches=fig.get_figheight()))
        finally:
            plt.close(fig)
    fields = list(dict.fromkeys(k for row in sources for k in row if k != 'reviewed_source_record'))
    with (output / 'figure_source.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k != 'reviewed_source_record'} for row in sources)
    mapping = dict(schema='GTA-AVT-minimal4-published-figure-source-v1', binding=binding, records=sources,
        csv_null_representation='empty field; JSON preserves null; never imputed zero')
    (output / 'figure_source.json').write_text(json.dumps(mapping, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    manifest = dict(schema='GTA-AVT-minimal4-published-figures-v1', source_level=result['source_verification_level'],
        stage=result['stage'], scope=SCOPE, case_ids=result['case_ids'], statistical_unit='released scan/case ID',
        patient_independence_verified=False, author_server_or_export_attested=False, author_private_marker_used=False,
        selection_time_independently_verified=False, source_marker_FILE_sha256=None,
        temporary_synthetic_watermark=watermark_fixture, binding=binding,
        common_reference_declaration_sha256=canonical(result['common_reference_runtime']),
        helper_sha256=sha(__file__), portable_reader_sha256=READER_SHA, original_plot_extraction_sha256=ORIGINAL_PLOT_SHA,
        released_style_sha256=OLD_PLOT_SHA, statistical_math_sha256=MATH_SHA, matplotlib_version=matplotlib.__version__,
        comparisons=[x[0] for x in COMPARISONS], bootstrap_recomputed=False,
        intervals_independently_recomputed_or_attested=False, source_record_count=len(sources), files=saved,
        source_CSV_sha256=sha(output / 'figure_source.csv'), source_JSON_sha256=sha(output / 'figure_source.json'),
        limitations=['Declarations and original metric/summary FILE hashes establish self-consistency only.',
            'Weights, data bytes, physical author servers and pre-TEST selection chronology are not independently attested.',
            'Two within-role comparisons remain separate; one seed and unverified case independence.',
            'All metric/group rows remain visible; GT-group voxel recall is not group precision, branches or physical length.',
            'Intervals are stored artifact values; plotting reconciles raw means/support but does not rerun bootstrap.',
            'Exported figures still require actual visual inspection.'])
    reader.recheck_files(bound); dependencies(repository)
    (output / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('analysis', 'analysis-sha256', 'declaration', 'declaration-sha256', 'selection-record', 'selection-sha256', 'output'):
        parser.add_argument('--' + name)
    parser.add_argument('--repository', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--reader-directory', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--execute-plot', action='store_true')
    args = parser.parse_args()
    if not args.execute_plot:
        print(json.dumps(dict(state='plan_only', sources_opened=False, heavy_imports=False, figures=3,
            fixed_slots=list(SLOTS), author_attestation=False, bootstrap_or_model_or_TEST_run=False,
            needs='explicit paired/declaration FILE SHA; TEST additionally declared-selection FILE SHA; fresh output')))
        return
    require(all((args.analysis, args.analysis_sha256, args.declaration, args.declaration_sha256, args.output)),
            'Explicit existing paired/declaration FILE+SHA and fresh output required')
    result, cases, binding, reader, bound = actual_inputs(args.analysis, args.analysis_sha256,
        args.declaration, args.declaration_sha256, args.repository, args.reader_directory, args.selection_record, args.selection_sha256)
    manifest = render(result, cases, args.output, binding, args.repository, reader, bound)
    print(json.dumps(dict(state='complete_published_declaration_figures', stage=result['stage'], files=len(manifest['files']),
        author_attestation=False, bootstrap_recomputed=False, requires_visual_inspection=True)))


if __name__ == '__main__':
    main()

