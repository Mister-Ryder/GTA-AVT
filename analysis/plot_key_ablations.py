"""Static scientific PNG/PDF exports from analyze_key_ablations analysis.json.

No metric estimation or inference is performed. All patient points and paired
intervals come from the reviewed analysis artifact. Self-test figures are
watermarked SYNTHETIC and deleted with their temporary fixtures.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import analyze_key_ablations as analysis


COLORS = {'blue': '#24679B', 'gold': '#A86D16', 'grey': '#565D63'}
COMPARISON_STYLE = {
    'main_A_minus_C': ('A soft - C fusion', COLORS['blue'], 'o'),
    'main_A_minus_constant': ('A soft - constant', COLORS['gold'], 's'),
    'ablations_full_minus_scale': ('Full soft - scale only', COLORS['blue'], 'o'),
    'ablations_full_minus_structure': ('Full soft - structure only', COLORS['gold'], 's'),
    'ablations_soft_minus_hard': ('Soft - hard anchor', COLORS['grey'], '^'),
}
GROUP_LABELS = ['Radius <=2 mm', 'Radius 2-4 mm', 'Radius 4-8 mm', 'Radius >8 mm',
                'Nonjunction', 'Junction vicinity']
WIDTH = 7.1  # approximately180mm, a two-column journal figure


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(result):
    if result.get('schema') != 'GTA-AVT-key-ablation-analysis-v1':
        raise ValueError('Only analyze_key_ablations analysis.json artifacts are accepted')
    if result.get('seed') != analysis.SEED or result.get('bootstrap_draws') != analysis.DRAWS:
        raise ValueError('Bootstrap provenance differs from the frozen analysis')
    scope = result.get('patient_scope')
    if scope not in analysis.FIXED_CASES or result.get('patient_ids') != analysis.FIXED_CASES[scope]:
        raise ValueError('Patient scope differs from the frozen analysis')
    if (result.get('stage') == 'test') != (scope == 'final_test12'):
        raise ValueError('Validation and final test cannot be mixed')
    if not analysis.valid_sha(result.get('index_sha256')):
        raise ValueError('Analysis input provenance is missing')
    ids = result['patient_ids']
    rows = result.get('per_case_metrics', [])
    by_method = {}
    for row in rows:
        method = row['method']
        if method not in analysis.METHODS or row.get('role') != analysis.METHODS[method][0] or row.get('stage') != result['stage']:
            raise ValueError('Invalid method/role/stage in patient table')
        if row.get('fraction_metric_value_unit') != 'percent':
            raise ValueError('Patient metric percentages must be explicitly declared')
        if row['case_id'] in by_method.setdefault(method, {}):
            raise ValueError('Duplicated patient/method in analysis table')
        if row.get('cohort') != analysis.cohort(row['case_id']):
            raise ValueError('Patient cohort identity differs')
        by_method[method][row['case_id']] = row
    for method in ('A_soft', 'C_concat', 'A_soft_constant'):
        if set(by_method.get(method, {})) != set(ids):
            raise ValueError('Patient Dice figure requires all main A/C/constant patients')
    for method, patients in by_method.items():
        if set(patients) != set(ids):
            raise ValueError('A method omits a stage patient')
        if any(not isinstance(row.get('dice'), (int, float)) or not np.isfinite(row['dice']) or not 0. <= row['dice'] <= 100. for row in patients.values()):
            raise ValueError('Patient Dice values must be percentages in [0,100]')
    # Independently reconcile displayed aggregate differences with patient rows.
    seen = set()
    for row in result['paired_summaries']:
        if row['cohort'] != 'all':
            continue
        key = (row['comparison'], row['metric'])
        if key in seen:
            raise ValueError('Duplicated overall paired summary')
        seen.add(key)
        if row['difference_unit'] != analysis.METRICS[row['metric']][0]:
            raise ValueError('Difference units disagree with the declared metric')
        a, b = row['method_a'], row['method_b']
        if a not in by_method or b not in by_method or analysis.METHODS[a][0] != analysis.METHODS[b][0]:
            raise ValueError('Cross-role or unavailable comparison cannot be plotted')
        valid = [(by_method[a][c].get(row['metric']), by_method[b][c].get(row['metric'])) for c in ids]
        valid = [(av, bv) for av, bv in valid if av is not None and bv is not None]
        if row['paired_cases'] != len(valid) or row['total_cases'] != len(ids):
            raise ValueError('Paired support does not reconcile with patient values')
        if valid and not np.isclose(np.mean([av - bv for av, bv in valid]), row['delta_a_minus_b'], atol=1e-9, rtol=1e-9):
            raise ValueError('Paired mean does not reconcile with patient-level differences')
    group_cases = {}
    for row in result['geometry_case_recall']:
        key = (row['method'], row['case_id'], row['group'])
        if key in group_cases:
            raise ValueError('Duplicated patient/method/geometry record')
        group_cases[key] = row['centerline_recall_percent']
    for row in result['geometry_paired_summaries']:
        if row['cohort'] != 'all':
            continue
        values = [(group_cases[(row['method_a'], c, row['group'])],
                   group_cases[(row['method_b'], c, row['group'])]) for c in ids]
        valid = [(av, bv) for av, bv in values if av is not None and bv is not None]
        if row['paired_cases'] != len(valid) or row['total_cases'] != len(ids) or row['difference_unit'] != 'pp':
            raise ValueError('Geometry paired support/units do not reconcile with patient group recalls')
        if valid and not np.isclose(np.mean([av - bv for av, bv in valid]), row['delta_a_minus_b'], atol=1e-9, rtol=1e-9):
            raise ValueError('Geometry mean is not the unweighted paired patient difference')
    return by_method


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#676D72', 'text.color': '#252B30', 'axes.labelcolor': '#252B30',
        'xtick.color': '#454B50', 'ytick.color': '#454B50', 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})


def caption(result):
    label = 'Held-out test' if result['stage'] == 'test' else 'Validation development'
    incomplete = ' | INCOMPLETE method set' if not result['complete_method_set'] else ''
    return '%s | %s | n=%d | one training seed%s' % (label, result['patient_scope'], len(result['patient_ids']), incomplete)


def watermark(fig, synthetic):
    if synthetic:
        fig.text(.5, .51, 'SYNTHETIC SELF-TEST - NOT RESULTS', ha='center', va='center',
                 fontsize=20, color='#8D2D2D', alpha=.26, rotation=24, zorder=100)


def limits(rows):
    values = [0.]
    for row in rows:
        for key in ('delta_a_minus_b', 'ci95_lower', 'ci95_upper'):
            if row.get(key) is not None:
                value = float(row[key])
                if not np.isfinite(value):
                    raise ValueError('Nonfinite estimate/interval in figure source')
                values.append(value)
    low, high = min(values), max(values)
    pad = max((high - low) * .15, .25)
    return low - pad, high + pad


def interval(ax, y, row, color, marker):
    if row['delta_a_minus_b'] is None:
        if row['paired_cases'] != 0:
            raise ValueError('Undefined estimate has nonzero paired support')
        return
    lo, hi = row['ci95_lower'], row['ci95_upper']
    if lo is None or hi is None or not np.isfinite([lo, hi]).all() or lo > hi:
        raise ValueError('Missing or invalid paired confidence bounds')
    ax.hlines(y, lo, hi, color=color, linewidth=1.25)
    ax.vlines([lo, hi], y - .055, y + .055, color=color, linewidth=.8)
    ax.plot(row['delta_a_minus_b'], y, marker=marker, markersize=4.8, color=color,
            markerfacecolor='white' if marker == 's' else color, markeredgewidth=1.1)


def zero_axis(ax):
    ax.axvline(0., color='#7C8287', linewidth=.8, linestyle='--', zorder=0)
    ax.grid(axis='x', color='#E5E8EA', linewidth=.5)
    ax.set_axisbelow(True)


def source_row(figure, panel, row, **extra):
    keys = ('comparison', 'metric', 'group', 'method_a', 'method_b', 'delta_a_minus_b',
            'ci95_lower', 'ci95_upper', 'paired_cases', 'total_cases', 'difference_unit',
            'no_gt_support_cases', 'unrecorded_group_cases')
    return dict(figure=figure, panel=panel, **{k: row.get(k) for k in keys}, **extra)


def patient_figure(result, by_method, sources, synthetic):
    name = '01_main_patient_dice'
    ids = result['patient_ids']
    fig, ax = plt.subplots(figsize=(WIDTH, max(3.25, .24 * len(ids) + 1.8)))
    fig.subplots_adjust(left=.21, right=.96, top=.78, bottom=.2)
    styles = [('C_concat', 'C fusion', COLORS['grey'], 'o', -.14),
              ('A_soft', 'A soft', COLORS['blue'], 'D', 0.),
              ('A_soft_constant', 'A constant', COLORS['gold'], 's', .14)]
    for i, case_id in enumerate(ids):
        positions, values = [], []
        for method, _, _, _, offset in styles:
            positions.append(i + offset)
            values.append(by_method[method][case_id]['dice'])
        ax.plot(values, positions, color='#B8BEC3', linewidth=.7, zorder=1)
    for method, label, color, marker, offset in styles:
        values = [by_method[method][c]['dice'] for c in ids]
        ax.scatter(values, np.arange(len(ids)) + offset, label=label, c=color if marker != 's' else 'white',
                   edgecolors=color, marker=marker, s=24, linewidths=1., zorder=3)
        for case_id, value in zip(ids, values):
            sources.append(dict(figure=name, panel='patient_dice', method=method, case_id=case_id,
                                cohort=analysis.cohort(case_id), value=value, value_unit='percent'))
    for i in range(1, len(ids)):
        if analysis.cohort(ids[i]) != analysis.cohort(ids[i - 1]):
            ax.axhline(i - .5, color='#D9DEE1', linewidth=.8)
    ax.set(yticks=np.arange(len(ids)), yticklabels=[c.replace('_', ' ') for c in ids],
           ylim=(len(ids) - .5, -.5), xlim=(0., 100.), xlabel='Patient Dice (%)')
    ax.grid(axis='x', color='#E6E9EB', linewidth=.5)
    ax.legend(loc='lower left', bbox_to_anchor=(0., 1.03), ncol=3, frameon=False, borderaxespad=0.)
    fig.suptitle('Patient Dice: main training role', y=.98, fontsize=11)
    fig.text(.21, .895, caption(result), fontsize=8)
    fig.text(.21, .025, 'Every patient is shown. Grey connectors link the same patient; vertical offsets separate methods.', fontsize=7.5)
    watermark(fig, synthetic)
    return name, fig


def dice_difference_figure(result, sources, synthetic):
    name = '02_paired_dice_differences'
    rows = [r for r in result['paired_summaries'] if r['cohort'] == 'all' and r['metric'] == 'dice']
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 4.5), sharex=True, gridspec_kw={'height_ratios': [2, 3]})
    fig.subplots_adjust(left=.37, right=.86, top=.81, bottom=.18, hspace=.65)
    xlim = limits(rows)
    for ax, role, title in zip(axes, ('main', 'ablations'),
            ('Main training role', 'Second training role: within-role full-A bridge')):
        selected = [r for r in rows if r['role'] == role]
        for i, row in enumerate(selected):
            label, color, marker = COMPARISON_STYLE[row['comparison']]
            interval(ax, i, row, color, marker)
            ax.text(1.03, i, 'n=%d/%d' % (row['paired_cases'], row['total_cases']),
                    transform=ax.get_yaxis_transform(), va='center', fontsize=8)
            sources.append(source_row(name, role, row))
        ax.set(yticks=np.arange(len(selected)), yticklabels=[COMPARISON_STYLE[r['comparison']][0] for r in selected],
               ylim=(max(len(selected), 1) - .5, -.5), xlim=xlim)
        if not selected:
            ax.text(.5, .5, 'No completed within-role contrast', ha='center', transform=ax.transAxes, fontsize=8)
        ax.set_title(title, loc='left', pad=7)
        zero_axis(ax)
    axes[-1].set_xlabel('Paired mean Dice difference, A - B (pp; + favors A)')
    fig.suptitle('Paired Dice differences', y=.985, fontsize=11)
    fig.text(.07, .915, caption(result), fontsize=8)
    fig.text(.07, .055, 'Points: mean patient differences; lines: 95% paired patient bootstrap CI (20,000 resamples).', fontsize=7.5)
    fig.text(.07, .02, 'Training roles are separated. One-seed patient intervals do not estimate training-seed stability.', fontsize=7.5)
    watermark(fig, synthetic)
    return name, fig


def geometry_figure(result, role, sources, synthetic):
    name = '03_main_geometry_and_fp' if role == 'main' else '04_second_role_geometry_and_fp'
    geometry = [r for r in result['geometry_paired_summaries'] if r['cohort'] == 'all' and r['role'] == role]
    fp = [r for r in result['paired_summaries'] if r['cohort'] == 'all' and r['role'] == role and r['metric'] == 'fp_ml']
    comparisons = [x[0] for x in analysis.COMPARISONS if any(r['comparison'] == x[0] for r in fp)]
    if not comparisons:
        return None
    fig, (ax, companion) = plt.subplots(1, 2, figsize=(WIDTH, 5.4), gridspec_kw={'width_ratios': [3.4, 1.3]})
    fig.subplots_adjust(left=.18, right=.89, top=.72, bottom=.28, wspace=1.05)
    offsets = np.linspace(-.22, .22, len(comparisons)) if len(comparisons) > 1 else [0.]
    handles = []
    for contrast, offset in zip(comparisons, offsets):
        label, color, marker = COMPARISON_STYLE[contrast]
        handles.append(plt.Line2D([0], [0], marker=marker, color=color, linewidth=1., markersize=4.5,
            markerfacecolor='white' if marker == 's' else color, label=label))
        for i, group in enumerate(analysis.GROUPS):
            candidates = [r for r in geometry if r['comparison'] == contrast and r['group'] == group]
            if len(candidates) != 1:
                raise ValueError('Every plotted contrast must retain every GT group, including undefined groups')
            row = candidates[0]
            interval(ax, i + offset, row, color, marker)
            support = '%d/%d' % (row['paired_cases'], row['total_cases'])
            if row['delta_a_minus_b'] is None:
                support += ' NA'
            ax.text(1.02, i + offset, support, transform=ax.get_yaxis_transform(),
                    va='center', color=color, fontsize=7)
            sources.append(source_row(name, 'GT_group_recall', row))
    all_geometry = [r for r in result['geometry_paired_summaries'] if r['cohort'] == 'all']
    ax.set(yticks=np.arange(len(analysis.GROUPS)), yticklabels=GROUP_LABELS,
           ylim=(len(analysis.GROUPS) - .55, -.55), xlim=limits(all_geometry),
           xlabel='GT-group recall difference\n(pp; + favors A)')
    ax.set_title('(a) GT-group recall', loc='left', pad=8)
    ax.text(1.02, 1.035, 'Paired n', transform=ax.transAxes, fontsize=7.5)
    zero_axis(ax)
    for i, contrast in enumerate(comparisons):
        row = next(r for r in fp if r['comparison'] == contrast)
        _, color, marker = COMPARISON_STYLE[contrast]
        interval(companion, i, row, color, marker)
        companion.text(1.04, i, '%d/%d' % (row['paired_cases'], row['total_cases']),
                       transform=companion.get_yaxis_transform(), va='center', color=color, fontsize=7)
        sources.append(source_row(name, 'global_FP_volume', row))
    all_fp = [r for r in result['paired_summaries'] if r['cohort'] == 'all' and r['metric'] == 'fp_ml']
    companion.set(yticks=[], ylim=(len(comparisons) - .55, -.55), xlim=limits(all_fp),
                  xlabel='Global FP difference\n(ml; - favors A)')
    companion.set_title('(b) Global FP', loc='left', pad=8)
    zero_axis(companion)
    title = 'Main training role' if role == 'main' else 'Second training role: within-role full-A bridge'
    fig.suptitle('Geometry recall and global false-positive tradeoff', y=.985, fontsize=11)
    fig.text(.045, .94, title, fontsize=8)
    fig.text(.045, .908, caption(result), fontsize=7.7)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.51, .88), ncol=1 if len(handles) > 2 else 2,
               frameon=False, fontsize=7.5, handlelength=2.)
    fig.text(.045, .115, 'Means weight supported patients equally. Lines: 95% paired patient bootstrap CI; n=defined pairs/total.', fontsize=7.3)
    fig.text(.045, .075, 'NA groups remain undefined; no zero imputation or GT-group precision. Global FP captures the tradeoff.', fontsize=7.3)
    fig.text(.045, .035, 'Radius/junction groups use operational GT geometry. Same reference metrics; training roles are never pooled.', fontsize=7.3)
    watermark(fig, synthetic)
    return name, fig


def plot_result(result, output, source_sha256, synthetic=False):
    by_method = validate(result)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Figure outputs are immutable; choose a new empty directory')
    output.mkdir(parents=True, exist_ok=True)
    style()
    sources, saved = [], []
    figures = [patient_figure(result, by_method, sources, synthetic),
               dice_difference_figure(result, sources, synthetic)]
    for role in ('main', 'ablations'):
        item = geometry_figure(result, role, sources, synthetic)
        if item is not None:
            figures.append(item)
    for name, fig in figures:
        try:
            fig.canvas.draw()
            for extension in ('png', 'pdf'):
                path = output / (name + '.' + extension)
                fig.savefig(path, dpi=300, metadata={'Creator': 'GTA-AVT static analysis helper'})
                saved.append(dict(file=path.name, sha256=sha(path), width_inches=float(fig.get_figwidth()),
                                  height_inches=float(fig.get_figheight()), dpi=300 if extension == 'png' else None))
        finally:
            plt.close(fig)
    fields = list(dict.fromkeys(k for row in sources for k in row))
    with (output / 'figure_source.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sources)
    manifest = dict(schema='GTA-AVT-key-ablation-figures-v1', source_analysis_sha256=source_sha256,
        plot_helper_sha256=sha(__file__), stage=result['stage'], patient_scope=result['patient_scope'],
        patient_ids=result['patient_ids'], complete_method_set=result['complete_method_set'],
        bootstrap=dict(unit='paired patient', draws=result['bootstrap_draws'], seed=result['seed'], interval='percentile95%'),
        synthetic=synthetic, files=saved, source_table_sha256=sha(output / 'figure_source.csv'),
        selected_execution_device=result['reference_identity']['selected_execution_device'],
        reference_environment_sha256=analysis.canonical_sha(result['reference_identity']['reference_environment']),
        limitations=['One seed; patient uncertainty only.', 'No cross-training-role pooling or SOTA ranking.',
                     'All patients shown; unsupported groups remain undefined.', 'Actual-data exports still require visual review.'])
    (output / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    return manifest


def self_test():
    """Reuse the analysis engine's temporary genuine-schema fixtures."""
    original = analysis.save_result
    summary = {}

    def save_and_plot(result, output):
        original(result, output)
        manifest = plot_result(result, Path(output) / 'SYNTHETIC_figures', sha(Path(output) / 'analysis.json'), synthetic=True)
        assert manifest['synthetic'] and len(manifest['files']) == 8
        from PIL import Image
        for row in manifest['files']:
            path = Path(output) / 'SYNTHETIC_figures' / row['file']
            if path.suffix == '.png':
                with Image.open(path) as image:
                    assert image.width >= 2000 and image.height >= 900
            else:
                assert path.read_bytes().startswith(b'%PDF-')
        primary_points = [r for r in read(Path(output) / 'analysis.json')['per_case_metrics'] if r['method'] in ('A_soft', 'C_concat', 'A_soft_constant')]
        assert len(primary_points) == len(result['patient_ids']) * 3
        summary.update(state='passed', temporary_figures=8, patient_points=len(primary_points),
                       all_temporary_figures_watermarked=True, synthetic_files_retained=False)

    analysis.save_result = save_and_plot
    try:
        analysis.self_test()
    finally:
        analysis.save_result = original
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', help='Actual analyze_key_ablations analysis.json')
    parser.add_argument('--output')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        if args.analysis or args.output:
            parser.error('--self-test cannot be combined with real input/output')
        print(json.dumps(self_test()))
        return
    if not args.analysis or not args.output:
        parser.error('Specify --analysis and a new --output directory')
    manifest = plot_result(read(args.analysis), args.output, sha(args.analysis))
    print(json.dumps(dict(state='complete', figures=len(manifest['files']), stage=manifest['stage'],
                         synthetic=False, actual_data_visual_review_required=True)))


if __name__ == '__main__':
    main()
