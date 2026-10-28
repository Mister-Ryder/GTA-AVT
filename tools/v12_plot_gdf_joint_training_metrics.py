"""Plot the isolated gdf-joint revision from exported objective CSVs only.

The shared 2k head has its own axis. Geometry and refine are separate figures
with phase-local successful-update coordinates; missing future phases are not
drawn. No old 92k workflow coordinates, smoothing or native-score placeholders.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'shared_head': '#8764A5', 'AR': '#176B9B', 'AR_GDF': '#C8782D'}
PHASE_BUDGET = {'shared_head': 2000, 'geometry': 20000, 'refine': 10000}


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def num(row, key):
    try:
        result = float(row.get(key, ''))
        return result if math.isfinite(result) else math.nan
    except (TypeError, ValueError):
        return math.nan


def phase_rows(rows, run, phase):
    return sorted((r for r in rows if r['run_label'] == run and r['phase'] == phase),
                  key=lambda r: num(r, 'successful_update'))


def draw(ax, rows, run, phase, key, label=None, color=None, scale=1.,
         marker=None, linestyle='-', alpha=1.):
    source = phase_rows(rows, run, phase)
    x = np.array([num(r, 'successful_update') / 1000 for r in source])
    y = np.array([num(r, key) * scale for r in source])
    good = np.isfinite(x) & np.isfinite(y)
    if good.any():
        ax.plot(x[good], y[good], label=label or run, color=color or COLORS[run],
                lw=1.15, marker=marker, markersize=4, linestyle=linestyle, alpha=alpha)
    return bool(good.any())


def finish_axis(ax, title, ylabel, observed_max):
    ax.set_title(title, loc='left', fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Phase-local successful updates (thousands)')
    ax.set_xlim(0, max(observed_max / 1000 * 1.025, .1))
    ax.grid(axis='y', color='#DFE5EA', lw=.65)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best', frameon=False, fontsize=8)
    else:
        ax.text(.5, .5, 'No recorded values for this metric', ha='center',
                va='center', transform=ax.transAxes, color='#59656E')


def save(fig, output, stem):
    files = []
    for extension in ('png', 'svg'):
        path = output / (stem + '.' + extension)
        fig.savefig(path, dpi=170, facecolor='white')
        files.append(path)
    plt.close(fig)
    return files


def shared_head(train, val, output):
    source = phase_rows(train, 'shared_head', 'shared_head')
    validations = phase_rows(val, 'shared_head', 'shared_head')
    if not source:
        return []
    observed_max = max(num(r, 'successful_update') for r in source)
    fig, aa = plt.subplots(2, 2, figsize=(14, 8.6))
    fig.subplots_adjust(left=.075, right=.98, bottom=.13, top=.83, hspace=.48, wspace=.27)
    fig.suptitle('GTA-AVT gdf-joint-r3: shared Gaussian-head pretraining',
                 x=.075, y=.975, ha='left', fontsize=17, fontweight='bold')
    fig.text(.075, .927, 'Shared head only | Planned 2,000 successful updates | P and S frozen | New head: 55 parameters', fontsize=10.5)
    fig.text(.075, .888, 'Recorded sparse TRAIN scalars and every recorded fixed-patch VAL observation; no smoothing', fontsize=10)
    draw(aa[0, 0], train, 'shared_head', 'shared_head', 'losses.gdf_mse', 'TRAIN GDF MSE', alpha=.65)
    draw(aa[0, 0], val, 'shared_head', 'shared_head', 'losses.gdf_mse', 'VAL GDF MSE', color='#C8782D', marker='o')
    finish_axis(aa[0, 0], 'A  Gaussian regression loss', 'MSE', observed_max)
    draw(aa[0, 1], val, 'shared_head', 'shared_head', 'losses.gdf_fg_mse', 'VAL foreground', marker='o')
    draw(aa[0, 1], val, 'shared_head', 'shared_head', 'losses.gdf_bg_mse', 'VAL background', color='#27836B', marker='s')
    finish_axis(aa[0, 1], 'B  Foreground / background field error', 'MSE', observed_max)
    draw(aa[1, 0], val, 'shared_head', 'shared_head', 'summary.dice.mean', 'Fixed-patch VAL Dice', scale=100, marker='o')
    finish_axis(aa[1, 0], 'C  Frozen-S segmentation diagnostic', 'Dice (%)', observed_max)
    diagnostic = [100 * num(r, 'summary.dice.mean') for r in validations]
    diagnostic = [v for v in diagnostic if math.isfinite(v)]
    if diagnostic:
        aa[1, 0].set_ylim(min(diagnostic) - .05, max(diagnostic) + .05)
    aa[1, 0].ticklabel_format(axis='y', style='plain', useOffset=False)
    draw(aa[1, 1], train, 'shared_head', 'shared_head', 'learning_rates.gdf', 'Head learning rate')
    finish_axis(aa[1, 1], 'D  Recorded head learning rate', 'Learning rate', observed_max)
    rates = [num(r, 'learning_rates.gdf') for r in source]
    rates = [v for v in rates if math.isfinite(v)]
    if rates:
        aa[1, 1].set_ylim(0, max(rates) * 1.1)
    fig.text(.075, .025, 'The shared 2k head is trained once. Its segmentation diagnostic is not evidence of S learning or native whole-case performance.', fontsize=9, color='#46525C')
    return save(fig, output, '01_shared_gaussian_head')


def paired_phase(train, val, phase, output):
    source = [r for r in train + val if r['run_label'] in ('AR', 'AR_GDF') and r['phase'] == phase]
    if not source:
        return []
    observed_max = max(num(r, 'successful_update') for r in source)
    fig, aa = plt.subplots(2, 3, figsize=(17, 9.2))
    fig.subplots_adjust(left=.065, right=.985, bottom=.14, top=.825, hspace=.47, wspace=.29)
    fig.suptitle('GTA-AVT gdf-joint-r3: ' + phase + ' matched-pair observations',
                 x=.065, y=.975, ha='left', fontsize=17, fontweight='bold')
    fig.text(.065, .928, 'AR: no GDF gradient | AR_GDF: calibrated GDF | P frozen; full S trainable; BN running statistics fixed', fontsize=10.5)
    fig.text(.065, .886, f'Planned phase budget: {PHASE_BUDGET[phase]:,} successful updates per arm | Only recorded observations are shown', fontsize=10)
    panels = (
        (train, 'losses.primary_segmentation_loss', 'A  TRAIN primary segmentation loss', 'Dice + ratio BCE', 1., None),
        (val, 'summary.dice.mean', 'B  Fixed-patch VAL Dice', 'Dice (%)', 100., 'o'),
        (train, 'gdf_weight', 'C  Applied GDF weight', 'GDF multiplier', 1., None),
        (val, 'losses.gdf_fg_mse', 'D  VAL foreground Gaussian error', 'MSE', 1., 'o'),
        (val, 'losses.gdf_bg_mse', 'E  VAL background Gaussian error', 'MSE', 1., 'o'))
    for ax, (rows, key, title, ylabel, scale, marker) in zip(aa.flat, panels):
        for run in ('AR', 'AR_GDF'):
            draw(ax, rows, run, phase, key, scale=scale, marker=marker,
                 alpha=.7 if rows is train else 1.)
        finish_axis(ax, title, ylabel, observed_max)
    ax = aa[1, 2]
    learning_keys = sorted({key for r in source for key in r if key.startswith('learning_rates.')})
    styles = ('-', '--', ':', '-.')
    for run in ('AR', 'AR_GDF'):
        for index, key in enumerate(learning_keys):
            draw(ax, train, run, phase, key, run + ' ' + key.split('.', 1)[1],
                 linestyle=styles[index % len(styles)])
    finish_axis(ax, 'F  Recorded optimizer-group rates', 'Learning rate', observed_max)
    fig.text(.065, .055, 'TRAIN curves use exported raw samples; VAL points are original observations. Lines do not cross phase resets; future phases have no placeholders.', fontsize=9, color='#46525C')
    fig.text(.065, .025, 'Patch Dice and field error do not substitute for final native VAL8. Total losses differ between arms; the primary segmentation term is shown separately.', fontsize=9, color='#46525C')
    return save(fig, output, ('02_' if phase == 'geometry' else '03_') + phase + '_matched_pair')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.input); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    train = read_rows(source / 'training_scalars.csv')
    val = read_rows(source / 'validation_summary.csv')
    if any(r['run_label'] not in COLORS or r['phase'] not in PHASE_BUDGET for r in train + val):
        raise ValueError('Only isolated gdf-joint shared_head/AR/AR_GDF phases are supported')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 11, 'axes.labelsize': 9, 'legend.fontsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'svg.fonttype': 'none', 'axes.unicode_minus': False})
    produced = shared_head(train, val, output)
    for phase in ('geometry', 'refine'):
        produced += paired_phase(train, val, phase, output)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'schema': 'GTA-AVT-gdf-joint-training-figures-1',
        'revision': '20261028-gdf-joint-r3', 'smoothing': False, 'interpolation': False,
        'axes': 'phase-local successful updates; shared head is separate; no old 92k workflow axis',
        'native_whole_case_results_plotted': False, 'planned_phase_budgets': PHASE_BUDGET,
        'input_csv_sha256': {name: sha(source / name) for name in ('training_scalars.csv', 'validation_summary.csv')},
        'figure_sha256': {p.name: sha(p) for p in produced}, 'matplotlib_version': matplotlib.__version__}
    (output / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'figures': list(manifest['figure_sha256']), 'train_rows': len(train), 'val_rows': len(val)}))


if __name__ == '__main__':
    main()
