"""Neutral scientific figures from the public scalar CSVs (no model execution).

Training observations are shown as exported raw scalar samples. Validation points
are shown at their original recorded successful-update coordinates. Lines never
bridge phase resets. Native whole-case endpoints are kept in a separate figure.
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
from matplotlib.lines import Line2D
import numpy as np

PHASES = ('base_warm', 'base', 'condition', 'gdf_head', 'geometry', 'refine')
BUDGET = dict(base_warm=20000, base=30000, condition=10000, gdf_head=2000, geometry=20000, refine=10000)
OFFSETS = {}; total = 0
for phase in PHASES:
    OFFSETS[phase] = total; total += BUDGET[phase]
COLORS = dict(zip(PHASES, ('#28799B', '#C98532', '#7771A6', '#B8839E', '#27836B', '#507A9B')))
RUN_COLORS = {'C0': '#7B8792', 'AR': '#176B9B', 'C0_GDF': '#CF792D', 'AR_GDF': '#238369'}


def rows(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def num(row, key):
    value = row.get(key)
    if value is None or value == '':
        return math.nan
    try:
        result = float(value)
    except (ValueError, TypeError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def phase_rows(data, run, phase):
    return sorted((r for r in data if r.get('run_label') == run and r.get('phase') == phase),
                  key=lambda r: num(r, 'successful_update'))


def series(ax, data, run, phase, metric, color, label=None, scale=1., local=False,
           marker=None, linestyle='-', alpha=1.):
    source = phase_rows(data, run, phase)
    x = np.asarray([num(r, 'successful_update') for r in source])
    y = np.asarray([scale * num(r, metric) for r in source])
    good = np.isfinite(x) & np.isfinite(y)
    if not good.any():
        return []
    x = (x[good] + (0 if local else OFFSETS.get(phase, 0))) / 1000.
    y = y[good]
    ax.plot(x, y, color=color, label=label, lw=1.15, alpha=alpha,
            marker=marker, markersize=3, linestyle=linestyle)
    return y.tolist()


def shade(ax):
    for phase in PHASES:
        start = OFFSETS[phase] / 1000.
        end = start + BUDGET[phase] / 1000.
        ax.axvspan(start, end, color=COLORS[phase], alpha=.045, zorder=-5)
        if start:
            ax.axvline(start, color='#A9B0B6', ls='--', lw=.7)
    ax.set_xlim(0, 92)
    ax.set_xticks((0, 20, 40, 50, 62, 82, 92))
    ax.set_xlabel('Executed post-P workflow updates (thousands)')
    ax.grid(axis='y', color='#E0E5EA', lw=.65)
    ax.set_axisbelow(True)


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 12, 'axes.labelsize': 10, 'legend.fontsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'svg.fonttype': 'none', 'axes.unicode_minus': False})


def save(fig, output, stem):
    for suffix in ('png', 'svg'):
        fig.savefig(output / (stem + '.' + suffix), dpi=170, facecolor='white')
    plt.close(fig)


def main_trajectory(train, val, run, output):
    fig, aa = plt.subplots(2, 3, figsize=(18, 10.5))
    fig.subplots_adjust(left=.06, right=.985, bottom=.14, top=.845, hspace=.42, wspace=.28)
    fig.suptitle('GTA-AVT ratio-r2: recorded main-run training and fixed-patch validation',
                 x=.06, y=.974, ha='left', fontsize=17, fontweight='bold')
    fig.text(.06, .93, 'Main run: ' + run + ' | Raw sampled TRAIN scalars and original VAL observations; no smoothing', fontsize=11)
    fig.text(.06, .895, 'base_warm 20k | base 30k | condition 10k | gdf_head 2k | geometry 20k | refine 10k', fontsize=10)
    for ax in aa.flat:
        shade(ax)
    ax = aa[0, 0]
    for phase in PHASES:
        series(ax, train, run, phase, 'losses.loss', COLORS[phase], phase, alpha=.6)
        series(ax, val, run, phase, 'losses.loss', COLORS[phase], marker='o', linestyle='--')
    ax.set_ylabel('Total loss (phase objectives differ)')
    ax.set_title('A  Total loss: TRAIN solid / VAL dashed', loc='left')
    ax.legend(frameon=False, ncol=2)
    ax = aa[0, 1]
    for phase in PHASES:
        if phase == 'gdf_head':
            continue
        series(ax, train, run, phase, 'losses.primary_segmentation_loss', '#89939E',
               'TRAIN samples' if phase == 'base_warm' else None, alpha=.6)
        series(ax, val, run, phase, 'losses.primary_segmentation_loss', '#176B9B',
               'Fixed VAL' if phase == 'base_warm' else None, marker='o')
    ax.set_ylabel('Primary Dice + ratio-BCE loss')
    ax.set_title('B  Primary segmentation loss (head warmup omitted)', loc='left')
    ax.legend(frameon=False)
    ax = aa[0, 2]
    for metric, color, label in (('dice', '#176B9B', 'Dice'), ('precision', '#8056A0', 'Precision'), ('recall', '#CF792D', 'Recall')):
        for phase in PHASES:
            series(ax, val, run, phase, 'summary.' + metric + '.mean', color,
                   label if phase == 'base_warm' else None, scale=100., marker='o')
    ax.set_ylabel('Fixed VAL64 patch metric (%)')
    ax.set_ylim(0, 102)
    ax.set_title('C  Fixed-patch validation metrics', loc='left')
    ax.legend(frameon=False, loc='lower right')
    ax = aa[1, 0]
    for key, color, label in (('learning_rates.shared', '#176B9B', 'Shared'),
                              ('learning_rates.gates', '#8056A0', 'Gates'),
                              ('learning_rates.gdf', '#238369', 'Gaussian head')):
        added = False
        for phase in PHASES:
            values = series(ax, train, run, phase, key, color, label if not added else None)
            added |= bool(values)
    ax.set_yscale('log'); ax.set_ylim(1e-8, 2e-3)
    ax.set_ylabel('Learning rate (log scale)')
    ax.set_title('D  Recorded optimizer-group learning rates', loc='left')
    ax.legend(frameon=False)
    ax = aa[1, 1]
    for key, color, label in (('alpha', '#8056A0', 'Condition strength alpha'), ('gdf_weight', '#238369', 'GDF weight')):
        for phase in PHASES:
            series(ax, train, run, phase, key, color, label if phase == 'base_warm' else None)
    ax.set_ylabel('Coefficient'); ax.set_ylim(-.05, 1.15)
    ax.set_title('E  Recorded condition and GDF schedules', loc='left')
    ax.legend(frameon=False)
    ax = aa[1, 2]
    for key, color, label in (('losses.gdf_fg_mse', '#238369', 'Foreground'), ('losses.gdf_bg_mse', '#CF792D', 'Background')):
        for phase in PHASES:
            series(ax, val, run, phase, key, color, label if phase == 'base_warm' else None, marker='o')
    ax.set_yscale('log'); ax.set_ylabel('Unweighted GDF MSE (log scale)')
    ax.set_title('F  Fixed-patch GDF errors', loc='left')
    ax.legend(frameon=False)
    fig.text(.06, .075, 'VAL64: eight fixed patches per case, averaged within case and then across eight cases. These are not native whole-case metrics.', fontsize=9.5)
    fig.text(.06, .047, 'The x-axis records executed workflow updates, including 2k head-only updates; it is not selected-weight lineage. Phase resets are not joined.', fontsize=9.5)
    fig.text(.06, .020, 'The selected base checkpoint is step27k after warm20k; later phases inherit it. Total-loss scales differ by supervision and weight schedule.', fontsize=9.5)
    save(fig, output, '01_main_training_patch_validation')


def attribute_trajectory(train, val, run, output):
    if not phase_rows(val, run, 'attribute'):
        return
    fig, aa = plt.subplots(1, 3, figsize=(17.4, 5.8))
    fig.subplots_adjust(left=.06, right=.985, bottom=.22, top=.77, wspace=.28)
    fig.suptitle('Attribute pretraining: recorded P metrics', x=.06, y=.955, ha='left', fontsize=17, fontweight='bold')
    fig.text(.06, .86, 'Main-run predictor pretraining | P support scores and radius regression are separate from S segmentation Dice', fontsize=10.5)
    ax = aa[0]
    series(ax, train, run, 'attribute', 'losses.loss', '#89939E', 'TRAIN samples', local=True, alpha=.6)
    series(ax, val, run, 'attribute', 'losses.loss', '#176B9B', 'Fixed VAL', local=True, marker='o')
    ax.set_title('A  Attribute total loss', loc='left'); ax.set_ylabel('Loss'); ax.legend(frameon=False)
    ax = aa[1]
    for key, color, label in (('support_dice', '#176B9B', 'Support Dice'), ('support_precision', '#8056A0', 'Support precision'), ('support_recall', '#CF792D', 'Support recall')):
        series(ax, val, run, 'attribute', 'attributes.summary.' + key + '.mean', color, label, local=True, scale=100., marker='o')
    ax.set_ylim(0, 102); ax.set_ylabel('P support metric (%)'); ax.set_title('B  Fixed-VAL support prediction', loc='left'); ax.legend(frameon=False, loc='lower right')
    ax = aa[2]
    for key, color, label in (('radius_mse', '#238369', 'P radius MSE'), ('constant_radius_mse', '#89939E', 'TRAIN constant reference')):
        series(ax, val, run, 'attribute', 'attributes.summary.' + key + '.mean', color, label, local=True, marker='o')
    ax.set_yscale('log'); ax.set_ylabel('Standardized log-radius MSE (log scale)'); ax.set_title('C  Fixed-VAL radius regression', loc='left'); ax.legend(frameon=False)
    for ax in aa:
        ax.set_xlim(0, 30); ax.set_xlabel('P successful updates (thousands)'); ax.grid(axis='y', color='#E0E5EA', lw=.65)
    fig.text(.06, .105, 'P is selected at29k under the stated readiness/selection rule and subsequently participates in joint base training.', fontsize=9.5)
    fig.text(.06, .065, 'This figure records the initial predictor phase only. It is neither a native segmentation endpoint nor the later frozen P state.', fontsize=9.5)
    save(fig, output, '02_attribute_pretraining')


def tail_comparison(val, output):
    phases = ('condition', 'gdf_head', 'geometry', 'refine')
    runs = sorted({r['run_label'] for r in val if r.get('phase') in phases})
    fig, aa = plt.subplots(1, 4, figsize=(19, 5.5))
    fig.subplots_adjust(left=.055, right=.99, top=.70, bottom=.22, wspace=.29)
    fig.suptitle('Local-scope controls: available fixed-patch validation observations', x=.055, y=.96, ha='left', fontsize=17, fontweight='bold')
    fig.text(.055, .855, 'Each phase uses its own successful-update axis. Unfinished trajectories stop at the snapshot; no future or missing values are filled.', fontsize=10)
    handles=[]
    for run in runs:
        color=RUN_COLORS.get(run,RUN_COLORS.get(next((x for x in ('C0_GDF','AR_GDF','C0','AR') if x in run),''),'#89939E'))
        handles.append(Line2D([0],[0],color=color,lw=1.3,marker='o',markersize=3,label=run))
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.052,.81),ncol=min(4,len(handles)),frameon=False)
    for ax, phase in zip(aa, phases):
        ys = []
        for run in runs:
            color = RUN_COLORS.get(run, None)
            if color is None:
                color = RUN_COLORS.get(next((x for x in ('C0_GDF', 'AR_GDF', 'C0', 'AR') if x in run), ''), '#89939E')
            ys += series(ax, val, run, phase, 'summary.dice.mean', color, run, scale=100., local=True, marker='o')
        ax.set_title(phase, loc='left'); ax.set_xlim(0, BUDGET[phase]/1000.)
        ax.set_xlabel('Phase updates (thousands)'); ax.set_ylabel('Fixed VAL64 Dice (%)')
        if ys:
            ax.set_ylim(max(0, min(ys)-1.), min(100, max(ys)+1.))
        ax.grid(axis='y', color='#E0E5EA', lw=.65)
    fig.text(.055, .1, 'All four arms retain P and attribute feature fusion. C0 uses static gates; no-GDF arms omit the GDF gradient to S after head warmup.', fontsize=9.5)
    fig.text(.055, .055, 'Y-axis ranges follow each panel\'s observed values. These are stage observations, not final matched native endpoint conclusions.', fontsize=9.5)
    save(fig, output, '03_local_controls_patch_validation')


def native_comparison(native, output):
    summary = rows(native / 'endpoint_summary.csv')
    case = rows(native / 'endpoint_per_case.csv')
    keys = ('dice', 'precision', 'recall', 'cldice', 'fine_centerline_recall')
    available = sorted({r['endpoint'] for r in summary if r.get('status') == 'available'})
    if not available:
        return
    for endpoint in available:
        endpoint_cases=[r for r in case if r['endpoint']==endpoint]
        if len(endpoint_cases)!=8 or len({r['case_id'] for r in endpoint_cases})!=8:
            raise ValueError('Native endpoints must contain eight unique case rows')
        for metric in keys:
            values=[num(r,metric+'_pct') for r in endpoint_cases]
            values=[x for x in values if math.isfinite(x)]
            expected=next(num(r,'mean_pct') for r in summary if r['endpoint']==endpoint and r['metric']==metric)
            if values and not math.isclose(float(np.mean(values)),expected,rel_tol=1e-10,abs_tol=1e-10):
                raise ValueError('Native summary and case rows disagree')
    fig, aa = plt.subplots(1, 2, figsize=(17.5, 6.3))
    fig.subplots_adjust(left=.065, right=.985, top=.76, bottom=.22, wspace=.25)
    fig.suptitle('Native VAL8: available objective endpoints', x=.065, y=.96, ha='left', fontsize=17, fontweight='bold')
    fig.text(.065, .86, 'Separate from training-time VAL64 patches | Case-equal whole-case scores | Missing controls are not imputed', fontsize=10.5)
    width = .8 / len(available); x=np.arange(len(keys))
    for index, endpoint in enumerate(available):
        lookup={r['metric']:num(r, 'mean_pct') for r in summary if r['endpoint']==endpoint and r.get('status')=='available'}
        y=[lookup.get(key, math.nan) for key in keys]
        aa[0].bar(x-.4+width/2+index*width,y,width,label=endpoint,color=RUN_COLORS.get(endpoint,'#89939E'))
    aa[0].set_xticks(x, ('Dice','Precision','Recall','clDice','Fine CL recall'));aa[0].set_ylim(0,105);aa[0].set_ylabel('Native case-equal mean (%)');aa[0].legend(frameon=False)
    ids=sorted({r['case_id'] for r in case});x=np.arange(len(ids));width=.8/len(available)
    for index,endpoint in enumerate(available):
        lookup={r['case_id']:num(r,'dice_pct') for r in case if r['endpoint']==endpoint}
        aa[1].bar(x-.4+width/2+index*width,[lookup.get(k,math.nan) for k in ids],width,label=endpoint,color=RUN_COLORS.get(endpoint,'#89939E'))
    aa[1].set_xticks(x,ids,rotation=30,ha='right');aa[1].set_ylim(0,105);aa[1].set_ylabel('Native Dice (%)');aa[1].legend(frameon=False)
    for ax in aa:ax.grid(axis='y',color='#E0E5EA',lw=.65);ax.set_axisbelow(True)
    fig.text(.065,.09,'F precedes condition and Gaussian training. F-to-final includes42k extra workflow updates, including2k head-only; it is not an isolated GDF contrast.',fontsize=9.5)
    fig.text(.065,.05,'Fine centerline recall uses GT native Lee skeleton sites with physical EDT radius<=2mm; it is not physical branch-length recall.',fontsize=9.5)
    save(fig,output,'04_native_available_endpoints')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--main-run',default='AR_GDF')
    parser.add_argument('--native',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);style()
    train=rows(args.input/'training_scalars.csv');val=rows(args.input/'validation_summary.csv')
    if not val:raise ValueError('No validation_summary.csv observations')
    if not any(r.get('run_label')==args.main_run for r in val):raise ValueError('Requested main run not present')
    present={r.get('phase') for r in val if r.get('run_label')==args.main_run}
    if not set(PHASES).issubset(present):
        raise ValueError('This plot protocol requires the complete ratio-r2 main trajectory; a new suffix needs a separate plot scope')
    main_trajectory(train,val,args.main_run,args.output)
    attribute_trajectory(train,val,args.main_run,args.output)
    tail_comparison(val,args.output)
    if args.native:native_comparison(args.native,args.output)
    inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (args.input/'training_scalars.csv',args.input/'validation_summary.csv') if p.is_file()}
    native_inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            ((args.native/'endpoint_summary.csv',args.native/'endpoint_per_case.csv') if args.native else ()) if p.is_file()}
    figures={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir()) if p.suffix in ('.png','.svg')}
    (args.output/'plot_manifest.json').write_text(json.dumps(dict(schema='GTA-AVT-training-figures-1',
        main_run=args.main_run,smoothing=False,phase_resets_joined=False,
        training_render='exported raw scalar samples',validation_render='all exported original validation observations',
        input_csv_sha256=inputs,native_csv_sha256=native_inputs,figure_sha256=figures,
        matplotlib_version=matplotlib.__version__,train_rows=len(train),validation_rows=len(val)),indent=2)+'\n')
    print(json.dumps(dict(status='complete',figures=sorted(p.name for p in args.output.glob('*.png')))))


if __name__=='__main__':main()
