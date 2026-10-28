"""Export objective training trajectories without publishing deployment metadata.

Accepts read-only snapshots whose ``phases`` map contains raw ``history`` rows,
all ``validations`` and/or the existing 200-update ``windows``. No model is
loaded. Raw points, aggregated windows and native endpoints remain distinct.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

LABEL = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
FIELD = re.compile(r'^[A-Za-z_][A-Za-z0-9_.-]*$')
HASH = re.compile(r'^(?:[0-9a-f]{40}|[0-9a-f]{64})$')
PHASES = ('attribute', 'base_warm', 'base', 'condition', 'gdf_head', 'geometry', 'refine', 'shared_head')
CONFIG_KEYS = {
    'seed', 'validation_seed', 'device', 'patch_size', 'batch_size', 'samples_per_epoch', 'val_samples',
    'num_workers', 'amp', 'checkpoint_every', 'validate_every', 'deterministic', 'dice_weight', 'bce_weight',
    'use_gdf', 'min_support_dice', 'min_support_recall', 'stage', 'max_steps', 'lr', 'weight_decay',
    'max_grad_norm', 'deep_supervision_weight', 'deep_supervision_weights', 'base_attribute_weight',
    'augment', 'alpha_ramp_steps', 'local_shared_lr', 'local_gate_lr', 'local_gdf_lr', 'gdf_head_lr',
    'gdf_weight_start', 'gdf_weight_end', 'gdf_ramp_steps', 'sampling', 'bce_class_balance', 'phase', 'arm',
    'lambda_gdf', 'availability_weight', 'radius_weight', 'gdf_delay_steps', 'require_attribute_readiness'}
ENUMS = {'stage': {'pretrain', 'base', 'gdf_head', 'local_adapt', 'full_joint'},
         'phase': set(PHASES), 'arm': {'C0', 'V_only', 'AR', 'AR_GDF', 'C0_GDF'},
         'device': {'cpu', 'cuda'}, 'bce_class_balance': {'none', 'airway_ratio'}}


def label(value, what):
    if not isinstance(value, str) or not LABEL.fullmatch(value):
        raise ValueError('Unsafe ' + what)
    return value


def finite(value, what):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('Expected finite number: ' + what)
    return value


def integer(value, what, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Expected nonnegative integer: ' + what)
    return value


def numeric_tree(value, prefix='', output=None):
    """Only numeric leaves; arbitrary strings/paths and narrative are discarded."""
    result = {} if output is None else output
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not FIELD.fullmatch(key):
                continue
            numeric_tree(item, prefix + '.' + key if prefix else key, result)
    elif isinstance(value, bool):
        result[prefix] = int(value)
    elif isinstance(value, (float, int)):
        result[prefix] = finite(value, prefix)
    elif value is None:
        result[prefix] = None
    elif isinstance(value, (tuple, list)) and value and all(isinstance(x, (float, int)) and not isinstance(x, bool) for x in value):
        values = [finite(x, prefix) for x in value]
        result[prefix + '.mean'] = sum(values) / len(values)
        for index, item in enumerate(values): result[prefix + '.item_' + str(index)] = item
    return result


def hashes(value, prefix='', output=None):
    result = {} if output is None else output
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and FIELD.fullmatch(key):
                hashes(item, prefix + '.' + key if prefix else key, result)
    elif isinstance(value, str) and HASH.fullmatch(value):
        result[prefix] = value
    return result


def configuration(config):
    result = {}
    for key in CONFIG_KEYS & set(config):
        value = config[key]
        if key in ENUMS:
            if value not in ENUMS[key]: raise ValueError('Unknown scientific enum: ' + key)
            result[key] = value
        elif isinstance(value, dict):
            result[key] = numeric_tree(value)
        elif isinstance(value, list):
            result[key] = [finite(x, key) for x in value]
        elif value is None or isinstance(value, bool): result[key] = value
        else: result[key] = finite(value, key)
    return result


def train_values(row):
    if isinstance(row.get('values'), dict):
        return numeric_tree(row['values'])
    result = {}
    for key in ('lr', 'alpha', 'gdf_weight', 'epoch', 'elapsed_seconds'):
        if key in row: result[key] = finite(row[key], key)
    for key in ('losses', 'learning_rates', 'routing', 'sampling'):
        numeric_tree(row.get(key, {}), key, result)
    return result


def write_csv(path, rows, required=()):
    names = list(dict.fromkeys(list(required) + [key for row in rows for key in row]))
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader(); writer.writerows(rows)


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def export_snapshots(snapshots, output_dir, stride=100):
    """Input items: {run_label: safe label, snapshot: parsed snapshot object}."""
    integer(stride, 'stride', 1)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()): raise FileExistsError('Use a fresh export directory')
    output.mkdir(parents=True, exist_ok=True)
    tables = {key: [] for key in ('training_scalars', 'training_windows', 'validation_summary',
                                 'validation_case_metrics', 'phase_endpoints', 'amp_skips', 'checkpoint_markers')}
    metadata, configurations, seen = [], {}, set()
    for item in snapshots:
        run = label(item['run_label'], 'run label')
        if run in seen: raise ValueError('Duplicate run label')
        seen.add(run); snapshot = item['snapshot']; phases = snapshot.get('phases', {})
        if not isinstance(phases, dict): raise ValueError('Snapshot requires phases mapping')
        run_info = dict(run_label=run, snapshot_sha256=hashlib.sha256(json.dumps(snapshot, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                        identities=hashes({key:snapshot.get(key) for key in ('plan', 'pipeline', 'process', 'source_identity')}), phases={})
        pipeline=snapshot.get('pipeline') or {}
        for key in ('status','current','error_type'):
            value=pipeline.get(key)
            if value is None or (isinstance(value,str) and LABEL.fullmatch(value)):
                run_info['pipeline_'+key+'_at_snapshot']=value
        configurations[run] = {}
        for phase, data in phases.items():
            if phase not in PHASES: raise ValueError('Unknown phase: ' + phase)
            prefix = dict(run_label=run, phase=phase)
            config, summary = data.get('config') or {}, data.get('summary') or {}
            configurations[run][phase] = configuration(config)
            raw = data.get('history', data.get('history_rows', [])) or []
            if not isinstance(raw, list): raise ValueError('History must be a list')
            previous = 0
            for row in raw:
                step = integer(row.get('successful_update', row.get('step')), 'successful update', 1)
                if step <= previous: raise ValueError('Raw successful updates must be strictly increasing')
                previous = step
            validations = {}
            for v in data.get('validations', []) or []:
                step = integer(v.get('step', v.get('successful_update')), 'validation update', 1)
                if step in validations: raise ValueError('Duplicate validation update')
                validations[step] = v['validation']
            for row in raw:
                step = row.get('successful_update', row.get('step'))
                if row.get('validation') and step not in validations: validations[step] = row['validation']
            maximum = max([r.get('successful_update',r.get('step')) for r in raw] or [0])
            skip_steps = {integer(e['successful_update'],'skip boundary',1) for e in data.get('amp_skip_events',[]) or []}
            retained = 0
            for row in raw:
                step = row.get('successful_update', row.get('step'))
                values = train_values(row)  # validate even points not retained
                if step not in (1, maximum) and step % stride and step not in validations and step not in skip_steps and not row.get('amp_skip_boundary'):
                    continue
                record = dict(prefix, successful_update=step, attempt=integer(row['attempt'], 'attempt', step) if row.get('attempt') is not None else None,
                              sample_kind='recorded_successful_update', **values)
                tables['training_scalars'].append(record); retained += 1
            if not raw:
                for kind in ('first', 'last'):
                    row = data.get(kind)
                    if row:
                        record = dict(prefix, successful_update=integer(row['step'], 'boundary step', 1),
                                      attempt=integer(row['attempt'],'boundary attempt',1) if row.get('attempt') is not None else None,
                                      sample_kind='recorded_boundary_scalar', **train_values(row))
                        if not any(r.get('run_label')==run and r.get('phase')==phase and r.get('successful_update')==record['successful_update'] for r in tables['training_scalars']):
                            tables['training_scalars'].append(record); retained += 1
            for window in data.get('windows', []) or []:
                first, last = integer(window['first'], 'window first', 1), integer(window['last'],'window last',1)
                if last < first: raise ValueError('Window order')
                for metric, stats in window.get('stats', {}).items():
                    if not FIELD.fullmatch(metric): raise ValueError('Unsafe window metric')
                    tables['training_windows'].append(dict(prefix, first_update=first, last_update=last,
                        count=integer(window['n'],'window count',1), metric=metric,
                        mean=finite(stats['mean'],metric), p10=finite(stats['p10'],metric), p90=finite(stats['p90'],metric),
                        defined_count=integer(stats['n'],'window defined count',1)))
            for step, value in sorted(validations.items()):
                scope = 'fixed sampled VAL patches; not native whole-case evaluation'
                vals = numeric_tree({key:value.get(key) for key in ('losses','loss_defined_batches','summary','selection_score','selection_key','routing_mean')})
                attrs = value.get('attributes') or {}
                numeric_tree({key:attrs.get(key) for key in ('summary','readiness')}, 'attributes', vals)
                tables['validation_summary'].append(dict(prefix, successful_update=step, scope=scope, **vals))
                for family, cases in (('segmentation',value.get('per_case') or []),('attributes',attrs.get('per_case') or [])):
                    ids = set()
                    for case in cases:
                        case_id = label(case['case_id'], 'case ID')
                        if case_id in ids: raise ValueError('Duplicate validation case ID')
                        ids.add(case_id)
                        numeric = numeric_tree({k:v for k,v in case.items() if k!='case_id'})
                        tables['validation_case_metrics'].append(dict(prefix, successful_update=step,
                            metric_family=family, case_id=case_id, **numeric))
            actual = summary.get('successful_updates', data.get('rows', data.get('raw_history_rows', maximum)))
            actual = integer(actual or 0, 'phase successful updates')
            attempts = summary.get('attempts')
            if attempts is None and data.get('last'): attempts = data['last'].get('attempt')
            attempts = integer(attempts,'phase attempts',actual) if attempts is not None else None
            endpoint = dict(prefix, successful_updates=actual, attempts=attempts,
                skipped_attempts_total=attempts-actual if attempts is not None else None,
                planned_successful_updates=config.get('max_steps'), elapsed_seconds=summary.get('elapsed_seconds'),
                last_recorded_validation_step=max(validations) if validations else None,
                status='complete' if summary and actual==config.get('max_steps') else 'partial',
                summary_present=int(bool(summary)))
            for key in ('elapsed_seconds','best_validation_score'):
                if summary.get(key) is not None: endpoint[key]=finite(summary[key],key)
            tables['phase_endpoints'].append(endpoint)
            if attempts is not None:
                tables['amp_skips'].append(dict(prefix, event_scope='phase_total', successful_update=actual,
                    skipped_attempts=attempts-actual, previous_attempt=None,current_attempt=attempts,
                    inference='attempt count minus successful updates; not a log of loss values on skipped batches'))
            for event in data.get('amp_skip_events', []) or []:
                before=integer(event['previous_attempt'],'previous attempt')
                after=integer(event.get('current_attempt',event.get('next_attempt')),'current attempt',before+1)
                indices=event['skipped_attempts'] if isinstance(event['skipped_attempts'],list) else None
                if indices is not None:
                    if indices!=list(range(before+1,after)): raise ValueError('AMP skipped attempt indices inconsistent')
                    skip=len(indices)
                else: skip=integer(event['skipped_attempts'],'skipped attempts',1)
                if skip!=after-before-1: raise ValueError('AMP event count inconsistent with consecutive success boundary')
                tables['amp_skips'].append(dict(prefix, event_scope='consecutive_success_boundary',
                    successful_update=integer(event['successful_update'],'skip boundary',1),
                    skipped_attempts=skip,previous_attempt=before,current_attempt=after,
                    skipped_attempt_indices=json.dumps(indices) if indices is not None else None,
                    inference='inferred from consecutive successful-row attempt counters; not explicit failed-batch log'))
            source_hash=data.get('complete_line_sha256')
            if source_hash is not None and (not isinstance(source_hash,str) or not re.fullmatch('[0-9a-f]{64}',source_hash)):
                raise ValueError('Expected source history SHA256')
            for gap in data.get('gaps',[]) or []:
                if not isinstance(gap,list) or len(gap)!=2 or any(type(x) is not int or x<0 for x in gap):
                    raise ValueError('History gaps require two nonnegative update indices')
            run_info['phases'][phase]=dict(source_history_sha256=source_hash,
                represented_successful_rows=data.get('raw_history_rows', data.get('rows')),
                retained_scalar_rows=retained,raw_points_available=bool(raw),window_size=snapshot.get('window'),
                window_count=len(data.get('windows') or []),validation_count=len(validations),
                malformed_lines=data.get('malformed_lines',0),history_gaps=data.get('gaps',[]),
                nonfinite_losses=data.get('nonfinite_losses',[]),identities=hashes(summary.get('lineage',{})))
            if data.get('nonfinite_losses'): raise ValueError('Nonfinite loss records require separate failure review, not silent scalar publication')
        markers=list(snapshot.get('checkpoint_markers',[]) or [])
        # Derive saved selections only from their actual downstream checkpoint
        # lineage, never from maxima/minima of the plotted validation curve.
        warm=((phases.get('base_warm') or {}).get('summary') or {}).get('lineage',{}).get('predictor_init',{})
        if 'attribute' in phases and warm.get('checkpoint_sha256') and warm.get('successful_updates'):
            markers.append(dict(phase='attribute',kind='selected_for_base_warm',successful_update=warm['successful_updates'],checkpoint_sha256=warm['checkpoint_sha256']))
        inherited=((phases.get('condition') or {}).get('summary') or {}).get('lineage',{}).get('segmentation_init',{})
        if 'base' in phases and inherited.get('parent_phase')=='base' and inherited.get('common_checkpoint_sha256') and inherited.get('common_successful_updates'):
            markers.append(dict(phase='base',kind='selected_for_condition',successful_update=inherited['common_successful_updates'],checkpoint_sha256=inherited['common_checkpoint_sha256']))
        marker_keys=set()
        for marker in markers:
            phase=marker['phase']
            if phase not in phases: raise ValueError('Checkpoint marker phase unavailable')
            kind=label(marker['kind'],'checkpoint kind')
            step=integer(marker['successful_update'],'checkpoint step',1)
            marker_key=(phase,kind,step)
            if marker_key in marker_keys:continue
            marker_keys.add(marker_key)
            observed=phases[phase].get('raw_history_rows',phases[phase].get('rows'))
            if observed is not None and step>integer(observed,'observed successful rows'):
                raise ValueError('Checkpoint marker exceeds recorded phase budget')
            record=dict(run_label=run,phase=phase,successful_update=step,kind=kind)
            digest=marker.get('checkpoint_sha256')
            if digest is not None:
                if not HASH.fullmatch(digest) or len(digest)!=64: raise ValueError('Checkpoint marker SHA256 required')
                record['checkpoint_sha256']=digest
            tables['checkpoint_markers'].append(record)
        metadata.append(run_info)
    headers=dict(training_scalars=('successful_update','attempt','sample_kind'),
        training_windows=('first_update','last_update','count','metric','mean','p10','p90','defined_count'),
        validation_summary=('successful_update','scope'),validation_case_metrics=('successful_update','metric_family','case_id'),
        phase_endpoints=('successful_updates','attempts','status'),amp_skips=('successful_update','event_scope','skipped_attempts'),
        checkpoint_markers=('successful_update','kind','checkpoint_sha256'))
    for name, rows in tables.items():
        write_csv(output/(name+'.csv'),rows,('run_label','phase')+headers[name])
    json_write(output/'scientific_configuration.json',configurations)
    document=dict(schema='GTA-AVT-objective-training-trajectory-1',raw_stride=stride,
        scopes=dict(training='recorded TRAIN successful-update scalars; no interpolation',
            windows='nonoverlapping recorded 200-update window mean/p10/p90 where available; not raw points',
            validation='fixed sampled VAL64; separate from native whole-case metrics',
            time='phase elapsed_seconds only unless a raw row actually records time; no invented per-update timestamps',
            amp='attempt-minus-success counts; failed-batch losses and exact failed input identities are not recorded',
            units='scores/losses retain original units; Dice/precision/recall fractions, not percent',
            privacy='source paths, raw plan strings, credentials, images, labels and model weights excluded'),
        runs=metadata,table_rows={k:len(v) for k,v in tables.items()})
    json_write(output/'trajectory_manifest.json',document)
    return document


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',action='append',required=True,help='RUN_LABEL=private_snapshot.json; repeat per run')
    parser.add_argument('--output',required=True)
    parser.add_argument('--stride',type=int,default=100)
    args=parser.parse_args(); inputs=[]
    for value in args.input:
        run,path=value.split('=',1)
        inputs.append(dict(run_label=run,snapshot=json.loads(Path(path).read_text(encoding='utf-8-sig'))))
    result=export_snapshots(inputs,args.output,args.stride)
    print(json.dumps(dict(runs=[v['run_label'] for v in result['runs']],table_rows=result['table_rows'])))


if __name__=='__main__':main()
