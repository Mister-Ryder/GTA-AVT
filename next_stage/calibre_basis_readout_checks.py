"""Small synthetic-only basis readout checks; no actual endpoint inputs."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    here = Path(__file__).parent; target = here / 'calibre_basis_readout.py'
    tree = ast.parse(target.read_text(encoding='utf-8'), feature_version=(3, 8))
    m = load(target, 'calibre_basis_readout_synthetic')
    fixture = load(here / 'boundary_field_readout_checks.py', 'basis_legacy_math_fixture')
    ids, old, ignored = fixture.fixture()
    rows = {'C0': old['C0'], 'A_only': old['A_only'], 'AR': old['AB']}
    for arm in m.ARMS:
        for row in rows[arm]:
            row['oracle'] = dict(available_radius_sites=100, unknown_sites=64 ** 3 - 100)
    interventions = []
    for row in rows['AR']:
        altered = copy.deepcopy(row['metrics']); altered['dice'] -= .002
        interventions.append(dict(case_id=row['case_id'], source=row['source'], slot=row['slot'],
            stratum=row['stratum'], primary=row['primary'], normal=copy.deepcopy(row['metrics']), permuted=altered,
            radius_moments=dict(count=100, sum=200., sum_squares=500.), zR_changed=True,
            mean_abs_logit_response=.01, binary_changed_voxels=10,
            permutation=dict(eligible_sites=100, shift=50, changed_radius_sites=20,
                own_A_sha256='a' * 64, correct_radius_multiset_sha256='b' * 64,
                permuted_radius_multiset_sha256='b' * 64, own_A_and_support_preserved=True)))
    result = m.analyze_seed({'seed': m.SEEDS[0]}, rows, interventions, ids)
    assert result['original_primary_screen_passed'] is True
    assert result['CT_uniform_transfer_safety']['passed'] is True and result['capacity_continue'] is True
    assert abs(result['mechanisms']['own_A_half_ring_radius']['mean_correct_minus_permuted_Dice_pp'] - .2) < 1e-12

    # This new guard may stop an otherwise passing primary; primary math stays intact.
    harmful = copy.deepcopy(rows)
    for row in harmful['AR']:
        if row['slot'] >= 8:
            row['metrics']['cldice'] = .6
    stopped = m.analyze_seed({'seed': m.SEEDS[0]}, harmful, interventions, ids)
    assert stopped['original_primary_screen_passed'] is True
    assert stopped['CT_uniform_transfer_safety']['passed'] is False and stopped['capacity_continue'] is False
    assert stopped['CT_uniform_transfer_safety']['comparisons']['C0']['metrics']['cldice']['mean_AR_minus_control_pp'] < 0

    # Fine pooling is covered/nodes in each case, never mean of patch ratios.
    unequal = copy.deepcopy(rows)
    for arm in m.ARMS:
        for row in unequal[arm]:
            if row['slot'] >= 8:
                n, k = (1, 1) if row['slot'] == 8 else (99, 0)
                row['metrics'].update(fine_nodes=n, fine_covered=k, fine_recall=k / n)
    fine = m.analyze_seed({'seed': m.SEEDS[0]}, unequal, interventions, ids)
    record = fine['CT_uniform_transfer_safety']['comparisons']['C0']['metrics']['fine_recall']
    assert record['AR_mean'] == 1 / 694 and record['mean_AR_minus_control_pp'] == 0.
    assert record['supported_cases'] == 12 and record['passed'] is True

    # Identically absent fine support stays null, not0 or an invented12-case mean.
    absent = copy.deepcopy(rows)
    for arm in m.ARMS:
        for row in absent[arm]:
            if row['slot'] >= 8:
                row['metrics'].update(fine_nodes=0, fine_covered=0, fine_recall=None)
    missing = m.analyze_seed({'seed': m.SEEDS[0]}, absent, interventions, ids)
    record = missing['CT_uniform_transfer_safety']['comparisons']['C0']['metrics']['fine_recall']
    assert record['AR_mean'] is None and record['mean_AR_minus_control_pp'] is None
    assert record['passed'] is None and missing['capacity_continue'] is None
    mismatch = m.case_arms(rows, ids)
    mismatch['AR']['per_case'][ids[0]]['uniform_known_nonempty']['cldice'] = None
    unequal_support = m.CT_uniform_safety(mismatch, ids)['comparisons']['C0']['metrics']['cldice']
    assert not unequal_support['supports_identical'] and unequal_support['passed'] is None
    assert unequal_support['per_case_AR_minus_control'] == {}

    # The tighter guard cannot rescue a primary mechanism below .1pp.
    weak = copy.deepcopy(interventions)
    for row in weak:
        row['permuted']['dice'] = row['normal']['dice'] - .0001
    failed = m.analyze_seed({'seed': m.SEEDS[0]}, rows, weak, ids)
    assert failed['original_primary_screen_passed'] is False
    assert failed['CT_uniform_transfer_safety']['passed'] is True and failed['capacity_continue'] is False
    unsupported = copy.deepcopy(interventions)
    for row in unsupported:
        if row['case_id'] == ids[0]:
            row['zR_changed'] = False
    assert m.analyze_seed({'seed': m.SEEDS[0]}, rows, unsupported, ids)['original_primary_screen_passed'] is None

    bank = dict(probe_cases=ids, fit_cases=['fit' + str(i) for i in range(24)])
    hashes = {'normalization.json': 'same'}
    with patch.object(m, 'sha', return_value=m.BANK_SHA), patch.object(m, 'read', return_value=bank), patch.object(m, 'load_seed', side_effect=[
        ({'seed': m.SEEDS[0]}, harmful, interventions, hashes), ({'seed': m.SEEDS[1]}, rows, interventions, hashes),
        ({'seed': m.SEEDS[2]}, rows, interventions, hashes)]):
        final = m.analyze(['one', 'two', 'three'], 'bank')
    assert final['all3_original_primary_screens_passed'] is True
    assert final['primary_screen_passed'] is False and final['capacity_continue'] is False
    assert final['all3_screens_passed'] is False and final['primary_failure_cannot_be_rescued']
    assert final['seeds_pooled'] is False and final['next_experiment_authorized'] is False

    # Old metric, source/row, primary-mechanism and population math is AST exact.
    original = ast.parse((here / 'calibre_coadapt_readout.py').read_text(encoding='utf-8'))
    names = ('require', 'sha', 'read', 'lines', 'encoded', 'finite', 'mean', 'variance',
        'validate_metrics', 'aggregate', 'case_arms', 'compare', 'mechanism', 'load_seed')
    for name in names:
        a, b = [next(n for n in tr.body if isinstance(n, ast.FunctionDef) and n.name == name) for tr in (original, tree)]
        assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False), name
    a = copy.deepcopy(next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'analyze_seed'))
    b = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'original_primary_analysis'))
    b.name = a.name
    assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
    try:
        m.metadata(Path('synthetic-never-accessed'), {'schema': 'GTA-AVT-calibre-coadapt-pilot-v1'}, bank)
    except ValueError:
        pass
    else:
        raise AssertionError('Old candidate metadata was accepted')
    assert m.WORKER_SHA == m.sha(here / 'calibre_basis_pilot.py')
    assert m.MODEL_SHA == m.sha(here / 'calibre_basis_stage_model.py')
    meta = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'metadata')
    assert 'morphology_convs.%s.%s' in ast.get_source_segment(target.read_text(encoding='utf-8'), meta)
    text = m.render(final)
    assert '不是CT部署' in text and 'CT_uniform' in text and '原主要检查' in text
    json.dumps(final, allow_nan=False)
    run = subprocess.run([sys.executable, '-B', '-I', '-S', str(target)], capture_output=True, text=True, check=True)
    assert json.loads(run.stdout)['status'] == 'inert'
    print(json.dumps(dict(status='PASS', groups=8, actual_results_read=False,
        old_metric_and_row_functions_AST_exact=len(names), original_primary_math_AST_exact=True,
        worker_recipe_and_model_pins=True, undefined_support_stays_null=True)))


if __name__ == '__main__':
    main()
