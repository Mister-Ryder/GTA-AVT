"""Small local checks only; no Torch, data, checkpoints or endpoints opened."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    path = Path(__file__).with_name('calibre_routing_pilot.py')
    old = path.with_name('calibre_basis_pilot.py')
    assert hashlib.sha256(old.read_bytes()).hexdigest() == '68840bce80d3576c353fe6c43c49c5ea7bfc7db44e772e9bd687283af1198f7c'
    trees = [ast.parse(p.read_bytes(), feature_version=(3, 8)) for p in (path, old)]
    compile(trees[0], str(path), 'exec')
    spec = importlib.util.spec_from_file_location('routing_worker_smoke', path)
    tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
    names = ('make_plan', 'radius_permutation', 'summarize_mechanism', 'CT_uniform_safety', 'prepare', 'batch', 'forward_loss')
    functions = lambda tree: {n.name: ast.dump(n, include_attributes=False) for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name in names}
    assert functions(trees[0]) == functions(trees[1]) and len(functions(trees[0])) == 7
    update = lambda tree: [ast.dump(n, include_attributes=False) for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'optimizer_update_with_amp_retries']
    assert update(trees[0]) == update(trees[1]) and len(update(trees[0])) == 1
    assert tool.RECIPE['updates_per_arm'] == 2000 and tool.RECIPE['trainable_parameters'] == 115458
    assert tool.RECIPE['shared_parameters'] == 114882 and tool.RECIPE['gate_parameters'] == 576
    assert tool.SEEDS == (20261030, 20261031, 20261032)

    probe = [source + '_%d' % index for source in ('Dongyang', 'KiTS', 'Rider') for index in range(4)]
    mechanism = dict(per_case={case: dict(supported=True, correct_minus_permuted_Dice_pp=.2) for case in probe})
    good = tool.SOURCE_safety(mechanism, probe)
    assert good['passed'] and all(v['complete_unchanged_support'] for v in good['source_groups'].values())
    zero = copy.deepcopy(mechanism)
    for case in probe[-4:]: zero['per_case'][case]['correct_minus_permuted_Dice_pp'] = 0.
    assert tool.SOURCE_safety(zero, probe)['passed']
    bad = copy.deepcopy(mechanism)
    for case in probe[-4:]: bad['per_case'][case]['correct_minus_permuted_Dice_pp'] = -.001
    assert sum(v['correct_minus_permuted_Dice_pp'] for v in bad['per_case'].values()) > 0
    assert tool.SOURCE_safety(bad, probe)['passed'] is False
    missing = copy.deepcopy(mechanism)
    missing['per_case'][probe[-1]].update(supported=False, correct_minus_permuted_Dice_pp=None)
    guard = tool.SOURCE_safety(missing, probe)
    assert guard['passed'] is False and guard['source_groups']['Rider']['case_equal_correct_minus_half_ring_Dice_pp'] is None
    assert len(guard['source_groups']['Rider']['expected_case_ids']) == 4
    assert tool.SOURCE_safety(mechanism, probe[:-1])['passed'] is False

    source = path.read_text(encoding='utf8')
    assert "candidate.CalibreRoutingStageModel(arm=arm, alpha=0.)" in source
    assert "first_coefficient_row_gradient_maxabs=first_row_grads" in source
    assert "coefficient_rows_changed=row_changes" in source
    assert "for row in range(3)" in source and 'axis_gain_RMS' in source
    assert 'legacy_axis1_only=' in source and 'route_coeff_sum_residual_maxabs' in source
    assert "capacity_continue=original_primary and uniform_guard['passed'] and source_guard['passed']" in source
    result = subprocess.run([sys.executable, '-B', '-I', '-S', str(path)], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['status'] == 'inert'
    assert 'torch' not in sys.modules
    print(json.dumps(dict(status='PASS', groups=5, original_source_and_7_math_functions_unchanged=True,
        same_sampler_forward_loss_and_AMP_update=True, SOURCE_zero_negative_null_and_missing_cases_checked=True,
        original_budget_and_seeds=True, three_axis_QA_and_trace_wired=True, default_inert=True,
        actual_data_or_GPU_read=False)))


if __name__ == '__main__':
    main()
