"""Small stdlib smoke checks; no models, data, formal endpoints or GPU opened."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    path = Path(__file__).with_name('calibre_basis_pilot.py')
    original = path.with_name('calibre_coadapt_pilot.py')
    require_sha = '231b99fefb5b17e9e71d7b24bf9a4ba74f8cb032fd05b6ab62b8fa11ecee63f8'
    assert hashlib.sha256(original.read_bytes()).hexdigest() == require_sha
    new_tree = ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))
    old_tree = ast.parse(original.read_text(encoding='utf-8'), feature_version=(3, 8))
    compile(new_tree, str(path), 'exec')
    spec = importlib.util.spec_from_file_location('basis_smoke', path)
    tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
    def functions(tree, names):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in names}
    names = ('make_plan', 'radius_permutation', 'summarize_mechanism', 'prepare', 'batch', 'forward_loss')
    assert functions(old_tree, names) == functions(new_tree, names)
    def update_loop(tree):
        return [ast.dump(node, include_attributes=False) for node in ast.walk(tree)
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'step']
    assert len(update_loop(old_tree)) == 1 and update_loop(old_tree) == update_loop(new_tree)
    assert tool.RECIPE['updates_per_arm'] == 2000 and tool.SEEDS == (20261030, 20261031, 20261032)
    assert tool.RECIPE['shared_parameters'] == 114882 and tool.RECIPE['trainable_parameters'] == 115074
    assert tool.RECIPE['gate_parameters'] == 192
    probe = ['case%d' % i for i in range(12)]
    def summary(value):
        return dict(per_case={case: dict(uniform_nonempty=dict(cldice=value, centerline_recall=value,
            fine_recall=value, patches=3, fine_nodes=7, fine_covered=5)) for case in probe})
    arms = dict(C0=summary(.7), A_only=summary(.8), AR=summary(.9))
    guard = tool.CT_uniform_safety(arms, probe)
    assert guard['passed'] is True and guard['stricter_only'] is True and guard['cannot_rescue_original_primary'] is True
    for arm in arms:
        arms[arm]['per_case'][probe[0]]['uniform_nonempty']['fine_recall'] = None
    guard = tool.CT_uniform_safety(arms, probe)
    record = guard['comparisons']['A_only']['metrics']['fine_recall']
    assert record['supported_cases'] == 11 and record['passed'] is True and record['supports_identical'] is True
    assert record['per_case_support'][probe[1]]['AR']['fine_nodes'] == 7
    bad = copy.deepcopy(arms); bad['AR']['per_case'][probe[1]]['uniform_nonempty']['fine_recall'] = None
    assert tool.CT_uniform_safety(bad, probe)['passed'] is False
    harm = copy.deepcopy(arms)
    for case in probe[1:]: harm['AR']['per_case'][case]['uniform_nonempty']['fine_recall'] = .79
    negative = tool.CT_uniform_safety(harm, probe)
    assert negative['passed'] is False and negative['comparisons']['A_only']['metrics']['fine_recall']['mean_AR_minus_control_pp'] < 0.
    missing = dict(C0=summary(None), A_only=summary(None), AR=summary(None))
    assert tool.CT_uniform_safety(missing, probe)['passed'] is False
    process = subprocess.run([sys.executable, '-B', '-I', '-S', str(path)], capture_output=True, text=True, check=True)
    assert json.loads(process.stdout)['status'] == 'inert'
    print(json.dumps(dict(status='PASS', groups=7, old_source_unchanged=True,
        same_training_loop_and_original_mechanism=True, default_inert=True,
        uniform_identical_defined_support_and_null_preservation=True,
        worker_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), actual_data_read=False)))


if __name__ == '__main__':
    main()
