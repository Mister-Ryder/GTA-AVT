"""Five small synthetic/default checks, with no actual run or endpoint reads."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    path = Path(__file__).with_name('calibre_routing_learning_summary.py')
    old = path.with_name('calibre_basis_learning_summary.py')
    assert hashlib.sha256(old.read_bytes()).hexdigest() == '7ced67ece05f3f447aaa932f02a04736a53081b8115da6dbb7622616041d99a3'
    trees = [ast.parse(p.read_bytes(), feature_version=(3, 8)) for p in (path, old)]
    compile(trees[0], str(path), 'exec')
    names = ('stats', 'gradient_stats', 'paired_blocks')
    functions = lambda tree: {n.name: ast.dump(n, include_attributes=False) for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name in names}
    assert functions(trees[0]) == functions(trees[1]) and len(functions(trees[0])) == 3
    spec = importlib.util.spec_from_file_location('routing_learning_smoke', path)
    tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
    g = tool.gradient_stats([None, 0., 2., float('nan'), 'bad', -1.])
    assert (g['finite_count'], g['missing_count'], g['nonfinite_or_nonnumeric_count'],
        g['exact_zero_count'], g['positive_maxabs_count'], g['invalid_negative_maxabs_count']) == (3, 1, 2, 1, 1, 1)
    rows = []
    for step in range(1, 2001):
        rows.append(dict(step=step, alpha=min(1., step/250.), loss=1. - step/10000., amp_retries=0,
            gate_norms=dict(bias=step/1000., availability=step/2000., radius=step/3000.),
            gradient_maxabs={name: .1 for name in tool.PARAMETERS},
            coefficient_row_gradient_maxabs={name: [0., None, .2] for name in
                ('calibre_bias','calibre_availability_weight','calibre_radius_weight')},
            fields=dict(legacy_axis1_only=dict(D_RMS=2.,gain_absmean=.1,gain_maxabs=.2),
                zA=dict(mean=.1,SD=.2,maxabs=1.),zR=dict(mean=.0,SD=.3,maxabs=2.),
                route_coeff_min=.2,route_coeff_max=1.8,route_coeff_sum_residual_maxabs=1e-6,
                routing_axes=[dict(axis=i+1,gain_RMS=.1*(i+1),gain_absmean=.05*(i+1),
                    gain_maxabs=.2*(i+1),direction_RMS=i+1.) for i in range(3)])))
    record = tool.summarize_arm(20261030, 'AR', rows)
    assert len(record['blocks']) == 8 and len(record['all2000_parameter_gradients']) == 15
    assert record['all2000_radius_axis_gradients']['1']['exact_zero_count'] == 2000
    assert record['all2000_radius_axis_gradients']['2']['missing_count'] == 2000
    assert record['all2000_radius_axis_gradients']['3']['positive_maxabs_count'] == 2000
    assert record['blocks'][0]['fields_pre_update']['routing_axes.2.direction_RMS']['mean'] == 3.
    assert record['blocks'][0]['gate_norms_post_update']['radius']['first'] == 1/3000.
    assert record['blocks'][1]['gate_norms_post_update']['radius']['previous_block_last'] == 250/3000.
    assert tool.nested({'x':[{'y':4}]},'x.0.y') == 4 and tool.nested({'x':[]},'x.0.y') is None
    paired_rows = {arm: copy.deepcopy(rows) for arm in tool.ARMS}
    for row in paired_rows['AR']: row['loss'] -= .05
    pairs = tool.paired_blocks(20261030, paired_rows)
    assert len(pairs) == 16 and all(row['negative_AR_lower_loss_steps'] == 250 for row in pairs)
    with tempfile.TemporaryDirectory(prefix='routing_learning_synthetic_') as tmp:
        root = Path(tmp); fake = root/'final.json'; fake.write_bytes(b'{}')
        try:
            tool.analyze([root/'a',root/'b',root/'c'],fake,'0'*64)
        except ValueError as exc:
            assert 'SHA differs' in str(exc)
        else:
            raise AssertionError('Wrong final-analysis digest accepted')
        result = dict(per_seed=[dict(seed=seed, arms={arm: copy.deepcopy(record) for arm in tool.ARMS},
            paired_loss_blocks=tool.paired_blocks(seed, paired_rows)) for seed in tool.SEEDS],
            no_selection_or_rescue=True, scientific_verdict_modified=False)
        original = tool.analyze
        tool.analyze = lambda *args: result
        try:
            output = root/'report'
            receipt = tool.execute(argparse.Namespace(inputs=['a','b','c'],final_analysis='fake',
                final_analysis_sha256='fake',output=str(output)))
        finally:
            tool.analyze = original
        assert receipt['block_rows'] == 72 and receipt['gradient_rows'] == 1296 and receipt['paired_rows'] == 48
        assert {p.name for p in output.iterdir()} == {'analysis.json','readout.txt','blocks.csv','gradients.csv','paired_loss.csv'}
        assert b'\r\n' not in (output/'analysis.json').read_bytes()
    process = subprocess.run([sys.executable,'-B','-I','-S',str(path)], capture_output=True,text=True,check=True)
    assert json.loads(process.stdout) == dict(status='inert',inputs_read=False,outputs_written=False)
    for filename,digest in [('calibre_routing_pilot.py',tool.WORKER_SHA),
            ('calibre_routing_stage_model.py',tool.MODEL_SHA),('calibre_routing_readout.py',tool.READER_SHA)]:
        assert hashlib.sha256(path.with_name(filename).read_bytes()).hexdigest() == digest
    print(json.dumps(dict(passed=True,groups=5,default_inert=True,original_stats_and_paired_math_exact=True,
        radius_row_zero_None_finite_distinct=True,all8blocks_and_3axes=True,final_SHA_rejection=True,
        five_file_synthetic_output_72_1296_48=True,actual_runs_or_GPU_read=False)))


if __name__ == '__main__':
    main()
