"""Five small synthetic/default groups; never open actual experiment results."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch


def main():
    path = Path(__file__).with_name('calibre_basis_learning_summary.py')
    old = path.with_name('calibre_coadapt_learning_summary.py')
    assert hashlib.sha256(old.read_bytes()).hexdigest() == '84025d2fa5050001a0645cf4d74f71369ae138b0228347c63c12dcafb2deb252'
    old_tree = ast.parse(old.read_text(encoding='utf-8'), feature_version=(3, 8))
    tree = ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8)); compile(tree, str(path), 'exec')
    names = ('finite', 'stats', 'gradient_stats', 'nested', 'summarize_arm', 'paired_blocks')
    def functions(t):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.walk(t)
            if isinstance(n, ast.FunctionDef) and n.name in names}
    assert functions(tree) == functions(old_tree)
    spec = importlib.util.spec_from_file_location('basis_learning_smoke', path)
    tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
    assert len(tool.PARAMETERS) == len(set(tool.PARAMETERS)) == 15
    assert tool.READER_SHA == '1f7fe250df11b96f831850c9eccafa5a4ba3e51d0b3dd224cea9953d95542d44'
    assert tool.WORKER_SHA == '68840bce80d3576c353fe6c43c49c5ea7bfc7db44e772e9bd687283af1198f7c'
    assert tool.MODEL_SHA == '6f5e6fcf84600c737fc8844c0fdce6839f980d19d734512686f374cce1a0c294'
    # Group1: original arithmetic and new15-tensor source identities.
    flags = tool.gradient_stats([None, 0., .2, -.1])
    assert flags['missing_count'] == 1 and flags['exact_zero_count'] == 1 and flags['positive_maxabs_count'] == 1
    assert flags['invalid_negative_maxabs_count'] == 1
    # Group2: None/zero/positive scalar observations retain separate counts.
    def logs(arm):
        rows = []
        for step in range(1, 2001):
            grad = {name: .1 for name in tool.PARAMETERS}
            grad['calibre_availability_weight'] = 0. if arm == 'C0' else .1
            grad['calibre_radius_weight'] = .2 if arm == 'AR' else 0.
            if step == 1: grad['structure_guidance.morphology_convs.0.bias'] = None
            rows.append(dict(step=step, alpha=min(1., step / 250.),
                loss=1. + step / 2000. + dict(C0=.2, A_only=.1, AR=0.)[arm], gradient_maxabs=grad,
                gate_norms=dict(bias=step / 10000., availability=0. if arm == 'C0' else step / 20000.,
                    radius=step / 30000. if arm == 'AR' else 0.),
                fields=dict(D_RMS=.4, gain_absmean=.01, gain_maxabs=.02,
                    zA=dict(mean=.1, SD=.2, maxabs=1.), zR=dict(mean=.2, SD=.3, maxabs=2.))))
        return rows
    rows = {arm: logs(arm) for arm in tool.ARMS}
    arms = {arm: tool.summarize_arm(20261030, arm, rows[arm]) for arm in tool.ARMS}
    assert all(len(v['blocks']) == 8 and v['blocks'][-1]['last_step'] == 2000 for v in arms.values())
    first = arms['AR']['blocks'][0]
    assert first['losses']['BCE']['finite_count'] == 0 and first['losses']['BCE']['missing_count'] == 250
    assert first['gate_norms_post_update']['bias']['previous_block_last'] == 0.
    assert arms['C0']['all2000_parameter_gradients']['calibre_radius_weight']['exact_zero_count'] == 2000
    assert arms['AR']['all2000_parameter_gradients']['structure_guidance.morphology_convs.0.bias']['missing_count'] == 1
    pairs = tool.paired_blocks(20261030, rows)
    assert len(pairs) == 16 and all(v['negative_AR_lower_loss_steps'] == 250 for v in pairs)
    assert math.isclose(pairs[0]['paired_same_step_loss']['mean'], -.2)
    # Group3: all8 windows, same-step loss arithmetic, raw missing components.
    incomplete = copy.deepcopy(rows['AR']); incomplete[-1]['gradient_maxabs'].pop(tool.PARAMETERS[-1])
    try:
        tool.summarize_arm(20261030, 'AR', incomplete)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing parameter observation was accepted')
    with tempfile.TemporaryDirectory(prefix='calibre_basis_summary_') as temporary:
        home = Path(temporary).resolve()
        assert home.parent == Path(tempfile.gettempdir()).resolve() and home.name.startswith('calibre_basis_summary_')
        bad_final = home / 'not_final.json'; bad_final.write_text('{}', encoding='utf-8')
        try:
            tool.analyze([home / 'missingA', home / 'missingB', home / 'missingC'], bad_final)
        except ValueError:
            pass
        else:
            raise AssertionError('Incomplete final readout was accepted')
        # Group4: final-before-run-access and complete gradient inventory guards.
        per_seed = []
        for seed in tool.SEEDS:
            record = copy.deepcopy(arms)
            for arm in tool.ARMS:
                for block in record[arm]['blocks']: block['seed'] = seed
            per_seed.append(dict(seed=seed, arms=record, paired_loss_blocks=tool.paired_blocks(seed, rows)))
        fake = dict(schema='GTA-AVT-calibre-basis-coadapt-learning-explanation-v1',
            source_sha256=tool.sha(path), final_readout_source_sha256=tool.READER_SHA,
            worker_sha256=tool.WORKER_SHA, model_sha256=tool.MODEL_SHA,
            parameter_tensors=15, trainable_parameters=115074, per_seed=per_seed,
            no_new_success_or_convergence_criterion=True, scientific_verdict_modified=False,
            new_training_authorized=False, GTA_goal_completed=False)
        output = home / 'five_scalar_files'
        with patch.object(tool, 'analyze', return_value=fake):
            receipt = tool.execute(argparse.Namespace(inputs=['synthetic'] * 3, final_analysis='synthetic', output=output))
        assert set(p.name for p in output.iterdir()) == {'analysis.json', 'readout.txt', 'blocks.csv', 'parameter_gradients.csv', 'paired_loss.csv'}
        assert receipt['block_rows'] == 72 and receipt['gradient_rows'] == 1080 and receipt['paired_rows'] == 48
        assert not any('branch_weight_norm' in p.read_text(encoding='utf-8') for p in output.iterdir())
        assert '参数张量' in (output / 'readout.txt').read_text(encoding='utf-8')
    process = subprocess.run([sys.executable, '-B', '-I', '-S', str(path)], capture_output=True, text=True, check=True)
    assert json.loads(process.stdout) == dict(status='inert', inputs_read=False, outputs_written=False)
    # Group5: exactly five scalar outputs and default no-input/no-write behavior.
    print(json.dumps(dict(status='PASS', groups=5, actual_results_read=False,
        source_sha256=tool.sha(path), block_rows=72, gradient_rows=1080, paired_rows=48)))


if __name__ == '__main__':
    main()
