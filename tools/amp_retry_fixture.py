"""Exercise the actual train.py AMP helper with injected finite-loss overflow."""
import argparse
import copy
import json
from pathlib import Path
import sys
import torch


def rng_state():
    return torch.get_rng_state().clone(), [state.clone() for state in torch.cuda.get_rng_state_all()]


def restore_rng(state):
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state_all(state[1])


def same_rng(a, b):
    return torch.equal(a[0], b[0]) and all(torch.equal(x, y) for x, y in zip(a[1], b[1]))


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        return a.shape == b.shape and torch.allclose(a, b, atol=1e-7, rtol=1e-6)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_tree(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    return a == b


def count_steps(optimizer):
    counter = {'calls': 0}
    original = optimizer.step

    def counted(*args, **kwargs):
        counter['calls'] += 1
        return original(*args, **kwargs)

    optimizer.step = counted
    return counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    sys.path.insert(0, str(project / 'code'))
    from train import optimizer_update_with_amp_retries, set_seed
    from common import atomic_json, source_hashes
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the actual enabled GradScaler overflow fixture')
    torch.set_num_threads(2)
    set_seed(131)
    initial = torch.nn.Sequential(torch.nn.Conv3d(1, 4, 1, bias=False), torch.nn.BatchNorm3d(4),
                                  torch.nn.Dropout3d(.25), torch.nn.Conv3d(4, 1, 1)).cuda().train()
    with torch.no_grad():
        initial[0].weight.mul_(.05)
        initial[3].weight.mul_(.05)
        initial[3].bias.zero_()
    image = torch.randn(2, 1, 8, 8, 8, device='cuda') * .1
    target = torch.randn_like(image) * .1
    before_rng = rng_state()

    def run(model, scale, injected_attempts):
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        calls = count_steps(optimizer)
        scaler = torch.cuda.amp.GradScaler(init_scale=scale)
        forwards, events = [], []
        injection = {'calls': 0}

        def inject(gradient):
            injection['calls'] += 1
            if injection['calls'] <= injected_attempts:
                return torch.full_like(gradient, float('inf'))
            return gradient

        hook = model[0].weight.register_hook(inject)

        def forward_loss():
            with torch.cuda.amp.autocast():
                output = model(image)
                loss = torch.nn.functional.mse_loss(output.float(), target.float())
            forwards.append(output.detach().clone())
            return loss, {'total': float(loss.detach())}

        try:
            details, update = optimizer_update_with_amp_retries(model, optimizer, scaler, forward_loss,
                max_attempts=8, gradient_clip=12., on_overflow=events.append)
        finally:
            hook.remove()
        return {'model': model, 'optimizer': optimizer, 'scaler': scaler, 'calls': calls,
                'forwards': forwards, 'events': events, 'update': update, 'rng': rng_state()}

    clean = run(copy.deepcopy(initial), 32768., 0)
    assert clean['calls']['calls'] == 1 and clean['update']['amp_attempts'] == 1
    restore_rng(before_rng)
    retried = run(copy.deepcopy(initial), 65536., 1)
    assert retried['calls']['calls'] == 1 and retried['update']['amp_attempts'] == 2
    assert len(retried['events']) == 1 and retried['scaler'].get_scale() == 32768.
    assert torch.equal(retried['forwards'][0], retried['forwards'][1]), 'Dropout/BN replay changed the exact same forward'
    assert torch.equal(clean['forwards'][0], retried['forwards'][1])
    assert same_tree(clean['model'].state_dict(), retried['model'].state_dict()), 'Successful model update differs from clean lower-scale reference'
    assert same_tree(clean['optimizer'].state_dict(), retried['optimizer'].state_dict()), 'Optimizer state differs from the one-update reference'
    assert same_rng(clean['rng'], retried['rng']), 'RNG after one successful update differs'
    for name, value in clean['model'].named_buffers():
        assert torch.equal(value, dict(retried['model'].named_buffers())[name]), 'BN buffer differs:' + name
    max_difference = max(float((value - retried['model'].state_dict()[name]).abs().max())
                         for name, value in clean['model'].state_dict().items())

    # Bounded all-overflow failure must make zero optimizer calls and restore
    # BatchNorm/RNG after every attempt, including the final failed attempt.
    restore_rng(before_rng)
    failing = copy.deepcopy(initial)
    failing_optimizer = torch.optim.AdamW(failing.parameters(), lr=3e-4)
    failed_calls = count_steps(failing_optimizer)
    failing_scaler = torch.cuda.amp.GradScaler(init_scale=65536.)
    failed_events = []
    hook = failing[0].weight.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
    start_state = copy.deepcopy(failing.state_dict())

    def failing_forward():
        with torch.cuda.amp.autocast():
            prediction = failing(image)
            loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
        return loss, {'total': float(loss.detach())}

    try:
        optimizer_update_with_amp_retries(failing, failing_optimizer, failing_scaler, failing_forward,
                                         max_attempts=8, on_overflow=failed_events.append)
    except FloatingPointError as error:
        assert 'after8 attempts (7 retries)' in str(error)
    else:
        raise AssertionError('Persistent AMP overflow did not stop at8attempts')
    finally:
        hook.remove()
    assert len(failed_events) == 8 and failed_calls['calls'] == 0
    assert same_tree(start_state, failing.state_dict()) and not failing_optimizer.state
    assert same_rng(before_rng, rng_state())

    # True FP32 invalid gradients and invalid forward losses never enter AMP
    # retry/skip logic or execute any optimizer step.
    fp_model = torch.nn.Linear(1, 1, bias=False)
    fp_optimizer = torch.optim.SGD(fp_model.parameters(), lr=.1)
    fp_calls = count_steps(fp_optimizer)
    disabled = torch.cuda.amp.GradScaler(enabled=False)
    hook = fp_model.weight.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))

    def fp_forward():
        loss = fp_model(torch.ones(1, 1)).square().sum()
        return loss, {'total': float(loss.detach())}

    try:
        optimizer_update_with_amp_retries(fp_model, fp_optimizer, disabled, fp_forward)
    except FloatingPointError as error:
        assert 'Nonfinite FP32 gradients' in str(error)
    else:
        raise AssertionError('True nonfinite FP32 gradient was hidden')
    hook.remove()
    assert fp_calls['calls'] == 0

    # Inject a norm-reduction overflow after otherwise finite gradients. This
    # must hard-fail before any optimizer step rather than misuse scaler.skip.
    original_clip = torch.nn.utils.clip_grad_norm_
    torch.nn.utils.clip_grad_norm_ = lambda *args, **kwargs: torch.tensor(float('inf'))
    try:
        optimizer_update_with_amp_retries(fp_model, fp_optimizer, disabled, fp_forward)
    except FloatingPointError as error:
        assert 'Nonfinite gradient norm despite finite gradient elements' in str(error)
    else:
        raise AssertionError('Finite-gradient norm overflow executed an update')
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
    assert fp_calls['calls'] == 0

    def invalid_loss():
        loss = fp_model.weight.sum() * torch.tensor(float('nan'))
        return loss, {}

    try:
        optimizer_update_with_amp_retries(fp_model, fp_optimizer, disabled, invalid_loss)
    except FloatingPointError as error:
        assert 'Nonfinite forward objective' in str(error)
    else:
        raise AssertionError('Invalid forward loss was retried')
    assert fp_calls['calls'] == 0
    result = {'state': 'passed', 'torch_version': torch.__version__, 'source_sha256': source_hashes(),
              'injected_finite_loss_overflow': 'passed', 'initial_scale_preserved_at65536': True,
              'reduced_scale': retried['scaler'].get_scale(), 'successful_optimizer_calls': retried['calls']['calls'],
              'retry_update': retried['update'], 'maximum_state_difference_vs_clean32768_reference': max_difference,
              'BN_buffers_exact_reference_match': True, 'CPU_CUDA_rng_exact_reference_match': True,
              'same_batch_dropout_forward_exact_replay': True, 'optimizer_state_matches_one_clean_update': True,
              'eight_attempts_seven_retries_then_fail': 'passed', 'persistent_overflow_optimizer_calls': failed_calls['calls'],
              'FP32_nonfinite_gradient_immediate_failure': 'passed', 'nonfinite_loss_immediate_failure': 'passed'}
    result['finite_gradient_norm_overflow_no_optimizer_step'] = 'passed'
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
