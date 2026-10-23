"""Synthetic contracts for the matched local-learning pair; no study reads.

Default CPU checks cover the proper loss, analytic gradients, identical head
learning before routing, recipe bounds and exact frozen sliding-formula parity.
--full-patch128 adds actual network integration/gradient checks, normally CUDA.
"""
import argparse
import ast
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
for directory in (CODE, HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def loss_checks():
    import torch
    from local_learning_train import local_soft_ce, effective_recipe, alpha_at
    # Patch0 has two supervised windows; patch1 has one; patch2 is unknown.
    # Their means, rather than all windows pooled, must each count1/3.
    logits = torch.tensor([[[[[0., 0.]]], [[[2., -1.]]], [[[0., 0.]]],
                            [[[0., 0.]]], [[[0., 0.]]]],
                           [[[[0., 0.]]], [[[1., 9.]]], [[[0., 0.]]],
                            [[[0., 0.]]], [[[0., 0.]]]],
                           [[[[9., 9.]]], [[[0., 0.]]], [[[0., 0.]]],
                            [[[0., 0.]]], [[[0., 0.]]]]], dtype=torch.float32, requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 1] = 1.
    target[1, 1] = 1.
    valid = torch.tensor([True, True, True, False, False, False]).reshape(3, 1, 1, 1, 2)
    loss, details = local_soft_ce(logits, target, valid)
    ce = lambda value: math.log(math.exp(value) + 4.) - value
    expected = ((ce(2.) + ce(-1.)) / 2. + ce(1.) + 0.) / 3.
    assert abs(float(loss) - expected) < 2e-7
    assert details["local_valid_patches"] == 2 and details["local_batch_patches"] == 3
    loss.backward()
    expected_gradient = logits.detach().softmax(1) - target
    expected_gradient[0] /= 6.
    expected_gradient[1, :, :, :, 0] /= 3.
    expected_gradient[1, :, :, :, 1] = 0.
    expected_gradient[2] = 0.
    assert torch.allclose(logits.grad, expected_gradient, atol=3e-8, rtol=1e-6)
    assert not bool(logits.grad[2].count_nonzero())
    half_logits = logits.detach().half().requires_grad_()
    half_loss, _ = local_soft_ce(half_logits, target.half(), valid)
    assert half_loss.dtype == torch.float32 and bool(torch.isfinite(half_loss))
    half_loss.backward()
    assert bool(torch.isfinite(half_logits.grad).all())
    empty_loss, _ = local_soft_ce(logits, torch.zeros_like(target), torch.zeros_like(valid))
    assert float(empty_loss) == 0.
    for invalid in ({"local_weight": .2}, {"gdf_weight": .1}, {"scale_weight": 0.},
                    {"alpha_start": 0}, {"lr": .0003}, {"max_steps": 1000},
                    {"detach_conditions": False}, {"conditioning": "scale_only"}):
        try:
            effective_recipe(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Protocol variation was accepted: " + str(invalid))
    assert effective_recipe({})["max_steps"] == 2000
    assert [alpha_at(x, 500, 500) for x in (0, 499, 500, 999, 1000)] == [0., 0., 0., 1., 1.]
    return dict(status="passed", analytic_loss=float(loss), expected_loss=expected,
                per_patch_then_batch_mean=True, zero_support_patch_counts_in_batch=True,
                fp16_logits_fp32_objective=True, bounded_recipe=True)


def head_learning_checks():
    import torch
    from local_learning_model import LocalLearningModel
    from local_learning_train import local_soft_ce
    torch.set_num_threads(1)
    torch.manual_seed(20261023)
    fixed = LocalLearningModel("constant")
    fixed_rng = torch.get_rng_state().clone()
    torch.manual_seed(20261023)
    routed = LocalLearningModel("local_scale")
    assert torch.equal(fixed_rng, torch.get_rng_state())
    assert fixed.state_dict().keys() == routed.state_dict().keys()
    assert all(torch.equal(value, routed.state_dict()[key]) for key, value in fixed.state_dict().items())
    names_a = [key for key, value in fixed.named_parameters() if value.requires_grad]
    names_b = [key for key, value in routed.named_parameters() if value.requires_grad]
    assert names_a == names_b and "local_head.weight" in names_a
    # A fixed synthetic nonzero endpoint, shared by both arms, demonstrates
    # feature gradients on the FIRST local-loss update (zero W would hide it).
    with torch.no_grad():
        fixed.local_head.weight[1, 0, 0, 0, 0] = .1
        fixed.local_head.weight[0, 1, 0, 0, 0] = -.2
    routed.load_state_dict(fixed.state_dict(), strict=True)
    features = torch.linspace(-.5, .5, 64 * 32 ** 3).reshape(1, 64, 32, 32, 32)
    valid = torch.zeros((1, 1, 32, 32, 32), dtype=torch.bool)
    valid[:, :, 7:25, 7:25, 7:25] = True
    target = torch.zeros((1, 5, 32, 32, 32))
    target[:, 1] = 1.
    input_valid = torch.ones((1, 1, 128, 128, 128), dtype=torch.bool)
    optimizers = [torch.optim.AdamW(model.local_head.parameters(), lr=.0001)
                  for model in (fixed, routed)]
    feature_gradient = None
    for step in range(3):
        outputs, gradients = [], []
        rng = torch.get_rng_state().clone()
        for model, optimizer in zip((fixed, routed), optimizers):
            torch.set_rng_state(rng)
            model.train()
            assert model.local_head.training and model.classifier.training
            model._stage_request = dict(mode="configured", override=None, alpha=0.)
            model._local_request = dict(input_valid=input_valid)
            feature = features.clone().requires_grad_()
            classifier = dict(generation=torch.zeros((1, 5)), structure=torch.zeros((1, 3)),
                              encoder_features=[feature])
            optimizer.zero_grad(set_to_none=True)
            model.conditions(classifier, "predicted")
            assert model._local_logits.requires_grad and not model._local_score.requires_grad
            loss, _ = local_soft_ce(model._local_logits, target, valid)
            loss.backward()
            assert bool(model.local_head.weight.grad.count_nonzero())
            assert bool(feature.grad.count_nonzero())
            feature_gradient = float(feature.grad.abs().max())
            outputs.append(model._local_logits.detach().clone())
            gradients.append(feature.grad.clone())
            optimizer.step()
            model._stage_request = model._local_request = None
            model._local_logits = model._local_score = None
        assert torch.equal(outputs[0], outputs[1]) and torch.equal(gradients[0], gradients[1])
        assert all(torch.equal(a, b) for a, b in zip(fixed.local_head.parameters(), routed.local_head.parameters()))
        assert torch.equal(rng, torch.get_rng_state()), "Head/support changed the RNG"
    return dict(status="passed", common_init_and_rng_exact=True,
                optimizer_parameter_order_exact=True, local_head_trainable_both_arms=True,
                synthetic_head_updates_per_arm=3, head_outputs_gradients_updates_exact=True,
                detached_consumption=True, nonzero_feature_gradient_max_abs=feature_gradient,
                scope="actual conditions/readout/CE on synthetic grid; no full-network backward")


def _source_function(path, name, namespace):
    source = ast.parse(Path(path).read_text(encoding="utf-8"))
    node = next(x for x in source.body if isinstance(x, ast.FunctionDef) and x.name == name)
    code = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")
    exec(code, namespace)
    return namespace[name]


def inference_formula_checks():
    """Run actual function ASTs without requiring native-image dependencies."""
    import numpy as np
    import torch
    namespace = dict(np=np, torch=torch, itertools=itertools)
    namespace["normalize"] = _source_function(CODE / "data.py", "normalize", namespace)
    namespace["tile_starts"] = _source_function(CODE / "inference.py", "tile_starts", namespace)
    old = _source_function(CODE / "inference.py", "sliding_probability", dict(namespace))
    new = _source_function(HERE / "local_learning_inference.py", "sliding_probability", dict(namespace))

    class Fixture(torch.nn.Module):
        def __init__(self, require_valid):
            super().__init__()
            self.require_valid = require_valid
            self.requests = []

        def forward(self, image, coordinate, condition_mode, **kwargs):
            self.requests.append((image.detach().clone(), coordinate.detach().clone(), kwargs))
            if self.require_valid:
                assert kwargs["input_valid"].dtype == torch.bool
                assert kwargs["input_valid"].shape == image.shape
            return {"logits": image * .3 + coordinate[:, 0, None, None, None, None] * .01}

    record = dict(crop_start_zyx=[3, 4, 5], full_shape_zyx=[99, 101, 103])
    rows = []
    for shape in ((11, 13, 9), (25, 16, 27)):
        image = np.linspace(-1024., 1200., int(np.prod(shape)), dtype=np.float32).reshape(shape)
        a, b = Fixture(False), Fixture(True)
        pa, na = old(a, image, record, torch.device("cpu"), patch=16, batch_size=2)
        pb, nb = new(b, image, record, torch.device("cpu"), patch=16, batch_size=2)
        assert na == nb and np.array_equal(pa, pb), "Tiling/blending changed"
        assert len(a.requests) == len(b.requests)
        for one, two in zip(a.requests, b.requests):
            assert torch.equal(one[0], two[0]) and torch.equal(one[1], two[1])
        if any(x < 16 for x in shape):
            assert int(b.requests[0][2]["input_valid"].sum()) == int(np.prod(shape))
        rows.append(dict(shape=list(shape), tiles=na, probability_exact=True))
    return dict(status="passed", formula_scope="actual sliding function ASTs; native recovery reused unchanged",
                cases=rows, normalized_images_and_coordinates_exact=True, ct_padding_validity=True,
                gt_entered_model=False)


def full_model_checks(device):
    import torch
    from local_learning_model import LocalLearningModel
    from local_learning_train import local_soft_ce
    torch.manual_seed(20261023)
    fixed = LocalLearningModel("constant").to(device)
    routed = LocalLearningModel("local_scale").to(device)
    routed.load_state_dict(fixed.state_dict(), strict=True)
    image = torch.linspace(-1., 1., 2 * 128 ** 3, device=device).reshape(2, 1, 128, 128, 128)
    valid = torch.ones_like(image, dtype=torch.bool)
    target = torch.zeros((2, 5, 32, 32, 32), device=device)
    target[:, 1] = 1.
    loss_valid = torch.zeros((2, 1, 32, 32, 32), dtype=torch.bool, device=device)
    loss_valid[:, :, 2:30, 2:30, 2:30] = True
    amp_modes = (False, True) if str(device).startswith("cuda") else (False,)
    for amp in amp_modes:
        # Include train-mode BN/Dropout and compare the shared alpha-zero path.
        fixed.train()
        routed.train()
        state = torch.get_rng_state().clone()
        cuda_state = torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else None
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
            a = fixed(image, input_valid=valid, alpha=0.)
            after = torch.get_rng_state().clone()
            after_cuda = torch.cuda.get_rng_state_all() if cuda_state is not None else None
            torch.set_rng_state(state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            b = routed(image, input_valid=valid, alpha=0.)
        assert torch.equal(after, torch.get_rng_state())
        if after_cuda is not None:
            assert all(torch.equal(x, y) for x, y in zip(after_cuda, torch.cuda.get_rng_state_all()))
        for key in a:
            if isinstance(a[key], list):
                assert all(torch.equal(x, y) for x, y in zip(a[key], b[key])), key
            else:
                assert torch.equal(a[key], b[key]), key
        assert all(torch.equal(value, routed.state_dict()[key]) for key, value in fixed.state_dict().items())
    # Normal consumption is detached even when alpha=1 and local head learns.
    routed.eval()
    with torch.no_grad():
        routed.local_head.weight.zero_()
        routed.local_head.bias.zero_()
        routed.local_head.weight[1, 0, 0, 0, 0] = .1
        routed.local_head.bias[1] = .8
        with torch.cuda.amp.autocast(enabled=str(device).startswith("cuda")):
            constant_same_weight = routed(image, input_valid=valid, alpha=0.)["logits"]
    routed.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=str(device).startswith("cuda")):
        result = routed(image, input_valid=valid, alpha=1.)
        segmentation = result["logits"].float().square().mean()
    logit_response = float((result["logits"].detach().float() - constant_same_weight.float()).abs().mean())
    assert logit_response > 0. and bool(torch.isfinite(result["logits"]).all())
    gradient_scale = 1024. if str(device).startswith("cuda") else 1.
    (segmentation * gradient_scale).backward()
    assert all(p.grad is None for p in routed.local_head.parameters())
    assert any(p.grad is not None and bool(p.grad.count_nonzero())
               for p in routed.classifier.encoder.parameters())
    # Separate localCE-only backward verifies shared encoder learning. No second
    # full segmentation graph is needed for this classification-path contract.
    routed.zero_grad(set_to_none=True)
    routed._stage_request = dict(mode="configured", override=None, alpha=1.)
    routed._local_request = dict(input_valid=valid)
    with torch.cuda.amp.autocast(enabled=str(device).startswith("cuda")):
        classifier = routed.classifier(image)
        routed.conditions(classifier, "predicted")
        local_loss, _ = local_soft_ce(routed._local_logits, target, loss_valid)
    (local_loss * gradient_scale).backward()
    assert bool(routed.local_head.weight.grad.count_nonzero())
    assert any(p.grad is not None and bool(p.grad.count_nonzero())
               for p in routed.classifier.encoder.parameters())
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in routed.parameters())
    routed._stage_request = routed._local_request = None
    routed._local_logits = routed._local_score = None
    return dict(status="passed", patch=128, batch_size=2, autocast_modes=list(amp_modes),
                train_mode_alpha0_outputs_buffers_rng_exact=True,
                segmentation_to_local_head_gradient=False, localCE_to_shared_encoder_gradient=True,
                amp_route_check_manual_loss_scale=gradient_scale,
                nonuniform_condition_same_weight_mean_abs_logit_response=logit_response,
                cuda_multistep_bitwise_claim=False, real_training_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--full-patch128", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Refusing to overwrite a previous QA receipt")
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = time.time()
    names = ("local_learning_model.py", "local_learning_train.py", "local_learning_inference.py",
             "local_learning_checks.py", "stage_model.py", "stage_train.py", "frozen_readout.py")
    report = dict(schema="GTA-AVT-matched-local-learning-synthetic-QA-v1", status="passed",
                  source_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names},
                  runtime=dict(python=sys.version, executable=sys.executable, torch=torch.__version__, argv=sys.argv),
                  study_data_read=False, real_training_started=False,
                  loss=loss_checks(), joint_head=head_learning_checks(), inference=inference_formula_checks(),
                  full_network128=(full_model_checks(args.device) if args.full_patch128 else dict(status="not_run")),
                  cuda_multistep_bitwise_claim=False, seconds=time.time() - started)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
