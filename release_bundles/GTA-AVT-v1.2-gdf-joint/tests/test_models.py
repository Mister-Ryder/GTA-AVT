"""Synthetic CPU architecture, conditional-consumption and gradient contracts.

These checks do not train on a dataset or establish experimental performance.
"""
import unittest

import torch
from gta_avt.models import AttributePredictor, BoundedCalibreRouting, GTAAVT


STATS = {"mu": 1., "sigma": .5, "pi_support": .2}


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_model(self):
        torch.manual_seed(42)
        return GTAAVT(AttributePredictor(STATS), arm="AR", use_gdf=True)

    def test_predictor_output_and_prior_initialization(self):
        p = AttributePredictor(STATS).eval()
        with torch.no_grad(): out = p(torch.randn(1, 1, 32, 32, 32))
        for key in ("support_logits", "radius_mean"):
            self.assertEqual(tuple(out[key].shape), (1, 1, 16, 16, 16))
        self.assertEqual([tuple(v.shape[1:]) for v in out["features"]],
                         [(64, 8, 8, 8), (128, 4, 4, 4), (256, 2, 2, 2)])
        self.assertTrue(torch.equal(out["radius_mean"], torch.zeros_like(out["radius_mean"])))
        self.assertNotIn("radius_log_variance", out)
        self.assertEqual(p.local_head.out_channels, 2)
        self.assertTrue(torch.allclose(out["support_score"], torch.full_like(out["support_score"], .2)))
        self.assertNotIn("a_logits", out)
        self.assertNotIn("z_radius", out)
        self.assertAlmostEqual(AttributePredictor({"mu": 1., "sigma": .5, "pi": .9}).statistics()["pi_support"], .01)

    def test_joint_base_passes_segmentation_gradient_into_attribute_encoder(self):
        model = self.make_model().train()
        output = model(torch.randn(1, 1, 32, 32, 32), alpha=0.)
        output["logits"].square().mean().backward()
        self.assertTrue(output["support_logits"].requires_grad)
        self.assertGreater(float(model.predictor.encoder[0].conv.weight.grad.abs().sum()), 0.)
        self.assertEqual(int(model.predictor.encoder[0].bn.num_batches_tracked), 1)
        self.assertIsNone(model.routing.support_weight.grad)

    def test_bounded_coefficients(self):
        gains = torch.tanh(torch.randn(2, 3, 64, 2, 2, 2) * 20)
        for alpha in (0., .3, 1.):
            coefficient = BoundedCalibreRouting.coefficients(gains, alpha)
            self.assertGreaterEqual(float(coefficient.min()), -1e-6)
            self.assertLessEqual(float(coefficient.max()), 2. + 1e-6)
            self.assertTrue(torch.allclose(coefficient.sum(1), torch.full_like(coefficient[:, 0], 4.), atol=1e-6))

    def test_subthreshold_support_is_consumed_without_hard_veto(self):
        torch.manual_seed(9)
        route = BoundedCalibreRouting().eval()
        with torch.no_grad():
            route.bias.fill_(.1); route.support_weight.fill_(.5)
        feature, attribute = torch.randn(1, 64, 8, 8, 8), torch.randn(1, 128, 4, 4, 4)
        v = torch.full((1, 1, 8, 8, 8), .2)
        mu = torch.ones_like(v)
        own, summary = route(feature, attribute, v, mu, "AR", 1.)
        neutral, reference = route(feature, attribute, v, mu, "C0", 1.)
        self.assertFalse(torch.allclose(own, neutral))
        self.assertGreater(float(summary["conditional_delta_rms"]), 0.)
        self.assertEqual(float(reference["conditional_delta_rms"]), 0.)
        self.assertAlmostEqual(float(summary["radius_axis_absmean"]), .2, places=6)
        self.assertGreater(float(neutral.detach().abs().sum()), 0.)
        self.assertLessEqual(float(summary["coefficient_sum_residual"]), 1e-6)

    def test_radius_consumption_clipping_and_zero_mean_reference(self):
        torch.manual_seed(9)
        route = BoundedCalibreRouting().eval()
        with torch.no_grad(): route.radius_weight.fill_(.7)
        feature, attribute = torch.randn(1, 64, 8, 8, 8), torch.randn(1, 128, 4, 4, 4)
        v = torch.full((1, 1, 8, 8, 8), .2)
        mu = torch.randn_like(v)
        own, trace = route(feature, attribute, v, mu, "AR", 1.)
        shifted, _ = route(feature, attribute, v, -mu, "AR", 1.)
        support_only, _ = route(feature, attribute, v, mu, "V_only", 1.)
        prior, trace_prior = route(feature, attribute, v, torch.zeros_like(mu), "AR", 1.)
        self.assertFalse(torch.allclose(own, shifted))
        self.assertTrue(torch.equal(prior, support_only))
        self.assertGreater(float(trace["radius_axis_absmean"]), 0.)
        self.assertEqual(float(trace_prior["radius_axis_absmean"]), 0.)
        clipped, _ = route(feature, attribute, v, torch.full_like(mu, 4.), "AR", 1.)
        extreme, _ = route(feature, attribute, v, torch.full_like(mu, 100.), "AR", 1.)
        self.assertTrue(torch.equal(clipped, extreme))
        own.square().mean().backward()
        self.assertGreater(float(route.radius_weight.grad.abs().sum()), 0.)

    def test_zero_initialized_complete_model_has_matched_forward(self):
        model = self.make_model().eval()
        x = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            ar = model(x)
            model.arm = "V_only"; v = model(x)
            model.arm = "C0"; c0 = model(x)
        self.assertTrue(torch.equal(ar["logits"], v["logits"]))
        self.assertTrue(torch.equal(ar["logits"], c0["logits"]))
        self.assertEqual(tuple(ar["logits"].shape), tuple(x.shape))
        self.assertEqual(tuple(ar["gdf"].shape), tuple(x.shape))
        self.assertTrue(bool((ar["gdf"] >= 0).all()))
        self.assertEqual(float(ar["routing"]["conditional_delta_rms"]), 0.)

    def test_exact_stage_parameter_and_module_modes(self):
        model = self.make_model()
        model.train()
        self.assertTrue(model.predictor.training)
        self.assertTrue(model.guidance_adapter.training)
        self.assertTrue(all(p.requires_grad for p in model.predictor.parameters()))
        self.assertFalse(model.routing.bias.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in model.gdf_head.parameters()))
        self.assertEqual(model.parameter_groups()["gdf"], [])
        model.set_stage("gdf_head").train()
        self.assertTrue(all(not module.training for module in model.modules()))
        self.assertTrue(all(not p.requires_grad for p in model.predictor.parameters()))
        self.assertEqual(set(model.active_parameter_names()), {"gdf_head.0.weight", "gdf_head.0.bias"})
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 55)
        model.set_stage("local_adapt").train()
        groups = model.parameter_groups()
        self.assertEqual({key: sum(p.numel() for p in values) for key, values in groups.items()},
                         {"shared": 114882, "gates": 576, "gdf": 55})
        self.assertEqual(len(model.active_parameter_names()), 17)
        self.assertTrue(all(not module.training for module in model.modules()))
        parameters = [p for group in groups.values() for p in group]
        self.assertEqual(len({id(p) for p in parameters}), len(parameters))
        self.assertEqual({id(p) for p in parameters}, {id(p) for p in model.parameters() if p.requires_grad})
        model.set_stage("base").train()
        self.assertTrue(model.guidance_adapter.training)
        self.assertTrue(model.predictor.training)

    def test_gdf_gradient_reaches_local_routing_and_preserves_frozen_state(self):
        model = self.make_model().set_stage("local_adapt").train()
        active = set(model.active_parameter_names())
        frozen = {name: value.detach().clone() for name, value in model.state_dict().items() if name not in active}
        output = model(torch.randn(1, 1, 32, 32, 32))
        self.assertFalse(output["radius_mean"].requires_grad)
        # GDF alone must update shared active routing, not only its output head.
        output["gdf"].square().mean().backward()
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if name not in active))
        self.assertGreater(float(model.routing.morphology_convs[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.routing.morph_fusion.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.gdf_head[0].weight.grad.abs().sum()), 0.)
        torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.001).step()
        self.assertTrue(all(torch.equal(value, model.state_dict()[name]) for name, value in frozen.items()))

    def test_gdf_warmup_has_no_segmentation_parameter_gradient(self):
        model = self.make_model().set_stage("gdf_head").train()
        output = model(torch.randn(1, 1, 32, 32, 32), alpha=0.)
        output["gdf"].mean().backward()
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if not name.startswith("gdf_head.")))
        self.assertTrue(all(p.grad is not None for p in model.gdf_head.parameters()))

    def test_explicit_override_is_continuous_but_strict(self):
        model = self.make_model()
        size = (16, 16, 16)
        field = torch.zeros(1, 1, *size)
        prediction = {"support_logits": field, "radius_mean": field}
        override = {"support": torch.full_like(field, .2), "radius_mean": field}
        support, _ = model._conditions(prediction, override, size)
        self.assertTrue(torch.equal(support, override["support"]))
        for changed in ({"support": field}, dict(override, support=field + 1.1),
                        dict(override, radius_log_variance=field),
                        dict(override, radius_mean=field + float("nan"))):
            with self.assertRaises(ValueError): model._conditions(prediction, changed, size)
        with self.assertRaises(ValueError): model.set_stage("joint")
        with self.assertRaises(ValueError): AttributePredictor(dict(STATS, sigma=0.))
        with self.assertRaises(ValueError): AttributePredictor(dict(STATS, pi_support=1.))
        with self.assertRaises(ValueError): model(torch.zeros(1, 1, 33, 32, 32))


if __name__ == "__main__":
    unittest.main()
