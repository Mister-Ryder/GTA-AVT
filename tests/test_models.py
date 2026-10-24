"""CPU architecture/gradient checks; these are not experimental efficacy tests."""
import unittest

import torch

from gta_avt.models import AttributePredictor, BoundedCalibreRouting, GTAAVT


STATS = {"log_radius_mean": 1., "log_radius_std": .5, "emission_prior": .2}


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_predictor_shape_and_zero_head(self):
        p = AttributePredictor(STATS).eval()
        with torch.no_grad():
            output = p(torch.randn(1, 1, 32, 32, 32))
        self.assertEqual(tuple(output["a_logits"].shape), (1, 1, 16, 16, 16))
        self.assertEqual([tuple(v.shape[1:]) for v in output["features"]],
                         [(64, 8, 8, 8), (128, 4, 4, 4), (256, 2, 2, 2)])
        self.assertTrue(torch.equal(output["z_radius"], torch.zeros_like(output["z_radius"])))
        self.assertTrue(torch.allclose(output["a_logits"].sigmoid(), torch.full_like(output["a_logits"], .2)))
        self.assertFalse(any("generation_head" in name for name, _ in p.named_parameters()))

    def test_bounded_coefficients(self):
        gains = torch.tanh(torch.randn(2, 3, 64, 2, 2, 2) * 20)
        for alpha in (0., .3, 1.):
            c = BoundedCalibreRouting.coefficients(gains, alpha)
            self.assertGreaterEqual(float(c.min()), -1e-6)
            self.assertLessEqual(float(c.max()), 2. + 1e-6)
            self.assertTrue(torch.allclose(c.sum(1), torch.full_like(c[:, 0], 4.), atol=1e-6))

    def test_local_radius_consumption_is_not_an_availability_veto(self):
        torch.manual_seed(9)
        route = BoundedCalibreRouting().eval()
        with torch.no_grad():
            route.radius_weight.fill_(.7)
        feature, attribute = torch.randn(1, 64, 8, 8, 8), torch.randn(1, 128, 4, 4, 4)
        a = torch.ones(1, 1, 8, 8, 8)
        z = torch.randn_like(a)
        own, _ = route(feature, attribute, a, z, "AR", 1.)
        shifted, _ = route(feature, attribute, a, -z, "AR", 1.)
        a_only, _ = route(feature, attribute, a, z, "A_only", 1.)
        neutral, _ = route(feature, attribute, torch.zeros_like(a), z, "AR", 1.)
        self.assertFalse(torch.allclose(own, shifted))
        self.assertTrue(torch.equal(a_only, neutral))
        self.assertGreater(float(neutral.detach().abs().sum()), 0.)
        own.square().mean().backward()
        self.assertGreater(float(route.radius_weight.grad.abs().sum()), 0.)

    def test_complete_model_freeze_parity_and_gradients(self):
        torch.manual_seed(42)
        m = GTAAVT(AttributePredictor(STATS), arm="AR", use_gdf=True)
        x = torch.randn(1, 1, 32, 32, 32)
        m.eval()
        with torch.no_grad():
            ar = m(x)
            m.arm = "C0"
            c0 = m(x)
        self.assertTrue(torch.equal(ar["logits"], c0["logits"]))
        self.assertEqual(tuple(ar["logits"].shape), tuple(x.shape))
        self.assertEqual(tuple(ar["gdf"].shape), tuple(x.shape))
        self.assertTrue(bool((ar["gdf"] >= 0).all()))
        m.arm = "AR"
        m.train()
        self.assertFalse(m.predictor.training)
        self.assertTrue(m.guidance_adapter.training)
        self.assertTrue(all(not module.training for module in m.predictor.modules()))
        out = m(x, alpha=.5)
        loss = out["logits"].square().mean() + out["gdf"].mean()
        loss = loss + sum(value.square().mean() for value in out["aux_logits"])
        loss.backward()
        self.assertTrue(all(p.grad is None for p in m.predictor.parameters()))
        self.assertTrue(all(p.grad is not None for p in m.guidance_adapter.parameters()))
        self.assertTrue(all(p.grad is not None for p in m.gdf_head.parameters()))
        self.assertIsNotNone(m.routing.bias.grad)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            AttributePredictor(dict(STATS, log_radius_std=0.))
        p = AttributePredictor(STATS)
        with self.assertRaises(ValueError):
            p(torch.zeros(1, 1, 33, 32, 32))


if __name__ == "__main__":
    unittest.main()
