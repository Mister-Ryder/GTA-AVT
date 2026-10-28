"""CT attributes, bounded local fusion and Gaussian supervision for GTA-AVT v1.2.

Lumen support is distinct from radius measurement validity. Production forwards
accept CT alone; validity is a loss-only mask. Neutral-base training jointly
forms P/S features; local adaptation then freezes their common state and BN.
"""
from __future__ import annotations

import math
from typing import Mapping
import torch
from torch import nn
from torch.nn import functional as F
from .vendor.AnatomicalClassifier import ConvBlock, SEBlock
from .vendor.AnatomicalFusionModules import AnatomicalGuidanceModule, GenerationGuidanceModule
from .vendor.model_arch_model10 import Encoder, Decoder, FusionModule


ARMS = ("C0", "V_only", "AR")
STAGES = ("base", "gdf_head", "local_adapt", "full_joint")
RADIUS_AXIS_CLIP = 4.
MODEL_CONFIG = {
    "name": "GTA-AVT-v1.2", "revision": "20261028-gdf-joint-r3", "input_channels": 1, "input_multiple": 16,
    "attribute_encoder_channels": [32, 64, 128, 256], "attribute_field_stride": 2,
    "seg_encoder_channels": [32, 64, 128, 256], "seg_decoder_channels": [128, 64, 32],
    "routing_site_channels": 64, "routing_coefficients": 576,
    "routing_dilations": [1, 1, 2, 3], "attribute_heads": ["support_logits", "radius_mean"],
    "radius_axis_clip": RADIUS_AXIS_CLIP,
    "support_semantics": "continuous lumen occupancy score; not calibrated confidence",
    "predictor_training": "joint with S in neutral base; frozen throughout post-condition Gaussian revision",
    "gaussian_head": "Conv3d(2,1,kernel_size=3,padding=1)+Sigmoid on final two segmentation logits; no detach",
    "gaussian_target": "gdf-joint-bounded-v1: casewise raw field minmax, then 0.5+0.5*u",
    "full_joint_scope": "all non-P parameters; BN running statistics remain fixed after base",
    "radius_consumption": "support * clipped deterministic standardized radius",
    "statistics_keys": ["mu", "sigma", "pi_support"],
}


def _check_image(x):
    if (not isinstance(x, torch.Tensor) or x.ndim != 5 or x.shape[1] != 1 or
            x.shape[0] < 1 or not x.is_floating_point() or
            any(n < 32 or n % 16 for n in x.shape[2:])):
        raise ValueError("Expected floating Bx1xDxHxW CT; dimensions >=32 and divisible by16")


class AttributePredictor(nn.Module):
    """CT -> soft lumen support and deterministic standardized log radius."""
    def __init__(self, statistics: Mapping):
        super().__init__()
        mean = float(statistics["mu"] if "mu" in statistics else statistics["log_radius_mean"])
        std = float(statistics["sigma"] if "sigma" in statistics else statistics["log_radius_std"])
        prior = float(statistics.get("pi_support", .01))
        if not (math.isfinite(mean) and math.isfinite(std) and std > 0 and math.isfinite(prior) and 0 < prior < 1):
            raise ValueError("Finite TRAIN mean, positive SD and support prior in(0,1) required")
        self.register_buffer("log_radius_mean", torch.tensor(mean, dtype=torch.float64))
        self.register_buffer("log_radius_std", torch.tensor(std, dtype=torch.float64))
        self.register_buffer("support_prior", torch.tensor(prior, dtype=torch.float64))
        self.encoder = nn.ModuleList([
            ConvBlock(1, 32), ConvBlock(32, 32, stride=2), ConvBlock(32, 64), ConvBlock(64, 64, stride=2),
            ConvBlock(64, 128), ConvBlock(128, 128, stride=2), ConvBlock(128, 256), ConvBlock(256, 256, stride=2)])
        self.attention_blocks = nn.ModuleList([SEBlock(c, reduction=8) for c in (64, 128, 256)])
        self.multi_scale_fusion = nn.Sequential(nn.Conv3d(448, 256, 1), nn.BatchNorm3d(256), nn.ReLU(inplace=True))
        self.local_head = nn.Conv3d(320, 2, 1)
        nn.init.zeros_(self.local_head.weight); nn.init.zeros_(self.local_head.bias)
        with torch.no_grad():
            self.local_head.bias[0] = math.log(prior) - math.log1p(-prior)

    def statistics(self):
        return {"mu": float(self.log_radius_mean), "sigma": float(self.log_radius_std), "pi_support": float(self.support_prior)}

    def forward(self, x):
        _check_image(x)
        output_size = tuple(n // 2 for n in x.shape[2:])
        features = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            if i in (3, 5, 7): features.append(x)
        features = [block(value) for block, value in zip(self.attention_blocks, features)]
        size = features[-1].shape[2:]
        fused = self.multi_scale_fusion(torch.cat((features[2],
            F.interpolate(features[1], size=size, mode="trilinear", align_corners=False),
            F.interpolate(features[0], size=size, mode="trilinear", align_corners=False)), 1))
        head = self.local_head(torch.cat((features[0], F.interpolate(fused,
            size=features[0].shape[2:], mode="trilinear", align_corners=False)), 1))
        with torch.autocast(device_type=x.device.type, enabled=False):
            field = F.interpolate(head.float(), size=output_size, mode="trilinear", align_corners=False)
            support_logits, radius_mean = field[:, :1], field[:, 1:2]
        return {"support_logits": support_logits, "radius_mean": radius_mean,
                "support_score": support_logits.sigmoid(), "features": features, "fused": fused}


class ActiveBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([Encoder(1, 16, 32, apply_pooling=False), Encoder(32, 32, 64),
                                      Encoder(64, 64, 128), Encoder(128, 128, 256)])
        self.decoders = nn.ModuleList([Decoder(256, 256, 384, 128, 128, num_groups=2),
            Decoder(128, 128, 192, 64, 64, num_groups=8), Decoder(64, 64, 96, 32, 32, num_groups=8)])
        self.fusion_module_decoder = FusionModule([128, 64, 32], out_channels=2, scale_factors=[1, 2, 4])


class BoundedCalibreRouting(AnatomicalGuidanceModule):
    """v0.1 four branches and three bounded contrasts with continuous conditions.

    The equivalent branch coefficients lie in[0,2] and sum to4 in real
    arithmetic; neither feature-energy conservation nor topology is guaranteed.
    """
    def __init__(self):
        super().__init__(64, 128)
        self.morphology_convs = nn.ModuleList([nn.Conv3d(64, 16, 3, padding=d, dilation=d) for d in (1, 1, 2, 3)])
        self.morph_fusion = nn.Conv3d(64, 64, 1)
        self.bias = nn.Parameter(torch.zeros(3, 64))
        self.support_weight = nn.Parameter(torch.zeros(3, 64))
        self.radius_weight = nn.Parameter(torch.zeros(3, 64))

    @staticmethod
    def coefficients(gains, alpha):
        g1, g2, g3 = gains.unbind(1)
        remaining, a = 1. - g1.abs(), float(alpha)
        return torch.stack((1 + a * (-g1 + remaining * g2), 1 + a * (-g1 - remaining * g2),
                            1 + a * (g1 + remaining * g3), 1 + a * (g1 - remaining * g3)), 1)

    @staticmethod
    def _summary(gains, static_gains, directions, alpha, fused, support, radius_axis):
        # Diagnose actual conditional versus same-weight static routing without
        # retaining four large coefficient maps or a second autograd graph.
        with torch.no_grad():
            g1, g2, g3 = gains.detach().unbind(1)
            s1, s2, s3 = static_gains.detach().unbind(1)
            d1, d2, d3 = (d.detach() for d in directions)
            remaining, static_remaining = 1. - g1.abs(), 1. - s1.abs()
            conditional = float(alpha) * (g1 - s1) * d1
            conditional = conditional + float(alpha) * (remaining * g2 - static_remaining * s2) * d2
            conditional = conditional + float(alpha) * (remaining * g3 - static_remaining * s3) * d3
            conditional_rms = conditional.square().mean().sqrt()
            del conditional
            all_routing = float(alpha) * g1 * d1
            all_routing = all_routing + float(alpha) * remaining * (g2 * d2 + g3 * d3)
            routing_rms = all_routing.square().mean().sqrt()
            del all_routing
            minimum, maximum = gains.new_tensor(float("inf")), gains.new_tensor(float("-inf"))
            total = torch.zeros_like(g1)
            for branch in range(4):
                pair = remaining * (g2 if branch < 2 else g3)
                if branch % 2: pair = -pair
                coefficient = 1. + float(alpha) * ((-g1 if branch < 2 else g1) + pair)
                minimum = torch.minimum(minimum, coefficient.amin())
                maximum = torch.maximum(maximum, coefficient.amax())
                total.add_(coefficient)
            return {
                "gain_absmean": gains.detach().abs().mean(dim=(0, 2, 3, 4, 5)),
                "gain_maxabs": gains.detach().abs().amax(dim=(0, 2, 3, 4, 5)),
                "conditional_gain_absmean": (gains.detach() - static_gains.detach()).abs().mean(dim=(0, 2, 3, 4, 5)),
                "conditional_delta_rms": conditional_rms, "routing_delta_rms": routing_rms,
                "fused_feature_rms": fused.detach().float().square().mean().sqrt(),
                "coefficient_min": minimum, "coefficient_max": maximum,
                "coefficient_sum_residual": (total - 4.).abs().amax(),
                "support_mean": support.detach().mean(),
                "radius_axis_absmean": radius_axis.detach().abs().mean(),
                "radius_axis_maxabs": radius_axis.detach().abs().amax(),
                "nonzero_radius_fraction": (radius_axis.detach() != 0).float().mean(),
            }

    def forward(self, feature, attribute, support, radius_mean, arm, alpha):
        if arm not in ARMS or not math.isfinite(float(alpha)) or not 0 <= float(alpha) <= 1:
            raise ValueError("Invalid routing arm or alpha")
        expected = (feature.shape[0], 1) + tuple(feature.shape[2:])
        if any(tuple(value.shape) != expected for value in (support, radius_mean)):
            raise ValueError("Dense conditions must match the actual SAAM grid")
        fused = super().forward(feature, attribute)
        h = torch.cat([conv(fused) for conv in self.morphology_convs], 1)
        factor = (1. - 1. / 3) * (1. + .1 / 3)
        F0 = fused + factor * self.morph_fusion(h)
        with torch.autocast(device_type=h.device.type, enabled=False):
            v = support.float() if arm != "C0" else torch.zeros_like(support, dtype=torch.float32)
            c_r = v * radius_mean.float().clamp(-RADIUS_AXIS_CLIP, RADIUS_AXIS_CLIP)
            if arm != "AR": c_r = torch.zeros_like(v)
            shape = (1, 3, 64, 1, 1, 1)
            static_gains = torch.tanh(self.bias.float().view(shape))
            gains = torch.tanh(self.bias.float().view(shape) + self.support_weight.float().view(shape) * v.unsqueeze(1)
                               + self.radius_weight.float().view(shape) * c_r.unsqueeze(1))
            g1, g2, g3 = gains.unbind(1)
            h0, h1, h2, h3 = h.float().split(16, dim=1)
            weight = self.morph_fusion.weight.float()
            d1 = factor * F.conv3d(torch.cat((-h0, -h1, h2, h3), 1), weight, bias=None)
            d2 = factor * F.conv3d(torch.cat((h0, -h1), 1), weight[:, :32], bias=None)
            d3 = factor * F.conv3d(torch.cat((h2, -h3), 1), weight[:, 32:], bias=None)
            routed = F0 + float(alpha) * g1 * d1
            routed = routed + float(alpha) * (1. - g1.abs()) * (g2 * d2 + g3 * d3)
        return routed, self._summary(gains, static_gains, (d1, d2, d3), alpha, fused, v, c_r)


class GTAAVT(nn.Module):
    """Joint P/S base, then frozen-pair local adaptation with explicit stages."""
    def __init__(self, predictor: AttributePredictor, arm="AR", use_gdf=True):
        super().__init__()
        if arm not in ARMS: raise ValueError("Expected C0, V_only or AR")
        self.arm, self.use_gdf = arm, bool(use_gdf)
        self.model_config = dict(MODEL_CONFIG, arm=arm, use_gdf=bool(use_gdf))
        self.predictor = predictor
        self.backbone = ActiveBackbone()
        self.generation_guidance = GenerationGuidanceModule(128, 256)
        self.routing = BoundedCalibreRouting()
        self.guidance_adapter = nn.Sequential(ConvBlock(256, 128, 1, padding=0), nn.Conv3d(128, 16, 1))
        self.ds_heads = nn.ModuleList([nn.Conv3d(64, 2, 1), nn.Conv3d(128, 2, 1)])
        self.output_feature = nn.Sequential(nn.Conv3d(22, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True))
        self.output_head = nn.Conv3d(32, 2, 1)
        self.gdf_head = nn.Sequential(nn.Conv3d(2, 1, 3, padding=1), nn.Sigmoid()) if self.use_gdf else None
        self.set_stage("base")

    @staticmethod
    def _gate_name(name):
        return name in ("routing.bias", "routing.support_weight", "routing.radius_weight")

    @classmethod
    def _local_name(cls, name):
        return name.startswith(("routing.morphology_convs.", "routing.morph_fusion.", "output_head.", "gdf_head.")) or cls._gate_name(name)

    def set_stage(self, stage):
        if stage not in STAGES: raise ValueError("Expected base, gdf_head, local_adapt or full_joint")
        if stage == "gdf_head" and self.gdf_head is None: raise ValueError("Gaussian head stage requires use_gdf=True")
        self.stage = stage
        for name, parameter in self.named_parameters():
            if stage == "base": active = not name.startswith("gdf_head.") and not self._gate_name(name)
            elif stage == "gdf_head": active = name.startswith("gdf_head.")
            elif stage == "full_joint": active = not name.startswith("predictor.")
            else: active = self._local_name(name)
            parameter.requires_grad_(active)
            parameter.grad = None
        if stage == "local_adapt":
            count, tensors = (115513, 17) if self.use_gdf else (115458, 15)
            active = [parameter for parameter in self.parameters() if parameter.requires_grad]
            if len(active) != tensors or sum(p.numel() for p in active) != count:
                raise RuntimeError("Local parameter scope differs from v0.1 plus Gaussian head")
        self.train(self.training)
        return self

    def train(self, mode=True):
        if not isinstance(mode, bool): raise ValueError("Training mode must be boolean")
        super().train(mode if self.stage == "base" else False)
        if self.stage != "base": self.predictor.eval()
        return self

    def active_parameter_names(self):
        return tuple(name for name, value in self.named_parameters() if value.requires_grad)

    def trainable_parameter_names(self):
        return self.active_parameter_names()

    def parameter_groups(self):
        result = {"shared": [], "gates": [], "gdf": []}
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                key = "gates" if self._gate_name(name) else "gdf" if name.startswith("gdf_head.") else "shared"
                result[key].append(parameter)
        return result

    @staticmethod
    def _foreground(logits):
        return logits[:, 1:2] - logits[:, :1]

    def _conditions(self, prediction, override, size):
        support, mean = prediction["support_logits"].float().sigmoid(), prediction["radius_mean"].float()
        if override is not None:
            keys = {"support", "radius_mean"}
            if not isinstance(override, dict) or set(override) != keys:
                raise ValueError("Diagnostic override requires support/radius_mean")
            fields = [torch.as_tensor(override[key], device=support.device, dtype=torch.float32).detach()
                      for key in ("support", "radius_mean")]
            if any(tuple(value.shape) != tuple(support.shape) for value in fields):
                raise ValueError("Override must match Bx1xinput/2 attribute grid")
            if not all(bool(torch.isfinite(value).all()) for value in fields): raise ValueError("Diagnostic conditions must be finite")
            support, mean = fields
            if bool(((support < 0.) | (support > 1.)).any()): raise ValueError("Support score must be in[0,1]")
        if support.shape[2:] != size: raise RuntimeError("Dense attribute and SAAM decoder grids differ")
        return support, mean

    def forward(self, x, alpha=1.0, condition_override=None):
        _check_image(x)
        if not math.isfinite(float(alpha)) or not 0 <= float(alpha) <= 1: raise ValueError("Routing alpha must be finite and in[0,1]")
        if self.stage == "base":
            prediction = self.predictor(x)
        else:
            with torch.no_grad(): prediction = self.predictor(x)
        features, decoded = [], []
        value = x
        for encoder in self.backbone.encoders:
            value = encoder(value); features.append(value)
        features.reverse()
        for i, decoder in enumerate(self.backbone.decoders):
            value = decoder(features[i + 1], value)
            if i == 0:
                op = self.generation_guidance
                fused = AnatomicalGuidanceModule.forward(op, value, prediction["features"][2])
                enhanced = fused * op.generation_enhance(fused) * (1. + .2 * .2)
                gate = op.uncertainty_gate(fused) * .2
                value = enhanced * (1. - gate) + fused * gate
            elif i == 1:
                support, mean = self._conditions(prediction, condition_override, value.shape[2:])
                value, routing = self.routing(value, prediction["features"][1], support, mean, self.arm, alpha)
            decoded.append(value)
        fused = self.backbone.fusion_module_decoder(decoded)
        guidance = F.interpolate(self.guidance_adapter(prediction["fused"]), size=fused.shape[2:], mode="trilinear", align_corners=False)
        output_feature = self.output_feature(torch.cat((fused, guidance), 1))
        segmentation_logits = self.output_head(output_feature)
        output = {"logits": self._foreground(segmentation_logits), "segmentation_logits": segmentation_logits,
            "aux_logits": [self._foreground(self.ds_heads[0](decoded[1])), self._foreground(self.ds_heads[1](decoded[0]))],
            "support_logits": prediction["support_logits"], "radius_mean": prediction["radius_mean"],
            "routing": routing}
        if self.gdf_head is not None: output["gdf"] = self.gdf_head(segmentation_logits)
        return output
