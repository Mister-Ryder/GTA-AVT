"""CT-only GTA-AVT v1 with separately pretrained, frozen geometric attributes.

The same predictor supplies its multi-scale features and dense conditions.
Trainable segmentation adapters consume frozen features; no duplicate classifier
or label-derived inference input is present. Historical primitives retain their
own provenance in :mod:`gta_avt.vendor`.
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


ARMS = ("C0", "A_only", "AR")
MODEL_CONFIG = {
    "name": "GTA-AVT-v1", "input_channels": 1, "input_multiple": 16,
    "attribute_encoder_channels": [32, 64, 128, 256],
    "attribute_field_stride": 2, "seg_encoder_channels": [32, 64, 128, 256],
    "seg_decoder_channels": [128, 64, 32], "routing_site_channels": 64,
    "routing_coefficients": 576, "routing_dilations": [1, 1, 2, 3],
    "applicability_threshold": .5, "predictor_frozen_during_segmentation": True,
    "statistics_keys": ["mu", "sigma", "pi"],
}


def _check_image(x):
    if (not isinstance(x, torch.Tensor) or x.ndim != 5 or x.shape[1] != 1 or
            x.shape[0] < 1 or not x.is_floating_point() or
            any(n < 32 or n % 16 for n in x.shape[2:])):
        raise ValueError("Expected normalized floating Bx1xDxHxW CT, dimensions >=32 and divisible by16")


class AttributePredictor(nn.Module):
    """Independent CT -> applicability logit and standardized log-radius field.

    ``statistics`` must be estimated using TRAIN only. Applicability estimates
    emission by the fixed geometry pipeline, not foreground probability or a
    calibrated uncertainty. Radius is defined only where that pipeline emits it.
    No applicability/validity mask is accepted as input to this predictor.
    """
    def __init__(self, statistics: Mapping):
        super().__init__()
        mean = float(statistics["mu"] if "mu" in statistics else statistics["log_radius_mean"])
        std = float(statistics["sigma"] if "sigma" in statistics else statistics["log_radius_std"])
        prior = float(statistics["pi"] if "pi" in statistics else statistics["emission_prior"])
        if not (math.isfinite(mean) and math.isfinite(std) and std > 0 and
                math.isfinite(prior) and 0 < prior < 1):
            raise ValueError("TRAIN log-radius mean/SD must be finite, SD positive, and emission prior in(0,1)")
        self.register_buffer("log_radius_mean", torch.tensor(mean, dtype=torch.float64))
        self.register_buffer("log_radius_std", torch.tensor(std, dtype=torch.float64))
        self.register_buffer("emission_prior", torch.tensor(prior, dtype=torch.float64))
        # Exact active historical encoder/SE/fusion, without unused global heads.
        self.encoder = nn.ModuleList([
            ConvBlock(1, 32), ConvBlock(32, 32, stride=2),
            ConvBlock(32, 64), ConvBlock(64, 64, stride=2),
            ConvBlock(64, 128), ConvBlock(128, 128, stride=2),
            ConvBlock(128, 256), ConvBlock(256, 256, stride=2)])
        self.attention_blocks = nn.ModuleList([SEBlock(c, reduction=8) for c in (64, 128, 256)])
        self.multi_scale_fusion = nn.Sequential(nn.Conv3d(448, 256, 1), nn.BatchNorm3d(256), nn.ReLU(inplace=True))
        self.local_head = nn.Conv3d(320, 2, 1)
        nn.init.zeros_(self.local_head.weight)
        nn.init.zeros_(self.local_head.bias)
        with torch.no_grad():
            self.local_head.bias[0] = math.log(prior) - math.log1p(-prior)

    def statistics(self):
        return {"mu": float(self.log_radius_mean), "sigma": float(self.log_radius_std),
                "pi": float(self.emission_prior)}

    def forward(self, x):
        _check_image(x)
        output_size = tuple(n // 2 for n in x.shape[2:])
        features = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            if i in (3, 5, 7):
                features.append(x)
        features = [block(value) for block, value in zip(self.attention_blocks, features)]
        size = features[-1].shape[2:]
        fused = self.multi_scale_fusion(torch.cat((features[2],
            F.interpolate(features[1], size=size, mode="trilinear", align_corners=False),
            F.interpolate(features[0], size=size, mode="trilinear", align_corners=False)), 1))
        head = self.local_head(torch.cat((features[0], F.interpolate(fused,
            size=features[0].shape[2:], mode="trilinear", align_corners=False)), 1))
        # Dense regression/interpolation arithmetic remains FP32 under AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            field = F.interpolate(head.float(), size=output_size, mode="trilinear", align_corners=False)
            a_logits, z_radius = field[:, :1], field[:, 1:2]
        return {"a_logits": a_logits, "z_radius": z_radius,
                "p_a": torch.sigmoid(a_logits), "features": features, "fused": fused}


class ActiveBackbone(nn.Module):
    """Only the four encoders, three decoders and decoder fusion actually used."""
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            Encoder(1, 16, 32, apply_pooling=False), Encoder(32, 32, 64),
            Encoder(64, 64, 128), Encoder(128, 128, 256)])
        self.decoders = nn.ModuleList([
            Decoder(256, 256, 384, 128, 128, num_groups=2),
            Decoder(128, 128, 192, 64, 64, num_groups=8),
            Decoder(64, 64, 96, 32, 32, num_groups=8)])
        self.fusion_module_decoder = FusionModule([128, 64, 32], out_channels=2, scale_factors=[1, 2, 4])


class BoundedCalibreRouting(AnatomicalGuidanceModule):
    """Four original SAAM branches, three bounded continuous contrasts.

    F = F0 + alpha*g1*D1 + alpha*(1-|g1|)*(g2*D2+g3*D3).
    ``alpha`` is in[0,1]. The effective four branch coefficients are bounded in
    [0,2] and sum to4 in real/same-precision arithmetic; this is not a claim of
    feature-energy conservation or bitwise equivalence under mixed precision.
    """
    def __init__(self):
        super().__init__(64, 128)
        self.morphology_convs = nn.ModuleList([
            nn.Conv3d(64, 16, 3, padding=d, dilation=d) for d in (1, 1, 2, 3)])
        self.morph_fusion = nn.Conv3d(64, 64, 1)
        self.bias = nn.Parameter(torch.zeros(3, 64))
        self.availability_weight = nn.Parameter(torch.zeros(3, 64))
        self.radius_weight = nn.Parameter(torch.zeros(3, 64))

    @staticmethod
    def coefficients(gains, alpha):
        """Return the four BxCxDxHxW maps for diagnostics on small fields."""
        g1, g2, g3 = gains.unbind(1)
        remaining = 1. - g1.abs()
        a = float(alpha)
        return torch.stack((1 + a * (-g1 + remaining * g2),
                            1 + a * (-g1 - remaining * g2),
                            1 + a * (g1 + remaining * g3),
                            1 + a * (g1 - remaining * g3)), 1)

    def forward(self, feature, attribute, a, z_radius, arm, alpha):
        fused = super().forward(feature, attribute)
        h = torch.cat([conv(fused) for conv in self.morphology_convs], 1)
        # Same fixed soft-structure factor as the historical constant control.
        factor = (1. - 1. / 3) * (1. + .1 / 3)
        F0 = fused + factor * self.morph_fusion(h)
        with torch.autocast(device_type=h.device.type, enabled=False):
            a = a.float() if arm != "C0" else torch.zeros_like(a, dtype=torch.float32)
            c_r = a * z_radius.float() if arm == "AR" else torch.zeros_like(a)
            shape = (1, 3, 64, 1, 1, 1)
            gains = torch.tanh(self.bias.float().view(shape) +
                self.availability_weight.float().view(shape) * a.unsqueeze(1) +
                self.radius_weight.float().view(shape) * c_r.unsqueeze(1))
            g1, g2, g3 = gains.unbind(1)
            h0, h1, h2, h3 = h.float().split(16, dim=1)
            weight = self.morph_fusion.weight.float()
            d1 = factor * F.conv3d(torch.cat((-h0, -h1, h2, h3), 1), weight, bias=None)
            d2 = factor * F.conv3d(torch.cat((h0, -h1), 1), weight[:, :32], bias=None)
            d3 = factor * F.conv3d(torch.cat((h2, -h3), 1), weight[:, 32:], bias=None)
            routed = F0 + float(alpha) * g1 * d1
            routed = routed + float(alpha) * (1. - g1.abs()) * (g2 * d2 + g3 * d3)
        # Small channel/axis summaries; do not retain four full coefficient maps.
        summary = {"gain_absmean": gains.detach().abs().mean(dim=(0, 2, 3, 4, 5)),
                   "gain_maxabs": gains.detach().abs().amax(dim=(0, 2, 3, 4, 5))}
        return routed, summary


class GTAAVT(nn.Module):
    """Full segmentation network using one frozen pretrained predictor.

    All segmentation layers/adapters remain trainable. Calling ``train()`` keeps
    the predictor in eval mode, including its batch-normalization statistics.
    ``condition_override`` is an explicit correspondence diagnostic; production
    training and inference both consume predictions from the same CT predictor.
    """
    def __init__(self, predictor: AttributePredictor, arm="AR", use_gdf=False):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("Expected C0, A_only or AR")
        self.arm, self.use_gdf = arm, bool(use_gdf)
        self.model_config = dict(MODEL_CONFIG, arm=arm, use_gdf=bool(use_gdf))
        self.predictor = predictor
        self.predictor.requires_grad_(False)
        self.predictor.eval()
        self.backbone = ActiveBackbone()
        self.generation_guidance = GenerationGuidanceModule(128, 256)
        self.routing = BoundedCalibreRouting()
        # This adapter belongs to S and is trained, unlike frozen P features.
        self.guidance_adapter = nn.Sequential(ConvBlock(256, 128, 1, padding=0), nn.Conv3d(128, 16, 1))
        self.ds_heads = nn.ModuleList([nn.Conv3d(64, 2, 1), nn.Conv3d(128, 2, 1)])
        self.output_feature = nn.Sequential(nn.Conv3d(22, 32, 3, padding=1),
                                            nn.BatchNorm3d(32), nn.ReLU(inplace=True))
        self.output_head = nn.Conv3d(32, 2, 1)
        self.gdf_head = nn.Sequential(nn.Conv3d(32, 1, 1), nn.Softplus()) if self.use_gdf else None

    def train(self, mode=True):
        super().train(mode)
        self.predictor.eval()
        return self

    @staticmethod
    def _foreground(logits):
        return logits[:, 1:2] - logits[:, :1]

    def _conditions(self, prediction, override, size):
        a = (prediction["a_logits"].sigmoid() >= .5).float()
        z_radius = prediction["z_radius"].float()
        if override is not None:
            if not isinstance(override, dict) or set(override) != {"a", "z_radius"}:
                raise ValueError("Diagnostic override requires exactly a and z_radius")
            a = torch.as_tensor(override["a"], device=a.device, dtype=torch.float32).detach()
            z_radius = torch.as_tensor(override["z_radius"], device=a.device, dtype=torch.float32).detach()
            if tuple(a.shape) != tuple(prediction["a_logits"].shape) or a.shape != z_radius.shape:
                raise ValueError("Override must match Bx1xinput/2 attribute grid")
            if not bool(torch.isfinite(a).all() and torch.isfinite(z_radius).all()) or bool(((a != 0) & (a != 1)).any()):
                raise ValueError("Override applicability must be exact binary and radius finite")
        if a.shape[2:] != size:
            raise RuntimeError("Dense attribute grid and actual SAAM decoder grid differ")
        return a, z_radius

    def forward(self, x, alpha=1.0, condition_override=None):
        _check_image(x)
        if not math.isfinite(float(alpha)) or not 0 <= float(alpha) <= 1:
            raise ValueError("Routing alpha must be finite and in[0,1]")
        with torch.no_grad():
            prediction = self.predictor(x)
        features = []
        decoded = []
        value = x
        for encoder in self.backbone.encoders:
            value = encoder(value)
            features.append(value)
        features.reverse()
        routing = None
        for i, decoder in enumerate(self.backbone.decoders):
            value = decoder(features[i + 1], value)
            if i == 0:
                op = self.generation_guidance
                fused = AnatomicalGuidanceModule.forward(op, value, prediction["features"][2])
                enhanced = fused * op.generation_enhance(fused) * (1. + .2 * .2)
                gate = op.uncertainty_gate(fused) * .2
                value = enhanced * (1. - gate) + fused * gate
            elif i == 1:
                a, z_radius = self._conditions(prediction, condition_override, value.shape[2:])
                value, routing = self.routing(value, prediction["features"][1], a, z_radius, self.arm, alpha)
            decoded.append(value)
        fused = self.backbone.fusion_module_decoder(decoded)
        guidance = F.interpolate(self.guidance_adapter(prediction["fused"]), size=fused.shape[2:],
                                 mode="trilinear", align_corners=False)
        output_feature = self.output_feature(torch.cat((fused, guidance), 1))
        output = {"logits": self._foreground(self.output_head(output_feature)),
                  "aux_logits": [self._foreground(self.ds_heads[0](decoded[1])),
                                 self._foreground(self.ds_heads[1](decoded[0]))],
                  "a_logits": prediction["a_logits"], "z_radius": prediction["z_radius"],
                  "routing": routing}
        if self.gdf_head is not None:
            output["gdf"] = self.gdf_head(output_feature)
        return output
