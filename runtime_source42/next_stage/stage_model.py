"""Isolated next-stage GTA, retaining the frozen historical operators.

The normal candidate changes only GAAM's scale condition: q0 is a fixed
interpolation coefficient, s is the global softmax fine component, and alpha
gradually interpolates from the same-weight constant calculation.
The optional diagnostic modes are explicit interventions, not normal inference.
"""
from pathlib import Path
import math
import sys

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import torch
from torch import nn
from torch.nn import functional as F
from faithful_model import FaithfulGTAAVT
from historical.func.AnatomicalFusionModules import AnatomicalGuidanceModule


DIAGNOSTIC_MODES = (
    "qpred_rpred", "qpred_runiform", "qfixed_rpred", "qfixed_runiform")


class StageModel(FaithfulGTAAVT):
    """Original GTA operators with controlled scale routing and a GDF head.

    All arms register the GDF head, including arms with GDF loss weight zero.
    Construction of that head restores the CPU RNG exactly, so the common
    legacy parameter initialization and subsequent Dropout RNG remain unchanged.
    A model instance is intended for single-threaded forward execution.
    """
    def __init__(self, conditioning="constant", detach_conditions=True,
                 alpha=0., q0=.2, s0=.2, gdf_seed=20261023):
        if conditioning not in ("constant", "scale_only"):
            raise ValueError("Unknown stage conditioning")
        if not (0. <= q0 <= 1. and 0. <= s0 <= 1.):
            raise ValueError("q0/s0 must be bounded control coefficients")
        super().__init__(method="faithful_soft", width=32)
        self.conditioning = conditioning
        self.detach_conditions = bool(detach_conditions)
        self.q0, self.s0 = float(q0), float(s0)
        self.alpha = self._valid_alpha(alpha)
        # The common model was already initialized. Do not advance its RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(gdf_seed))
            self.gdf_head = nn.Conv3d(32, 1, 1)
        self.gdf_seed = int(gdf_seed)
        self._stage_request = None
        self._projection_feature = None
        self.last_condition_trace = None
        self._projection_hook = self.anatomical_output_fusion[-1].register_forward_pre_hook(
            self._capture_projection_feature)

    @staticmethod
    def _valid_alpha(value):
        value = float(value)
        if not math.isfinite(value) or not 0. <= value <= 1.:
            raise ValueError("alpha must be finite and in [0,1]")
        return value

    def set_alpha(self, alpha):
        self.alpha = self._valid_alpha(alpha)

    def _capture_projection_feature(self, module, inputs):
        self._projection_feature = inputs[0]

    @staticmethod
    def _intervene(probability, mode, q_reference):
        q = probability[:, :1]
        if mode.startswith("qfixed_"):
            q = torch.full_like(q, q_reference)
        if mode.endswith("_runiform"):
            r = torch.full_like(probability[:, 1:], 1. / (probability.shape[1] - 1))
        else:
            nonempty = probability[:, 1:]
            mass = nonempty.sum(1, keepdim=True)
            r = nonempty / mass.clamp_min(torch.finfo(nonempty.dtype).tiny)
            r = torch.where(mass > 0., r, torch.full_like(r, 1. / r.shape[1]))
        return torch.cat((q, (1. - q) * r), 1)

    @staticmethod
    def _external_probability(value, reference, name):
        value = torch.as_tensor(value, device=reference.device, dtype=torch.float32)
        if value.shape != reference.shape:
            raise ValueError("External %s shape mismatch" % name)
        if not bool(torch.isfinite(value).all()) or bool((value < 0.).any()):
            raise ValueError("External probabilities must be finite and nonnegative")
        if not bool(torch.allclose(value.sum(1), torch.ones_like(value[:, 0]),
                                   atol=1e-5, rtol=1e-5)):
            raise ValueError("External probabilities must sum to one; mask unknown upstream")
        return value

    def conditions(self, classifier, condition_mode):
        # super.forward passes only the inherited 'predicted' marker.
        request = self._stage_request
        if request is None:
            raise RuntimeError("Stage conditions outside a forward request")
        scale = classifier["generation"].float().softmax(1)
        structure = classifier["structure"].float().softmax(1)
        raw_scale, raw_structure = scale, structure
        override = request["override"]
        if override:
            if set(override) - {"scale", "structure"}:
                raise ValueError("Unknown condition_override key")
            if "scale" in override:
                scale = self._external_probability(override["scale"], scale, "scale")
            if "structure" in override:
                structure = self._external_probability(override["structure"], structure, "structure")
        mode = request["mode"]
        if mode in DIAGNOSTIC_MODES:
            scale = self._intervene(scale, mode, self.q0)
            structure = self._intervene(structure, mode, 1. / 3.)
        elif mode == "constant" or self.conditioning == "constant":
            scale = torch.full_like(scale, 1. / 5.)
            structure = torch.full_like(structure, 1. / 3.)
        else:
            # This scale vector is the raw head prediction. soft_generation
            # deliberately consumes only s; its q0 is a separate fixed control.
            structure = torch.full_like(structure, 1. / 3.)
        if self.detach_conditions:
            scale, structure = scale.detach(), structure.detach()
        self.last_condition_trace = {
            "mode": mode, "alpha": request["alpha"],
            "scale_head_probability": raw_scale.detach(),
            "structure_head_probability": raw_structure.detach(),
            "supplied_scale": scale.detach(),
            "supplied_structure": structure.detach(),
            "normal_scale_q0": self.q0,
            "normal_scale_s0": self.s0,
            "consumed_probability_detached": self.detach_conditions,
            "diagnostic_intervention": mode in DIAGNOSTIC_MODES,
        }
        return scale, structure

    def soft_generation(self, feature, attribute, probability):
        operator = self.generation_guidance
        fused = AnatomicalGuidanceModule.forward(operator, feature, attribute)
        enhancement = operator.generation_enhance(fused)
        gate = operator.uncertainty_gate(fused)
        request = self._stage_request
        if request["mode"] in DIAGNOSTIC_MODES:
            q = probability[:, 0, None, None, None, None]
            s = probability[:, 1, None, None, None, None]
            enhanced = fused * enhancement * (1. + .2 * s)
            return enhanced * (1. - gate * q) + fused * gate * q
        shape = (fused.shape[0], 1, 1, 1, 1)
        qref = torch.full(shape, self.q0, dtype=torch.float32, device=fused.device)
        sref = torch.full(shape, self.s0, dtype=torch.float32, device=fused.device)
        # Preserve the frozen operator's AMP promotion/order: the first
        # product is half under autocast; the probability arithmetic is FP32.
        masked = fused * enhancement
        base = (masked * (1. + .2 * sref) *
                (1. - gate * qref) + fused * gate * qref)
        if request["mode"] == "constant" or self.conditioning == "constant":
            return base
        if request["alpha"] == 0.:
            return base
        s = probability[:, 1, None, None, None, None]
        conditional = (masked * (1. + .2 * s) *
                       (1. - gate * qref) + fused * gate * qref)
        if request["alpha"] == 1.:
            return conditional
        # Difference of the same G(F;q0,s); all convolutions are reused.
        return base + request["alpha"] * (conditional - base)

    def soft_structure(self, feature, attribute, probability):
        # The configured candidate always receives uniform structure; only
        # explicit diagnostic interventions can vary the structure probabilities.
        return super().soft_structure(feature, attribute, probability)

    def forward(self, image, coordinate=None, condition_mode=None,
                condition_override=None, alpha=None):
        mode = "configured" if condition_mode is None else condition_mode
        if mode not in ("configured", "predicted", "constant") + DIAGNOSTIC_MODES:
            raise ValueError("Unknown stage condition_mode")
        if mode == "predicted":
            mode = "configured"
        if condition_override is not None and not isinstance(condition_override, dict):
            raise ValueError("condition_override must be a probability dictionary")
        if self._stage_request is not None:
            raise RuntimeError("Reentrant StageModel forward is not supported")
        self._stage_request = dict(mode=mode, override=condition_override,
                                   alpha=self.alpha if alpha is None else self._valid_alpha(alpha))
        self._projection_feature = None
        try:
            output = super().forward(image, coordinate, condition_mode="predicted")
            feature = self._projection_feature
            if feature is None or feature.shape[1] != 32:
                raise RuntimeError("Expected the original full-resolution32-channel projection input")
            raw = self.gdf_head(feature)
            output["gdf_raw"] = raw
            # Target is log1p(cumulative Gaussian field)/fixed field_scale.
            output["gdf_prediction"] = F.softplus(raw.float())
            return output
        finally:
            self._projection_feature = None
            self._stage_request = None

