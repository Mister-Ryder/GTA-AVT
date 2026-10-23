"""Matched local-supervision pair A: identical learning, optional consumption.

Both arms compute the same local head with gradients to the shared classifier.
Only GAAM's consumed fine-score differs after the common alpha-zero stage.
The probability used in that calculation is always detached. Old operators,
global auxiliary heads, the fixed q0, and uniform SAAM remain unchanged.
"""
import torch

try:
    from .stage_model import StageModel
    from .frozen_readout import (make_readout_head, align_classifier_features,
                                 ct_support_tensor)
except ImportError:
    from stage_model import StageModel
    from frozen_readout import (make_readout_head, align_classifier_features,
                                ct_support_tensor)

from historical.func.AnatomicalFusionModules import AnatomicalGuidanceModule


class LocalLearningModel(StageModel):
    def __init__(self, conditioning="constant", detach_conditions=True,
                 alpha=0., q0=.2, s0=.2, gdf_seed=20261023,
                 local_seed=20261023):
        if conditioning not in ("constant", "local_scale"):
            raise ValueError("Expected constant or local_scale arm")
        if not detach_conditions:
            raise ValueError("Local consumption must remain detached")
        super().__init__("constant" if conditioning == "constant" else "scale_only",
                         detach_conditions=True, alpha=alpha, q0=q0, s0=s0,
                         gdf_seed=gdf_seed)
        self.local_conditioning = conditioning
        self.local_head = make_readout_head(local_seed)
        self.local_seed = int(local_seed)
        self._local_request = None
        self._local_logits = None
        self._local_score = None
        self.last_local_trace = None

    def load_common_state(self, state):
        """Old constant initialization may omit GDF; only new heads may miss."""
        if not isinstance(state, dict) or not state:
            raise ValueError("Expected a common tensor state dictionary")
        if any(key.startswith("local_head.") for key in state):
            raise ValueError("Common initialization must not already contain a local head")
        result = self.load_state_dict(state, strict=False)
        missing = set(result.missing_keys)
        allowed = ({"local_head.weight", "local_head.bias"},
                   {"local_head.weight", "local_head.bias", "gdf_head.weight", "gdf_head.bias"})
        if result.unexpected_keys or missing not in allowed:
            raise ValueError("Common checkpoint mismatch: " + str(result))
        return result

    def load_readout_state(self, state):
        if set(state) != {"weight", "bias"}:
            raise ValueError("Readout state must contain only weight/bias")
        reference = self.local_head.state_dict()
        for key in reference:
            if (state[key].shape != reference[key].shape or state[key].dtype != torch.float32
                    or not bool(torch.isfinite(state[key]).all())):
                raise ValueError("Invalid local readout " + key)
        self.local_head.load_state_dict(state, strict=True)
        self.local_head.requires_grad_(True)

    def conditions(self, classifier, condition_mode):
        scale, structure = super().conditions(classifier, condition_mode)
        feature = classifier["encoder_features"][0]
        if tuple(feature.shape[1:]) != (64, 32, 32, 32):
            raise ValueError("Local learning fixes input128 and classifier grid32")
        # This branch executes in BOTH arms, including alpha0. In contrast to
        # frozen B, no no_grad/detach is applied before the local loss logits.
        with torch.cuda.amp.autocast(enabled=False):
            self._local_logits = self.local_head(align_classifier_features(feature))
        probability = self._local_logits.detach().float().softmax(1)
        with torch.no_grad():
            valid = ct_support_tensor(self._local_request["input_valid"])
            self._local_score = torch.where(
                valid, probability[:, 1:2], torch.full_like(probability[:, 1:2], self.s0))
            self.last_local_trace = dict(
                head_computed=True, head_trainable=True,
                probability_detached=not self._local_score.requires_grad,
                support_count=valid.flatten(1).sum(1),
                score_mean=self._local_score.flatten(1).mean(1),
                score_sd=self._local_score.flatten(1).std(1, unbiased=False))
        return scale, structure

    def soft_generation(self, feature, attribute, probability):
        request = self._stage_request
        if (self.local_conditioning == "constant" or request["mode"] == "constant"
                or request["alpha"] == 0.):
            return super().soft_generation(feature, attribute, probability)
        if request["mode"] != "configured":
            raise ValueError("Learning pair permits only configured/constant consumption")
        operator = self.generation_guidance
        fused = AnatomicalGuidanceModule.forward(operator, feature, attribute)
        enhancement = operator.generation_enhance(fused)
        gate = operator.uncertainty_gate(fused)
        if self._local_score.shape != (fused.shape[0], 1) + tuple(fused.shape[2:]):
            raise ValueError("Local score and GAAM grid differ")
        shape = (fused.shape[0], 1, 1, 1, 1)
        qref = torch.full(shape, self.q0, dtype=torch.float32, device=fused.device)
        sref = torch.full(shape, self.s0, dtype=torch.float32, device=fused.device)
        masked = fused * enhancement
        base = masked * (1. + .2 * sref) * (1. - gate * qref) + fused * gate * qref
        conditional = (masked * (1. + .2 * self._local_score) *
                       (1. - gate * qref) + fused * gate * qref)
        if request["alpha"] == 1.:
            return conditional
        return base + request["alpha"] * (conditional - base)

    def forward(self, image, coordinate=None, condition_mode=None,
                condition_override=None, alpha=None, input_valid=None):
        if self._local_request is not None:
            raise RuntimeError("Reentrant local-learning forward is unsupported")
        if tuple(image.shape[1:]) != (1, 128, 128, 128):
            raise ValueError("Local learning requires B,1,128,128,128")
        if condition_mode not in (None, "configured", "predicted", "constant"):
            raise ValueError("Local learning does not expose global q/r interventions")
        if condition_override is not None:
            raise ValueError("No GT/externally overridden probability enters this model")
        if input_valid is None:
            raise ValueError("Explicit CT input_valid is required, including inference padding")
        input_valid = torch.as_tensor(input_valid, device=image.device)
        if input_valid.shape != image.shape or input_valid.dtype != torch.bool:
            raise ValueError("input_valid must match image shape with bool dtype")
        self._local_request = dict(input_valid=input_valid)
        self._local_logits = self._local_score = None
        try:
            result = super().forward(image, coordinate, condition_mode=condition_mode,
                                     condition_override=None, alpha=alpha)
            if self._local_logits is None:
                raise RuntimeError("Both arms must compute local logits")
            result["local_logits"] = self._local_logits
            return result
        finally:
            self._local_request = None
            self._local_logits = self._local_score = None

