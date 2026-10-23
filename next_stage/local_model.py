"""One bounded local-scale revision of the frozen StageModel.

The new readout is frozen, consumes detached classifier features, and changes
only GAAM's spatial fine-score. The constant/alpha-zero route executes the
unchanged parent calculation without calling the readout or CT-support helper.
Oracle fields and composition interventions are diagnostic inputs, not inference.
"""
import torch

try:
    from .stage_model import StageModel, DIAGNOSTIC_MODES
    from .frozen_readout import (make_readout_head, readout_probabilities,
                                 ct_support_tensor)
except ImportError:
    from stage_model import StageModel, DIAGNOSTIC_MODES
    from frozen_readout import (make_readout_head, readout_probabilities,
                                ct_support_tensor)

from historical.func.AnatomicalFusionModules import AnatomicalGuidanceModule


LOCAL_MODES = ("configured", "r_mean", "r_repaired", "s_repaired")


def local_intervention(probability, eligible, mode="configured"):
    """Return a raw fine-score and an auditable same-patch fixed mapping.

    eligible is CT support for normal use; a GT-supported subset may be supplied
    only to explicit diagnostics. Re-pair whole r vectors among eligible
    lexicographic indices by a fixed half-ring rotation. q stays at its original
    position. This preserves the r multiset, not the q-r joint distribution or
    raw-s histogram. s_repaired instead preserves the consumed-s multiset.
    """
    if mode not in LOCAL_MODES:
        raise ValueError("Unknown local intervention")
    if probability.ndim != 5 or probability.shape[1] != 5:
        raise ValueError("Expected B,5,D,H,W probability field")
    expected = (probability.shape[0], 1) + tuple(probability.shape[2:])
    if tuple(eligible.shape) != expected or eligible.dtype != torch.bool:
        raise ValueError("Local support must be a B,1,D,H,W boolean field")
    p = probability.detach().float()
    q = p[:, :1]
    nonempty = p[:, 1:]
    mass = nonempty.sum(1, keepdim=True)
    r = nonempty / mass.clamp_min(torch.finfo(torch.float32).tiny)
    r = torch.where(mass > 0., r, torch.full_like(r, .25))
    score = p[:, 1:2].clone()
    supplied = p.clone()
    # -1 means no spatial remapping; zero/one eligible point is unchanged.
    mapping = torch.full(expected, -1, dtype=torch.long, device=p.device)
    if mode == "configured":
        return score, supplied, mapping
    for batch in range(p.shape[0]):
        indices = eligible[batch, 0].reshape(-1).nonzero().reshape(-1)
        count = indices.numel()
        if count <= 1:
            continue
        rflat = r[batch].reshape(4, -1)
        qflat = q[batch, 0].reshape(-1)
        destination = supplied[batch].reshape(5, -1)
        if mode == "r_mean":
            replacement = rflat[:, indices].mean(1, keepdim=True).expand(-1, count)
        else:
            source = indices.roll(shifts=max(1, count // 2)) if count > 1 else indices
            mapping[batch, 0].reshape(-1)[indices] = source
            if mode == "s_repaired":
                score[batch, 0].reshape(-1)[indices] = p[batch, 1].reshape(-1)[source]
                continue
            replacement = rflat[:, source]
        destination[1:, indices] = (1. - qflat[indices])[None] * replacement
        score[batch, 0].reshape(-1)[indices] = destination[1, indices]
    # For s_repaired supplied remains the original simplex intentionally: only
    # a consumed scalar was remapped; inventing a new full probability is wrong.
    return score.detach(), supplied.detach(), mapping


class LocalStageModel(StageModel):
    """Frozen 325-parameter readout; original heads/operators remain shared."""
    def __init__(self, conditioning="constant", detach_conditions=True,
                 alpha=0., q0=.2, s0=.2, gdf_seed=20261023,
                 local_seed=20261023):
        if conditioning not in ("constant", "local_scale"):
            raise ValueError("Unknown local conditioning")
        if not detach_conditions:
            raise ValueError("Revision B requires detached local consumption")
        super().__init__("constant" if conditioning == "constant" else "scale_only",
                         detach_conditions=True, alpha=alpha, q0=q0, s0=s0,
                         gdf_seed=gdf_seed)
        self.local_conditioning = conditioning
        # Helper initializes under fork_rng and restores common RNG exactly.
        self.local_head = make_readout_head(local_seed)
        self.local_head.requires_grad_(False)
        self.local_head.eval()
        self.local_seed = int(local_seed)
        self._local_request = None
        self._local_score = None
        self.last_local_trace = None

    def train(self, mode=True):
        super().train(mode)
        # No BN/Dropout exists in the head; explicit eval documents the contract.
        self.local_head.eval()
        return self

    def load_common_state(self, state):
        """Load the unchanged StageModel state, allowing only the new head gap."""
        if any(key.startswith("local_head.") for key in state):
            raise ValueError("Expected common StageModel state, not local state")
        incompatible = self.load_state_dict(state, strict=False)
        if set(incompatible.missing_keys) != {"local_head.weight", "local_head.bias"} or incompatible.unexpected_keys:
            raise ValueError("Common checkpoint mismatch: " + str(incompatible))
        return incompatible

    def load_readout_state(self, state):
        """Only load the frozen head's weight/bias, never any old model state."""
        if set(state) != {"weight", "bias"}:
            raise ValueError("Readout state must contain exactly weight and bias")
        expected = self.local_head.state_dict()
        for key in expected:
            if state[key].shape != expected[key].shape or not bool(torch.isfinite(state[key]).all()):
                raise ValueError("Invalid readout parameter " + key)
        self.local_head.load_state_dict(state, strict=True)
        self.local_head.requires_grad_(False)
        self.local_head.eval()

    def _local_is_active(self):
        stage = self._stage_request
        return (stage is not None and self.local_conditioning == "local_scale"
                and stage["mode"] not in ("constant",) + DIAGNOSTIC_MODES
                and stage["alpha"] != 0.)

    def conditions(self, classifier, condition_mode):
        scale, structure = super().conditions(classifier, condition_mode)
        if not self._local_is_active():
            return scale, structure
        request = self._local_request
        if request is None:
            raise RuntimeError("Local conditions outside a local forward")
        feature = classifier["encoder_features"][0]
        if tuple(feature.shape[1:]) != (64, 32, 32, 32):
            raise ValueError("Active local revision requires 128-cubed input")
        # The no_grad context is essential even with frozen head weights:
        # otherwise segmentation could backpropagate through its input features.
        with torch.no_grad():
            valid = ct_support_tensor(request["input_valid"])
            external = request["probability"]
            if external is None:
                probability = readout_probabilities(self.local_head, feature.detach())
            else:
                reference = feature.new_empty((feature.shape[0], 5, 32, 32, 32), dtype=torch.float32)
                probability = self._external_probability(external, reference, "local_scale")
            eligible = valid
            if request["diagnostic_valid"] is not None:
                diagnostic = torch.as_tensor(request["diagnostic_valid"], device=valid.device)
                if diagnostic.shape != valid.shape or diagnostic.dtype != torch.bool:
                    raise ValueError("Diagnostic validity must be B,1,32,32,32 bool")
                eligible = valid & diagnostic
            score, supplied, mapping = local_intervention(probability, eligible, request["mode"])
            # Diagnostic masks only select the intervention/statistics region;
            # they never control normal consumption or add a GT-dependent gate.
            self._local_score = torch.where(valid, score, torch.full_like(score, self.s0)).detach()
            trace = dict(mode=request["mode"], oracle_input=external is not None,
                         probability_detached=not self._local_score.requires_grad,
                         normal_support="CT-known complete16mm window only",
                         head_evaluated=external is None,
                         support_count=valid.flatten(1).sum(1).detach(),
                         score_mean=self._local_score.flatten(1).mean(1).detach(),
                         score_sd=self._local_score.flatten(1).std(1, unbiased=False).detach())
            support_weight = valid.float().flatten(1)
            support_count = support_weight.sum(1).clamp_min(1.)
            support_score = self._local_score.flatten(1)
            support_mean = (support_score * support_weight).sum(1) / support_count
            trace["supported_score_mean"] = support_mean.detach()
            trace["supported_score_sd"] = (((support_score - support_mean[:, None]).square() *
                                            support_weight).sum(1) / support_count).sqrt().detach()
            if request["trace"]:
                trace.update(raw_probability=probability.detach(), supplied_probability=supplied,
                             consumed_score=self._local_score, ct_support=valid,
                             diagnostic_eligible=eligible, source_flat_index=mapping)
            self.last_local_trace = trace
        return scale, structure

    def soft_generation(self, feature, attribute, probability):
        if not self._local_is_active():
            # Do not duplicate even one old operation in the reusable control.
            return super().soft_generation(feature, attribute, probability)
        if self._local_score is None:
            raise RuntimeError("Missing detached local score")
        operator = self.generation_guidance
        fused = AnatomicalGuidanceModule.forward(operator, feature, attribute)
        enhancement = operator.generation_enhance(fused)
        gate = operator.uncertainty_gate(fused)
        if self._local_score.shape != (fused.shape[0], 1) + tuple(fused.shape[2:]):
            raise ValueError("Local score and GAAM nominal grids differ")
        shape = (fused.shape[0], 1, 1, 1, 1)
        qref = torch.full(shape, self.q0, dtype=torch.float32, device=fused.device)
        sref = torch.full(shape, self.s0, dtype=torch.float32, device=fused.device)
        masked = fused * enhancement
        base = masked * (1. + .2 * sref) * (1. - gate * qref) + fused * gate * qref
        conditional = (masked * (1. + .2 * self._local_score) *
                       (1. - gate * qref) + fused * gate * qref)
        residual = conditional - base
        if self._local_request["trace"]:
            self.last_local_trace["unscaled_GAAM_residual_rms"] = residual.detach().float().square().flatten(1).mean(1).sqrt()
        if self._stage_request["alpha"] == 1.:
            return conditional
        return base + self._stage_request["alpha"] * residual

    def forward(self, image, coordinate=None, condition_mode=None,
                condition_override=None, alpha=None, input_valid=None,
                local_probability=None, local_mode="configured",
                local_diagnostic_valid=None, trace_local=False):
        if self._local_request is not None:
            raise RuntimeError("Reentrant LocalStageModel forward is not supported")
        if local_mode not in LOCAL_MODES:
            raise ValueError("Unknown local mode")
        if local_mode == "configured" and local_diagnostic_valid is not None:
            raise ValueError("Normal local inference cannot receive a GT diagnostic mask")
        actual_alpha = self.alpha if alpha is None else self._valid_alpha(alpha)
        active = (self.local_conditioning == "local_scale" and actual_alpha != 0.
                  and condition_mode not in ("constant",) + DIAGNOSTIC_MODES)
        if active:
            if condition_override is not None:
                raise ValueError("Active local mode uses local_probability, not global condition_override")
            if tuple(image.shape[1:]) != (1, 128, 128, 128):
                raise ValueError("Active local revision requires B,1,128,128,128")
            if input_valid is None:
                input_valid = torch.ones_like(image, dtype=torch.bool)
            else:
                input_valid = torch.as_tensor(input_valid, device=image.device)
                if input_valid.shape != image.shape or input_valid.dtype != torch.bool:
                    raise ValueError("input_valid must match the image with boolean CT validity")
        self._local_request = dict(input_valid=input_valid, probability=local_probability,
                                   diagnostic_valid=local_diagnostic_valid,
                                   mode=local_mode, trace=bool(trace_local))
        self._local_score = None
        self.last_local_trace = dict(mode="inactive", head_evaluated=False,
                                     probability_detached=True)
        try:
            return super().forward(image, coordinate, condition_mode=condition_mode,
                                   condition_override=condition_override, alpha=alpha)
        finally:
            self._local_score = None
            self._local_request = None

