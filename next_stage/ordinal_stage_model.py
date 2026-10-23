"""Ordered-radius CDF projection with the same signed condition-feature interaction.

The original constant GTA operator G0 is preserved exactly. The only new
parameters are128 GAAM-channel weights and128 biases, all initialized to zero.
Only the consumed scalar changes from raw fine to the mean of three ordered
radius CDF masses. Constant uniform-five-class consumption is .4.
Normal conditions come from CT predictions only; explicit overrides are for
registered same-prediction re-pair diagnostics, never normal GT conditioning.
"""
import torch
from torch import nn
from stage_model import StageModel


class OrdinalStageModel(StageModel):
    def __init__(self, conditioning='constant', detach_conditions=True, alpha=0.,
                 q0=.2, s0=.2, gdf_seed=20261023):
        if conditioning not in ('constant', 'scale_only') or not detach_conditions:
            raise ValueError('Candidate requires constant/scale_only detached conditions')
        if q0 != .2 or s0 != .2:
            raise ValueError('The candidate preserves original q0=s0=.2')
        super().__init__(conditioning, True, alpha, q0, s0, gdf_seed)
        # zeros() advances no RNG; identical modules/parameter order in both arms.
        self.interaction_weight = nn.Parameter(torch.zeros(128, dtype=torch.float32))
        self.interaction_bias = nn.Parameter(torch.zeros(128, dtype=torch.float32))
        self.last_interaction_trace = None
        self._interaction_request = None

    @staticmethod
    def ordinal_condition(probability):
        """Mean of nonempty P(radius<=2), P(radius<=4), P(radius<=8).

        Input is the existing five-class image prediction, not a GT attribute.
        Empty mass contributes zero. This is an ordered composition score, not
        a clinical diameter, a new target, normalized r, or calibrated certainty.
        """
        if probability.ndim != 2 or probability.shape[1] != 5:
            raise ValueError('Expected the original Bx5 scale probabilities')
        p = probability.float()
        return p[:, 1:2] + (2. * p[:, 2:3] + p[:, 3:4]) / 3.

    def load_common_state(self, state):
        """Load the old common constant state; only declared new keys may miss."""
        result = self.load_state_dict(state, strict=False)
        required = {'interaction_weight', 'interaction_bias'}
        allowed = required | {'gdf_head.weight', 'gdf_head.bias'}
        if result.unexpected_keys or not required.issubset(result.missing_keys) or not set(result.missing_keys).issubset(allowed):
            raise ValueError('Common checkpoint differs beyond candidate/GDF new parameters')
        return result

    def interaction_diagnostics(self):
        """Detached scalars for the current forward, before optimizer update.

        alpha*gain is a channel-relative residual multiplier, not a measured
        logit effect, gradient attribution or clinical/segmentation benefit.
        """
        trace = self.last_interaction_trace
        if trace is None:
            raise RuntimeError('No candidate forward available for diagnostics')
        result = dict(interaction_diagnostics_phase='forward_before_optimizer_update',
                      interaction_alpha=float(trace['alpha']),
                      interaction_diagnostic_override=bool(trace['override_supplied']))
        with torch.no_grad():
            for name, tensor in (('raw_fine', trace['raw_fine']), ('raw_ordinal', trace['raw_ordinal']), ('c', trace['consumed_c'])):
                value = tensor.detach().float()
                result.update({'interaction_'+name+'_mean': float(value.mean()),
                               'interaction_'+name+'_min': float(value.min()),
                               'interaction_'+name+'_max': float(value.max()),
                               'interaction_'+name+'_population_std': float(value.std(unbiased=False))})
            gain = trace['gain'].detach().float()
            multiplier = float(trace['alpha']) * gain
            result.update(interaction_gain_mean=float(gain.mean()),
                          interaction_gain_absmean=float(gain.abs().mean()),
                          interaction_gain_maxabs=float(gain.abs().max()),
                          interaction_alpha_gain_absmean=float(multiplier.abs().mean()),
                          interaction_alpha_gain_maxabs=float(multiplier.abs().max()))
            for name, parameter in (('weight', self.interaction_weight), ('bias', self.interaction_bias)):
                value = parameter.detach().float()
                result['interaction_'+name+'_norm'] = float(value.norm())
                result['interaction_'+name+'_maxabs'] = float(value.abs().max())
        return result

    def soft_generation(self, feature, attribute, probability):
        request = self._stage_request
        own = self._interaction_request
        if request is None or own is None:
            raise RuntimeError('Interaction outside a model forward request')
        alpha = request['alpha']
        # Parent conditions/trace remain normal. Only its local G0 calculation
        # temporarily uses alpha0, then the request is restored even on failure.
        old_alpha = request['alpha']
        request['alpha'] = 0.
        try:
            g0 = super().soft_generation(feature, attribute, probability)
        finally:
            request['alpha'] = old_alpha
        if g0.shape[1] != 128:
            raise RuntimeError('Expected the original128-channel GAAM output')
        probability = self.last_condition_trace['scale_head_probability'].float().detach()
        raw_fine = probability[:, 1:2]
        raw_ordinal = self.ordinal_condition(probability)
        if request['mode'] == 'constant' or self.conditioning == 'constant':
            c = torch.full_like(raw_ordinal, .4)
        else:
            c = raw_ordinal
        if own['override'] is not None:
            value = torch.as_tensor(own['override'], dtype=torch.float32, device=g0.device)
            if value.shape == (g0.shape[0],):
                value = value[:, None]
            if value.shape != (g0.shape[0], 1) or not bool(torch.isfinite(value).all()) or bool(((value < 0.) | (value > 1.)).any()):
                raise ValueError('Diagnostic consumed c must be finite B/B1 fractions')
            c = value.detach()
        argument = (self.interaction_bias[None, :] + self.interaction_weight[None, :] * c)
        gain = torch.tanh(argument.float())
        self.last_interaction_trace = dict(
            raw_fine=raw_fine, raw_ordinal=raw_ordinal, consumed_c=c.detach(), alpha=alpha,
            gain=gain.detach(), g0_shape=tuple(g0.shape), g0_dtype=str(g0.dtype),
            g0=g0.detach() if own['trace_tensor'] else None,
            diagnostic=own['diagnostic'], override_supplied=own['override'] is not None,
            conditioning=self.conditioning, q0=.2, s0=.2, constant_c=.4,
            readout='mean_three_nonempty_ordered_radius_CDF_masses',
            projection_weights=(0., 1., 2./3., 1./3., 0.), consumed_c_detached=True)
        result = g0 if alpha == 0. else g0 + float(alpha) * g0 * gain[:, :, None, None, None]
        self.last_interaction_trace['adapted'] = result.detach() if own['trace_tensor'] else None
        return result

    def forward(self, image, coordinate=None, condition_mode=None,
                condition_override=None, alpha=None, consumed_c_override=None,
                diagnostic=False, trace_interaction=False):
        if condition_override is not None:
            raise ValueError('Attribute/GT overrides are not part of this candidate')
        if condition_mode not in (None, 'configured', 'predicted', 'constant'):
            raise ValueError('Original probability interventions do not define this candidate')
        if consumed_c_override is not None and not diagnostic:
            raise ValueError('Consumed-c override requires explicit diagnostic=True')
        if self._interaction_request is not None:
            raise RuntimeError('Reentrant candidate forward is not supported')
        self._interaction_request = dict(override=consumed_c_override,
            diagnostic=bool(diagnostic), trace_tensor=bool(trace_interaction))
        self.last_interaction_trace = None
        try:
            return super().forward(image, coordinate, condition_mode, None, alpha)
        finally:
            self._interaction_request = None
