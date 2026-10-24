"""Fixed-reference coordinates for the original frozen ordinal interaction.

The sole consumer change is b+w*(raw_c-.4). The reference is the declared
constant input, not an empirical mean. Raw predictions, projection, override
domain and null-support metadata remain inherited. No state or RNG is added.
"""
import torch
from ordinal_stage_model import OrdinalStageModel
from frozen_ordinal_stage_model import FrozenOrdinalStageModel


class CenteredFrozenOrdinalStageModel(FrozenOrdinalStageModel):
    def soft_generation(self, feature, attribute, probability):
        request = self._stage_request
        own = self._interaction_request
        if request is None or own is None:
            raise RuntimeError('Interaction outside a model forward request')
        alpha = request['alpha']
        # Start after OrdinalStageModel in the inherited MRO: the original
        # StageModel G0, avoiding a second ordinal interaction in this override.
        old_alpha = request['alpha']
        request['alpha'] = 0.
        try:
            g0 = super(OrdinalStageModel, self).soft_generation(feature, attribute, probability)
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
        argument = (self.interaction_bias[None, :] + self.interaction_weight[None, :] * (c - .4))
        gain = torch.tanh(argument.float())
        self.last_interaction_trace = dict(
            raw_fine=raw_fine, raw_ordinal=raw_ordinal, consumed_c=c.detach(), alpha=alpha,
            gain=gain.detach(), g0_shape=tuple(g0.shape), g0_dtype=str(g0.dtype),
            g0=g0.detach() if own['trace_tensor'] else None,
            diagnostic=own['diagnostic'], override_supplied=own['override'] is not None,
            conditioning=self.conditioning, q0=.2, s0=.2, constant_c=.4,
            readout='mean_three_nonempty_ordered_radius_CDF_masses',
            projection_weights=(0., 1., 2./3., 1./3., 0.), consumed_c_detached=True)
        self.last_interaction_trace.update(reference_c=.4, centered_c=(c - .4).detach())
        result = g0 if alpha == 0. else g0 + float(alpha) * g0 * gain[:, :, None, None, None]
        self.last_interaction_trace['adapted'] = result.detach() if own['trace_tensor'] else None
        return result
