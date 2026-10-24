"""Relax only the product-field consumer constraint with one availability axis.

The original C0 and A_only operator paths are unchanged. AB retains the product
axis and adds a zero-initialized availability-axis coefficient per128 channels.
No head, G0, normalization, loss, sampling or optimizer is defined or changed.
"""
import torch
from torch import nn
from torch.nn import functional as F
from boundary_field_model import BoundaryFieldStageModel as ProductBoundaryFieldStageModel
from boundary_field_model import HEAD_BUFFERS


class BoundaryFieldStageModel(ProductBoundaryFieldStageModel):
    def __init__(self, arm='C0', alpha=0., q0=.2, s0=.2, gdf_seed=20261023):
        super().__init__(arm, alpha, q0, s0, gdf_seed)
        # zeros advances no RNG; controls never train or consume this parameter.
        self.interaction_availability_weight = nn.Parameter(
            torch.zeros(128, dtype=torch.float32), requires_grad=arm == 'AB')

    def load_common_state(self, state):
        expected = set(self.state_dict())
        new = set(HEAD_BUFFERS) | {'interaction_weight', 'interaction_bias', 'interaction_availability_weight'}
        optional = {'gdf_head.weight', 'gdf_head.bias'}
        missing = expected - set(state)
        if set(state) - expected or not new <= missing or missing - new - optional:
            raise ValueError('Common checkpoint differs beyond declared boundary/interaction/GDF new state')
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or set(result.missing_keys) != missing:
            raise ValueError('Unexpected common checkpoint loading result')
        with torch.no_grad():
            self.interaction_availability_weight.zero_()
        return result

    def soft_generation(self, feature, attribute, probability):
        if self.boundary_arm != 'AB':
            return super().soft_generation(feature, attribute, probability)
        own = self._boundary_request
        if own is None:
            raise RuntimeError('Separable field interaction outside a forward request')
        original_trace = own['trace_tensor']
        own['trace_tensor'] = True
        try:
            # Parent computes exactly the frozen G0 and original product field.
            # Its unused old adapted output/graph is immediately discarded.
            super().soft_generation(feature, attribute, probability)
        finally:
            own['trace_tensor'] = original_trace
        trace = self.last_interaction_trace
        g0 = trace['g0']
        if g0 is None or g0.requires_grad or trace['consumed_A'] is None:
            raise RuntimeError('Expected frozen G0 and a defined consumed availability field')
        trace['adapted'] = None
        with torch.cuda.amp.autocast(enabled=False):
            availability_raw8 = trace['consumed_A'].double() * self.boundary_B_fit_constant
            availability_raw = F.interpolate(availability_raw8, size=g0.shape[2:],
                mode='trilinear', align_corners=False)
            availability_normalized64 = (availability_raw - self.boundary_field_mean) / self.boundary_field_std
            zA = availability_normalized64.float()
            zAB = trace['centered_c']
            argument = (self.interaction_bias[None, :, None, None, None] +
                self.interaction_weight[None, :, None, None, None] * zAB +
                self.interaction_availability_weight[None, :, None, None, None] * zA)
            gain = torch.tanh(argument.float())
        alpha = trace['alpha']
        result = g0 if alpha == 0. else g0 + float(alpha) * g0 * gain
        trace.update(availability_raw_c8=availability_raw8.detach(),
            availability_raw_field=availability_raw.detach(),
            availability_normalized_field64=availability_normalized64.detach(),
            availability_centered_c=zA.detach(), gain=gain.detach(),
            separable_consumer=True,
            consumer_argument='b+w*zAB+v*zA; both fields share the same fixed fit24 mean/std',
            g0=g0.detach() if original_trace else None,
            adapted=result.detach() if original_trace else None)
        return result

    def forward(self, image, coordinate=None, alpha=None, boundary_override=None,
                diagnostic=False, trace_interaction=False):
        if self.boundary_arm == 'AB' and isinstance(boundary_override, dict) and 'raw_c' in boundary_override:
            raise ValueError('Separable AB requires A/B fields; raw_c alone cannot identify zA')
        return super().forward(image, coordinate, alpha, boundary_override, diagnostic, trace_interaction)
