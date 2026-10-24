"""Frozen common G0 with a detached CT-only local boundary condition field.

This is a new diagnostic operator, not evidence of GTA efficacy. Only the
existing128 interaction weights and128 biases may learn. Independent heads,
encoder, all G0 parameters and all normalization buffers remain fixed/eval.
"""
import math
import torch
from torch.nn import functional as F
from ordinal_stage_model import OrdinalStageModel
from centered_frozen_ordinal_stage_model import CenteredFrozenOrdinalStageModel


HEAD_SCHEMA = 'GTA-AVT-boundary-independent-heads-v1'
ARMS = ('C0', 'A_only', 'AB')
HEAD_BUFFERS = ('boundary_A_weight', 'boundary_A_bias', 'boundary_feature_mean',
    'boundary_feature_std', 'boundary_B_coefficients', 'boundary_B_fit_constant',
    'boundary_field_mean', 'boundary_field_std')


class BoundaryFieldStageModel(CenteredFrozenOrdinalStageModel):
    def __init__(self, arm='C0', alpha=0., q0=.2, s0=.2, gdf_seed=20261023):
        if arm not in ARMS:
            raise ValueError('Expected C0, A_only or AB boundary field arm')
        super().__init__('constant', True, alpha, q0, s0, gdf_seed)
        self.boundary_arm = arm
        # Deterministic buffers add no RNG use and no trainable parameters.
        self.register_buffer('boundary_A_weight', torch.zeros(64, dtype=torch.float64))
        self.register_buffer('boundary_A_bias', torch.zeros((), dtype=torch.float64))
        self.register_buffer('boundary_feature_mean', torch.zeros(64, dtype=torch.float64))
        self.register_buffer('boundary_feature_std', torch.ones(64, dtype=torch.float64))
        self.register_buffer('boundary_B_coefficients', torch.zeros(65, dtype=torch.float64))
        self.register_buffer('boundary_B_fit_constant', torch.zeros((), dtype=torch.float64))
        self.register_buffer('boundary_field_mean', torch.zeros((), dtype=torch.float64))
        self.register_buffer('boundary_field_std', torch.ones((), dtype=torch.float64))
        self._boundary_heads_installed = False
        self._boundary_request = None
        self._boundary_encoder_feature = None
        self._boundary_feature_hook = self.classifier.register_forward_hook(self._capture_boundary_feature)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.interaction_weight.requires_grad_(True)
        self.interaction_bias.requires_grad_(True)
        self.train(False)

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError('Training mode must be boolean')
        # Eval mode does not prevent autograd through the fixed decoder to w/b.
        super().train(False)
        return self

    def load_common_state(self, state):
        expected = set(self.state_dict())
        new = set(HEAD_BUFFERS) | {'interaction_weight', 'interaction_bias'}
        optional = {'gdf_head.weight', 'gdf_head.bias'}
        missing = expected - set(state)
        if set(state) - expected or not new <= missing or missing - new - optional:
            raise ValueError('Common checkpoint differs beyond boundary/interaction/GDF new state')
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or set(result.missing_keys) != missing:
            raise ValueError('Unexpected common checkpoint loading result')
        return result

    def install_heads(self, checkpoint, mean, std):
        """Install already-qualified fixed heads and predeclared fit24 AB moments.

        No files, GT labels, fitting or moment calculation occur in this model.
        mean/std are one common CT-predicted ALL512-cell reference for all arms.
        """
        if checkpoint.get('schema') != HEAD_SCHEMA or checkpoint.get('optimization_qualified') is not True:
            raise ValueError('Qualified independent-head checkpoint required')
        a, b = checkpoint['availability'], checkpoint['B']
        if a.get('dtype') != 'float64' or a.get('output') != 'sigmoid' or b.get('intercept_index') != 0 or b.get('clip') != [0., 1.] or b.get('refitted') is not False:
            raise ValueError('Prepared head semantics differ')
        mean, std = float(mean), float(std)
        if not math.isfinite(mean) or not 0. <= mean <= 1. or not math.isfinite(std) or std <= 0.:
            raise ValueError('Fixed finite fit24 mean and positive standard deviation required')
        raw = dict(boundary_A_weight=a['standardized_weights'], boundary_A_bias=a['bias'],
            boundary_feature_mean=a['feature_mean'], boundary_feature_std=a['feature_std'],
            boundary_B_coefficients=b['coefficients'], boundary_B_fit_constant=b['fit_constant'],
            boundary_field_mean=mean, boundary_field_std=std)
        values = {}
        for name, value in raw.items():
            buffer = getattr(self, name)
            value = torch.as_tensor(value, dtype=torch.float64, device=buffer.device)
            if value.shape != buffer.shape or not bool(torch.isfinite(value).all()):
                raise ValueError('Prepared buffer shape or finite check failed: ' + name)
            values[name] = value.detach()
        if bool((values['boundary_feature_std'] <= 0.).any()) or not 0. <= float(values['boundary_B_fit_constant']) <= 1.:
            raise ValueError('Invalid feature scale or B fit prior')
        with torch.no_grad():
            for name, value in values.items():
                getattr(self, name).copy_(value)
        self._boundary_heads_installed = True
        return self

    def _capture_boundary_feature(self, module, inputs, output):
        if self._boundary_request is not None:
            feature = output['encoder_features'][0]
            if tuple(feature.shape[1:]) != (64, 32, 32, 32):
                raise ValueError('Boundary head requires original128-patch encoder feature64x32^3')
            self._boundary_encoder_feature = feature.detach()

    @staticmethod
    def pool_boundary_features(feature):
        if tuple(feature.shape[1:]) != (64, 32, 32, 32):
            raise ValueError('Expected Bx64x32x32x32 frozen encoder feature')
        value = F.avg_pool3d(feature.float(), kernel_size=4, stride=4)
        return value.permute(0, 2, 3, 4, 1).contiguous().double()

    def _predict_boundary(self):
        if not self._boundary_heads_installed or self._boundary_encoder_feature is None:
            raise RuntimeError('Install fixed heads and run the normal CT classifier first')
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            feature = self.pool_boundary_features(self._boundary_encoder_feature)
            logits = ((feature - self.boundary_feature_mean) / self.boundary_feature_std) @ self.boundary_A_weight + self.boundary_A_bias
            availability = torch.sigmoid(logits).unsqueeze(1)
            ambiguity = (feature @ self.boundary_B_coefficients[1:] + self.boundary_B_coefficients[0]).clamp(0., 1.).unsqueeze(1)
        return availability.detach(), ambiguity.detach()

    @staticmethod
    def _external_field(value, reference):
        value = torch.as_tensor(value, dtype=torch.float64, device=reference.device)
        if value.shape != reference.shape or not bool(torch.isfinite(value).all()) or bool(((value < 0.) | (value > 1.)).any()):
            raise ValueError('Diagnostic field must be finite Bx1x8x8x8 in[0,1]')
        return value.detach()

    def soft_generation(self, feature, attribute, probability):
        request, own = self._stage_request, self._boundary_request
        if request is None or own is None:
            raise RuntimeError('Boundary interaction outside a forward request')
        alpha = request['alpha']
        request['alpha'] = 0.
        try:
            # Exactly the existing StageModel G0, bypassing BOTH scalar adapters.
            g0 = super(OrdinalStageModel, self).soft_generation(feature, attribute, probability)
        finally:
            request['alpha'] = alpha
        if g0.shape[1] != 128:
            raise RuntimeError('Expected original128-channel GAAM output')
        own_A, own_B = self._predict_boundary()
        if self.boundary_arm == 'C0':
            own_raw = torch.ones_like(own_A) * self.boundary_field_mean
        elif self.boundary_arm == 'A_only':
            own_raw = own_A * self.boundary_B_fit_constant
        else:
            own_raw = own_A * own_B
        consumed_A = None if self.boundary_arm == 'C0' else own_A
        consumed_B = (None if self.boundary_arm == 'C0' else
            torch.ones_like(own_B) * self.boundary_B_fit_constant if self.boundary_arm == 'A_only' else own_B)
        raw = own_raw
        override = own['override']
        if override is not None:
            if 'raw_c' in override:
                raw = self._external_field(override['raw_c'], own_raw)
                consumed_A = consumed_B = None
            else:
                consumed_A = self._external_field(override['A'], own_A) if 'A' in override else own_A
                default_B = torch.ones_like(own_B) * self.boundary_B_fit_constant if self.boundary_arm == 'A_only' else own_B
                consumed_B = self._external_field(override['B'], own_B) if 'B' in override else default_B
                raw = consumed_A * consumed_B
        with torch.cuda.amp.autocast(enabled=False):
            resized_raw = F.interpolate(raw.double(), size=g0.shape[2:], mode='trilinear', align_corners=False)
            if self.boundary_arm == 'C0' and override is None:
                # Exact constant-reference null, not an epsilon comparison.
                resized_raw = torch.ones_like(resized_raw) * self.boundary_field_mean
                normalized64 = torch.zeros_like(resized_raw)
            else:
                normalized64 = (resized_raw - self.boundary_field_mean) / self.boundary_field_std
            z = normalized64.float()
            argument = self.interaction_bias[None, :, None, None, None] + self.interaction_weight[None, :, None, None, None] * z
            gain = torch.tanh(argument.float())
        result = g0 if alpha == 0. else g0 + float(alpha) * g0 * gain
        normal_probability = self.last_condition_trace['scale_head_probability'].float().detach()
        self.last_interaction_trace = dict(raw_fine=normal_probability[:, 1:2],
            raw_ordinal=self.ordinal_condition(normal_probability).detach(),
            own_A=own_A, own_B=own_B, own_raw_c8=own_raw.detach(),
            consumed_A=consumed_A, consumed_B=consumed_B, consumed_raw_c8=raw.detach(),
            consumed_c=resized_raw.detach(), normalized_field64=normalized64.detach(), centered_c=z.detach(),
            gain=gain.detach(), alpha=alpha, g0_shape=tuple(g0.shape), g0_dtype=str(g0.dtype),
            g0=g0.detach() if own['trace_tensor'] else None,
            adapted=result.detach() if own['trace_tensor'] else None,
            diagnostic=own['diagnostic'], override_supplied=override is not None,
            boundary_arm=self.boundary_arm, conditioning='constant_G0_with_detached_boundary_field',
            readout='CT_predicted_availability_times_boundary_ambiguity',
            field_reference_mean=float(self.boundary_field_mean), field_reference_std=float(self.boundary_field_std),
            constant_reference_exact_null=self.boundary_arm == 'C0' and override is None,
            q0=.2, s0=.2, consumed_c_detached=True,
            interpolation='raw F64 c8 trilinear align_corners=False, then normalize F64, then cast z FP32')
        return result

    def forward(self, image, coordinate=None, alpha=None, boundary_override=None,
                diagnostic=False, trace_interaction=False):
        if not self._boundary_heads_installed:
            raise RuntimeError('Prepared heads and fixed fit24 field moments must be installed')
        if self._boundary_request is not None:
            raise RuntimeError('Reentrant boundary forward is unsupported')
        if boundary_override is not None:
            if not diagnostic or not isinstance(boundary_override, dict) or not boundary_override:
                raise ValueError('Explicit nonempty diagnostic boundary override required')
            if set(boundary_override) - {'A', 'B', 'raw_c'} or ('raw_c' in boundary_override and len(boundary_override) != 1):
                raise ValueError('Use raw_c alone, or A/B component overrides')
        self._boundary_request = dict(override=boundary_override, diagnostic=bool(diagnostic), trace_tensor=bool(trace_interaction))
        self._boundary_encoder_feature = None
        self.last_interaction_trace = None
        try:
            return super().forward(image, coordinate, condition_mode='constant', alpha=alpha,
                diagnostic=diagnostic, trace_interaction=trace_interaction)
        finally:
            self._boundary_encoder_feature = None
            self._boundary_request = None
