"""Bounded three-axis routing of the same four existing SAAM branches.

This is a TRAIN diagnostic, not a deployable CT-only GTA model. The original
constant StageModel is the zero-adapter starting point. Only its existing
four SAAM branches, fusion projection, final two-channel projection and
576 zero coefficients may learn; every module stays in evaluation mode.
This only expands routing freedom; no new CNN, condition or oracle is added.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from stage_model import StageModel


ARMS = ('C0', 'A_only', 'AR')
PROBE_SITE = 'SAAM64_bounded_three_axis_branch_routing_with_partial_calibre_oracle'
COEFFICIENT_NAMES = ('calibre_bias', 'calibre_availability_weight',
                     'calibre_radius_weight')
MOMENT_BUFFERS = ('calibre_log_radius_mean', 'calibre_log_radius_std')
EXISTING_PARAMETER_NAMES = (
    'structure_guidance.morph_fusion.weight',
    'structure_guidance.morph_fusion.bias',
    'anatomical_output_fusion.3.weight',
    'anatomical_output_fusion.3.bias',
    'structure_guidance.morphology_convs.0.weight',
    'structure_guidance.morphology_convs.0.bias',
    'structure_guidance.morphology_convs.1.weight',
    'structure_guidance.morphology_convs.1.bias',
    'structure_guidance.morphology_convs.2.weight',
    'structure_guidance.morphology_convs.2.bias',
    'structure_guidance.morphology_convs.3.weight',
    'structure_guidance.morphology_convs.3.bias')


def valid_moments(mean, std):
    mean, std = float(mean), float(std)
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0.:
        raise ValueError('Fixed finite fit-only log-radius mean and positive SD required')
    return mean, std


class CalibreRoutingStageModel(StageModel):
    """F0 + alpha*g1*D1 + alpha*(1-|g1|)*(g2*D2+g3*D3).

    A=0 includes unsupported/neutral oracle sites; it is not a vessel-negative
    label. The upstream helper retains the original unknown/state/reason maps.
    All three arms use the identical trainable parameter names and optimizer
    groups. In C0 both conditional axes are zero; in A_only only radius is zero.
    """
    def __init__(self, arm='C0', alpha=0., gdf_seed=20261023):
        if arm not in ARMS:
            raise ValueError('Expected C0, A_only or AR calibre arm')
        super().__init__('constant', True, alpha, .2, .2, gdf_seed)
        self.calibre_arm = arm
        # No randomly initialized module or new RNG draw is introduced.
        self.calibre_bias = nn.Parameter(torch.zeros(3, 64, dtype=torch.float32))
        self.calibre_availability_weight = nn.Parameter(torch.zeros(3, 64, dtype=torch.float32))
        self.calibre_radius_weight = nn.Parameter(torch.zeros(3, 64, dtype=torch.float32))
        self.register_buffer('calibre_log_radius_mean', torch.zeros((), dtype=torch.float64))
        self.register_buffer('calibre_log_radius_std', torch.ones((), dtype=torch.float64))
        self._calibre_moments_installed = False
        self._calibre_request = None
        self.last_calibre_trace = None
        self._configure_trainable_parameters()
        self.train(False)

    def _configure_trainable_parameters(self):
        named = dict(self.named_parameters())
        expected = set(EXISTING_PARAMETER_NAMES) | set(COEFFICIENT_NAMES)
        if not expected <= set(named):
            raise RuntimeError('Original SAAM/final projection names differ')
        if sum(named[n].numel() for n in EXISTING_PARAMETER_NAMES[:2]) != 4160:
            raise RuntimeError('Expected existing64x64+bias SAAM projection')
        if sum(named[n].numel() for n in EXISTING_PARAMETER_NAMES[2:4]) != 66:
            raise RuntimeError('Expected existing32-to2 final projection')
        if sum(named[n].numel() for n in EXISTING_PARAMETER_NAMES[4:]) != 110656:
            raise RuntimeError('Expected original four64-to16+bias3x3x3 branches')
        branches = self.structure_guidance.morphology_convs
        if (len(branches) != 4 or any(tuple(c.weight.shape) != (16, 64, 3, 3, 3) or
                tuple(c.bias.shape) != (16,) for c in branches) or
                [tuple(c.dilation) for c in branches] != [(1,1,1), (1,1,1), (2,2,2), (3,3,3)]):
            raise RuntimeError('Original branch order/shapes/receptive fields differ')
        if (any(tuple(named[n].shape) != (3, 64) for n in COEFFICIENT_NAMES) or
                sum(named[n].numel() for n in COEFFICIENT_NAMES) != 576):
            raise RuntimeError('Expected three3x64 routing coefficient tensors')
        for name, parameter in named.items():
            parameter.requires_grad_(name in expected)

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError('Training mode must be boolean')
        # Eval fixes all BN/Dropout behavior while permitting these conv/gate
        # derivatives and the downstream input Jacobian to participate.
        super().train(False)
        return self

    def active_parameter_names(self):
        return tuple(name for name, value in self.named_parameters() if value.requires_grad)

    def trainable_parameter_names(self):
        return self.active_parameter_names()

    def coadapt_parameter_groups(self):
        """Return parameters only; the external trainer owns LR/decay/optimizer."""
        named = dict(self.named_parameters())
        return {'existing': [named[n] for n in EXISTING_PARAMETER_NAMES],
                'coefficients': [named[n] for n in COEFFICIENT_NAMES]}

    def trainable_groups(self):
        groups = self.coadapt_parameter_groups()
        return {'shared': groups['existing'], 'gates': groups['coefficients']}

    def load_common_state(self, state):
        expected = set(self.state_dict())
        added = set(COEFFICIENT_NAMES) | set(MOMENT_BUFFERS)
        optional = {'gdf_head.weight', 'gdf_head.bias'}
        missing = expected - set(state)
        if set(state) - expected or not added <= missing or missing - added - optional:
            raise ValueError('Common checkpoint differs beyond calibre coefficients/moments/GDF')
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or set(result.missing_keys) != missing:
            raise ValueError('Unexpected common checkpoint loading result')
        return result

    def install_moments(self, log_radius_mean, log_radius_std):
        """Install one precomputed fit24 physical-log-radius reference for all arms."""
        mean, std = valid_moments(log_radius_mean, log_radius_std)
        with torch.no_grad():
            self.calibre_log_radius_mean.fill_(mean)
            self.calibre_log_radius_std.fill_(std)
        self._calibre_moments_installed = True
        return self

    @staticmethod
    def _oracle(value, reference, name):
        if not isinstance(value, dict) or set(value) != {'A', 'logR', 'valid'}:
            raise ValueError(name + ' must contain exactly A/logR/valid')
        shape = (reference.shape[0], 1) + tuple(reference.shape[2:])
        device = reference.device
        availability = torch.as_tensor(value['A'], device=device, dtype=torch.float32).detach()
        log_radius = torch.as_tensor(value['logR'], device=device, dtype=torch.float64).detach()
        valid = torch.as_tensor(value['valid'], device=device)
        if valid.dtype != torch.bool:
            raise ValueError('Oracle validity must be explicit boolean')
        valid = valid.detach()
        if any(tuple(v.shape) != shape for v in (availability, log_radius, valid)):
            raise ValueError('Oracle must match actual Bx1x64x64x64 SAAM grid')
        if not bool(torch.isfinite(availability).all()) or not bool(torch.isfinite(log_radius).all()):
            raise ValueError('Oracle inputs must be finite, with explicit neutral0 for unknowns')
        if bool(((availability != 0.) & (availability != 1.)).any()):
            raise ValueError('Oracle A must be exact binary applicability, not a probability')
        if bool(((availability == 1.) & ~valid).any()) or bool((log_radius[~valid] != 0.).any()):
            raise ValueError('A-positive requires measured radius; invalid radius must be neutral0')
        radius = torch.exp(log_radius[valid])
        if not bool(torch.isfinite(radius).all()) or bool((radius <= 0.).any()):
            raise ValueError('Valid logR must represent finite positive radius in mm')
        return availability, log_radius, valid

    @staticmethod
    def context_direction(h, operator, probability):
        """Existing local/context branch contrast, with both branch h and W differentiable."""
        if h.ndim != 5 or h.shape[1] != 64:
            raise RuntimeError('Expected four original16-channel branch responses')
        if (tuple(operator.weight.shape) != (64, 64, 1, 1, 1) or
                tuple(operator.kernel_size) != (1, 1, 1) or
                tuple(operator.stride) != (1, 1, 1) or
                tuple(operator.padding) != (0, 0, 0) or
                tuple(operator.dilation) != (1, 1, 1) or operator.groups != 1):
            raise RuntimeError('Original64x64 morph_fusion1x1 differs')
        p = probability.detach().float()
        if (p.shape != (h.shape[0], 3) or not bool(torch.isfinite(p).all()) or
                not torch.equal(p, torch.full_like(p, 1. / 3.))):
            raise RuntimeError('Original constant three-class structure probability required')
        with torch.cuda.amp.autocast(enabled=False):
            h = h.float()
            signed = torch.cat((-h[:, :16], -h[:, 16:32],
                                h[:, 32:48], h[:, 48:64]), dim=1)
            # W stays on the graph. Bias is intentionally excluded from D,
            # while the ordinary F0 path retains its original W AND bias.
            contrast = F.conv3d(signed, operator.weight.float(), bias=None)
            empty = p[:, 0, None, None, None, None]
            junction = p[:, 2, None, None, None, None]
            direction = ((1. - empty) * (1. + .1 * junction)) * contrast
        if not bool(torch.isfinite(direction).all()):
            raise RuntimeError('Nonfinite co-adapted context direction')
        return direction

    def soft_structure(self, feature, attribute, probability):
        own = self._calibre_request
        if own is None:
            raise RuntimeError('Oracle co-adaptation outside an explicit request')
        operator = self.structure_guidance.morph_fusion
        captured = {}

        def capture(module, inputs):
            if len(inputs) != 1 or 'h' in captured:
                raise RuntimeError('Expected exactly one original morph_fusion call')
            captured['h'] = inputs[0]

        hook = operator.register_forward_pre_hook(capture)
        try:
            F0 = super().soft_structure(feature, attribute, probability)
            if 'h' not in captured:
                raise RuntimeError('Original SAAM fusion input was not captured')
            h = captured.pop('h')
        finally:
            hook.remove()
            captured.clear()
        if tuple(F0.shape[1:]) != (64, 64, 64, 64):
            raise RuntimeError('Expected actual128-patch SAAM output64x64x64x64')
        own_A, own_R, own_valid = self._oracle(own['oracle'], F0, 'Own oracle')
        consumed_R, consumed_valid = own_R, own_valid
        if own['radius_override'] is not None:
            override = own['radius_override']
            if not isinstance(override, dict) or set(override) != {'logR', 'valid'}:
                raise ValueError('Radius override contains exactly logR/valid and preserves own A')
            _, consumed_R, consumed_valid = self._oracle(
                dict(A=own_A, logR=override['logR'], valid=override['valid']), F0,
                'Diagnostic radius override')
        with torch.cuda.amp.autocast(enabled=False):
            zA = torch.zeros_like(own_A) if self.calibre_arm == 'C0' else own_A
            if self.calibre_arm == 'AR':
                standardized = (consumed_R - self.calibre_log_radius_mean) / self.calibre_log_radius_std
                zR = torch.where(own_A == 1., standardized, torch.zeros_like(standardized)).float()
            else:
                zR = torch.zeros_like(own_A)
            gains = []
            for axis in range(3):
                argument = (self.calibre_bias[axis][None, :, None, None, None] +
                            self.calibre_availability_weight[axis][None, :, None, None, None] * zA +
                            self.calibre_radius_weight[axis][None, :, None, None, None] * zR)
                gains.append(torch.tanh(argument.float()))
            g1, g2, g3 = gains
            remaining = 1. - g1.abs()
        # D1 uses the parent's exact signed-input/FP32 calculation order.
        D1 = self.context_direction(h, operator, probability)
        D2, D3 = self.within_pair_directions(h, operator, probability)
        del h
        alpha = self._stage_request['alpha']
        if alpha == 0.:
            result = F0
        else:
            # Keep the old axis1 arithmetic order, then add the two new axes.
            result = F0 + float(alpha) * g1 * D1
            result = result + float(alpha) * remaining * (g2 * D2 + g3 * D3)
        trace_tensor = own['trace']
        route_stats = self.route_statistics(g1, g2, g3, alpha)
        self.last_calibre_trace = dict(
            arm=self.calibre_arm, probe_site=PROBE_SITE, alpha=alpha,
            oracle_only=True, normal_CT_inference_supported=False,
            own_A=own_A, own_logR=own_R, own_valid=own_valid,
            consumed_logR=consumed_R, consumed_valid=consumed_valid,
            A=own_A, logR=consumed_R,
            zA=zA.detach(), zR=zR.detach(),
            zA_mean=float(zA.detach().mean()), zA_std=float(zA.detach().std(unbiased=False)),
            zR_mean=float(zR.detach().mean()), zR_std=float(zR.detach().std(unbiased=False)),
            # Legacy gain/D fields refer ONLY to axis1, never total routing gain.
            gain=g1.detach() if trace_tensor else None,
            gain_absmean=float(g1.detach().abs().mean()),
            gain_maxabs=float(g1.detach().abs().max()),
            gain_semantics='Legacy axis1 g1 only; not total routing gain',
            D_shape=tuple(D1.shape), D_dtype=str(D1.dtype),
            D_RMS=float(D1.detach().square().mean().sqrt()),
            D_semantics='Legacy axis1 D1 only',
            g1=g1.detach() if trace_tensor else None,
            D=D1.detach() if trace_tensor else None,
            D1=D1.detach() if trace_tensor else None,
            axis_gain_absmean=[float(g.detach().abs().mean()) for g in gains],
            axis_gain_maxabs=[float(g.detach().abs().max()) for g in gains],
            axis_gain_RMS=[float(g.detach().square().mean().sqrt()) for g in gains],
            axis_direction_RMS=[float(d.detach().square().mean().sqrt()) for d in (D1,D2,D3)],
            route_coeff_min=route_stats['min'],
            route_coeff_max=route_stats['max'],
            route_coeff_sum_residual_maxabs=route_stats['sum_residual_maxabs'],
            route_coeff_interpretation='Real/same-precision branch coefficients; AMP F0/D is not a bitwise expansion',
            F0_shape=tuple(F0.shape), F0_dtype=str(F0.dtype),
            radius_override_supplied=own['radius_override'] is not None,
            radius_override_preserves_own_A=True,
            log_radius_mean=float(self.calibre_log_radius_mean),
            log_radius_std=float(self.calibre_log_radius_std),
            D_weight_differentiable=operator.weight.requires_grad,
            unknown_handling='Explicit applicability0 and neutral radius axis0; not a background label')
        return result

    @staticmethod
    def within_pair_directions(h, operator, probability):
        """Original W/h, same constant factor, no fusion bias or detach.

        The ordinary context_direction is evaluated first and validates all
        original h/W/probability structure. These are the two additional axes.
        """
        if h.ndim != 5 or h.shape[1] != 64 or tuple(operator.weight.shape) != (64,64,1,1,1):
            raise RuntimeError('Expected the original four16-channel branches and fusion W')
        p = probability.detach().float()
        if (p.shape != (h.shape[0], 3) or not bool(torch.isfinite(p).all()) or
                not torch.equal(p, torch.full_like(p, 1. / 3.))):
            raise RuntimeError('Original constant three-class structure probability required')
        with torch.cuda.amp.autocast(enabled=False):
            h = h.float()
            empty = p[:, 0, None, None, None, None]
            junction = p[:, 2, None, None, None, None]
            factor = (1. - empty) * (1. + .1 * junction)
            D2 = factor * F.conv3d(torch.cat((h[:, :16], -h[:, 16:32]), dim=1),
                                  operator.weight[:, :32].float(), bias=None)
            D3 = factor * F.conv3d(torch.cat((h[:, 32:48], -h[:, 48:64]), dim=1),
                                  operator.weight[:, 32:].float(), bias=None)
        if not bool(torch.isfinite(D2).all()) or not bool(torch.isfinite(D3).all()):
            raise RuntimeError('Nonfinite within-pair routing direction')
        return D2, D3

    @staticmethod
    def route_statistics(g1, g2, g3, alpha):
        """Scalar FP32 diagnostics without retaining four full coefficient maps."""
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            g1, g2, g3 = g1.detach().float(), g2.detach().float(), g3.detach().float()
            remaining = 1. - g1.abs()
            total = torch.zeros_like(g1)
            minimum, maximum = float('inf'), -float('inf')
            for branch in range(4):
                difference = g2 if branch < 2 else g3
                coefficient = remaining * difference
                if branch % 2:
                    coefficient.neg_()
                coefficient.add_(g1, alpha=-1. if branch < 2 else 1.)
                coefficient.mul_(float(alpha)).add_(1.)
                minimum = min(minimum, float(coefficient.min()))
                maximum = max(maximum, float(coefficient.max()))
                total.add_(coefficient)
            residual = float((total - 4.).abs().max())
        return dict(min=minimum, max=maximum, sum_residual_maxabs=residual)

    def forward(self, image, coordinate=None, oracle=None, oracle_mode=False,
                radius_override=None, diagnostic=False, alpha=None, trace=False):
        if oracle_mode is not True:
            raise ValueError('This diagnostic requires explicit oracle_mode=True, not CT-only inference')
        if not self._calibre_moments_installed:
            raise RuntimeError('Install one fixed fit-only physical-log-radius reference first')
        if oracle is None:
            raise ValueError('Explicit partial oracle required for every arm, including C0')
        if radius_override is not None and (self.calibre_arm != 'AR' or diagnostic is not True):
            raise ValueError('Only diagnostic AR radius override is supported')
        if self._calibre_request is not None:
            raise RuntimeError('Reentrant calibre co-adaptation is unsupported')
        self._calibre_request = dict(oracle=oracle, radius_override=radius_override,
                                    diagnostic=bool(diagnostic), trace=bool(trace))
        self.last_calibre_trace = None
        try:
            return super().forward(image, coordinate, condition_mode='constant', alpha=alpha)
        finally:
            self._calibre_request = None
