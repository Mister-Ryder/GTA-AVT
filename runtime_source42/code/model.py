"""GTA-AVT-v1: an explicitly adapted GTA topology, not the original airway model.

Preserved design: separate image-only attribute encoder, deep scale guidance,
middle structure guidance, optional early spatial guidance, and final guidance
fusion. Replaced design: soft physical-radius/structure mixtures instead of
hard depth/shape argmax, GroupNorm for small batches, bounded residual modulation.
The common segmentation backbone is identical in every comparison.
"""
import torch
from torch import nn
from torch.nn import functional as F

SOFT_FILM_METHODS = ('baseline', 'multitask', 'gta_constant', 'gta_scale', 'gta_structure', 'gta_joint', 'gta_scan')
from faithful_model import FaithfulGTAAVT, FAITHFUL_METHODS
METHODS = SOFT_FILM_METHODS + FAITHFUL_METHODS


def block(a, b):
    return nn.Sequential(nn.Conv3d(a, b, 3, padding=1, bias=False), nn.GroupNorm(8, b), nn.LeakyReLU(.01, inplace=True),
                         nn.Conv3d(b, b, 3, padding=1, bias=False), nn.GroupNorm(8, b), nn.LeakyReLU(.01, inplace=True))


class Backbone(nn.Module):
    def __init__(self, width):
        super().__init__()
        channels = [width * 2 ** i for i in range(4)]
        self.encoders = nn.ModuleList([block(1, channels[0])] + [block(a, b) for a, b in zip(channels[:-1], channels[1:])])
        self.up = nn.ModuleList([nn.ConvTranspose3d(a, b, 2, stride=2) for a, b in zip(channels[:0:-1], channels[-2::-1])])
        self.decoders = nn.ModuleList([block(2 * x, x) for x in channels[-2::-1]])
        self.output = nn.Conv3d(width, 1, 1)
        self.deep_supervision = nn.ModuleList([nn.Conv3d(width * 2, 1, 1), nn.Conv3d(width * 4, 1, 1)])


class AttributeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([block(1, 16), block(16, 32), block(32, 64), block(64, 128)])
        self.scale = nn.Linear(128, 5)
        self.structure = nn.Linear(128, 3)

    def forward(self, image):
        features = []
        for index, layer in enumerate(self.blocks):
            image = layer(F.max_pool3d(image, 2) if index else image)
            features.append(image)
        pooled = F.adaptive_avg_pool3d(image, 1).flatten(1)
        return features, self.scale(pooled), self.structure(pooled)


class Guidance(nn.Module):
    def __init__(self, channels, attribute_channels, conditions):
        super().__init__()
        self.projection = nn.Conv3d(attribute_channels, channels, 1)
        self.fusion = nn.Sequential(nn.Conv3d(2 * channels, channels, 1), nn.GroupNorm(8, channels), nn.LeakyReLU(.01, inplace=True))
        self.modulation = nn.Linear(conditions, channels * 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, feature, attribute, condition):
        attribute = F.interpolate(self.projection(attribute), size=feature.shape[2:], mode='trilinear', align_corners=False)
        fused = self.fusion(torch.cat([feature, attribute], 1))
        gamma, beta = self.modulation(condition).chunk(2, 1)
        gamma, beta = [x[:, :, None, None, None] for x in (gamma, beta)]
        return feature + .2 * (fused * (1. + torch.tanh(gamma)) + torch.tanh(beta))


class SoftFiLMGTAAVT(nn.Module):
    def __init__(self, method='gta_joint', width=32):
        super().__init__()
        if method not in SOFT_FILM_METHODS:
            raise ValueError(method)
        self.method = method
        # Instantiate first so the common backbone has paired initialization.
        self.backbone = Backbone(width)
        if method != 'baseline':
            self.attributes = AttributeEncoder()
            self.scale_guidance = Guidance(width * 4, 128, 5)
            self.structure_guidance = Guidance(width * 2, 64, 3)
            self.location_guidance = Guidance(width * 2, 32, 3)
            self.final_guidance = nn.Conv3d(16, width, 1)
            self.final_fusion = nn.Sequential(nn.Conv3d(width * 2, width, 1), nn.GroupNorm(8, width), nn.LeakyReLU(.01, inplace=True))
            self.shared_scale_head = nn.Linear(width * 8, 5)
            self.shared_structure_head = nn.Linear(width * 8, 3)

    def forward(self, image, coordinate=None, condition_mode='predicted'):
        if any(x % 8 for x in image.shape[2:]):
            raise ValueError('Patch dimensions must be divisible by8')
        output = {}
        if self.method != 'baseline':
            attributes, scale, structure = self.attributes(image)
            output.update(scale_logits=scale, structure_logits=structure)
            scale_condition, structure_condition = scale.softmax(1), structure.softmax(1)
            if self.method in ('gta_constant', 'gta_structure', 'multitask') or condition_mode == 'constant':
                scale_condition = torch.full_like(scale_condition, 1. / 5)
            if self.method in ('gta_constant', 'gta_scale', 'multitask') or condition_mode == 'constant':
                structure_condition = torch.full_like(structure_condition, 1. / 3)
            if condition_mode == 'permuted':
                scale_condition, structure_condition = scale_condition.flip(1), structure_condition.flip(1)
        guided = self.method not in ('baseline', 'multitask')
        features = []
        x = image
        for index, layer in enumerate(self.backbone.encoders):
            x = layer(F.max_pool3d(x, 2) if index else x)
            if guided and index == 1 and self.method == 'gta_scan':
                if coordinate is None:
                    raise ValueError('Scan-coordinate ablation requires CT metadata coordinates')
                x = self.location_guidance(x, attributes[1], coordinate)
            features.append(x)
        decoded = []
        if self.method == 'multitask':
            pooled = F.adaptive_avg_pool3d(x, 1).flatten(1)
            output['scale_logits'] = self.shared_scale_head(pooled)
            output['structure_logits'] = self.shared_structure_head(pooled)
        for index, (up, layer) in enumerate(zip(self.backbone.up, self.backbone.decoders)):
            x = layer(torch.cat([up(x), features[-index - 2]], 1))
            if guided and index == 0:
                x = self.scale_guidance(x, attributes[3], scale_condition)
            if guided and index == 1:
                x = self.structure_guidance(x, attributes[2], structure_condition)
            decoded.append(x)
        if guided:
            guidance = self.final_guidance(attributes[0])
            x = x + .2 * self.final_fusion(torch.cat([x, guidance], 1))
        output['logits'] = self.backbone.output(x)
        output['aux_logits'] = [self.backbone.deep_supervision[0](decoded[1]),
                                self.backbone.deep_supervision[1](decoded[0])]
        return output


def GTAAVT(method='gta_joint', width=32):
    """Stable constructor used by train/evaluate; model families stay explicit."""
    if method in FAITHFUL_METHODS:
        return FaithfulGTAAVT(method, width)
    return SoftFiLMGTAAVT(method, width)
