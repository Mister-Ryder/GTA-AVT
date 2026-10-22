"""Traceable original-operator GTA migration to AVT.

Vendored modules are byte-identical. This wrapper follows the active historical
enhanced decoder path, with no-space/no-Gaussian as explicit component ablations.
Hard variants retain detached argmax/confidence arithmetic. Soft variants use
the SAME GAAM/SAAM convolutional operators with AVT composition arithmetic.
Max mixture mass is never described here as calibrated uncertainty.
"""
import torch
from torch import nn
from torch.nn import functional as F
from historical.func.AnatomicalClassifier import AirwayAnatomicalClassifier
from historical.func.AnatomicalFusionModules import (
    AnatomicalGuidanceModule, GenerationGuidanceModule, StructureGuidanceModule)
from historical.func.model_arch_model10 import SegAirwayModel

FAITHFUL_METHODS = (
    'faithful_baseline', 'faithful_concat', 'faithful_constant', 'faithful_joint',
    'faithful_scale', 'faithful_structure', 'faithful_soft', 'faithful_soft_scale',
    'faithful_soft_structure', 'faithful_soft_constant')


def foreground_logit(two_class_logits):
    return two_class_logits[:, 1:2] - two_class_logits[:, 0:1]


def radius_to_historical_generation(classes):
    # The original fine-vessel multiplier tests gen_pred>=5. Physical radius,
    # not graph depth, now determines which AVT class activates that operator.
    table = torch.tensor([0, 5, 1, 2, 3], device=classes.device, dtype=torch.long)
    return table[classes]


def junction_to_historical_structure(classes):
    table = torch.tensor([0, 1, 3], device=classes.device, dtype=torch.long)
    return table[classes]


class FaithfulGTAAVT(nn.Module):
    def __init__(self, method='faithful_joint', width=32):
        super().__init__()
        if method not in FAITHFUL_METHODS or width != 32:
            raise ValueError('Historical anchor requires a faithful method and fixed original width32')
        self.method = method
        self.use_prior = method != 'faithful_baseline'
        # Common backbone FIRST preserves paired B/C/A initialization.
        self.base_model = SegAirwayModel(in_channels=1, out_channels=2)
        # Construct same-shaped heads before the6-vs22 input projection so even
        # B/C/A deep-supervision and final32->2 weights begin identically.
        self.ds_heads = nn.ModuleList([nn.Conv3d(64, 2, 1), nn.Conv3d(128, 2, 1)])
        output_projection = nn.Conv3d(32, 2, 1)
        self.anatomical_output_fusion = nn.Sequential(
            nn.Conv3d(22 if self.use_prior else 6, 32, 3, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True), output_projection)
        if self.use_prior:
            self.classifier = AirwayAnatomicalClassifier(
                in_channels=1, generation_classes=5, structure_classes=3, location_classes=7)
            # C registers the same objects to make ALL C/A initialization exact,
            # but executes only their original base fusion. Specialized kernels
            # are inactive in C and excluded from its active-parameter report.
            self.generation_guidance = GenerationGuidanceModule(128, 256)
            self.structure_guidance = StructureGuidanceModule(64, 128)

    def conditions(self, classifier, condition_mode):
        soft = self.method.startswith('faithful_soft')
        if soft:
            # FP32 composition arithmetic avoids tiny softmax derivatives being
            # rounded to zero in half-precision conditional scalars.
            scale = classifier['generation'].float().softmax(1)
            structure = classifier['structure'].float().softmax(1)
        else:
            with torch.no_grad():
                scale = classifier['generation'].softmax(1)
                structure = classifier['structure'].softmax(1)
        if self.method in ('faithful_constant', 'faithful_soft_constant') or condition_mode == 'constant':
            scale, structure = torch.full_like(scale, 1. / 5), torch.full_like(structure, 1. / 3)
        if self.method in ('faithful_structure', 'faithful_soft_structure'):
            scale = torch.full_like(scale, 1. / 5)
        if self.method in ('faithful_scale', 'faithful_soft_scale'):
            structure = torch.full_like(structure, 1. / 3)
        if condition_mode == 'permuted':
            scale, structure = scale.flip(1), structure.flip(1)
        return scale, structure

    def soft_generation(self, feature, attribute, probability):
        operator = self.generation_guidance
        fused = AnatomicalGuidanceModule.forward(operator, feature, attribute)
        enhancement = operator.generation_enhance(fused)
        availability_gate = operator.uncertainty_gate(fused)
        empty, fine = probability[:, 0, None, None, None, None], probability[:, 1, None, None, None, None]
        enhanced = fused * enhancement * (1. + .2 * fine)
        # Original learned gate is retained but now multiplies predicted empty
        # composition. It is an availability gate, not calibrated uncertainty.
        return enhanced * (1. - availability_gate * empty) + fused * availability_gate * empty

    def soft_structure(self, feature, attribute, probability):
        operator = self.structure_guidance
        fused = AnatomicalGuidanceModule.forward(operator, feature, attribute)
        morphology = operator.morph_fusion(torch.cat([conv(fused) for conv in operator.morphology_convs], 1))
        empty = probability[:, 0, None, None, None, None]
        junction = probability[:, 2, None, None, None, None]
        return fused + (1. - empty) * morphology * (1. + .1 * junction)

    def forward(self, image, coordinate=None, condition_mode='predicted'):
        if image.ndim != 5 or image.shape[1] != 1 or any(n % 16 for n in image.shape[2:]):
            raise ValueError('Historical classifier requires single-channel 3D patches divisible by16')
        if condition_mode not in ('predicted', 'constant', 'permuted'):
            raise ValueError('Unknown condition mode')
        classifier = self.classifier(image) if self.use_prior else None
        if self.use_prior:
            scale_probability, structure_probability = self.conditions(classifier, condition_mode)
            with torch.no_grad():
                generation = radius_to_historical_generation(scale_probability.argmax(1))
                structure = junction_to_historical_structure(structure_probability.argmax(1))
        features = []
        x = image
        for encoder in self.base_model.encoders:
            x = encoder(x)
            features.append(x)
        features.reverse()
        decoded = []
        for index, decoder in enumerate(self.base_model.decoders):
            x = decoder(features[index + 1], x)
            if self.use_prior and index < 2:
                attribute = classifier['encoder_features'][2 - index]
                if self.method == 'faithful_concat':
                    operator = self.generation_guidance if index == 0 else self.structure_guidance
                    x = AnatomicalGuidanceModule.forward(operator, x, attribute)
                elif self.method.startswith('faithful_soft'):
                    x = (self.soft_generation(x, attribute, scale_probability) if index == 0 else
                         self.soft_structure(x, attribute, structure_probability))
                elif index == 0:
                    x = self.generation_guidance(x, attribute, generation, scale_probability)
                else:
                    x = self.structure_guidance(x, attribute, structure, structure_probability)
            decoded.append(x)
        fused = self.base_model.fusion_module_decoder(decoded)
        if self.use_prior:
            guidance = F.interpolate(classifier['guidance_map'], size=fused.shape[2:], mode='trilinear', align_corners=False)
            fused = torch.cat([fused, guidance], 1)
        output = {'logits': foreground_logit(self.anatomical_output_fusion(fused)),
                  'aux_logits': [foreground_logit(self.ds_heads[0](decoded[1])),
                                 foreground_logit(self.ds_heads[1](decoded[0]))]}
        if self.use_prior:
            output.update(scale_logits=classifier['generation'], structure_logits=classifier['structure'])
        return output
