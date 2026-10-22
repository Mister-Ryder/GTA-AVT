"""Validation-only runtime mechanism intervention; no weights/source edits."""
from pathlib import Path
import types


FORMULA = {
    'scale': '[q_scale, (1-q_scale)/4, (1-q_scale)/4, (1-q_scale)/4, (1-q_scale)/4]',
    'structure': '[q_structure, (1-q_structure)/2, (1-q_structure)/2]',
    'q_definition': 'each original image-only head\'s predicted p_empty',
}


def _availability_only_conditions(self, classifier, condition_mode):
    import torch
    if condition_mode != 'predicted':
        raise ValueError('availability_only cannot combine with other condition modes')
    scale = classifier['generation'].float().softmax(1)
    structure = classifier['structure'].float().softmax(1)
    q_scale = scale[:, :1]
    q_structure = structure[:, :1]
    return (torch.cat([q_scale, ((1. - q_scale) / 4.).expand(-1, 4)], dim=1),
            torch.cat([q_structure, ((1. - q_structure) / 2.).expand(-1, 2)], dim=1))


def attach_availability_only(model, method, split):
    if method != 'faithful_soft' or split != 'val':
        raise ValueError('availability_only is validation-only and requires the trained faithful_soft model')
    if not hasattr(model, 'conditions'):
        raise ValueError('Expected original faithful conditions interface')
    model.conditions = types.MethodType(_availability_only_conditions, model)
    from reference_weights_support import file_hash
    return dict(name='availability_only', stage='runtime-validation-diagnostic', separately_trained_model=False,
                changes_weights=False, uses_GT=False, formula=FORMULA,
                intervention_source_sha256=file_hash(Path(__file__)))
