"""Original ordinal consumer with the entire common classifier fixed.

Only classifier parameter gradients and training mode change. Its encoder,
attention, all heads and guidance map remain in the original forward paths.
No tensor, parameter, projection, condition or interaction formula is added.
"""
from ordinal_stage_model import OrdinalStageModel


class FrozenOrdinalStageModel(OrdinalStageModel):
    def __init__(self, conditioning='constant', detach_conditions=True, alpha=0.,
                 q0=.2, s0=.2, gdf_seed=20261023):
        super().__init__(conditioning, detach_conditions, alpha, q0, s0, gdf_seed)
        self.classifier.requires_grad_(False)
        self.classifier.eval()

    def train(self, mode=True):
        super().train(mode)
        self.classifier.eval()
        return self
