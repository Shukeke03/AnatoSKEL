"""AnatoSKEL composition: global detector, local CT encoder and shape/pose backend."""
import torch
from torch import Tensor, nn

from anatoskel.config import ModelConfig
from anatoskel.models.backend import AnatomicalReconstructor
from anatoskel.models.detector import CoarseDetector
from anatoskel.models.encoder import ResUNet


class AnatoSKEL(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = (config or ModelConfig()).validate()
        self.detector = CoarseDetector(self.config)
        self.fine_encoder = ResUNet(self.config, fine=True)
        self.fine_mask_features = nn.Conv3d(self.config.fine_channels[0], self.config.mask_dim, 1)
        self.backend = AnatomicalReconstructor(self.config)

    def forward(self, volume: Tensor, affine: Tensor) -> dict:
        """Run whole-scan detection; explicit ROI preparation precedes reconstruction."""
        return self.encode_coarse(volume, affine)

    def encode_coarse(self, volume: Tensor, affine: Tensor) -> dict:
        return self.detector(volume, affine)

    def encode_fine(self, volume: Tensor, affine: Tensor) -> dict:
        return self.fine_encoder(volume, affine)

    def fine_masks(self, context: dict, queries: Tensor) -> Tensor:
        """One query per packed bone ROI; predicted masks also condition refinement."""
        embedding = self.detector.mask_embeddings(queries).reshape(-1, self.config.mask_dim)
        features = self.fine_mask_features(context["decoded"])
        if len(embedding) != len(features):
            raise ValueError("Packed ROI count must equal the number of bone queries")
        return torch.einsum("bc,bcdhw->bdhw", embedding, features).unsqueeze(1)

    def reconstruct(self, context: dict, queries: Tensor, class_ids: Tensor,
                    anchors_mm: Tensor, refinements: bool = True) -> dict:
        return self.backend(context, queries, class_ids, anchors_mm, refinements=refinements)

    def field(self, context: dict, state: dict, points_world: Tensor) -> dict:
        return self.backend.field(context, state, points_world)
