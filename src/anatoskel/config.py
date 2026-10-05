"""Explicit architecture presets; smoke is for software checks, not research results."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    coarse_channels: tuple[int, ...] = (16, 32, 64, 128, 256)
    fine_channels: tuple[int, ...] = (32, 64, 128, 256, 320)
    coarse_blocks: tuple[int, ...] = (1, 2, 2, 2, 2)
    fine_blocks: tuple[int, ...] = (2, 2, 2, 2, 2)
    query_dim: int = 256
    heads: int = 8
    decoder_layers: int = 4
    num_queries: int = 32
    num_classes: int = 26
    mask_dim: int = 64
    beta_dim: int = 64
    class_dim: int = 16
    field_hidden: int = 256
    sample_channels: int = 32
    attention_blocks: int = 2
    window_size: int = 4
    max_global_tokens: int = 1024
    refinement_rounds: int = 2
    refinement_points: int = 2048
    distance_scale_mm: float = 8.0
    coordinate_scale_mm: float = 100.0
    gradient_checkpointing: bool = True

    @classmethod
    def smoke(cls):
        return cls(coarse_channels=(8, 16, 24, 32, 32),
                   fine_channels=(8, 16, 24, 32, 40),
                   coarse_blocks=(1, 1, 1, 1, 1), fine_blocks=(1, 1, 1, 1, 1),
                   query_dim=32, heads=4, num_queries=4, mask_dim=8,
                   beta_dim=8, class_dim=4, field_hidden=32, sample_channels=4,
                   attention_blocks=1, max_global_tokens=64, refinement_points=32,
                   gradient_checkpointing=False)

    def validate(self):
        for channels, counts in ((self.coarse_channels, self.coarse_blocks),
                                 (self.fine_channels, self.fine_blocks)):
            if len(channels) != 5 or len(counts) != 5 or min(counts) < 1:
                raise ValueError("Encoders require five nonempty stages")
        if any(c % self.heads for c in (self.query_dim, self.coarse_channels[-1],
                                        self.fine_channels[-2], self.fine_channels[-1])):
            raise ValueError("Attention widths must be divisible by heads")
        if self.distance_scale_mm <= 0 or self.coordinate_scale_mm <= 0:
            raise ValueError("Coordinate and distance scales must be positive")
        if self.num_classes != 26:
            raise ValueError("This release uses the 26 VerSe vertebral identities")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown model settings: {sorted(unknown)}")
        values = dict(values)
        for key in ("coarse_channels", "fine_channels", "coarse_blocks", "fine_blocks"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values).validate()


def load_config(path: str | Path) -> dict:
    with open(path) as stream:
        result = yaml.safe_load(stream)
    if not isinstance(result, dict):
        raise ValueError("Configuration must be a mapping")
    return result
