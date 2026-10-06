"""Versioned architecture specification, separate from v0.1 checkpoints."""
from dataclasses import asdict, dataclass, fields


@dataclass
class V1Config:
    architecture: str = "anatoskel_v1"
    feature_size: int = 48
    swin_depths: tuple[int, ...] = (2, 2, 2, 2)
    swin_heads: tuple[int, ...] = (3, 6, 12, 24)
    window_size: int = 7
    query_dim: int = 384
    heads: int = 6
    ffn_hidden: int = 1024
    num_queries: int = 64
    num_classes: int = 26
    context_tokens: int = 256
    context_layers: int = 2
    global_layers: int = 3
    local_layers: int = 3
    query_points: int = 16
    latent_tokens: int = 256
    latent_dim: int = 128
    shape_width: int = 256
    shape_heads: int = 8
    shape_layers: int = 4
    shape_ffn_hidden: int = 704
    latent_predictor_layers: int = 2
    diffusion_layers: int = 6
    diffusion_train_steps: int = 1000
    diffusion_sample_steps: int = 20
    adapter_tail_steps: int = 4
    adapter_hidden: int = 96
    shape_scale_mm: float = 64.0
    tsdf_truncation_mm: float = 8.0
    occupancy_temperature_mm: float = 1.0
    field_hidden: int = 128
    detail_channels: int = 32
    contrastive_dim: int = 128
    local_size: int = 128
    local_spacing_mm: float = 1.0
    tile_overlap: int = 32
    global_max_size: int = 128
    gradient_checkpointing: bool = True
    pretrained_encoder: str | None = None

    def validate(self):
        if self.architecture != "anatoskel_v1":
            raise ValueError("Expected anatoskel_v1 architecture")
        if len(self.swin_depths) != 4 or len(self.swin_heads) != 4:
            raise ValueError("Swin needs four stages")
        if self.feature_size % 12:
            raise ValueError("VoCo-compatible feature_size must be divisible by 12")
        if any(self.feature_size * 2**i % h for i, h in enumerate(self.swin_heads)):
            raise ValueError("Swin widths must be divisible by their attention heads")
        if self.query_dim % self.heads or self.shape_width % self.shape_heads:
            raise ValueError("Attention widths must be divisible by heads")
        if not 0 < self.adapter_tail_steps <= self.diffusion_sample_steps:
            raise ValueError("Invalid adapter sampling tail")
        if self.local_size % 32 or not 0 <= self.tile_overlap < self.local_size:
            raise ValueError("Tile size must be divisible by 32 and exceed overlap")
        if min(self.shape_scale_mm, self.tsdf_truncation_mm,
               self.local_spacing_mm, self.occupancy_temperature_mm) <= 0:
            raise ValueError("Physical scales must be positive")
        if self.num_classes != 26:
            raise ValueError("VerSe identity head needs 26 classes")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown v1 model settings: {sorted(unknown)}")
        for key in ("swin_depths", "swin_heads"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values).validate()

    @classmethod
    def smoke(cls):
        return cls(feature_size=12, swin_depths=(1, 1, 1, 1), query_dim=48,
                   heads=6, ffn_hidden=128, num_queries=4, context_tokens=8,
                   context_layers=1, global_layers=1, local_layers=1,
                   query_points=4, latent_tokens=8, latent_dim=16,
                   shape_width=32, shape_heads=4, shape_layers=1,
                   shape_ffn_hidden=64, latent_predictor_layers=1,
                   diffusion_layers=2, diffusion_train_steps=32,
                   diffusion_sample_steps=4, adapter_tail_steps=2,
                   adapter_hidden=12, field_hidden=32, detail_channels=8,
                   contrastive_dim=16, local_size=32, tile_overlap=8,
                   global_max_size=32, gradient_checkpointing=False).validate()
