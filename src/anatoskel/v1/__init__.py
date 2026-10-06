"""AnatoSKEL v1: observation-aware transformer and latent diffusion."""

from .config import V1Config

__all__ = ["AnatoSKELV1", "V1Config"]


def __getattr__(name):
    if name == "AnatoSKELV1":
        from .network import AnatoSKELV1
        return AnatoSKELV1
    raise AttributeError(name)
