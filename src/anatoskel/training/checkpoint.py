"""Portable, versioned checkpoints with provenance and RNG state."""
from pathlib import Path
import random

import numpy as np
import torch

from anatoskel.config import ModelConfig

SCHEMA_VERSION = 1


def save_checkpoint(path, model, config, *, optimizer=None, step=0, kind="model",
                    provenance=None, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, "kind": kind,
               "model_config": config.to_dict(), "state_dict": model.state_dict(),
               "step": step, "torch_version": str(torch.__version__),
               "provenance": provenance or {}, "rng_state": torch.get_rng_state(),
               "python_rng_state": random.getstate()}
    numpy_state = np.random.get_state()
    payload["numpy_rng_state"] = (numpy_state[0], numpy_state[1].tolist(),
                                  *numpy_state[2:])
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    payload.update(extra or {})
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, map_location="cpu", expected_kind=None):
    # Only tensors and primitive metadata are serialized; no executable objects.
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema")
    if expected_kind and payload["kind"] != expected_kind:
        raise ValueError(f"Expected a {expected_kind} checkpoint")
    ModelConfig.from_dict(payload["model_config"])
    return payload
