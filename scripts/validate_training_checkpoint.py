"""Audit a saved training state without modifying it or downloading any data.

The gradient probe uses artificial points solely to verify numerical paths.
It does not establish surface accuracy or anatomical validity. Real-data
validation of prior zero crossings and reconstruction remains a separate step.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch

from anatoskel.config import ModelConfig
from anatoskel.data import label_to_class
from anatoskel.models import AnatoSKEL
from anatoskel.models.backend import (AnatomicalShapePrior, axis_angle_to_matrix,
                                     canonical_to_world, sample_volume, world_to_canonical)
from anatoskel.runtime import choose_device
from anatoskel.training.checkpoint import load_checkpoint


def finite_tensors(value, path="checkpoint"):
    """Reject corrupt floating model/optimizer/latent tensors, including nested state."""
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite tensor at {path}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite scalar at {path}")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tensors(item, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            finite_tensors(item, f"{path}[{index}]")


def validate_frames(frames):
    valid_count = 0
    worst_error = 0.
    labels = set()
    for index, frame in enumerate(frames):
        rotation = torch.tensor(frame["rotation"], dtype=torch.float64).reshape(3, 3)
        translation = torch.tensor(frame["translation"], dtype=torch.float64)
        if translation.shape != (3,) or not torch.isfinite(translation).all():
            raise ValueError(f"Malformed canonical translation at bone {index}")
        error = float((rotation.T @ rotation - torch.eye(3)).abs().max())
        worst_error = max(worst_error, error)
        if not torch.isfinite(rotation).all() or error > 1e-3 or abs(float(torch.linalg.det(rotation)) - 1) > 1e-3:
            raise ValueError(f"Canonical rotation is not SO(3) at bone {index}")
        label_to_class(frame["label_id"])
        labels.add(int(frame["label_id"]))
        if frame.get("pose_valid", False):
            if frame.get("method") not in ("trimmed_rigid_icp", "multistart_trimmed_rigid_icp"):
                raise ValueError("Trusted weak pose flag requires an atlas registration method")
            rms = frame.get("fit_rms_mm")
            if rms is None or not np.isfinite(rms) or rms > 3 or frame.get("fit_ambiguous", True):
                raise ValueError("Pose-valid frame has missing/failed fit-quality metadata")
            valid_count += 1
    return {"bones": len(frames), "label_ids": sorted(labels),
            "weak_pose_valid_bones": valid_count,
            "max_rotation_orthogonality_error": worst_error}


def validate_random_states(payload):
    torch.Generator(device="cpu").set_state(payload["rng_state"].cpu())
    random.Random().setstate(payload["python_rng_state"])
    state = payload["numpy_rng_state"]
    np.random.RandomState().set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    for state in payload.get("cuda_rng_state", []):
        if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
            raise ValueError("Malformed CUDA RNG state")


def validate_optimizer(payload, config, codes):
    """Actually load optimizer parameter groups and check momentum tensor shapes."""
    if payload["kind"] == "prior":
        module = AnatomicalShapePrior(config)
        module.load_state_dict(payload["state_dict"], strict=True)
        if codes is None:
            raise ValueError("Prior auto-decoder checkpoint is missing its optimized shape codes")
        code_parameter = torch.nn.Parameter(codes.clone())
        optimizer = torch.optim.Adam([*module.parameters(), code_parameter], lr=0.001)
    else:
        module = AnatoSKEL(config)
        module.load_state_dict(payload["state_dict"], strict=True)
        frozen = payload.get("run_config", {}).get("train", {}).get("freeze_prior", True)
        if frozen:
            module.backend.freeze_prior(pretrained=payload.get("prior_trained", False))
        optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=0.001)
    optimizer.load_state_dict(payload["optimizer"])
    checked = 0
    for parameter, state in optimizer.state.items():
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in state and state[key].shape != parameter.shape:
                raise ValueError(f"Optimizer {key} shape does not match its parameter")
        if "step" in state and float(state["step"]) > payload["step"]:
            raise ValueError("Optimizer parameter step exceeds checkpoint training step")
        checked += 1
    if payload["step"] > 0 and checked == 0:
        raise ValueError("No optimizer momentum state despite completed training steps")
    return {"parameter_groups": len(optimizer.param_groups), "momentum_states": checked}


def gradient_probe(payload, config, codes, frames, device, amp):
    prior = AnatomicalShapePrior(config).to(device)
    if payload["kind"] == "prior":
        prior.load_state_dict(payload["state_dict"])
    else:
        prefix = "backend.prior."
        prior.load_state_dict({key[len(prefix):]: value for key, value in payload["state_dict"].items()
                               if key.startswith(prefix)}, strict=True)
    # A frozen decoder must still transmit derivatives to patient shape/pose.
    prior.requires_grad_(False)
    selected = next((index for index, frame in enumerate(frames)
                     if 20 <= frame["label_id"] <= 25), 0)
    if frames:
        frame = frames[selected]
        label = frame["label_id"]
        rotation = torch.tensor(frame["rotation"], dtype=torch.float32, device=device).reshape(1, 1, 3, 3)
        origin = torch.tensor(frame["translation"], dtype=torch.float32, device=device).reshape(1, 1, 3)
    else:
        label = 20
        rotation = torch.eye(3, device=device).reshape(1, 1, 3, 3)
        origin = torch.zeros(1, 1, 3, device=device)
    initial_code = codes[selected] if codes is not None and len(codes) else torch.zeros(config.beta_dim)
    beta = initial_code.to(device).reshape(1, 1, -1).detach().requires_grad_(True)
    omega = torch.zeros(1, 1, 3, device=device, requires_grad=True)
    delta_t = torch.zeros(1, 1, 3, device=device, requires_grad=True)
    generator = torch.Generator().manual_seed(71)
    canonical = (torch.rand(1, 1, 256, 3, generator=generator) * 80 - 40).to(device)
    world = canonical_to_world(canonical, rotation, origin).detach()
    with torch.autocast(device_type=device.type, enabled=False):
        predicted_rotation = rotation @ axis_angle_to_matrix(omega)
    points = world_to_canonical(world, predicted_rotation, origin + delta_t)
    precision_context = torch.autocast("cuda", dtype=torch.bfloat16) \
        if device.type == "cuda" and amp else nullcontext()
    with precision_context:
        field = prior(points, beta, torch.tensor([[label_to_class(label)]], device=device))
    weights = torch.linspace(0.5, 1.5, 256, device=device)
    loss = (field.float() * weights).mean()
    gradients = torch.autograd.grad(loss, (beta, omega, delta_t))
    report = {"label_id": label, "amp": amp and device.type == "cuda",
              "field_min": float(field.detach().min()), "field_max": float(field.detach().max()),
              "probe_points": "artificial canonical points; not medical accuracy evaluation"}
    if not torch.isfinite(field).all():
        raise ValueError("Nonfinite frozen-prior probe output")
    for name, gradient in zip(("beta", "rotation_update", "translation_update"), gradients, strict=True):
        norm = float(gradient.float().norm())
        report[f"{name}_gradient_norm"] = norm
        if not torch.isfinite(gradient).all() or norm <= 1e-12:
            raise ValueError(f"Frozen prior has invalid/zero {name} gradients")
    if device.type == "cuda":
        volume = torch.randn(1, 4, 8, 8, 8, device=device, dtype=torch.bfloat16, requires_grad=True)
        voxel = torch.tensor([[[[3.25, 2.5, 4.125]]]], device=device, requires_grad=True)
        sampled = sample_volume(volume, voxel)
        volume_gradient, voxel_gradient = torch.autograd.grad(sampled.square().mean(), (volume, voxel))
        if not torch.isfinite(volume_gradient).all() or not torch.isfinite(voxel_gradient).all():
            raise ValueError("CUDA 3D sampling backward failed finite-gradient check")
        report["cuda_grid_sample_fp32_backward"] = "passed"
    return report


def validate(args):
    payload = load_checkpoint(args.checkpoint)
    config = ModelConfig.from_dict(payload["model_config"])
    if payload["kind"] not in ("prior", "model"):
        raise ValueError("Expected a prior or reconstruction-model training checkpoint")
    if payload["step"] < args.min_steps:
        raise ValueError(f"Checkpoint has {payload['step']} steps; require at least {args.min_steps}")
    origin = payload["provenance"]
    if origin.get("synthetic", False) and not args.allow_synthetic:
        raise ValueError("Synthetic software-check weights are not a real-data training checkpoint")
    patients = origin.get("training_patients")
    if not patients or len(patients) != len(set(patients)):
        raise ValueError("Checkpoint training patient provenance is empty or duplicated")
    manifest_hash = origin.get("manifest_sha256", "")
    if len(manifest_hash) != 64 or any(char not in "0123456789abcdef" for char in manifest_hash):
        raise ValueError("Missing or malformed training manifest SHA256")
    finite_tensors(payload)
    frames, codes = payload.get("canonical_frames", []), payload.get("shape_codes")
    frame_report = validate_frames(frames)
    if bool(payload.get("pose_supervision_valid", False)) != bool(frame_report["weak_pose_valid_bones"]):
        raise ValueError("Global pose-supervision flag disagrees with per-bone fit flags")
    warnings = []
    if not frame_report["weak_pose_valid_bones"]:
        warnings.append("No quality-qualified atlas pose labels; pose axes remain provisional")
    if args.require_atlas and (not frames or any("rigid_icp" not in f.get("method", "") for f in frames)):
        raise ValueError("Require training-atlas canonical registration, not independent PCA gauges")
    coverage = None
    if codes is not None:
        if not frames:
            raise ValueError("Saved shape codes require at least one canonical training frame")
        if codes.shape != (len(frames), config.beta_dim):
            raise ValueError("Shape-code count/width differs from canonical frames/config")
        coverage = float((codes.float().norm(dim=-1) > 1e-10).float().mean())
        if coverage < 0.999:
            message = f"Only {coverage:.1%} of initialized shape codes have been updated"
            if not origin.get("synthetic", False):
                raise ValueError(message + "; do not freeze this partially visited prior")
            warnings.append(message)
        visits = payload.get("prior_samples_seen", payload["step"])
        if payload["kind"] == "prior" and visits < len(frames) and not origin.get("synthetic", False):
            raise ValueError("Prior has fewer optimizer steps than training bones; finish a full bone pass before freezing")
    if payload["kind"] == "model" and not payload.get("prior_trained", False):
        warnings.append("Main model marks its prior as uninitialized; bootstrap requires separate quality checks")
    requested_labels = payload.get("run_config", {}).get("inference", {}).get("label_ids", [])
    unsupported_labels = sorted(set(requested_labels) - set(frame_report["label_ids"]))
    if unsupported_labels:
        warnings.append(f"Requested inference identities lack fitted prior bones: {unsupported_labels}")
    optimizer_report = validate_optimizer(payload, config, codes)
    validate_random_states(payload)
    cursor = {"next_epoch": payload.get("next_epoch"), "next_scan": payload.get("next_scan")}
    if cursor["next_epoch"] is None or cursor["next_scan"] is None:
        if args.require_resumable:
            raise ValueError("Checkpoint lacks epoch/scan continuation cursor")
        warnings.append("No continuation cursor; optimizer/RNG load alone cannot exactly resume scan order")
    logs = payload.get("training_log", [])
    if logs:
        if logs[-1]["step"] != payload["step"]:
            raise ValueError("Latest training log step differs from checkpoint step")
        if any(second["step"] <= first["step"] for first, second in zip(logs, logs[1:])):
            raise ValueError("Training log step ordering is not strictly increasing")
        recent = logs[-min(50, len(logs)):]
        if payload["kind"] == "model" and not any(record.get("fine_bones", 0) > 0 for record in recent):
            raise ValueError("No bone reconstruction supervision in recent training steps")
    else:
        warnings.append("Checkpoint has no embedded training-loss history")
    device = choose_device(args.device)
    probe = gradient_probe(payload, config, codes, frames, device, args.amp)
    if probe["field_min"] >= 0 or probe["field_max"] <= 0:
        if args.require_zero_crossing:
            raise ValueError("Prior probe has no TSDF sign crossing; inspect real canonical bone surfaces before freezing")
        warnings.append("Artificial prior probe has no sign crossing; surface-quality validation is still required")
    return {"checkpoint": str(Path(args.checkpoint).resolve()), "kind": payload["kind"],
            "step": payload["step"], "device": str(device), "synthetic": origin.get("synthetic", False),
            "training_patients": len(patients), "canonical_frames": frame_report,
            "updated_shape_code_fraction": coverage, "distance_scale_mm": config.distance_scale_mm,
            "coordinate_scale_mm": config.coordinate_scale_mm, "optimizer": optimizer_report,
            "continuation_cursor": cursor, "prior_gradient_probe": probe, "warnings": warnings,
            "status": "numerical_and_metadata_checks_passed",
            "quality_limit": "This does not verify CT segmentation or reconstructed-surface accuracy"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--min-steps", type=int, default=10)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--require-atlas", action="store_true")
    parser.add_argument("--require-resumable", action="store_true")
    parser.add_argument("--require-zero-crossing", action="store_true",
                        help="Require both TSDF signs in the artificial prior probe; this is only a coarse quality gate")
    parser.add_argument("--amp", action="store_true", help="Probe CUDA BF16 prior and FP32 grid-sampling backward")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.min_steps < 1:
        parser.error("--min-steps must be positive")
    try:
        report = validate(args)
    except (ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
