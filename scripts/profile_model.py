"""Measure real forward/backward memory on synthetic volumes, without data access.

Example (run on the rented CUDA server, not as a medical training experiment):
    python scripts/profile_model.py --bones 1 --steps 2 --output profile_b1.json
    python scripts/profile_model.py --bones 2 --steps 2 --output profile_b2.json

The coarse and fine passes follow the trainer's separate backward boundaries.
Peak numbers include optimizer state, checkpointed convolutions, mixed-precision
attention and FP32 geometric sampling. A random frozen prior is used solely to
match the memory layout of prior-backed training; it is never called pretrained.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from anatoskel.config import ModelConfig, load_config
from anatoskel.models import AnatoSKEL
from anatoskel.runtime import choose_device, doctor
from anatoskel.training.losses import detector_loss, field_loss
from anatoskel.training.pipeline import pack_context


def _synthetic_roi(size: int, label: int, center: torch.Tensor) -> dict:
    coordinates = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    distance = torch.stack(torch.meshgrid(coordinates, coordinates, coordinates,
                                          indexing="ij"), -1).square().sum(-1).sqrt()
    inside = distance < min(12.0, size / 4)
    affine = torch.eye(4)
    affine[:3, 3] = center - (size - 1) / 2
    return {"volume": torch.where(inside, 0.6, -0.8).unsqueeze(0),
            "affine": affine, "labels": inside.long() * label,
            "ct_valid_mask": torch.ones_like(inside), "valid_mask": torch.ones_like(inside)}


def _memory(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"peak_allocated_gib": None, "peak_reserved_gib": None}
    return {"peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def run_profile(config: ModelConfig, *, device: torch.device, coarse_shape: tuple,
                local_size: int, bones: int, points: int, steps: int) -> dict:
    """Run the actual trainable graph and fail on OOM or non-finite gradients."""
    torch.manual_seed(42)
    model = AnatoSKEL(config).to(device).train()
    model.backend.prior.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=1e-4, weight_decay=0.01)
    affine = torch.eye(4)
    affine[:3, :3] *= 3.0
    volume = torch.randn((1, 1, *coarse_shape), device=device) * 0.2
    labels = torch.zeros(coarse_shape, dtype=torch.long)
    middle = tuple(size // 2 for size in coarse_shape)
    for index in range(bones):
        z = min(max(middle[0] + index * 8, 3), coarse_shape[0] - 4)
        labels[z - 2:z + 3, middle[1] - 3:middle[1] + 4,
               middle[2] - 3:middle[2] + 4] = 20 + index
    valid = torch.ones_like(labels, dtype=torch.bool)
    centers = torch.stack([torch.tensor([0.0, 0.0, index * 35.0]) for index in range(bones)])
    rois = [_synthetic_roi(local_size, 20 + index, centers[index]) for index in range(bones)]
    local_points = (torch.rand((1, bones, points, 3)) * 2 - 1) * min(20.0, local_size / 3)
    points_world = (local_points + centers[None, :, None]).to(device)
    target = ((local_points.norm(dim=-1) - min(12.0, local_size / 4))
              / config.distance_scale_mm).clamp(-1, 1).to(device)
    class_ids = torch.tensor([[19 + index for index in range(bones)]], device=device)
    anchors = centers[None].to(device)
    rows = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _reset_peak(device)
        start = time.perf_counter()
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with amp:
            detection = model.encode_coarse(volume, affine[None].to(device))
            coarse_loss, _, _ = detector_loss(detection, labels, valid, affine)
            coarse_loss += 0.1 * detection["visibility_logits"].float().square().mean()
        queries = detection["queries"][:, :bones].detach()
        coarse_loss.backward()
        del detection
        coarse_memory = _memory(device)
        _reset_peak(device)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with amp:
            context = pack_context(model, rois, queries, device)
            state = model.reconstruct(context, queries, class_ids, anchors)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with amp:
            prediction = model.field(context, state, points_world)
        fine_loss = field_loss(prediction["field"], target, prediction["valid"], state["beta"])
        truth = torch.stack([(r["labels"] == 20 + index).float()
                             for index, r in enumerate(rois)]).unsqueeze(1).to(device)
        observed = context["annotation_valid_mask"]
        fine_loss += 0.2 * F.binary_cross_entropy_with_logits(context["mask_logits"][observed].float(),
                                                             truth[observed])
        fine_loss += 0.1 * state["visibility_logits"].float().square().mean()
        fine_loss.backward()
        if not torch.isfinite(coarse_loss) or not torch.isfinite(fine_loss):
            raise RuntimeError("Profile produced a non-finite loss")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        fine_memory = _memory(device)
        _reset_peak(device)
        optimizer.step()
        optimizer_memory = _memory(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        phases = {"coarse": coarse_memory, "fine": fine_memory, "optimizer": optimizer_memory}
        row = {"step": step + 1, "seconds": time.perf_counter() - start,
               "coarse_loss": float(coarse_loss.detach()), "fine_loss": float(fine_loss.detach()),
               "gradient_norm_before_clip": float(norm), "memory": phases}
        rows.append(row)
        # Match the next trainer iteration without keeping profile-only outputs alive.
        del context, state, prediction, truth, queries, coarse_loss, fine_loss
        gc.collect()
    peak = max((phase["peak_allocated_gib"] for row in rows for phase in row["memory"].values()
                if phase["peak_allocated_gib"] is not None), default=None)
    return {"status": "passed", "purpose": "synthetic software memory profiling only",
            "architecture": config.to_dict(), "coarse_shape_dhw": list(coarse_shape),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "local_size": local_size, "bones_per_step": bones, "points_per_bone": points,
            "precision": "bf16 autocast plus fp32 geometry" if device.type == "cuda" else "fp32",
            "prior": "random frozen weights, never a pretrained or validated prior",
            "peak_allocated_gib": peak, "steps": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "research.yaml"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Small local software architecture")
    parser.add_argument("--coarse-shape", nargs=3, type=int, default=(128, 128, 192), metavar=("D", "H", "W"))
    parser.add_argument("--local-size", type=int, default=128)
    parser.add_argument("--bones", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output")
    args = parser.parse_args()
    if min(args.coarse_shape) < 16 or args.local_size < 16 or args.points < 1 or args.steps < 1:
        parser.error("Shapes must be at least 16; points and steps must be positive")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    result = {"status": "failed", "purpose": "synthetic software memory profiling only"}
    try:
        device = choose_device(args.device)
        if device.type == "cuda":
            result["runtime"] = doctor(require_cuda=True)
            result["gpu_total_gib"] = torch.cuda.get_device_properties(device).total_memory / 2**30
            result["gpu_free_before_model_gib"] = torch.cuda.mem_get_info(device)[0] / 2**30
        config = ModelConfig.smoke() if args.smoke else ModelConfig.from_dict(load_config(args.config).get("model", {}))
        result.update(run_profile(config, device=device, coarse_shape=tuple(args.coarse_shape),
                                  local_size=args.local_size, bones=args.bones,
                                  points=args.points, steps=args.steps))
    except (RuntimeError, ValueError) as error:
        result["error"] = str(error)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
