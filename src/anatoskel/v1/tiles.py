"""RAS-aligned differentiable CT tiles and one-instance continuous-field fusion."""
from __future__ import annotations

import itertools
import math
import torch
from torch.nn import functional as F


def plan_tiles(anchor_mm, bounds_mm, *, size=128, spacing_mm=1., overlap=32,
               margin_mm=16., max_tiles=64):
    """Discrete coverage offsets share a differentiable instance reference."""
    anchor = torch.as_tensor(anchor_mm)
    bounds = torch.as_tensor(bounds_mm, device=anchor.device, dtype=anchor.dtype)
    if bounds.shape != (2, 3) or torch.any(bounds[1] < bounds[0]):
        raise ValueError("Bounds must be ordered [2,3] RAS millimetres")
    extent = (size-1)*spacing_mm
    step = (size-overlap)*spacing_mm
    if step <= 0:
        raise ValueError("Tile overlap must be smaller than size")
    low = bounds[0].detach()-margin_mm
    high = bounds[1].detach()+margin_mm
    axes = []
    for axis in range(3):
        # Keep a single tile at anchor when possible, expand without moving the
        # shared latent frame. Offset planning is discrete, extraction is not.
        left = max(0, math.ceil(float((anchor[axis].detach()-extent/2-low[axis])/step)))
        right = max(0, math.ceil(float((high[axis]-anchor[axis].detach()-extent/2)/step)))
        axes.append(range(-left, right+1))
    offsets = list(itertools.product(*axes))
    if len(offsets) > max_tiles:
        raise ValueError("Predicted extent exceeds tile budget; coverage cannot be silently dropped")
    return anchor[None]+anchor.new_tensor(offsets)*step


def extract_tile(scan, center_mm, *, size=128, spacing_mm=1.):
    """grid_sample retains the gradient to a predicted ROI centre.

    The source CT may stay on CPU; differentiable device copies connect centre
    predictions to feature/geometry losses without uploading the full CT.
    """
    with torch.autocast(device_type=scan["volume"].device.type, enabled=False):
        return _extract_tile(scan, center_mm, size=size, spacing_mm=spacing_mm)


def _extract_tile(scan, center_mm, *, size, spacing_mm):
    condition = scan.get("condition_mask", scan["ct_valid_mask"]) & scan["ct_valid_mask"]
    volume = scan["volume"].float().masked_fill(~condition[None], 0)
    center = torch.as_tensor(center_mm).to(device=volume.device, dtype=torch.float32)
    affine = torch.eye(4, device=volume.device, dtype=torch.float32)
    affine[:3, :3] *= spacing_mm
    affine[:3, 3] = center-(size-1)*spacing_mm/2
    axes = [torch.arange(size, device=volume.device, dtype=torch.float32)]*3
    xyz = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).flip(-1)
    world = xyz*spacing_mm+affine[:3, 3]
    inverse = torch.linalg.inv(scan["affine"].to(volume).float())
    voxel = world @ inverse[:3, :3].T + inverse[:3, 3]
    shape = voxel.new_tensor(volume.shape[-3:][::-1])
    grid = (2*(voxel+.5)/shape-1)[None]
    def sample(value, mode):
        return F.grid_sample(value.float().reshape(1, 1, *value.shape[-3:]),
                             grid, mode=mode, align_corners=False)[0]
    ct_valid = sample(scan["ct_valid_mask"], "nearest")[0] > .5
    visible = ct_valid & (sample(condition, "nearest")[0] > .5)
    support = sample(condition, "bilinear").clamp_min(1e-6)
    return {"volume": sample(volume, "bilinear")/support*visible[None], "affine": affine,
            "ct_valid_mask": ct_valid, "condition_mask": visible,
            "valid_mask": visible, "center_mm": center,
            "metadata": {"units": "mm", "coordinate_system": "RAS"}}


def tile_weights(points_world, tile_affine, size):
    with torch.autocast(device_type=points_world.device.type, enabled=False):
        return _tile_weights(points_world, tile_affine, size)


def _tile_weights(points_world, tile_affine, size):
    inverse = torch.linalg.inv(tile_affine.float())
    voxel = points_world.float() @ inverse[:3, :3].T+inverse[:3, 3]
    relative = (voxel-(size-1)/2)/(size/2)
    inside = (relative.abs() < 1).all(-1)
    # Zero at voxel-CELL support edges (-.5,size-.5), positive at every edge
    # voxel centre. This avoids a hard weight jump at an internal tile seam.
    weight = torch.cos(relative.clamp(-1, 1)*math.pi/2).square().prod(-1)*inside
    return weight, voxel


def blend_residuals(residuals, weights, *, required=None):
    numerator = (residuals.float()*weights.float()).sum(0)
    denominator = weights.float().sum(0)
    covered = denominator > 0
    if required is not None and bool((required & ~covered).any()):
        raise ValueError("Observed points lack tile coverage")
    result = numerator/denominator.clamp_min(torch.finfo(torch.float32).tiny)
    return result, covered


def overlap_consistency(residuals, weights):
    losses = []
    for first in range(len(residuals)):
        for second in range(first+1, len(residuals)):
            overlap = (weights[first] > 0) & (weights[second] > 0)
            if overlap.any():
                losses.append((residuals[first][overlap]-residuals[second][overlap]).abs().mean())
    return torch.stack(losses).mean() if losses else residuals.sum()*0
