"""Acquisition, annotation and conditioning domains; no fabricated partial GT."""
from __future__ import annotations

import copy
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import torch

from anatoskel.data.scans import validate_affine


def input_scan(scan):
    """Remove annotation data before any prediction or CT feature extraction."""
    metadata = {k: v for k, v in scan["metadata"].items()
                if k not in ("label_bounds", "original_truncated", "annotation_domain_truncated")}
    condition = scan["ct_valid_mask"] & scan.get("condition_mask", scan["ct_valid_mask"])
    return dict(scan, labels=None, volume=scan["volume"].masked_fill(~condition[None], 0),
                valid_mask=condition, ct_valid_mask=condition,
                condition_mask=condition, metadata=metadata)


def synthetic_crop(scan, *, seed=42, label_id=None, keep_fraction=None):
    """Physically crop CT BEFORE encoding. The caller retains the original GT.

    Returned input contains no labels or hidden full-scan extent. Selecting an
    augmentation plane using GT is permitted; no GT enters the forward model.
    """
    rng = np.random.default_rng(seed)
    axis = int(rng.integers(3))
    low_side = bool(rng.integers(2))
    fraction = float(rng.uniform(.55, .85) if keep_fraction is None else keep_fraction)
    if not 0 < fraction < 1:
        raise ValueError("Crop keep_fraction must lie strictly between zero and one")
    shape = scan["volume"].shape[-3:]
    low, high = 0, shape[axis]
    if label_id is not None and scan.get("labels") is not None:
        indices = torch.nonzero(scan["labels"] == label_id)
        if not len(indices):
            raise ValueError("Synthetic crop label is absent")
        low = int(indices[:, axis].min())
        high = int(indices[:, axis].max()) + 1
    keep = max(1, round((high-low)*fraction))
    slices = [slice(None)]*3
    start = np.zeros(3, dtype=int)
    if low_side:
        stop = min(shape[axis], low+keep)
        slices[axis] = slice(0, stop)
    else:
        start[axis] = max(0, high-keep)
        slices[axis] = slice(int(start[axis]), shape[axis])
    slices = tuple(slices)
    result = input_scan(scan)
    affine = scan["affine"].clone()
    affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ torch.as_tensor(
        start[::-1].copy(), dtype=affine.dtype)
    valid = scan["ct_valid_mask"][slices].clone()
    volume = scan["volume"][(slice(None), *slices)].clone()
    volume = volume.masked_fill(~valid[None], 0)
    metadata = copy.deepcopy(result["metadata"])
    metadata.update(synthetic_condition_crop=True, original_shape_xyz=list(valid.shape[::-1]),
                    original_affine=affine.tolist(),
                    ct_voxel_domain={"shape_xyz": list(valid.shape[::-1]),
                                     "affine": affine.tolist()},
                    condition_seed=int(seed))
    result.update(volume=volume, affine=affine, valid_mask=valid, ct_valid_mask=valid,
                  condition_mask=valid, artificial_padding_mask=torch.zeros_like(valid),
                  metadata=metadata)
    return result


def prepare_bone_supervision(scan, label_id, anchor_mm, *, n_points=8192,
                             n_surface=4096, scale_mm=64., tau_mm=8.,
                             margin_mm=1., seed=42):
    """Reliable partial TSDF in mm and true voxel-face surface samples.

    Interfaces are constructed ONLY between two known voxels. Acquisition and
    annotation boundaries never create a surface. Tight GT bounds are a target
    optimization and are never returned as network input.
    """
    if scan.get("labels") is None:
        raise ValueError("Supervision requires annotations")
    if min(n_points, n_surface) < 1 or min(scale_mm, tau_mm) <= 0:
        raise ValueError("Positive point counts and metric scales required")
    affine = validate_affine(scan["affine"], orthogonal=True)
    labels = scan["labels"].cpu().numpy()
    known = (scan["valid_mask"] & scan["ct_valid_mask"]).cpu().numpy()
    bone = (labels == label_id) & known
    indices = np.argwhere(bone)
    if not len(indices):
        raise ValueError("No annotated bone in the known domain")
    # Erosion is global so a tight computational crop cannot change completeness.
    lo, hi = indices.min(0), indices.max(0)+1
    spacing = np.linalg.norm(affine[:3, :3], axis=0)[::-1]
    margin = np.ceil((tau_mm+margin_mm+2*spacing.max())/spacing).astype(int)
    starts = np.maximum(0, lo-margin)
    stops = np.minimum(labels.shape, hi+margin)
    region = tuple(slice(int(a), int(b)) for a, b in zip(starts, stops))
    bone, known = bone[region], known[region]
    interior = ndimage.binary_erosion(known, border_value=0)
    complete = not bool((bone & ~interior).any())
    surface_dhw, normals_world = [], []
    for axis in range(3):
        first, second = [slice(None)]*3, [slice(None)]*3
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        first, second = tuple(first), tuple(second)
        interface = (bone[first] != bone[second]) & known[first] & known[second]
        location = np.argwhere(interface).astype(float)
        if not len(location):
            continue
        outward = np.where(bone[first][interface], 1., -1.)
        location[:, axis] += .5
        direction = affine[:3, 2-axis] / spacing[axis]
        surface_dhw.append(location)
        normals_world.append(outward[:, None]*direction[None])
    if not surface_dhw:
        raise ValueError("Bone has no trusted anatomical surface")
    surface_dhw = np.concatenate(surface_dhw)
    normals_world = np.concatenate(normals_world)
    surface_world = (surface_dhw[:, ::-1] + starts[::-1]) @ affine[:3, :3].T + affine[:3, 3]
    rng = np.random.default_rng(seed)
    candidates = np.argwhere(known)
    distances = ndimage.distance_transform_edt(
        np.pad(known, 1, constant_values=False), sampling=spacing)[1:-1, 1:-1, 1:-1]
    # EDT measures unknown voxel centres. Unknown support begins at voxel
    # faces; subtracting the half diagonal gives a conservative physical lower
    # bound even for anisotropic grids and corners.
    distances = np.maximum(0., distances-.5*np.linalg.norm(spacing))
    # Half near-surface and half uniform known-domain points prevent a
    # background-dominated loss without introducing unknown-domain negatives.
    near_mask = ndimage.binary_dilation(bone, iterations=2) & known
    near = np.argwhere(near_mask)
    chosen = np.concatenate((near[rng.integers(len(near), size=n_points//2)],
                             candidates[rng.integers(len(candidates),
                                                     size=n_points-n_points//2)]))
    world = (chosen[:, ::-1]+starts[::-1]) @ affine[:3, :3].T + affine[:3, 3]
    distance, _ = cKDTree(surface_world).query(world)
    occupancy = bone[tuple(chosen.T)]
    sdf_valid = distances[tuple(chosen.T)] > np.minimum(distance, tau_mm)+margin_mm
    target = np.minimum(distance, tau_mm)*np.where(occupancy, -1., 1.)
    picked = rng.integers(len(surface_world), size=n_surface)
    anchor = torch.as_tensor(anchor_mm).detach().cpu().numpy()
    def tensor(array, dtype=torch.float32):
        return torch.as_tensor(np.asarray(array).copy(), dtype=dtype)
    return {"points_world": tensor(world), "points_u": tensor((world-anchor)/scale_mm),
            "target_sdf_mm": tensor(target), "sdf_valid": tensor(sdf_valid, torch.bool),
            "occupancy": tensor(occupancy),
            "occupancy_valid": torch.ones(n_points, dtype=torch.bool),
            "surface_world": tensor(surface_world[picked]),
            "surface_u": tensor((surface_world[picked]-anchor)/scale_mm),
            "normals": tensor(normals_world[picked]), "complete": complete,
            "label_id": int(label_id), "anchor_mm": tensor(anchor)}
