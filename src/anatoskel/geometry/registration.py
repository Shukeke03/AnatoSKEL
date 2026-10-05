"""Optional trimmed rigid atlas fitting, with explicitly weak pose labels.

An atlas fixes the canonical gauge for each class. Its PCA-initialized axes do
not become anatomical axes by registration, and partial or ambiguous fits are
never advertised as trusted pose supervision.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import torch

from anatoskel.data.scans import (ScanDataset, canonical_from_mask, label_to_class,
                                  read_manifest, validate_affine)
from .meshes import kabsch, transform_points


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _points(values, max_points: int) -> np.ndarray:
    points = _numpy(values).astype(np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 12 or not np.isfinite(points).all():
        raise ValueError("Atlas registration needs at least 12 finite Nx3 points")
    if max_points < 12:
        raise ValueError("max_points must be at least 12")
    if np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-8) < 3:
        raise ValueError("Atlas surface points must span three spatial dimensions")
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points, dtype=int)]
    return points


def surface_points(mask_dhw, affine, max_points: int = 2048) -> np.ndarray:
    """Deterministically subsample boundary voxel centres in patient RAS mm."""
    mask = _numpy(mask_dhw).astype(bool)
    if mask.ndim != 3:
        raise ValueError("Boundary sampling requires a 3D mask")
    surface = mask & ~ndimage.binary_erosion(mask, border_value=0)
    dhw = np.argwhere(surface)
    return _points(transform_points(dhw[:, ::-1], affine), max_points)


def _pca(points):
    # Robust initialization only; final rigid updates use nearest-neighbour
    # correspondences and trimmed fitting. Keep gross peripheral outliers out
    # of the initial covariance without claiming segmentation-error detection.
    median = np.median(points, axis=0)
    radial = np.linalg.norm(points - median, axis=1)
    keep = radial <= np.quantile(radial, 0.9)
    core = points[keep]
    centre = core.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov((core - centre).T))
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order]
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1
    eigenvalues = eigenvalues[order]
    separation = min((eigenvalues[0] - eigenvalues[1]) / max(eigenvalues[0], 1e-8),
                     (eigenvalues[1] - eigenvalues[2]) / max(eigenvalues[1], 1e-8))
    return centre, basis, float(separation)


def _trimmed_rms(distances, fraction=0.85):
    count = max(3, int(np.ceil(fraction * len(distances))))
    kept = np.partition(distances, count - 1)[:count]
    return float(np.sqrt(np.mean(kept ** 2)))


def fit_atlas(template_points, observed_world, partial: bool = False,
              max_rms_mm: float = 3, max_points: int = 2048,
              max_iterations: int = 40) -> dict[str, Any]:
    """Fit a template-canonical surface to observed-world surface with trimmed ICP.

    Four proper PCA-sign rotations initialize the fit. The quality flag is only
    a weak numerical label; it says nothing about clinical/anatomical axis
    validity. Observable symmetry or near-equivalent fits disable pose labels.
    """
    if max_rms_mm <= 0 or not np.isfinite(max_rms_mm) or max_iterations < 1:
        raise ValueError("Registration tolerance and iteration count must be positive")
    template, observed = _points(template_points, max_points), _points(observed_world, max_points)
    source_centre, source_basis, source_separation = _pca(template)
    target_centre, target_basis, target_separation = _pca(observed)
    target_tree = cKDTree(observed)
    candidates = []
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        if np.prod(signs) < 0:
            continue
        rotation = target_basis @ np.diag(signs) @ source_basis.T
        translation = target_centre - rotation @ source_centre
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            transformed = template @ rotation.T + translation
            distances, nearest = target_tree.query(transformed)
            count = max(12, int(np.ceil(0.85 * len(template))))
            kept = np.argsort(distances)[:count]
            try:
                update = kabsch(transformed[kept], observed[nearest[kept]])
            except ValueError:
                break
            delta_rotation, delta_translation = update["rotation"], update["translation"]
            rotation = delta_rotation @ rotation
            translation = delta_rotation @ translation + delta_translation
            if np.linalg.norm(delta_rotation - np.eye(3)) < 1e-5 and np.linalg.norm(delta_translation) < 1e-5:
                break
        transformed = template @ rotation.T + translation
        forward = target_tree.query(transformed)[0]
        backward = cKDTree(transformed).query(observed)[0]
        rms = float(np.sqrt(np.mean(forward ** 2)))
        trimmed_rms = 0.5 * (_trimmed_rms(forward) + _trimmed_rms(backward))
        coverage = float(min(np.mean(forward <= max_rms_mm), np.mean(backward <= max_rms_mm)))
        candidates.append({"rotation": rotation, "translation": translation,
                           "rms_mm": rms, "trimmed_rms_mm": trimmed_rms,
                           "symmetric_rms_mm": float(np.sqrt((np.mean(forward ** 2) + np.mean(backward ** 2)) / 2)),
                           "inlier_coverage": coverage, "iterations": iterations,
                           "score": trimmed_rms + max_rms_mm * (1 - coverage)})
    candidates.sort(key=lambda candidate: candidate["score"])
    best = candidates[0]
    near_equivalent = []
    tolerance = max(0.15, 0.1 * best["score"])
    for other in candidates[1:]:
        cosine = np.clip((np.trace(other["rotation"] @ best["rotation"].T) - 1) / 2, -1, 1)
        angle = float(np.degrees(np.arccos(cosine)))
        if angle > 15 and other["score"] - best["score"] <= tolerance:
            near_equivalent.append(angle)
    symmetric = min(source_separation, target_separation) < 0.03
    ambiguous = bool(symmetric or near_equivalent)
    quality = best["rms_mm"] <= max_rms_mm and best["trimmed_rms_mm"] <= max_rms_mm and best["inlier_coverage"] >= 0.8
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = best["rotation"], best["translation"]
    return {key: value for key, value in best.items() if key != "score"} | {
        "transform": transform, "pose_valid": bool(quality and not partial and not ambiguous),
        "partial": bool(partial), "ambiguous": ambiguous, "near_symmetric_axes": symmetric,
        "near_equivalent_rotations_deg": near_equivalent, "trimmed_fit": True,
        "method": "multistart_trimmed_rigid_icp", "weak_pose_label": True,
        "anatomical_axes_validated": False, "axis_separation": min(source_separation, target_separation),
        "candidate_scores": [candidate["score"] for candidate in candidates]}


def load_atlas(path: str | Path, training_patients) -> dict[str, Any]:
    """Load a mm atlas and reject any patient provenance outside the train split."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("units") != "mm":
        raise ValueError("Atlas must explicitly declare units='mm'")
    patients = payload.get("training_patients")
    if not isinstance(patients, list) or not patients or not all(isinstance(p, str) and p for p in patients):
        raise ValueError("Atlas must list nonempty training_patients")
    if len(set(patients)) != len(patients):
        raise ValueError("Atlas contains duplicate training patient identities")
    if not set(patients) <= set(training_patients):
        raise ValueError("Atlas provenance includes patients outside this training split")
    classes = payload.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ValueError("Atlas must have at least one class template")
    normalized = {}
    for key, values in classes.items():
        try:
            label = int(key)
        except (ValueError, TypeError) as exc:
            raise ValueError("Atlas class keys must be original VerSe label IDs") from exc
        class_id = label_to_class(label)
        if str(label) != key or not isinstance(values, dict):
            raise ValueError("Atlas class keys must be normalized label strings")
        points = _points(values.get("points_canonical", values.get("points")), max_points=10_000_000)
        normalized[key] = dict(values, points=points, label_id=label, class_id=class_id)
    return dict(payload, classes=normalized, anatomical_axes_validated=False)


def build_atlas(manifest: str | Path, out_path: str | Path, max_points: int = 2048) -> Path:
    """Build per-class training exemplars; axes remain provisional PCA gauges.

    Only complete training bones with stable PCA initialization are eligible.
    A median-volume exemplar fixes each class's canonical coordinate gauge;
    this is not a population-mean shape or clinically validated anatomical atlas.
    """
    records = read_manifest(manifest)
    training_patients = sorted({record.patient_id for record in records if record.split == "train"})
    if not training_patients:
        raise ValueError("Atlas construction requires training patients")
    candidates = {}
    for scan in ScanDataset(records, split="train"):
        if scan["labels"] is None:
            raise ValueError("Atlas construction requires training masks")
        affine = validate_affine(scan["affine"])
        volume_per_voxel = abs(np.linalg.det(affine[:3, :3]))
        for label in torch.unique(scan["labels"]).tolist():
            label = int(label)
            if label == 0 or scan["metadata"]["original_truncated"].get(str(label), False):
                continue
            mask = scan["labels"] == label
            frame = canonical_from_mask(mask, affine)
            if not frame["geometry_stable"]:
                continue
            world = surface_points(mask, affine, max_points)
            rotation, translation = _numpy(frame["rotation"]), _numpy(frame["translation"])
            canonical = (world - translation) @ rotation
            candidates.setdefault(label, []).append({
                "points": canonical, "volume_mm3": float(mask.sum()) * volume_per_voxel,
                "patient_id": scan["metadata"]["patient_id"], "scan_id": scan["metadata"]["scan_id"],
                "source_rotation": rotation.tolist(), "source_translation_mm": translation.tolist(),
                "frame_status": "provisional_pca", "anatomical_axes_validated": False})
    if not candidates:
        raise ValueError("No full stable training bones are available for a provisional atlas")
    classes = {}
    for label, entries in candidates.items():
        median_volume = np.median([entry["volume_mm3"] for entry in entries])
        exemplar = min(entries, key=lambda entry: abs(entry["volume_mm3"] - median_volume))
        classes[str(label)] = {key: value.tolist() if isinstance(value, np.ndarray) else value
                               for key, value in exemplar.items()}
        classes[str(label)]["available_training_instances"] = len(entries)
    payload = {"units": "mm", "training_patients": training_patients, "classes": classes,
               "method": "training_only_median_volume_exemplars", "canonical_status": "provisional_pca_gauge",
               "anatomical_axes_validated": False}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return out_path
