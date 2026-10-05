"""Patient RAS-mm meshes, without padding or claims of watertight reconstruction."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes
import torch

from anatoskel.data.scans import validate_affine


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def transform_points(points, affine) -> np.ndarray:
    points = _numpy(points).astype(np.float64)
    affine = validate_affine(affine)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Points must be finite Nx3 xyz coordinates")
    return points @ affine[:3, :3].T + affine[:3, 3]


def kabsch(source, target) -> dict[str, Any]:
    """Fit an SE(3) transform to *matched* point pairs; never invent correspondences.

    Returned rotation maps source-column xyz to target-column xyz. This utility
    alone does not establish an anatomical atlas or a registration confidence.
    """
    source, target = _numpy(source).astype(float), _numpy(target).astype(float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("Kabsch needs at least three matched Nx3 source/target pairs")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Correspondences must be finite")
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_zero, target_zero = source - source_mean, target - target_mean
    if np.linalg.matrix_rank(source_zero, tol=1e-8) < 2 or np.linalg.matrix_rank(target_zero, tol=1e-8) < 2:
        raise ValueError("Collinear correspondences cannot determine a rigid frame")
    u, _, vh = np.linalg.svd(source_zero.T @ target_zero)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vh.T @ u.T))
    rotation = vh.T @ correction @ u.T
    translation = target_mean - rotation @ source_mean
    residual = source @ rotation.T + translation - target
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, translation
    return {"rotation": rotation, "translation": translation, "transform": matrix,
            "rms_mm": float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
            "method": "kabsch_matched_correspondences", "anatomical_axes_validated": False}


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    name: str = "bone"
    label_id: int | None = None
    class_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def boundary_edges(self) -> int:
        if not len(self.faces):
            return 0
        edges = np.sort(np.concatenate((self.faces[:, (0, 1)], self.faces[:, (1, 2)],
                                        self.faces[:, (2, 0)])), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return int(np.sum(counts == 1))


def _extract(values_dhw, affine, *, level, gradient_direction, valid_mask=None,
             name="bone", label_id=None, class_id=None, metadata=None) -> Mesh:
    affine = validate_affine(affine)
    values = _numpy(values_dhw).astype(np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Mesh extraction needs a finite scalar DHW volume")
    extraction_mask = None
    if valid_mask is not None:
        valid = _numpy(valid_mask).astype(bool)
        if valid.shape != values.shape:
            raise ValueError("Coverage mask must match scalar field shape")
        # Avoid cells crossing interior invalid regions. Outer array boundaries
        # are naturally open: no padding is added before marching cubes.
        if not valid.all():
            valid = ndimage.binary_erosion(valid, structure=np.ones((3, 3, 3)), border_value=1)
        extraction_mask = valid.transpose(2, 1, 0).copy()
    info = dict(metadata or {}, extraction_affine=affine.tolist(), units="mm",
                coordinate_system="RAS", boundary_padding_added=False)
    empty = Mesh(np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64),
                 name, label_id, class_id, info)
    if min(values.shape) < 2 or not (float(values.min()) < level < float(values.max())):
        empty.metadata["empty_reason"] = "no_level_crossing"
        return empty
    try:
        vertices_xyz, faces, _, _ = marching_cubes(values.transpose(2, 1, 0).copy(), level=level,
                                                   gradient_direction=gradient_direction,
                                                   allow_degenerate=False, mask=extraction_mask)
    except (ValueError, RuntimeError) as exc:
        if "surface" not in str(exc).lower():
            raise
        empty.metadata["empty_reason"] = "no_observed_surface"
        return empty
    vertices_world = transform_points(vertices_xyz, affine)
    # Marching cubes was performed in xyz, so only affine handedness matters.
    if np.linalg.det(affine[:3, :3]) < 0:
        faces = faces[:, (0, 2, 1)]
    return Mesh(vertices_world, faces.astype(np.int64), name, label_id, class_id, info)


def mesh_mask(mask, affine, valid_mask=None, *, name="bone", label_id=None,
              class_id=None, metadata=None) -> Mesh:
    """Extract the visible instance mask; no cap is added at scan boundaries."""
    return _extract(_numpy(mask).astype(np.float32), affine, level=0.5,
                    gradient_direction="ascent", valid_mask=valid_mask, name=name,
                    label_id=label_id, class_id=class_id, metadata=metadata)


def mesh_field(field_dhw, affine, valid_mask=None, *, name="bone", label_id=None,
               class_id=None, metadata=None) -> Mesh:
    """Extract a negative-inside implicit field at zero in its own grid affine."""
    return _extract(field_dhw, affine, level=0.0, gradient_direction="descent",
                    valid_mask=valid_mask, name=name, label_id=label_id,
                    class_id=class_id, metadata=metadata)


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def export_meshes(meshes: list[Mesh], out_dir: str | Path, metadata: dict | None = None) -> dict[str, Path]:
    """Write one OBJ object per bone and JSON with frames/visibility/relationships.

    The caller supplies source affine, transforms, and relationships in metadata;
    export never manufactures anatomical axes or watertightness.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_path, json_path = out_dir / "bones.obj", out_dir / "bones.json"
    objects, offset = [], 0
    with obj_path.open("w") as stream:
        stream.write("# AnatoSKEL patient RAS coordinates, units millimetres\n")
        for index, mesh in enumerate(meshes):
            vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                raise ValueError("Mesh vertices must be finite Nx3 points")
            if faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0) or np.any(faces >= len(vertices)):
                raise ValueError("Mesh faces must index valid vertices")
            safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", mesh.name) or f"bone_{index}"
            stream.write(f"o {safe_name}\n")
            for vertex in vertices:
                stream.write("v %.8f %.8f %.8f\n" % tuple(vertex))
            for face in faces:
                stream.write("f %d %d %d\n" % tuple(face + offset + 1))
            offset += len(vertices)
            boundary_edges = mesh.boundary_edges
            objects.append({"name": safe_name, "label_id": mesh.label_id, "class_id": mesh.class_id,
                            "vertices": len(vertices), "faces": len(faces),
                            "boundary_edges": boundary_edges,
                            "closed_surface": bool(len(faces) and boundary_edges == 0),
                            "metadata": mesh.metadata})
    payload = {"units": "mm", "coordinate_system": "RAS", "meshes": objects,
               "source": metadata or {}, "boundary_capping_performed": False}
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n")
    return {"obj": obj_path, "json": json_path}
