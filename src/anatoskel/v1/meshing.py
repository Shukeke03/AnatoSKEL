"""Observed-domain mesh extraction with open acquisition boundaries.

The scalar field is never padded, and unknown voxels are never converted into
positive SDF samples. Triangle clipping trims surfaces; it does not create caps.
"""
from __future__ import annotations

import numpy as np

from anatoskel.data.scans import validate_affine
from anatoskel.geometry.meshes import Mesh, mesh_field, transform_points


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _clip_plane(polygon: np.ndarray, axis: int, bound: float, lower: bool) -> np.ndarray:
    if len(polygon) == 0:
        return polygon
    output = []
    previous = polygon[-1]
    previous_inside = previous[axis] >= bound if lower else previous[axis] <= bound
    for current in polygon:
        current_inside = current[axis] >= bound if lower else current[axis] <= bound
        if current_inside != previous_inside:
            fraction = (bound-previous[axis])/(current[axis]-previous[axis])
            intersection = previous+fraction*(current-previous)
            intersection[axis] = bound
            output.append(intersection)
        if current_inside:
            output.append(current)
        previous, previous_inside = current, current_inside
    return np.asarray(output, dtype=np.float64).reshape(-1, 3)


def _clip_acquisition(mesh: Mesh, original_domain: dict) -> Mesh:
    if not isinstance(original_domain, dict) or not {"shape_xyz", "affine"} <= set(original_domain):
        raise ValueError("Original acquisition domain needs shape_xyz and affine")
    shape = np.asarray(original_domain["shape_xyz"], dtype=np.float64)
    if (shape.shape != (3,) or not np.isfinite(shape).all() or (shape < 1).any()
            or not np.equal(shape, shape.round()).all()):
        raise ValueError("Acquisition shape_xyz must contain three positive integers")
    source_affine = validate_affine(original_domain["affine"])
    info = dict(mesh.metadata, original_acquisition_domain={
        "shape_xyz": shape.astype(int).tolist(), "affine": source_affine.tolist(),
    }, acquisition_clip_performed=True, clipping_adds_caps=False)
    if len(mesh.faces) == 0:
        return Mesh(mesh.vertices, mesh.faces, mesh.name, mesh.label_id, mesh.class_id, info)
    source_voxel = transform_points(mesh.vertices, np.linalg.inv(source_affine))
    lower, upper = np.full(3, -.5), shape-.5
    if ((source_voxel >= lower) & (source_voxel <= upper)).all():
        return Mesh(mesh.vertices, mesh.faces, mesh.name, mesh.label_id, mesh.class_id, info)
    triangles = []
    for face in mesh.faces:
        polygon = source_voxel[face]
        if (polygon.max(0) < lower).any() or (polygon.min(0) > upper).any():
            continue
        if not ((polygon >= lower) & (polygon <= upper)).all():
            for axis in range(3):
                polygon = _clip_plane(polygon, axis, lower[axis], lower=True)
                polygon = _clip_plane(polygon, axis, upper[axis], lower=False)
                if len(polygon) < 3:
                    break
        # Convex clipped triangles remain convex polygons; a fan preserves
        # original winding. There is intentionally no closing boundary face.
        for index in range(1, len(polygon)-1):
            triangles.extend((polygon[0], polygon[index], polygon[index+1]))
    if not triangles:
        info["empty_reason"] = "outside_acquisition_domain"
        return Mesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.int64),
                    mesh.name, mesh.label_id, mesh.class_id, info)
    points = np.asarray(triangles, dtype=np.float64)
    # Adjacent triangles compute the same cut-edge intersection from opposite
    # directions. Weld numerical roundoff only, without smoothing the surface.
    _, first, inverse = np.unique(points.round(8), axis=0, return_index=True, return_inverse=True)
    vertices = transform_points(points[first], source_affine)
    faces = inverse.reshape(-1, 3)
    triangle = vertices[faces]
    area2 = np.linalg.norm(np.cross(triangle[:, 1]-triangle[:, 0],
                                    triangle[:, 2]-triangle[:, 0]), axis=-1)
    distinct = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    faces = faces[distinct & (area2 > 1e-12)]
    used, compact = np.unique(faces, return_inverse=True)
    vertices, faces = vertices[used], compact.reshape(-1, 3).astype(np.int64)
    return Mesh(vertices, faces, mesh.name, mesh.label_id, mesh.class_id, info)


def mesh_observed(field_dhw, affine, scan_valid, original_domain=None, *, name="bone",
                  label_id=None, class_id=None, metadata=None) -> Mesh:
    """Extract a negative-inside mm field only where CT acquisition is valid.

    ``scan_valid`` is acquisition coverage in the extraction grid, never an
    annotation mask. The legacy extractor conservatively excludes cells next
    to invalid coverage and leaves outer array boundaries open. If supplied,
    ``original_domain`` clips triangles to the original (possibly oblique or
    reflected) CT voxel-cell support ``[-.5, shape_xyz-.5]``.
    """
    values, valid = _numpy(field_dhw), _numpy(scan_valid)
    if values.ndim != 3 or valid.shape != values.shape:
        raise ValueError("SDF and acquisition coverage must be matching DHW volumes")
    if not np.isin(valid, (False, True)).all():
        raise ValueError("Acquisition coverage must be binary")
    info = dict(metadata or {}, reconstruction_domain="observed_acquisition",
                boundary_capping_performed=False, unknown_sdf_filled=False,
                annotation_mask_used=False)
    mesh = mesh_field(values, affine, valid_mask=valid.astype(bool), name=name,
                      label_id=label_id, class_id=class_id, metadata=info)
    return _clip_acquisition(mesh, original_domain) if original_domain is not None else mesh
