"""Visible bone surfaces remain open and respect the actual CT physical FOV."""
import numpy as np
import pytest
import torch

from anatoskel.v1.meshing import mesh_observed


def sphere_field(shape=(32, 32, 32), center_xyz=(16., 16., 16.), radius=10.):
    z, y, x = np.indices(shape)
    points = np.stack((x, y, z), -1)
    return (np.linalg.norm(points-np.asarray(center_xyz), axis=-1)-radius).astype(np.float32)


def test_scan_truncated_sphere_retains_open_boundary_without_cap():
    field = sphere_field((32, 32, 18), center_xyz=(16., 16., 16.))
    domain = {"shape_xyz": [18, 32, 32], "affine": np.eye(4)}
    mesh = mesh_observed(field, np.eye(4), np.ones_like(field, dtype=bool), domain,
                         name="L1", label_id=20, class_id=19)
    assert len(mesh.faces) > 0
    assert mesh.boundary_edges > 0
    assert not mesh.metadata["boundary_capping_performed"]
    assert not mesh.metadata["unknown_sdf_filled"]
    # No triangle lies entirely on a fabricated flat cap at the x array edge.
    triangle_x = mesh.vertices[mesh.faces, 0]
    assert not np.all(np.isclose(triangle_x, 17), axis=1).any()
    assert mesh.label_id == 20 and mesh.class_id == 19


def test_complete_sphere_is_closed_without_expanding_acquisition_support():
    field = sphere_field()
    mesh = mesh_observed(field, np.eye(4), np.ones_like(field, dtype=bool),
                         {"shape_xyz": [32]*3, "affine": np.eye(4)})
    assert mesh.boundary_edges == 0
    assert mesh.vertices.min() >= 0
    assert mesh.vertices.max() <= 31


def test_unknown_field_changes_cannot_fabricate_an_interior_cap():
    field = sphere_field()
    valid = np.ones_like(field, dtype=bool)
    valid[:, :, 18:] = False
    first = mesh_observed(field, np.eye(4), valid)
    changed = field.copy()
    changed[~valid] = 1e6  # Would fabricate a cap if unknown values were used as background.
    second = mesh_observed(changed, np.eye(4), valid)
    assert np.array_equal(first.faces, second.faces)
    assert np.array_equal(first.vertices, second.vertices)
    assert first.boundary_edges > 0
    assert first.vertices[:, 0].max() <= 17


def test_oblique_reflected_domain_clips_triangles_exactly_and_keeps_boundary_open():
    field = sphere_field()
    angle = np.deg2rad(27)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    source = np.eye(4)
    source[:3, :3] = rotation @ np.diag([-1., 1.4, 1.2])
    shape = np.array([12, 24, 24])
    # Put the source FOV centre at the sphere centre. Its narrow reflected x
    # dimension cuts the sphere into an open band, with oblique cut planes.
    source[:3, 3] = np.array([16., 16., 16.])-source[:3, :3] @ ((shape-1)/2)
    mesh = mesh_observed(field, np.eye(4), np.ones_like(field, dtype=bool),
                         {"shape_xyz": shape, "affine": source})
    assert len(mesh.faces) > 0 and mesh.boundary_edges > 0
    inverse = np.linalg.inv(source)
    voxel = mesh.vertices @ inverse[:3, :3].T+inverse[:3, 3]
    assert (voxel >= -.5-1e-7).all()
    assert (voxel <= shape-.5+1e-7).all()
    clipped_vertex = np.isclose(voxel[:, 0], -.5) | np.isclose(voxel[:, 0], shape[0]-.5)
    assert clipped_vertex.any()
    # Clipping only existing triangles must not produce plane-only cap faces.
    for bound in (-.5, shape[0]-.5):
        assert not np.all(np.isclose(voxel[mesh.faces, 0], bound), axis=1).any()


def test_coverage_does_not_depend_on_annotation_metadata_and_accepts_tensors():
    field = torch.from_numpy(sphere_field())
    valid = torch.ones_like(field, dtype=torch.bool)
    first = mesh_observed(field, torch.eye(4), valid)
    second = mesh_observed(field, torch.eye(4), valid,
                          metadata={"label_bounds": {}, "annotation_domain_truncated": True})
    assert np.array_equal(first.vertices, second.vertices)
    assert np.array_equal(first.faces, second.faces)
    assert not second.metadata["annotation_mask_used"]


def test_empty_observed_domain_and_invalid_original_domain():
    field = sphere_field()
    mesh = mesh_observed(field, np.eye(4), np.zeros_like(field, dtype=bool))
    assert len(mesh.vertices) == 0 and len(mesh.faces) == 0
    with pytest.raises(ValueError, match="shape_xyz"):
        mesh_observed(field, np.eye(4), np.ones_like(field, dtype=bool), {"affine": np.eye(4)})
    with pytest.raises(ValueError, match="binary"):
        mesh_observed(field, np.eye(4), np.full_like(field, .5))
