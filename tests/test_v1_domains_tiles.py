"""Acquisition-domain supervision and shared-coordinate tile regression tests."""
import numpy as np
import pytest
import torch

from anatoskel.v1.domains import input_scan, prepare_bone_supervision, synthetic_crop
from anatoskel.v1.tiles import (blend_residuals, extract_tile, overlap_consistency,
                               plan_tiles, tile_weights)


def scan_example(shape=(24, 24, 24), spacing=(1., 1., 1.), truncated=False):
    labels = torch.zeros(shape, dtype=torch.long)
    labels[slice(0 if truncated else 6, 15), 6:18, 6:18] = 20
    valid = torch.ones(shape, dtype=torch.bool)
    affine = torch.eye(4, dtype=torch.float64)
    affine[:3, :3] = torch.diag(torch.tensor(spacing, dtype=torch.float64))
    affine[:3, 3] = torch.tensor([10., -20., 30.])
    z, y, x = torch.meshgrid(*(torch.arange(n) for n in shape), indexing="ij")
    volume = (x.float() + 2*y.float() + 3*z.float())[None]
    return {"volume": volume, "labels": labels, "valid_mask": valid.clone(),
            "ct_valid_mask": valid, "condition_mask": valid.clone(),
            "artificial_padding_mask": torch.zeros_like(valid), "affine": affine,
            "metadata": {"units": "mm", "coordinate_system": "RAS", "scan_id": "example",
                         "original_shape_xyz": list(shape[::-1]),
                         "original_affine": affine.tolist(),
                         "ct_voxel_domain": {"shape_xyz": list(shape[::-1]),
                                             "affine": affine.tolist()},
                         "label_bounds": {"20": [[6, 18], [6, 18], [6, 15]]},
                         "original_truncated": {"20": truncated},
                         "annotation_domain_truncated": {"20": truncated}}}


def supervise(scan, **kwargs):
    return prepare_bone_supervision(scan, 20, torch.tensor([22., -8., 42.]),
                                    n_points=4096, n_surface=4096, seed=121, **kwargs)


def test_partial_surface_does_not_create_a_scan_boundary_cap():
    scan = scan_example(truncated=True)
    result = supervise(scan)
    assert result["complete"] is False
    # Bone reaches the lower acquisition z boundary. No observed -z face exists.
    assert not (result["normals"][:, 2] < -.9).any()
    assert (result["normals"][:, 2] > .9).any()  # Its genuine upper surface remains.
    assert result["occupancy_valid"].all()  # Visible edge voxels still have occupancy labels.
    assert result["sdf_valid"].any()
    boundary = torch.isclose(result["points_world"][:, 2], torch.tensor(30.))
    assert boundary.any()
    assert not result["sdf_valid"][boundary].any()


def test_anisotropic_unknown_boundary_is_not_an_edt_voxel_centre():
    scan = scan_example(spacing=(1., 1., 4.), truncated=True)
    result = supervise(scan, margin_mm=1.)
    boundary = torch.isclose(result["points_world"][:, 2], torch.tensor(30.))
    # Here physical unknown support begins 2 mm from the edge voxel centre.
    # A lateral surface >=1.5 mm away cannot be certified with a 1 mm margin.
    unsafe = boundary & (result["target_sdf_mm"].abs() >= 1.5)
    assert unsafe.any()
    assert not result["sdf_valid"][unsafe].any()


def test_completeness_uses_scan_and_annotation_domains_not_tight_gt_crop():
    scan = scan_example(shape=(64, 64, 64))
    # A tight computational supervision crop around this small bone is permitted.
    complete = supervise(scan)
    assert complete["complete"] is True
    # Remove an annotation half-space that cuts the bone, while CT remains present.
    scan["valid_mask"][:, :, :10] = False
    scan["labels"][~scan["valid_mask"]] = 0
    partial = supervise(scan)
    assert partial["complete"] is False
    assert not (partial["normals"][:, 0] < -.9).any()  # No fake annotation cut face.
    # Inference observes CT regardless of annotation holes.
    inputs = input_scan(scan)
    assert inputs["labels"] is None
    assert inputs["valid_mask"].all()
    assert not {"label_bounds", "original_truncated", "annotation_domain_truncated"} & set(
        inputs["metadata"])


def test_synthetic_crop_removes_hidden_gt_and_preserves_patient_coordinates():
    scan = scan_example(shape=(24, 25, 26))
    affine_before = scan["affine"].clone()
    volume_before = scan["volume"].clone()
    saw_shift = False
    for seed in range(8):
        cropped = synthetic_crop(scan, seed=seed, label_id=20, keep_fraction=.5)
        assert cropped["labels"] is None
        assert not {"label_bounds", "original_truncated", "annotation_domain_truncated"} & set(
            cropped["metadata"])
        assert tuple(cropped["metadata"]["original_shape_xyz"]) == cropped["volume"].shape[-3:][::-1]
        assert np.allclose(cropped["metadata"]["original_affine"], cropped["affine"])
        origin_voxel = torch.linalg.solve(affine_before[:3, :3],
                                         cropped["affine"][:3, 3]-affine_before[:3, 3])
        start_xyz = origin_voxel.round().long()
        assert torch.allclose(origin_voxel, start_xyz.double())
        saw_shift |= bool(start_xyz.any())
        start_zyx = start_xyz.flip(0).tolist()
        shape_zyx = cropped["volume"].shape[-3:]
        region = tuple(slice(start, start+size) for start, size in zip(start_zyx, shape_zyx))
        assert torch.equal(cropped["volume"], volume_before[(slice(None), *region)])
        assert cropped["ct_valid_mask"].shape == cropped["volume"].shape[-3:]
        assert torch.equal(cropped["condition_mask"], cropped["ct_valid_mask"])
    assert saw_shift
    assert torch.equal(scan["volume"], volume_before)
    assert torch.equal(scan["affine"], affine_before)
    assert scan["labels"] is not None


def test_tile_extraction_retains_roi_centre_gradient_in_world_mm():
    scan = scan_example()
    center = torch.tensor([22., -8., 42.], requires_grad=True)
    tile = extract_tile(input_scan(scan), center, size=8, spacing_mm=1.)
    assert tile["valid_mask"].all()
    tile["volume"].mean().backward()
    assert torch.allclose(center.grad, torch.tensor([1., 2., 3.]), atol=2e-5)
    assert tile["affine"].requires_grad


def test_tile_condition_holes_cannot_leak_hidden_ct_through_interpolation():
    scan = scan_example()
    scan["condition_mask"][:, :, 12:] = False
    center = torch.tensor([21.8, -8., 42.])  # Some known samples straddle the hidden neighbour.
    first = extract_tile(scan, center, size=4)
    changed = dict(scan, volume=scan["volume"].clone())
    changed["volume"][:, :, :, 12:] = 1e6
    second = extract_tile(changed, center, size=4)
    assert first["valid_mask"].any()
    assert torch.equal(first["valid_mask"], second["valid_mask"])
    assert torch.allclose(first["volume"], second["volume"], atol=1e-5)


def test_tile_plan_covers_extent_and_shares_anchor_gradient():
    anchor = torch.tensor([0., 0., 0.], requires_grad=True)
    bounds = torch.tensor([[-100., -20., -20.], [100., 20., 20.]])
    centers = plan_tiles(anchor, bounds, size=64, spacing_mm=1., overlap=16, margin_mm=0)
    half = 63/2
    assert centers[:, 0].min()-half <= -100
    assert centers[:, 0].max()+half >= 100
    centers.sum().backward()
    assert torch.equal(anchor.grad, torch.full((3,), float(len(centers))))
    with pytest.raises(ValueError, match="budget"):
        plan_tiles(anchor, bounds, size=64, overlap=16, max_tiles=1)


def test_blended_field_is_continuous_across_internal_tile_edges():
    first, second = torch.eye(4), torch.eye(4)
    second[0, 3] = 4
    # Internal physical tile support edge and its two neighbouring points.
    x = torch.tensor([8.5-1e-4, 8.5, 8.5+1e-4])
    points = torch.stack((x, torch.full_like(x, 4), torch.full_like(x, 4)), -1)
    weights = torch.stack([tile_weights(points, affine, 9)[0] for affine in (first, second)])
    residuals = torch.stack((torch.zeros(3), torch.ones(3)))
    blended, covered = blend_residuals(residuals, weights, required=torch.ones(3, dtype=torch.bool))
    assert covered.all()
    assert (blended[1:]-blended[:-1]).abs().max() < 1e-3
    # Check the former hard cutoff at edge voxel centre too.
    points[:, 0] = torch.tensor([8-1e-4, 8., 8+1e-4])
    weights = torch.stack([tile_weights(points, affine, 9)[0] for affine in (first, second)])
    blended, _ = blend_residuals(residuals, weights)
    assert (blended[1:]-blended[:-1]).abs().max() < 1e-3


def test_edge_voxel_centres_covered_and_holes_are_explicit():
    points = torch.tensor([[0., 4., 4.], [8., 4., 4.], [20., 4., 4.]])
    weights, _ = tile_weights(points, torch.eye(4), 9)
    assert (weights[:2] > 0).all()
    residuals = torch.tensor([[2., 3., 4.]], requires_grad=True)
    blended, covered = blend_residuals(residuals, weights[None])
    assert torch.equal(covered, torch.tensor([True, True, False]))
    assert torch.allclose(blended[:2], residuals[0, :2])
    blended[:2].sum().backward()
    assert torch.equal(residuals.grad, torch.tensor([[1., 1., 0.]]))
    with pytest.raises(ValueError, match="coverage"):
        blend_residuals(residuals, weights[None], required=torch.ones(3, dtype=torch.bool))
    consistent = torch.tensor([[2., 3.], [2., 3.]])
    assert overlap_consistency(consistent, torch.ones_like(consistent)) == 0
    assert overlap_consistency(consistent+torch.tensor([[0.], [1.]]),
                               torch.ones_like(consistent)) == 1


def test_repeated_physical_points_are_not_contrastive_negatives():
    from anatoskel.v1.losses import local_contrastive

    class Projection:
        def __init__(self):
            self.local_contrast_projection = torch.nn.Linear(2, 4)

    feature = torch.randn(1, 2, 8, 8, 8)
    context = {"decoded": feature, "valid_mask": torch.ones(1, 1, 8, 8, 8, dtype=torch.bool),
               "affine": torch.eye(4)[None]}
    state = {"z": torch.randn(1, 8, 16, requires_grad=True)}
    duplicate_points = torch.full((1, 6, 3), 3.)
    loss = local_contrastive(Projection(), state, [context, context], duplicate_points)
    assert torch.isclose(loss, torch.tensor(0.))
