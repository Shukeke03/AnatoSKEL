"""Physical-coordinate and censorship checks, independent of model architecture."""
import json

import nibabel as nib
import numpy as np
import pytest
import torch

from anatoskel.data import (ScanDataset, canonical_from_mask, coarse_resample,
                           crop_at_center, crop_bone, generate_synthetic_dataset,
                           label_to_class, load_scan, mask_tsdf, read_manifest,
                           sample_supervision)
from anatoskel.geometry import export_meshes, kabsch, mesh_field, mesh_mask, transform_points
from anatoskel.data.scans import nifti_grids_match


def save_nifti(path, values, affine, units="mm"):
    image = nib.Nifti1Image(values, affine)
    image.header.set_xyzt_units(units)
    nib.save(image, path)


def test_nifti_xyz_tensor_dhw_and_reflected_affine(tmp_path):
    x, y, z = np.indices((5, 6, 7))
    ct = (x + 10 * y + 100 * z).astype(np.float32)
    mask = np.zeros(ct.shape, dtype=np.uint16)
    mask[2, 3, 4] = 28
    affine = np.array([[0, -2, 0, 10], [1.5, 0, 0, -7],
                       [0, 0, -3, 22], [0, 0, 0, 1]], dtype=float)
    save_nifti(tmp_path / "ct.nii.gz", ct, affine)
    save_nifti(tmp_path / "mask.nii.gz", mask, affine)
    scan = load_scan({"patient_id": "p", "scan_id": "s", "split": "train",
                      "ct": str(tmp_path / "ct.nii.gz"), "mask": str(tmp_path / "mask.nii.gz")})
    assert scan["volume"].shape == (1, 7, 6, 5)
    assert scan["labels"][4, 3, 2] == 28
    assert scan["volume"][0, 4, 3, 2].item() == pytest.approx(2 * (432 + 1000) / 3000 - 1)
    np.testing.assert_allclose(scan["affine"], affine)
    np.testing.assert_allclose(transform_points([[2, 3, 4]], affine), [[4, -4, 10]])
    assert label_to_class(1) == 0 and label_to_class(25) == 24 and label_to_class(28) == 25
    with pytest.raises(ValueError, match="Unsupported"):
        label_to_class(26)


def test_manifest_patient_leakage_duplicates_and_relative_paths(tmp_path):
    record = {"patient_id": "same_patient", "scan_id": "one", "split": "train", "ct": "ct.nii.gz"}
    manifest = tmp_path / "scans.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    assert read_manifest(manifest)[0].ct == (tmp_path / "ct.nii.gz").resolve()
    second = dict(record, scan_id="two", split="test", ct="other.nii.gz")
    manifest.write_text(json.dumps(record) + "\n" + json.dumps(second) + "\n")
    with pytest.raises(ValueError, match="Patient leakage"):
        read_manifest(manifest)
    second = dict(record, patient_id="different", scan_id="two")
    manifest.write_text(json.dumps(record) + "\n" + json.dumps(second) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(manifest)


def test_unknown_labels_and_missing_physical_units_fail(tmp_path):
    affine = np.eye(4)
    save_nifti(tmp_path / "ct.nii.gz", np.zeros((4, 4, 4), np.float32), affine, "unknown")
    record = {"patient_id": "p", "scan_id": "s", "split": "train", "ct": str(tmp_path / "ct.nii.gz")}
    with pytest.raises(ValueError, match="Spatial units"):
        load_scan(record)
    save_nifti(tmp_path / "ct.nii.gz", np.zeros((4, 4, 4), np.float32), affine)
    labels = np.zeros((4, 4, 4), np.uint16)
    labels[1, 1, 1] = 26
    save_nifti(tmp_path / "mask.nii.gz", labels, affine)
    with pytest.raises(ValueError, match="Unsupported mask labels"):
        load_scan(dict(record, mask=str(tmp_path / "mask.nii.gz")))


def test_missing_mask_units_inherit_only_identical_explicit_ct_grid(tmp_path):
    affine = np.diag([0.002, 0.003, 0.004, 1.0])
    ct_path, mask_path = tmp_path / "ct.nii.gz", tmp_path / "mask.nii.gz"
    save_nifti(ct_path, np.zeros((4, 4, 4), np.float32), affine, "meter")
    labels = np.zeros((4, 4, 4), np.uint8)
    labels[1, 1, 1] = 20
    save_nifti(mask_path, labels, affine, "unknown")
    record = {"patient_id": "p", "scan_id": "s", "split": "train",
              "ct": str(ct_path), "mask": str(mask_path)}
    scan = load_scan(record)
    np.testing.assert_allclose(scan["affine"], np.diag([2.0, 3.0, 4.0, 1.0]), atol=1e-6)
    assert scan["labels"][1, 1, 1] == 20
    assert nib.load(mask_path).header.get_xyzt_units()[0] == "unknown"
    shifted = affine.copy()
    shifted[0, 3] = 0.001
    save_nifti(mask_path, labels, shifted, "unknown")
    with pytest.raises(ValueError, match="matching explicit CT grid"):
        load_scan(record)


def test_grid_rounding_guard_checks_full_field_of_view():
    ct = np.diag([0.82, 0.82, 0.6, 1.0])
    mask = ct.copy()
    mask[2, 2] += 0.00009
    assert nifti_grids_match((556, 512, 1284), ct, mask)
    assert not nifti_grids_match((556, 512, 3000), ct, mask)
    shifted = ct.copy()
    shifted[0, 3] = 0.2
    assert not nifti_grids_match((556, 512, 1284), ct, shifted)


def test_resample_and_exact_physical_crop_preserve_frames(tmp_path):
    manifest = generate_synthetic_dataset(tmp_path, 32)
    scan = ScanDataset(manifest, "train")[0]
    coarse = coarse_resample(scan, 2.0)
    np.testing.assert_allclose(np.linalg.norm(coarse["affine"][:3, :3], axis=0), [2, 2, 2])
    assert coarse["metadata"]["original_affine"] == scan["metadata"]["original_affine"]
    centre = np.array([3.25, -2.2, 1.75])
    crop = crop_at_center(scan, centre, size=16, spacing_mm=1.0)
    np.testing.assert_allclose(transform_points([[7.5, 7.5, 7.5]], crop["affine"])[0], centre)
    assert crop["volume"].shape == (1, 16, 16, 16)
    crop_outside = crop_at_center(scan, [100, 0, 0], size=16)
    assert not crop_outside["valid_mask"].any()
    assert crop_outside["artificial_padding_mask"].all()
    assert crop_outside["metadata"]["ct_voxel_domain"] == scan["metadata"]["ct_voxel_domain"]
    crop_gt = crop_bone(scan, 20, size=32, spacing_mm=1)
    assert crop_gt["metadata"]["crop_label_id"] == 20
    assert not crop_gt["metadata"]["crop_artificial_boundary"]


def test_metric_tsdf_spacing_sign_and_shear_rejection():
    mask = torch.zeros((7, 7, 7), dtype=torch.bool)
    mask[:, :, 3] = True
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    sdf = mask_tsdf(mask, affine, 8)
    assert sdf[3, 3, 3] == pytest.approx(-0.25)
    assert sdf[3, 3, 4] == pytest.approx(0.25)
    assert sdf[3, 3, 6] == pytest.approx(0.75)
    sheared = affine.copy()
    sheared[0, 1] = 0.5
    with pytest.raises(ValueError, match="resample shear"):
        mask_tsdf(mask, sheared)


def test_shear_resampling_is_orthogonal_and_retains_coverage(tmp_path):
    affine = np.eye(4)
    affine[0, 1] = 0.25
    ct = np.ones((16, 16, 16), np.float32) * 400
    labels = np.zeros(ct.shape, np.uint16)
    labels[4:12, 4:12, 4:12] = 20
    save_nifti(tmp_path / "ct.nii.gz", ct, affine)
    save_nifti(tmp_path / "mask.nii.gz", labels, affine)
    scan = load_scan({"patient_id": "p", "scan_id": "s", "split": "train",
                      "ct": str(tmp_path / "ct.nii.gz"), "mask": str(tmp_path / "mask.nii.gz")})
    result = coarse_resample(scan, 1.0)
    basis = result["affine"][:3, :3].numpy()
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-6)
    assert result["valid_mask"].any() and result["artificial_padding_mask"].any()
    assert torch.isfinite(mask_tsdf(result["labels"] == 20, result["affine"])).all()


def test_censored_supervision_and_provisional_frames(tmp_path):
    scan = ScanDataset(generate_synthetic_dataset(tmp_path, 48), "train")[0]
    samples = sample_supervision(scan, 20, n_points=128, seed=1)
    voxels = samples["voxel_dhw"].numpy()
    expected = transform_points(voxels[:, ::-1], scan["affine"])
    np.testing.assert_allclose(samples["points_world"], expected, atol=1e-5)
    assert samples["valid"].all() and samples["target"].min() < 0 and samples["target"].max() > 0
    assert samples["class_id"] == 19
    frame = canonical_from_mask(scan["labels"] == 20, scan["affine"])
    assert np.linalg.det(frame["rotation"].numpy()) == pytest.approx(1)
    assert not frame["pose_valid"] and frame["provisional"]
    assert not frame["atlas_registration_done"]
    partial = canonical_from_mask(scan["labels"] == 20, scan["affine"], partial=True)
    assert not partial["geometry_stable"]
    assert len(read_manifest(tmp_path / "manifest.jsonl")) == 3
    assert len({record.patient_id for record in read_manifest(tmp_path / "manifest.jsonl")}) == 3


def test_kabsch_requires_correspondences_and_recovers_rigid_transform():
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    translation = np.array([2, -3, 4])
    result = kabsch(source, source @ rotation.T + translation)
    np.testing.assert_allclose(result["rotation"], rotation, atol=1e-8)
    np.testing.assert_allclose(result["translation"], translation, atol=1e-8)
    assert result["rms_mm"] < 1e-8 and not result["anatomical_axes_validated"]
    with pytest.raises(ValueError, match="Collinear"):
        kabsch([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0], [1, 0, 0], [2, 0, 0]])


def signed_volume(mesh):
    v, f = mesh.vertices, mesh.faces
    return np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6


def test_mesh_reflection_winding_mm_and_open_truncated_surface(tmp_path):
    d, h, w = np.indices((17, 17, 17))
    sphere = (d - 8) ** 2 + (h - 8) ** 2 + (w - 8) ** 2 <= 16
    affine = np.diag([-2.0, 1.5, 1.0, 1.0])
    affine[:3, 3] = [30, -10, 4]
    full = mesh_mask(sphere, affine, name="L1", label_id=20)
    assert signed_volume(full) > 0 and full.boundary_edges == 0
    np.testing.assert_allclose(full.vertices.mean(axis=0), [14, 2, 12], atol=1e-5)
    field = np.sqrt((d - 8) ** 2 + (h - 8) ** 2 + (w - 0) ** 2) - 5
    partial = mesh_field(field, np.eye(4), name="L2", label_id=21,
                         metadata={"original_truncated": True})
    assert partial.boundary_edges > 0
    triangles = partial.vertices[partial.faces]
    on_cut = triangles[np.all(np.isclose(triangles[:, :, 0], 0), axis=1)]
    # Exact iso-valued lattice nodes can make near-zero triangles, but never a cap.
    cap_area = np.linalg.norm(np.cross(on_cut[:, 1] - on_cut[:, 0], on_cut[:, 2] - on_cut[:, 0]), axis=1).sum() / 2
    assert cap_area < 1e-6
    paths = export_meshes([full, partial], tmp_path, metadata={"original_affine": affine})
    assert paths["obj"].read_text().count("\no ") == 2
    report = json.loads(paths["json"].read_text())
    assert report["units"] == "mm" and report["coordinate_system"] == "RAS"
    assert not report["boundary_capping_performed"]
    assert not report["meshes"][1]["closed_surface"]


def test_original_truncation_is_separate_from_crop_padding(tmp_path):
    labels = np.zeros((24, 24, 24), np.uint16)
    labels[:7, 7:17, 7:17] = 20
    affine = np.eye(4)
    save_nifti(tmp_path / "ct.nii.gz", np.zeros(labels.shape, np.float32), affine)
    save_nifti(tmp_path / "mask.nii.gz", labels, affine)
    scan = load_scan({"patient_id": "p", "scan_id": "s", "split": "train",
                      "ct": str(tmp_path / "ct.nii.gz"), "mask": str(tmp_path / "mask.nii.gz")})
    assert scan["metadata"]["original_truncated"]["20"]
    crop = crop_bone(scan, 20, size=32, spacing_mm=1)
    assert crop["metadata"]["original_truncated"]["20"]
    assert not crop["metadata"]["crop_artificial_boundary"]
    assert crop["artificial_padding_mask"].any()
    assert not crop["valid_mask"][crop["artificial_padding_mask"]].any()


def test_annotation_validity_does_not_remove_observable_ct(tmp_path):
    shape, affine = (24, 24, 24), np.eye(4)
    labels = np.zeros(shape, np.uint16)
    labels[7:12, 8:14, 8:14] = 20
    annotation = np.zeros(shape, np.uint8)
    annotation[:14] = 1
    save_nifti(tmp_path / "ct.nii.gz", np.zeros(shape, np.float32), affine)
    save_nifti(tmp_path / "mask.nii.gz", labels, affine)
    save_nifti(tmp_path / "annotation.nii.gz", annotation, affine)
    scan = load_scan({"patient_id": "p", "scan_id": "s", "split": "test",
                      "ct": str(tmp_path / "ct.nii.gz"), "mask": str(tmp_path / "mask.nii.gz"),
                      "valid_mask": str(tmp_path / "annotation.nii.gz")})
    assert scan["ct_valid_mask"].all()
    assert not scan["valid_mask"].all()
    crop = crop_at_center(scan, [18, 12, 12], size=6, spacing_mm=1)
    assert crop["ct_valid_mask"].all() and not crop["valid_mask"].any()
    outside = crop_at_center(scan, [24, 12, 12], size=8, spacing_mm=1)
    assert outside["ct_valid_mask"].any() and not outside["ct_valid_mask"].all()
    assert not outside["ct_valid_mask"][outside["artificial_padding_mask"]].any()
    coarse = coarse_resample(scan, 2)
    assert int(coarse["ct_valid_mask"].sum()) > int(coarse["valid_mask"].sum())


def test_explicit_ct_coverage_intersects_annotation_validity(tmp_path):
    shape, affine = (16, 16, 16), np.eye(4)
    ct_valid = np.ones(shape, np.uint8)
    ct_valid[:5] = 0
    annotation = np.ones(shape, np.uint8)
    save_nifti(tmp_path / "ct.nii.gz", np.zeros(shape, np.float32), affine)
    save_nifti(tmp_path / "ct_coverage.nii.gz", ct_valid, affine)
    save_nifti(tmp_path / "annotation.nii.gz", annotation, affine)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"patient_id": "p", "scan_id": "s", "split": "test",
                                    "ct": "ct.nii.gz", "valid_mask": "annotation.nii.gz",
                                    "ct_valid_mask": "ct_coverage.nii.gz"}) + "\n")
    scan = ScanDataset(manifest)[0]
    assert torch.equal(scan["valid_mask"], scan["ct_valid_mask"])
    assert not scan["ct_valid_mask"][:, :, :5].any()
    assert not scan["artificial_padding_mask"].any()  # Missing CT is not synthetic padding.
    cropped = crop_at_center(scan, [4, 8, 8], size=8, spacing_mm=1)
    assert not cropped["ct_valid_mask"].all()
    assert torch.equal(cropped["valid_mask"], cropped["ct_valid_mask"])
