"""Atlas provenance and weak-label ICP quality, not anatomical validation."""
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anatoskel.data import generate_synthetic_dataset, read_manifest
from anatoskel.geometry import build_atlas, fit_atlas, load_atlas


def asymmetric_cloud(seed=4):
    rng = np.random.default_rng(seed)
    core = rng.normal(size=(500, 3)) * [12, 7, 3]
    process = rng.normal(size=(100, 3)) * [2, 2, 3] + [8, 14, 8]
    return np.concatenate((core, process))


def test_trimmed_rigid_icp_recovers_rigid_transform_with_outliers():
    template = asymmetric_cloud()
    rotation = Rotation.from_euler("xyz", [0.3, -0.4, 0.8]).as_matrix()
    translation = np.array([40, -25, 70])
    target = template @ rotation.T + translation
    rng = np.random.default_rng(8)
    observed = np.concatenate((target, rng.normal(size=(30, 3)) * 2 + [150, 160, 170]))
    result = fit_atlas(template, observed, max_rms_mm=1, max_points=2048)
    np.testing.assert_allclose(result["rotation"], rotation, atol=1e-5)
    np.testing.assert_allclose(result["translation"], translation, atol=1e-5)
    assert np.linalg.det(result["rotation"]) == pytest.approx(1)
    assert result["trimmed_fit"] and result["rms_mm"] < 1e-5
    assert result["inlier_coverage"] > 0.9
    assert result["pose_valid"] and result["weak_pose_label"]
    assert not result["anatomical_axes_validated"]


def test_partial_and_symmetry_never_produce_trusted_pose():
    cloud = asymmetric_cloud()
    result = fit_atlas(cloud, cloud + [2, 3, 4], partial=True)
    assert result["partial"] and not result["pose_valid"]
    coordinates = np.array(np.meshgrid(np.arange(-3, 4), np.arange(-3, 4),
                                       np.arange(-3, 4), indexing="ij")).reshape(3, -1).T
    symmetric = coordinates[np.isclose(np.linalg.norm(coordinates, axis=1), 3)]
    symmetry = fit_atlas(symmetric, symmetric + [10, 20, 30])
    assert symmetry["ambiguous"] and not symmetry["pose_valid"]


def test_atlas_train_only_provenance_units_and_labels(tmp_path):
    points = asymmetric_cloud().tolist()
    atlas_path = tmp_path / "atlas.json"
    payload = {"units": "mm", "training_patients": ["p_train"],
               "classes": {"20": {"points": points}}}
    atlas_path.write_text(json.dumps(payload))
    atlas = load_atlas(atlas_path, ["p_train"])
    assert atlas["classes"]["20"]["class_id"] == 19
    assert isinstance(atlas["classes"]["20"]["points"], np.ndarray)
    with pytest.raises(ValueError, match="outside"):
        load_atlas(atlas_path, ["p_other"])
    payload["units"] = "voxels"
    atlas_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="units"):
        load_atlas(atlas_path, ["p_train"])
    payload["units"] = "mm"
    payload["classes"] = {"26": {"points": points}}
    atlas_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unsupported"):
        load_atlas(atlas_path, ["p_train"])


def test_build_atlas_uses_training_exemplars_and_marks_provisional_axes(tmp_path):
    manifest = generate_synthetic_dataset(tmp_path / "data", size=32)
    path = build_atlas(manifest, tmp_path / "atlas.json", max_points=128)
    records = read_manifest(manifest)
    train_patients = [r.patient_id for r in records if r.split == "train"]
    atlas = load_atlas(path, train_patients)
    assert atlas["training_patients"] == train_patients
    assert set(atlas["classes"]) == {"20", "21"}
    assert atlas["canonical_status"] == "provisional_pca_gauge"
    assert not atlas["anatomical_axes_validated"]
    for entry in atlas["classes"].values():
        assert entry["patient_id"] in train_patients
        assert entry["frame_status"] == "provisional_pca"
        assert len(entry["points"]) <= 128


def test_malformed_or_degenerate_atlas_points_fail():
    with pytest.raises(ValueError, match="finite"):
        fit_atlas(np.zeros((8, 3)), np.zeros((8, 3)))
    planar = np.column_stack((np.arange(20), np.arange(20) ** 2, np.zeros(20)))
    with pytest.raises(ValueError, match="three spatial"):
        fit_atlas(planar, planar)
