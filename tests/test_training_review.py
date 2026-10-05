"""Independent integration checks for physical budgets and real optimization paths."""
from pathlib import Path

import numpy as np
import torch

from anatoskel.data import generate_synthetic_dataset
from anatoskel.training.checkpoint import load_checkpoint
from anatoskel.training.pipeline import coarse_scan, train_model, train_prior


def test_coarse_budget_changes_only_a_common_physical_spacing():
    affine = torch.eye(4, dtype=torch.float64)
    affine[:3, :3] = torch.diag(torch.tensor([0.7, 1.2, 3.0], dtype=torch.float64))
    scan = {"volume": torch.zeros(1, 18, 38, 84), "labels": None,
            "valid_mask": torch.ones(18, 38, 84, dtype=torch.bool),
            "affine": affine, "metadata": {}}
    output = coarse_scan(scan, {"coarse_spacing_mm": 1, "coarse_max_dim": 28,
                                "coarse_max_voxels": 6000})
    assert output["volume"].numel() <= 6000
    assert max(output["volume"].shape[-3:]) <= 28
    spacing = np.linalg.norm(output["affine"].numpy()[:3, :3], axis=0)
    assert np.allclose(spacing, spacing[0])
    # Extent can round outward by one output voxel, without independently
    # squeezing the scan axes to a cube and distorting anatomical proportions.
    original_extent = (np.array([84, 38, 18]) - 1) * np.array([0.7, 1.2, 3.0])
    output_extent = (np.array(output["volume"].shape[-3:][::-1]) - 1) * spacing
    assert np.all(output_extent >= original_extent - 1e-6)
    assert np.all(output_extent <= original_extent + spacing + 1e-6)


def test_synthetic_prior_then_two_stage_training_and_portable_checkpoint(tmp_path):
    torch.set_num_threads(2)
    manifest = generate_synthetic_dataset(tmp_path / "fixtures", size=32)
    configuration = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
    prior_result = train_prior(configuration, manifest, tmp_path / "prior", device="cpu",
                               epochs=1, max_steps=1)
    result = train_model(configuration, manifest, tmp_path / "training", device="cpu",
                         prior_path=prior_result["checkpoint"], max_steps=1)
    assert result["steps"] == 1
    assert result["last"]["fine_bones"] == 2
    assert np.isfinite(result["last"]["coarse_loss"] + result["last"]["fine_loss"])
    checkpoint = load_checkpoint(result["checkpoint"], expected_kind="model")
    assert checkpoint["step"] == 1
    assert checkpoint["prior_trained"]
    assert checkpoint["training_status"] == "synthetic_check"
    assert checkpoint["provenance"]["training_patients"] == ["synthetic_patient_0"]
    assert checkpoint["pose_supervision_valid"] is False
    # The reconstruction branch has undergone optimization rather than merely
    # serializing a forward-pass graph or a random initialized checkpoint.
    assert checkpoint["state_dict"]["backend.residual.output.weight"].abs().sum() > 0
