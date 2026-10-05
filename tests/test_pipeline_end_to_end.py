"""Software-path tests on ellipsoids; these never establish medical accuracy."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from anatoskel.cli import main
from anatoskel.data import ScanDataset, generate_synthetic_dataset
from anatoskel.inference import infer_manifest, load_model, predict_scan, select_detections
from anatoskel.training.checkpoint import load_checkpoint
from anatoskel.training.pipeline import train_model, train_prior


@pytest.fixture(scope="module")
def software_workflow(tmp_path_factory):
    torch.set_num_threads(1)
    directory = tmp_path_factory.mktemp("pipeline_software")
    manifest = generate_synthetic_dataset(directory / "fixtures")
    settings = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml").read_text())
    settings["train"]["epochs"] = 3
    # These fixtures span only a few millimetres. Use an explicit software-test
    # normalization scale so a small warm-start can resolve their tiny geometry.
    settings["model"]["coordinate_scale_mm"] = 10.0
    config = directory / "software.yaml"
    config.write_text(yaml.safe_dump(settings))
    # This warm-start only makes a zero crossing possible for extraction tests.
    # Its training data remain synthetic and no reconstruction quality is claimed.
    prior = train_prior(config, manifest, directory / "prior", device="cpu", epochs=100,
                        max_steps=200)
    first = train_model(config, manifest, directory / "first", device="cpu",
                        prior_path=prior["checkpoint"], max_steps=1)
    return {"directory": directory, "manifest": manifest, "settings": settings,
            "config": config, "prior": prior, "first": first}


def test_detection_selection_filters_no_object_threshold_and_duplicate_ids():
    logits = torch.full((1, 6, 27), -10.0)
    logits[0, 0, 19] = 3  # VerSe 20, weaker duplicate.
    logits[0, 1, 19] = 6
    logits[0, 2, 26] = 9  # No object.
    logits[0, 3, 25] = 9  # VerSe T13, outside the configured lumbar set.
    logits[0, 4] = 0
    logits[0, 4, 21] = 0.1  # A weak maximum still fails confidence filtering.
    logits[0, 5, 20] = 5  # VerSe 21.
    selected = select_detections({"class_logits": logits}, threshold=0.5)
    assert [(query, label) for query, label, _ in selected] == [(1, 20), (5, 21)]


def test_resume_reached_limit_and_increment_match_uninterrupted_training(software_workflow):
    workflow = software_workflow
    first_path = Path(workflow["first"]["checkpoint"])
    original_sha = hashlib.sha256(first_path.read_bytes()).hexdigest()
    reached = train_model(workflow["config"], workflow["manifest"], workflow["directory"] / "noop",
                          device="cpu", resume=first_path, max_steps=1)
    assert reached["steps"] == 1
    assert reached["reason"] == "requested_step_limit_already_reached"
    assert hashlib.sha256(first_path.read_bytes()).hexdigest() == original_sha
    resumed = train_model(workflow["config"], workflow["manifest"], workflow["directory"] / "resumed",
                          device="cpu", resume=first_path, max_steps=2)
    uninterrupted = train_model(workflow["config"], workflow["manifest"], workflow["directory"] / "whole",
                                device="cpu", prior_path=workflow["prior"]["checkpoint"], max_steps=2)
    resumed_payload = load_checkpoint(resumed["checkpoint"])
    whole_payload = load_checkpoint(uninterrupted["checkpoint"])
    assert resumed["steps"] == uninterrupted["steps"] == 2
    assert len(resumed_payload["training_log"]) == 2
    for key, value in whole_payload["state_dict"].items():
        assert torch.equal(value, resumed_payload["state_dict"][key]), key


def _forced_single_class_checkpoint(workflow):
    """Force routing only: the encoder, pose/refinement and field stay real."""
    payload = load_checkpoint(workflow["first"]["checkpoint"])
    state = payload["state_dict"]
    state["detector.classifier.weight"].zero_()
    state["detector.classifier.bias"].fill_(-20)
    state["detector.classifier.bias"][19] = 20  # VerSe L1=20.
    payload["software_test_forced_classifier"] = True
    path = workflow["directory"] / "forced_classifier_software_only.pt"
    torch.save(payload, path)
    return path


def test_predicted_roi_actual_backend_field_and_physical_mesh(software_workflow):
    workflow = software_workflow
    model, _, device = load_model(_forced_single_class_checkpoint(workflow), "cpu")
    scan = ScanDataset(workflow["manifest"], split="test")[0]
    scan["labels"] = None
    scan["metadata"]["original_truncated"] = {}
    prediction = predict_scan(model, scan, workflow["settings"], device)
    assert prediction["predicted_roi"]
    assert [bone["label_id"] for bone in prediction["detections"]] == [20]
    assert torch.isfinite(prediction["fields"]).all()
    assert prediction["fields"].min() < 0 < prediction["fields"].max()
    surface = prediction["meshes"][0]
    assert len(surface.faces) > 0
    assert surface.metadata["predicted_roi"]
    assert surface.metadata["coordinate_system"] == "RAS"
    assert surface.metadata["units"] == "mm"
    assert surface.metadata["boundary_padding_added"] is False
    inverse = np.linalg.inv(scan["affine"].numpy())
    source_xyz = surface.vertices @ inverse[:3, :3].T + inverse[:3, 3]
    shape_xyz = np.array(scan["volume"].shape[-3:][::-1])
    assert np.all(source_xyz >= -0.5 - 1e-5)
    assert np.all(source_xyz <= shape_xyz - 0.5 + 1e-5)


def test_cli_annotation_free_inference_exports_obj_and_synthetic_guard(software_workflow, monkeypatch, capsys):
    workflow = software_workflow
    checkpoint = _forced_single_class_checkpoint(workflow)
    # Real inference manifests need only CT; masks are absent from the records.
    records = [json.loads(line) for line in workflow["manifest"].read_text().splitlines()]
    for record in records:
        record.pop("mask")
        record["ct"] = str((workflow["manifest"].parent / record["ct"]).resolve())
    manifest = workflow["directory"] / "ct_only.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="synthetic"):
        infer_manifest(checkpoint, manifest, workflow["directory"] / "guard", device="cpu")
    monkeypatch.setattr("sys.argv", ["anatoskel", "infer", "--checkpoint", str(checkpoint),
        "--manifest", str(manifest), "--output", str(workflow["directory"] / "cli_infer"),
        "--device", "cpu", "--allow-synthetic"])
    main()
    report = json.loads(capsys.readouterr().out)
    assert len(report) == 1
    obj = Path(report[0]["obj"]).read_text()
    metadata = json.loads(Path(report[0]["json"]).read_text())
    assert "o vertebra_20\n" in obj
    assert "\nv " in obj and "\nf " in obj
    assert metadata["units"] == "mm"
    assert metadata["source"]["training_status"] == "synthetic_check"
    assert metadata["source"]["predicted_roi"]
    assert metadata["source"]["original_truncated"] == {}
    assert metadata["meshes"][0]["label_id"] == 20
    assert metadata["meshes"][0]["faces"] > 0
    assert metadata["boundary_capping_performed"] is False
