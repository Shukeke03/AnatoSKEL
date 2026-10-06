"""Four-stage software workflow, stage gradients, and checkpoint continuation."""
import json

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from anatoskel.data.scans import generate_synthetic_dataset
from anatoskel.training.checkpoint import load_checkpoint
from anatoskel.v1.config import V1Config
from anatoskel.v1.pipeline import load_v1_model, train_v1


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def config_file(tmp_path, **updates):
    training = {"epochs": 3, "seed": 42, "learning_rate": 1e-4,
                "points_per_bone": 64, "surface_points": 64, "normal_points": 4,
                "label_ids": [20], "max_bones_per_step": 1,
                "max_tiles_per_bone": 64, "roi_warmup_steps": 100,
                "roi_jitter_mm": 0., "synthetic_crop_probability": 0.,
                "anchor_jitter_mm": 4., "amp": False, "checkpoint_every_steps": 1,
                "software_check": True}
    training.update(updates)
    path = tmp_path/"settings.yaml"
    path.write_text(yaml.safe_dump({"model": V1Config.smoke().to_dict(), "train": training}))
    return path


def make_partial_manifest(original_manifest):
    parent = original_manifest.parent
    first = json.loads(original_manifest.read_text().splitlines()[0])
    partial = dict(first, scan_id="synthetic_partial_train")
    for key in ("ct", "mask"):
        image = nib.load(parent/first[key])
        values = np.asarray(image.dataobj)[:, :, :11]  # Cut through L1; retain known lateral surface.
        name = f"synthetic_partial_{key}.nii.gz"
        cropped = nib.Nifti1Image(values, image.affine)
        cropped.header.set_xyzt_units("mm")
        nib.save(cropped, parent/name)
        partial[key] = name
    path = parent/"partial_manifest.jsonl"
    path.write_text(json.dumps(partial)+"\n"+json.dumps(first)+"\n")
    return path


def changed(before, after, prefix):
    return any(not torch.equal(value, after[key]) for key, value in before.items()
               if key.startswith(prefix) and torch.is_floating_point(value))


def test_crop_calibration_reencodes_full_gt_in_condition_predicted_frame(tmp_path, monkeypatch):
    from anatoskel.v1 import pipeline

    manifest = generate_synthetic_dataset(tmp_path/"fixtures", size=32)
    settings = config_file(tmp_path, synthetic_crop_probability=1.)
    shape = train_v1(settings, manifest, tmp_path/"shape", stage="shape", device="cpu", max_steps=1)
    ct = train_v1(settings, manifest, tmp_path/"ct", stage="ct", device="cpu",
                  init=shape["checkpoint"], max_steps=1)
    frames = []
    original = pipeline._supervision

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        frames.append(result)
        return result

    monkeypatch.setattr(pipeline, "_supervision", capture)
    result = train_v1(settings, manifest, tmp_path/"diffusion", stage="diffusion",
                      device="cpu", init=ct["checkpoint"], max_steps=1)
    payload = load_checkpoint(result["checkpoint"])
    recipe = payload["residual_calibration_recipe"]
    assert recipe["full_instances"] == 1 and recipe["crop_instances"] == 1
    assert int(payload["state_dict"]["diffusion.stats_count"]) == 2*V1Config.smoke().latent_tokens
    # Both targets retain the same COMPLETE patient surface, while their
    # coordinates change with the CT-only reference predicted after cropping.
    full, cropped = frames[:2]
    assert full["complete"] and cropped["complete"]
    assert torch.equal(full["surface_world"], cropped["surface_world"])
    for frame in (full, cropped):
        assert torch.allclose(frame["surface_u"]*64+frame["anchor_mm"],
                              frame["surface_world"], atol=1e-5)
    assert not torch.allclose(full["anchor_mm"], cropped["anchor_mm"])


def test_four_stage_training_and_partial_targets_checkpoint_resume(tmp_path, monkeypatch):
    manifest = generate_synthetic_dataset(tmp_path/"fixtures", size=32)
    settings = config_file(tmp_path)
    shape_result = train_v1(settings, manifest, tmp_path/"shape", stage="shape",
                            device="cpu", max_steps=1)
    shape_payload = load_checkpoint(shape_result["checkpoint"])
    assert shape_payload["schema_version"] == 2
    assert shape_payload["state_dict"]["shape_trained"]
    assert not shape_payload["state_dict"]["anchor_trained"]
    assert shape_payload["training_status"] == "synthetic_check"
    shape_resumed = train_v1(settings, manifest, tmp_path/"shape", stage="shape",
                             device="cpu", resume=shape_result["checkpoint"], max_steps=2)
    assert shape_resumed["steps"] == 2
    shape_payload = load_checkpoint(shape_resumed["checkpoint"])
    ct_result = train_v1(settings, manifest, tmp_path/"ct", stage="ct", device="cpu",
                         init=shape_result["checkpoint"], max_steps=1)
    ct_payload = load_checkpoint(ct_result["checkpoint"])
    assert ct_payload["state_dict"]["anchor_trained"]
    assert changed(shape_payload["state_dict"], ct_payload["state_dict"], "anchor_predictor.")
    assert not changed(shape_payload["state_dict"], ct_payload["state_dict"], "shape.")
    diffusion_result = train_v1(settings, manifest, tmp_path/"diffusion", stage="diffusion",
                                device="cpu", init=ct_result["checkpoint"], max_steps=1)
    diffusion_payload = load_checkpoint(diffusion_result["checkpoint"])
    assert diffusion_payload["state_dict"]["diffusion_trained"]
    assert diffusion_payload["state_dict"]["diffusion.stats_ready"]
    assert changed(ct_payload["state_dict"], diffusion_payload["state_dict"], "diffusion.")
    assert not changed(ct_payload["state_dict"], diffusion_payload["state_dict"], "anchor_predictor.")
    assert not changed(ct_payload["state_dict"], diffusion_payload["state_dict"], "shape.")

    # Trace diffusion calls: true partial must sample without constructing a GT
    # velocity target; the complete replay case uses the real v objective.
    from anatoskel.v1.diffusion import ResidualDiffusion
    calls = {"sample": 0, "loss": 0}
    original_sample, original_loss = ResidualDiffusion.sample, ResidualDiffusion.loss

    def sample(self, *args, **kwargs):
        calls["sample"] += 1
        assert kwargs["differentiable_tail"]
        return original_sample(self, *args, **kwargs)

    def loss(self, *args, **kwargs):
        calls["loss"] += 1
        assert kwargs["adapter_replay"]
        assert kwargs["complete"].all()
        return original_loss(self, *args, **kwargs)

    monkeypatch.setattr(ResidualDiffusion, "sample", sample)
    monkeypatch.setattr(ResidualDiffusion, "loss", loss)
    partial_manifest = make_partial_manifest(manifest)
    partial_result = train_v1(settings, partial_manifest, tmp_path/"partial", stage="partial",
                              device="cpu", init=diffusion_result["checkpoint"], max_steps=2)
    partial_payload = load_checkpoint(partial_result["checkpoint"])
    assert partial_payload["state_dict"]["adapters_trained"]
    assert calls == {"sample": 1, "loss": 1}
    before, after = diffusion_payload["state_dict"], partial_payload["state_dict"]
    adapter_keys = [key for key in before if ".adapter." in key]
    assert any(not torch.equal(before[key], after[key]) for key in adapter_keys)
    # In partial stage every trainable tensor is an adapter; fixed CT/AE/base
    # parameters and residual statistics must remain unchanged.
    assert all(torch.equal(value, after[key]) for key, value in before.items()
               if ".adapter." not in key and key != "adapters_trained")
    model, payload, device = load_v1_model(partial_result["checkpoint"], device="cpu")
    assert device.type == "cpu" and payload["stage"] == "partial"
    assert model.diffusion.stats_ready

    # Export executes actual CT tile encoding and diffusion. A known analytic
    # field makes this a meshing/provenance software test, not an accuracy test.
    monkeypatch.setattr(ResidualDiffusion, "sample", original_sample)
    monkeypatch.setattr(ResidualDiffusion, "loss", original_loss)
    from anatoskel.v1 import inference
    from anatoskel.v1.network import AnatoSKELV1
    real_field = AnatoSKELV1.field

    def sphere(self, state, contexts, points_world, **kwargs):
        result = real_field(self, state, contexts, points_world, **kwargs)
        result["sdf_mm"] = torch.linalg.vector_norm(
            points_world-state["anchors_mm"][:, None], dim=-1)-4
        return result

    def select_one(detection, threshold, allowed):
        # Tighten using the predicted CT query, never an annotation ROI.
        center = detection["centers_mm"][0, 0]
        detection["visible_bounds_mm"][0, 0] = torch.stack((center-3, center+3))
        return [(0, 20, .99)]

    monkeypatch.setattr(AnatoSKELV1, "field", sphere)
    monkeypatch.setattr(inference, "select_detections", select_one)
    with pytest.raises(ValueError, match="allow-synthetic"):
        inference.infer_manifest(partial_result["checkpoint"], manifest, tmp_path/"exports",
                                 device="cpu", split="test")
    exported = inference.infer_manifest(partial_result["checkpoint"], manifest, tmp_path/"exports",
                                        device="cpu", split="test", allow_synthetic=True)
    assert len(exported) == 1
    obj = (tmp_path/"exports"/"synthetic_test"/"bones_observed.obj").read_text()
    assert "\nv " in obj and "\nf " in obj
    report = json.loads((tmp_path/"exports"/"synthetic_test"/"bones.json").read_text())
    assert report["boundary_capping_performed"] is False
    assert report["source"]["completion_exported"] is False
    assert report["source"]["training_status"] == "synthetic_check"
    assert report["source"]["clinical_validation"] is False
    assert report["source"]["observed_domain_only"] is True
    assert not {"label_bounds", "original_truncated", "annotation_domain_truncated"} & set(
        report["source"])
    assert report["meshes"][0]["metadata"]["diffusion_steps"] == 4
    from anatoskel.data import ScanDataset
    scan = ScanDataset(manifest, split="test")[0]
    changed_annotations = dict(scan, labels=None, valid_mask=torch.zeros_like(scan["valid_mask"]))
    original = inference.predict_scan(model, scan, payload["run_config"], device)
    changed_prediction = inference.predict_scan(model, changed_annotations, payload["run_config"], device)
    assert torch.equal(original["fields"][0], changed_prediction["fields"][0])
    assert torch.equal(original["valid"][0], changed_prediction["valid"][0])


def test_stage_resume_rejects_manifest_or_stage_change(tmp_path):
    manifest = generate_synthetic_dataset(tmp_path/"fixtures", size=16)
    settings = config_file(tmp_path)
    result = train_v1(settings, manifest, tmp_path/"shape", stage="shape", device="cpu", max_steps=1)
    stopped = train_v1(settings, manifest, tmp_path/"same", stage="shape", device="cpu",
                       resume=result["checkpoint"], max_steps=1)
    assert stopped["steps"] == 1 and stopped["checkpoint"] == result["checkpoint"]
    with pytest.raises(ValueError, match="positive"):
        train_v1(settings, manifest, tmp_path/"zero", stage="shape", device="cpu", max_steps=0)
    with pytest.raises(ValueError, match="stage or manifest"):
        train_v1(settings, manifest, tmp_path/"wrong", stage="ct", device="cpu",
                 resume=result["checkpoint"], max_steps=2)
    changed_manifest = manifest.parent/"changed_manifest.jsonl"
    changed_manifest.write_text(manifest.read_text()+"\n")
    with pytest.raises(ValueError, match="stage or manifest"):
        train_v1(settings, changed_manifest, tmp_path/"wrong", stage="shape", device="cpu",
                 resume=result["checkpoint"], max_steps=2)
