"""CT-only v1 inference: one diffusion sample per bone and open observed OBJ."""
from pathlib import Path
import json

import numpy as np
from scipy import ndimage
import torch

from anatoskel.data import ScanDataset
from anatoskel.data.scans import label_to_class
from anatoskel.geometry import export_meshes
from anatoskel.inference import select_detections
from anatoskel.models.backend import sample_volume, world_to_voxel
from .domains import input_scan
from .pipeline import load_v1_model, encode_scan, prepare_instances, amp_context
from .meshing import mesh_observed


def _grid(tiles, spacing_mm):
    lower = torch.stack([tile["affine"][:3, 3].detach().cpu() for tile in tiles]).min(0).values
    upper = torch.stack([tile["affine"][:3, 3].detach().cpu()
                         +(tile["volume"].shape[-1]-1)*spacing_mm for tile in tiles]).max(0).values
    shape_xyz = torch.ceil((upper-lower)/spacing_mm).long()+1
    affine = torch.eye(4)
    affine[:3, :3] *= spacing_mm
    affine[:3, 3] = lower
    return tuple(int(n) for n in shape_xyz.flip(0)), affine


@torch.no_grad()
def fused_grid(model, state, contexts, tiles, source_scan, device, chunk_points=2048):
    shape, affine = _grid(tiles, model.config.local_spacing_mm)
    total = int(np.prod(shape))
    if total > 128_000_000:
        raise ValueError("Output grid budget exceeded; predicted geometry needs review")
    memory = model.shape.prepare_memory(state["z"])
    values = torch.empty(total)
    coverage = torch.empty(total, dtype=torch.bool)
    scan_valid = torch.empty(total, dtype=torch.bool)
    source_mask = source_scan["ct_valid_mask"][None, None].float()
    source_affine = source_scan["affine"][None].float()
    for start in range(0, total, chunk_points):
        flat = torch.arange(start, min(start+chunk_points, total))
        xyz = torch.stack((flat % shape[2], flat//shape[2] % shape[1],
                           flat//(shape[2]*shape[1])), -1).float()
        world_cpu = xyz @ affine[:3, :3].T+affine[:3, 3]
        voxel = world_to_voxel(world_cpu[None, None], source_affine)
        acquisition = sample_volume(source_mask, voxel, mode="nearest")[0, 0, :, 0] > .5
        field = model.field(state, contexts, world_cpu[None].to(device), shape_memory=memory)
        values[start:start+len(flat)] = field["sdf_mm"][0].float().cpu()
        coverage[start:start+len(flat)] = field["valid"][0].cpu()
        scan_valid[start:start+len(flat)] = acquisition
    coverage, scan_valid = coverage.reshape(shape), scan_valid.reshape(shape)
    if bool((scan_valid & ~coverage).any()):
        raise RuntimeError("Artificial tile coverage gap inside observed CT")
    return values.reshape(shape), scan_valid, affine


def _continuing_faces(field, valid, affine, source):
    """Only expand a ROI boundary if actual CT continues beyond that face."""
    result = []
    source_mask = source["ct_valid_mask"][None, None].float()
    for axis in range(3):
        for side in (0, -1):
            face = [slice(None)]*3
            face[axis] = side
            face = tuple(face)
            positions = ((field[face] < 0) & valid[face]).nonzero()
            if not len(positions):
                continue
            full = torch.empty((len(positions), 3))
            other = [i for i in range(3) if i != axis]
            full[:, other] = positions.float()
            full[:, axis] = -1 if side == 0 else field.shape[axis]
            world = full.flip(-1) @ affine[:3, :3].T+affine[:3, 3]
            voxel = world_to_voxel(world[None, None], source["affine"][None].float())
            continues = sample_volume(source_mask, voxel, mode="nearest")[0, 0, :, 0] > .5
            if continues.any():
                result.append((2-axis, side))
    return result


@torch.no_grad()
def predict_scan(model, scan, settings, device, *, score_threshold=None):
    if not all(bool(getattr(model, name)) for name in ("shape_trained", "anchor_trained", "diffusion_trained")):
        raise ValueError("Inference requires completed shape, CT and diffusion training stages")
    input_ct = input_scan(scan)
    options = settings.get("inference", {})
    with amp_context(device, settings["train"].get("amp", True)):
        _, _, detection = encode_scan(model, input_ct, device)
    chosen = select_detections(detection, options.get("score_threshold", .5)
                               if score_threshold is None else score_threshold,
                               options.get("label_ids", [20, 21, 22, 23, 24, 25]))
    meshes, details, fields, valids, grids = [], [], [], [], []
    for query, label, score in chosen:
        local_detection = dict(detection)
        local_detection["visible_bounds_mm"] = detection["visible_bounds_mm"].clone()
        for attempt in range(int(options.get("coverage_attempts", 3))):
            with amp_context(device, settings["train"].get("amp", True)):
                state, contexts, _, tiles = prepare_instances(model, input_ct, local_detection,
                    [(query, label)], device, max_tiles=int(options.get("max_tiles_per_bone", 64)))
                state = model.sample_instances(state, seed=int(options.get("seed", 42))+query)
                field, valid, affine = fused_grid(model, state, contexts, tiles, scan, device,
                                                   int(options.get("field_chunk_points", 2048)))
            continuing = _continuing_faces(field, valid, affine, scan)
            if not continuing:
                break
            shape_xyz = torch.tensor(field.shape[::-1])
            lower = affine[:3, 3].clone()
            upper = lower+(shape_xyz-1)*model.config.local_spacing_mm
            for axis, side in continuing:
                if side == 0:
                    lower[axis] -= model.config.local_spacing_mm*model.config.tile_overlap
                else:
                    upper[axis] += model.config.local_spacing_mm*model.config.tile_overlap
            local_detection["visible_bounds_mm"][0, query] = torch.stack((lower, upper)).to(device)
        else:
            raise RuntimeError("Bone reaches an artificial ROI boundary; refusing a misleading observed OBJ")
        info = {"query_index": query, "score": score, "predicted_roi": True,
                "anchor_mm": state["anchors_mm"][0].cpu(), "tile_count": len(tiles),
                "diffusion_steps": model.config.diffusion_sample_steps,
                "adapter_enabled": bool(model.adapters_trained),
                "observed_domain_only": True, "coverage_gap": False,
                "visibility_scores": state["visibility_logits"][0].sigmoid().float().cpu(),
                "visibility_calibrated": False}
        mesh = mesh_observed(field, affine, valid,
            original_domain=scan["metadata"].get("ct_voxel_domain"),
            name=f"vertebra_{label}", label_id=label, class_id=label_to_class(label), metadata=info)
        meshes.append(mesh)
        details.append({"label_id": label, "score": score, "vertices": len(mesh.vertices),
                        "faces": len(mesh.faces), "boundary_edges": mesh.boundary_edges,
                        "tile_count": len(tiles)})
        fields.append(field)
        valids.append(valid)
        grids.append(affine)
    return {"meshes": meshes, "detections": details, "fields": fields,
            "valid": valids, "grid_affines": grids,
            "reason": None if chosen else "no_confident_bones"}


def infer_manifest(checkpoint, manifest, output, *, device="auto", split="test",
                   allow_synthetic=False, score_threshold=None):
    model, payload, device = load_v1_model(checkpoint, device)
    if payload.get("training_status") == "synthetic_check" and not allow_synthetic:
        raise ValueError("Synthetic weights require --allow-synthetic")
    reports = []
    for scan in ScanDataset(manifest, split):
        prediction = predict_scan(model, scan, payload["run_config"], device,
                                   score_threshold=score_threshold)
        metadata = {key: value for key, value in scan["metadata"].items()
                    if key not in ("label_bounds", "original_truncated", "annotation_domain_truncated")}
        paths = export_meshes(prediction["meshes"], Path(output)/scan["metadata"]["scan_id"],
            metadata={**metadata, "architecture": "anatoskel_v1", "version": "1.0.0",
                      "checkpoint": str(checkpoint), "model_prediction": True,
                      "training_status": payload.get("training_status"),
                      "clinical_validation": False,
                      "observed_domain_only": True, "completion_exported": False,
                      "empty_reason": prediction["reason"]})
        # Keep the explicit observed filename while retaining the common exporter.
        observed = paths["obj"].with_name("bones_observed.obj")
        paths["obj"].replace(observed)
        paths["obj"] = observed
        reports.append({"scan_id": scan["metadata"]["scan_id"],
                        "detections": prediction["detections"],
                        **{key: str(value) for key, value in paths.items()}})
    return reports


def evaluate_manifest(checkpoint, manifest, output, *, device="auto", split="val",
                      allow_synthetic=False):
    model, payload, device = load_v1_model(checkpoint, device)
    if payload.get("training_status") == "synthetic_check" and not allow_synthetic:
        raise ValueError("Synthetic weights require --allow-synthetic")
    records = []
    allowed = set(payload["run_config"].get("inference", {}).get("label_ids", [20, 21, 22, 23, 24, 25]))
    for scan in ScanDataset(manifest, split):
        if scan["labels"] is None:
            raise ValueError("Evaluation requires annotations")
        prediction = predict_scan(model, scan, payload["run_config"], device)
        native_affine = scan["affine"].numpy()
        validity = scan["valid_mask"].numpy()
        present = set(int(x) for x in torch.unique(scan["labels"])) & allowed
        for label in sorted(present):
            truth = (scan["labels"].numpy() == label) & validity
            predicted = np.zeros_like(truth)
            found = next((i for i, item in enumerate(prediction["detections"])
                          if item["label_id"] == label), None)
            if found is not None:
                mapping = np.linalg.inv(prediction["grid_affines"][found].numpy()) @ native_affine
                binary = (prediction["fields"][found] < 0) & prediction["valid"][found]
                predicted = ndimage.affine_transform(binary.numpy().astype(np.uint8),
                    mapping[:3, :3][::-1, ::-1], mapping[:3, 3][::-1],
                    output_shape=truth.shape, order=0, mode="constant", cval=0).astype(bool) & validity
            denominator = int(predicted.sum()+truth.sum())
            records.append({"scan_id": scan["metadata"]["scan_id"], "label_id": label,
                            "detected": found is not None,
                            "dice": 2*int((predicted & truth).sum())/denominator if denominator else 1.})
    result = {"architecture": "anatoskel_v1", "split": split, "bones": records,
              "mean_dice": float(np.mean([item["dice"] for item in records])) if records else None,
              "clinical_validation": False}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2)+"\n")
    return result
