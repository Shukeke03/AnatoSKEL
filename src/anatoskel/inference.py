"""Predicted-ROI inference and observed-domain OBJ export."""
from pathlib import Path

import numpy as np
import torch

from anatoskel.config import ModelConfig
from anatoskel.data import ScanDataset, crop_at_center, label_to_class
from anatoskel.geometry import export_meshes, mesh_field
from anatoskel.models import AnatoSKEL
from anatoskel.runtime import choose_device
from anatoskel.training.checkpoint import load_checkpoint
from anatoskel.training.losses import VERSE_IDS
from anatoskel.training.pipeline import coarse_scan, pack_context


def load_model(checkpoint, device="auto"):
    device = choose_device(device)
    payload = load_checkpoint(checkpoint, expected_kind="model")
    model = AnatoSKEL(ModelConfig.from_dict(payload["model_config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload, device


def select_detections(detection, threshold=0.5, label_ids=(20, 21, 22, 23, 24, 25)):
    """One slot per unique identity; no forced slot-to-vertebra ordering."""
    probabilities = detection["class_logits"][0].float().softmax(-1)
    scores, classes = probabilities.max(-1)
    chosen = {}
    allowed = set(label_ids)
    for query, (score, cls) in enumerate(zip(scores.tolist(), classes.tolist())):
        if cls >= len(VERSE_IDS) or score < threshold or VERSE_IDS[cls] not in allowed:
            continue
        label = VERSE_IDS[cls]
        if label not in chosen or score > chosen[label][1]:
            chosen[label] = (query, score)
    return [(query, label, score) for label, (query, score) in sorted(chosen.items())]


@torch.no_grad()
def grid_fields(model, context, state, rois, chunk_points=32768):
    shape = tuple(rois[0]["volume"].shape[-3:])
    device = state["beta"].device
    axes = [torch.arange(n, device=device, dtype=torch.float32) for n in shape]
    xyz = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).flip(-1)
    affine = torch.stack([r["affine"] for r in rois]).to(device).float()
    fields, valid = [], []
    for voxel_chunk in xyz.split(chunk_points):
        world = torch.einsum("nij,pj->npi", affine[:, :3, :3], voxel_chunk) + affine[:, None, :3, 3]
        output = model.field(context, state, world[None])
        fields.append(output["field"][0].cpu())
        valid.append(output["valid"][0].cpu())
    return torch.cat(fields, 1).reshape(len(rois), *shape), \
        torch.cat(valid, 1).reshape(len(rois), *shape)


def _boundary_negative(field, valid):
    boundary = torch.zeros_like(valid)
    boundary[0] = boundary[-1] = True
    boundary[:, 0] = boundary[:, -1] = True
    boundary[:, :, 0] = boundary[:, :, -1] = True
    return bool(((field < 0) & valid & boundary).any())


@torch.no_grad()
def predict_scan(model, scan, settings, device, *, score_threshold=None, label_ids=None):
    # Inference sees physical CT coverage, never annotation validity or GT labels.
    scan = dict(scan, labels=None, valid_mask=scan["ct_valid_mask"],
                metadata={**scan["metadata"], "original_truncated": {}})
    inference_cfg, data_cfg = settings.get("inference", {}), settings["data"]
    coarse = coarse_scan(scan, data_cfg)
    detection = model.encode_coarse(coarse["volume"][None].to(device),
                                     coarse["affine"][None].to(device).float())
    chosen = select_detections(detection,
        inference_cfg.get("score_threshold", 0.5) if score_threshold is None else score_threshold,
        inference_cfg.get("label_ids", [20, 21, 22, 23, 24, 25]) if label_ids is None else label_ids)
    if not chosen:
        return {"meshes": [], "detections": [], "reason": "no_confident_bones",
                "predicted_roi": True}
    query_indices = [c[0] for c in chosen]
    queries = detection["queries"][:, query_indices]
    anchors = detection["centers_mm"][:, query_indices]
    class_ids = torch.tensor([[label_to_class(c[1]) for c in chosen]], device=device)
    centers = anchors[0].cpu()
    del detection
    size, spacing = data_cfg["local_size"], data_cfg["local_spacing_mm"]
    # Expand only the physical sampling box when a predicted surface reaches an
    # artificial ROI boundary. Original CT coverage remains the extraction mask.
    expansion = 1.0
    for attempt in range(3):
        rois = [crop_at_center(scan, center, size=size, spacing_mm=spacing * expansion)
                for center in centers]
        context = pack_context(model, rois, queries, device)
        state = model.reconstruct(context, queries, class_ids, anchors)
        fields, valid = grid_fields(model, context, state, rois,
                                    inference_cfg.get("field_chunk_points", 32768))
        clipped = [_boundary_negative(f, v) for f, v in zip(fields, valid)]
        if not any(clipped) or attempt == 2:
            break
        expansion *= 1.5
    transforms = torch.eye(4, device=device).repeat(len(chosen), 1, 1)
    transforms[:, :3, :3] = state["rotation"][0]
    transforms[:, :3, 3] = state["translation"][0]
    relations = []
    for i in range(len(chosen) - 1):
        relations.append({"from_label": chosen[i][1], "to_label": chosen[i + 1][1],
                          "transform": (torch.linalg.inv(transforms[i]) @ transforms[i + 1]).cpu()})
    meshes, details = [], []
    for index, (query, label, score) in enumerate(chosen):
        info = {"query_index": query, "score": score, "predicted_roi": True,
                "rigid_transform": transforms[index].cpu(), "beta": state["beta"][0, index].cpu(),
                "pose_axes_validated": False, "visibility_scores": state["visibility_logits"][0, index].sigmoid().cpu(),
                "artificial_roi_truncated": clipped[index], "roi_expansion": expansion,
                "observed_fraction": float(valid[index].float().mean())}
        surface = mesh_field(fields[index], rois[index]["affine"], valid_mask=valid[index],
                              name=f"vertebra_{label}", label_id=label,
                              class_id=label_to_class(label), metadata=info)
        surface.metadata["surface_open"] = surface.boundary_edges > 0
        meshes.append(surface)
        details.append({"label_id": label, "score": score, "vertices": len(surface.vertices),
                        "faces": len(surface.faces), "artificial_roi_truncated": clipped[index]})
    return {"meshes": meshes, "detections": details, "relations": relations,
            "predicted_roi": True, "fields": fields, "valid": valid, "rois": rois}


def infer_manifest(checkpoint, manifest, output, *, device="auto", split="test",
                   allow_synthetic=False, score_threshold=None):
    model, payload, device = load_model(checkpoint, device)
    if payload.get("training_status") == "synthetic_check" and not allow_synthetic:
        raise ValueError("This checkpoint was trained on synthetic fixtures; use --allow-synthetic for software checks")
    reports = []
    for scan in ScanDataset(manifest, split=split):
        # Avoid using annotations to choose ROIs or identities at inference.
        prediction = predict_scan(model, scan, payload["run_config"], device,
                                   score_threshold=score_threshold)
        paths = export_meshes(prediction["meshes"], Path(output) / scan["metadata"]["scan_id"],
            metadata={**scan["metadata"], "original_truncated": {},
                      "checkpoint": str(Path(checkpoint).resolve()),
                      "training_status": payload.get("training_status"),
                      "pose_axes_validated": False, "predicted_roi": True,
                      "relations": prediction.get("relations", []),
                      "empty_reason": prediction.get("reason"),
                      "model_config": model.config.to_dict()})
        reports.append({"scan_id": scan["metadata"]["scan_id"],
                        "detections": prediction["detections"], **{k: str(v) for k, v in paths.items()}})
    return reports


def evaluate_manifest(checkpoint, manifest, output, *, device="auto", split="val",
                      allow_synthetic=False):
    """Predicted ROI volumetric Dice in native CT coordinates; missed bones score 0."""
    from scipy import ndimage
    model, payload, device = load_model(checkpoint, device)
    if payload.get("training_status") == "synthetic_check" and not allow_synthetic:
        raise ValueError("Synthetic checkpoint evaluation requires --allow-synthetic")
    records = []
    allowed = payload["run_config"].get("inference", {}).get("label_ids", [20, 21, 22, 23, 24, 25])
    for scan in ScanDataset(manifest, split=split):
        if scan["labels"] is None:
            raise ValueError("Evaluation requires masks")
        prediction = predict_scan(model, dict(scan, labels=None), payload["run_config"], device)
        fields_by_label = {d["label_id"]: i for i, d in enumerate(prediction["detections"])}
        native_labels, native_valid = scan["labels"].numpy(), scan["valid_mask"].numpy()
        native_affine = scan["affine"].numpy()
        for label in sorted(set(allowed) & {int(x) for x in torch.unique(scan["labels"]) if int(x)}):
            truth = (native_labels == label) & native_valid
            predicted = np.zeros_like(truth)
            if label in fields_by_label:
                index = fields_by_label[label]
                roi = prediction["rois"][index]
                mapping = np.linalg.inv(roi["affine"].numpy()) @ native_affine
                # scipy arrays are DHW: reverse the xyz mapping on both sides.
                matrix = mapping[:3, :3][::-1, ::-1]
                offset = mapping[:3, 3][::-1]
                binary = ((prediction["fields"][index] < 0) & prediction["valid"][index]).numpy()
                predicted = ndimage.affine_transform(binary.astype(np.uint8), matrix, offset,
                              output_shape=truth.shape, order=0, mode="constant", cval=0).astype(bool)
                predicted &= native_valid
            denominator = int(truth.sum() + predicted.sum())
            dice = 2 * int((truth & predicted).sum()) / denominator if denominator else 1.
            records.append({"scan_id": scan["metadata"]["scan_id"], "label_id": label,
                            "dice": dice, "detected": label in fields_by_label,
                            "predicted_roi": True})
    result = {"split": split, "bones": records,
              "mean_dice": float(np.mean([r["dice"] for r in records])) if records else None,
              "training_status": payload.get("training_status"),
              "clinical_validation": False}
    import json
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2))
    return result
