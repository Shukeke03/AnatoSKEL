"""Training-split prior fitting and memory-conscious coarse/fine optimization."""
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from anatoskel.config import ModelConfig, load_config
from anatoskel.data import (ScanDataset, canonical_from_mask, coarse_resample, crop_at_center,
                           crop_bone, label_to_class, read_manifest, sample_supervision)
from anatoskel.data.scans import bone_voxels_world
from anatoskel.models import AnatoSKEL
from anatoskel.geometry import fit_atlas, load_atlas, surface_points
from anatoskel.models.backend import AnatomicalShapePrior, world_to_canonical
from anatoskel.runtime import choose_device
from .checkpoint import load_checkpoint, save_checkpoint
from .losses import detector_loss, field_loss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def provenance(manifest):
    records = read_manifest(manifest)
    return {"manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
            "training_patients": sorted({r.patient_id for r in records if r.split == "train"}),
            "synthetic": all(r.patient_id.startswith("synthetic_") for r in records)}


def coarse_scan(scan, data_config):
    """Keep the entire FOV with a common physical voxel scale and bounded volume."""
    spacing = float(data_config.get("coarse_spacing_mm", 3))
    shape = np.array(scan["volume"].shape[-3:][::-1])
    affine = scan["affine"].numpy()
    extent = (shape - 1) * np.linalg.norm(affine[:3, :3], axis=0)
    max_dim = int(data_config.get("coarse_max_dim", 192))
    max_voxels = int(data_config.get("coarse_max_voxels", 4_000_000))
    if max_dim < 16 or max_voxels < 4096:
        raise ValueError("Coarse budget is too small")
    spacing = max(spacing, float(extent.max()) / (max_dim - 1))
    while np.prod(np.ceil(extent / spacing).astype(int) + 1) > max_voxels:
        spacing *= 1.05
    result = coarse_resample(scan, spacing)
    # If resampling rounds just beyond the budget, enlarge isotropic spacing.
    while result["volume"].numel() > max_voxels or max(result["volume"].shape[-3:]) > max_dim:
        spacing *= 1.05
        result = coarse_resample(scan, spacing)
    return result


def pack_context(model, rois, queries, device):
    volume = torch.stack([r["volume"] for r in rois]).to(device)
    affine = torch.stack([r["affine"] for r in rois]).to(device).float()
    context = model.encode_fine(volume, affine)
    context["valid_mask"] = torch.stack([r["ct_valid_mask"] for r in rois]).unsqueeze(1).to(device)
    context["annotation_valid_mask"] = torch.stack([r["valid_mask"] for r in rois]).unsqueeze(1).to(device)
    context["bone_batch_shape"] = (1, len(rois))
    context["mask_logits"] = model.fine_masks(context, queries)
    context = model.backend.prepare_context(context)
    # The field only needs the projected pyramid. Autograd retains its own
    # required tensors; avoid duplicate high-resolution cache references.
    context.pop("features")
    context.pop("decoded")
    return context


def _amp(device, enabled):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and enabled \
        else nullcontext()


def training_dataset(settings, manifest):
    cache_dir = settings.get("train", {}).get("cache_dir")
    if cache_dir:
        summary = json.loads((Path(cache_dir) / "cache.json").read_text())
        if summary["data_config"] != settings["data"] or summary["status"] != "complete":
            raise ValueError("Training cache is incomplete or has different physical data settings")
    return ScanDataset(manifest, split="train", cache_dir=cache_dir)


def train_prior(config_path, manifest, output, *, device="auto", epochs=20, max_steps=None,
                atlas_path=None):
    """Auto-decoder warm-start. PCA frames remain explicitly provisional.

    This stage supplies a functional shape-prior warm-start, not trusted pose
    labels. Atlas correspondence fitting is required before anatomical-pose claims.
    """
    settings = load_config(config_path)
    if epochs < 1 or (max_steps is not None and max_steps < 1):
        raise ValueError("epochs and max_steps must be positive")
    config = ModelConfig.from_dict(settings.get("model", {}))
    device = choose_device(device)
    seed_everything(settings.get("train", {}).get("seed", 42))
    train_cfg = settings.get("train", {})
    dataset = training_dataset(settings, manifest)
    atlas = load_atlas(atlas_path, provenance(manifest)["training_patients"]) if atlas_path else None
    prior = AnatomicalShapePrior(config).to(device)
    samples, frames = [], []
    allowed = set(train_cfg.get("fine_label_ids", range(1, 29)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for scan_index, scan in enumerate(dataset):
        if scan["labels"] is None:
            raise ValueError("Prior fitting requires training masks")
        for label in torch.unique(scan["labels"]):
            label = int(label)
            if label == 0 or label not in allowed:
                continue
            if scan["metadata"]["original_truncated"].get(str(label), False):
                continue
            roi = crop_bone(scan, label, size=settings["data"]["local_size"],
                            spacing_mm=settings["data"]["local_spacing_mm"])
            try:
                bone_mask, bone_affine, _ = bone_voxels_world(scan, label)
                frame = canonical_from_mask(bone_mask, bone_affine)
            except ValueError:
                continue
            if atlas is not None:
                if str(label) not in atlas["classes"]:
                    continue
                fitted = fit_atlas(atlas["classes"][str(label)]["points"],
                    surface_points(bone_mask, bone_affine))
                frame.update(rotation=torch.from_numpy(fitted["rotation"]),
                             translation=torch.from_numpy(fitted["translation"]),
                             pose_valid=fitted["pose_valid"], method="trimmed_rigid_icp",
                             fit_rms_mm=fitted["rms_mm"], fit_ambiguous=fitted["ambiguous"])
            elif not frame["geometry_stable"]:
                continue
            supervision = sample_supervision(roi, label,
                n_points=settings["train"]["points_per_bone"],
                distance_scale_mm=config.distance_scale_mm, seed=len(samples))
            rotation = frame["rotation"].float().reshape(1, 1, 3, 3)
            translation = frame["translation"].float().reshape(1, 1, 3)
            canonical = world_to_canonical(supervision["points_world"][None, None], rotation,
                                            translation)[0, 0]
            samples.append((canonical, supervision["target"], supervision["class_id"]))
            frames.append({"scan_id": scan["metadata"]["scan_id"], "label_id": label,
                           "method": frame["method"], "pose_valid": frame["pose_valid"],
                           "fit_rms_mm": frame.get("fit_rms_mm"),
                           "fit_ambiguous": frame.get("fit_ambiguous", True),
                           "rotation": rotation.flatten().tolist(),
                           "translation": translation.flatten().tolist()})
        print(json.dumps({"stage":"prior_samples", "scan":scan_index+1,
                          "scans":len(dataset), "bones":len(samples)}), flush=True)
    if not samples:
        raise ValueError("No full, stable training bones available for shape-prior fitting")
    codes = torch.nn.Parameter(torch.zeros(len(samples), config.beta_dim, device=device))
    optimizer = torch.optim.Adam([*prior.parameters(), codes], lr=0.001)
    batch_size = int(train_cfg.get("prior_batch_size", 1))
    if batch_size < 1:
        raise ValueError("prior_batch_size must be positive")
    point_bank = torch.stack([s[0] for s in samples]).to(device)
    target_bank = torch.stack([s[1] for s in samples]).to(device)
    class_bank = torch.tensor([s[2] for s in samples], device=device)
    logs, step, samples_seen = [], 0, 0
    for epoch in range(epochs):
        for indices in torch.randperm(len(samples)).split(batch_size):
            indices = indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            with _amp(device, train_cfg.get("amp", True)):
                predicted = prior(point_bank[indices][None], codes[indices][None],
                                  class_bank[indices][None])
                loss = F.smooth_l1_loss(predicted[0].float(), target_bank[indices], beta=0.1) \
                    + 1e-4 * codes[indices].square().mean()
            loss.backward()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite prior training loss")
            torch.nn.utils.clip_grad_norm_([*prior.parameters(), codes], 1, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            samples_seen += len(indices)
            logs.append({"step": step, "prior_loss": float(loss.detach())})
            if step % 20 == 0:
                print(json.dumps({"stage":"prior_train", "epoch":epoch+1,
                                  "step":step, "loss":float(loss.detach())}), flush=True)
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break
    path = Path(output) / "prior.pt"
    save_checkpoint(path, prior, config, optimizer=optimizer, step=step, kind="prior",
                    provenance=provenance(manifest),
                    extra={"shape_codes": codes.detach().cpu(), "canonical_frames": frames,
                           "prior_samples_seen":samples_seen, "prior_batch_size":batch_size,
                           "canonical_status": "training_atlas_weak_registration" if atlas else "provisional_pca",
                           "pose_supervision_valid": any(f["pose_valid"] for f in frames),
                           "atlas_sha256": hashlib.sha256(Path(atlas_path).read_bytes()).hexdigest() if atlas else None})
    (Path(output) / "prior_log.json").write_text(json.dumps(logs, indent=2))
    return {"checkpoint": str(path), "steps": step, "bones": len(samples),
            "canonical_status": "training_atlas_weak_registration" if atlas else "provisional_pca"}


def train_model(config_path, manifest, output, *, device="auto", prior_path=None,
                resume=None, max_steps=None, allow_uninitialized_prior=False):
    settings = load_config(config_path)
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    config = ModelConfig.from_dict(settings.get("model", {}))
    train_cfg, data_cfg = settings["train"], settings["data"]
    seed_everything(train_cfg.get("seed", 42))
    device = choose_device(device)
    model = AnatoSKEL(config).to(device)
    origin = provenance(manifest)
    dataset = training_dataset(settings, manifest)
    if not len(dataset):
        raise ValueError("Training split is empty")
    prior_trained, canonical_frames, code_targets = False, [], None
    if prior_path:
        prior = load_checkpoint(prior_path, expected_kind="prior")
        if ModelConfig.from_dict(prior["model_config"]).to_dict() != config.to_dict():
            raise ValueError("Prior and reconstruction model configurations differ")
        if not set(prior["provenance"]["training_patients"]) <= set(origin["training_patients"]):
            raise ValueError("Prior was fitted using patients outside this training split")
        if prior["step"] < 1:
            raise ValueError("Prior checkpoint contains no optimization steps")
        model.backend.prior.load_state_dict(prior["state_dict"])
        prior_trained = True
        canonical_frames = prior.get("canonical_frames", [])
        code_targets = prior.get("shape_codes")
    elif not resume and not allow_uninitialized_prior:
        raise ValueError("Load --prior first, or explicitly bootstrap with --allow-uninitialized-prior")
    resume_payload = None
    if resume:
        resume_payload = load_checkpoint(resume, expected_kind="model")
        if resume_payload["model_config"] != config.to_dict():
            raise ValueError("Resume architecture differs from the requested configuration")
        if resume_payload["provenance"]["manifest_sha256"] != origin["manifest_sha256"]:
            raise ValueError("Resume manifest differs; use an explicit transfer-learning workflow")
        model.load_state_dict(resume_payload["state_dict"])
        prior_trained = resume_payload.get("prior_trained", False)
        canonical_frames = resume_payload.get("canonical_frames", [])
        code_targets = resume_payload.get("shape_codes")
    if train_cfg.get("freeze_prior", True):
        if not prior_trained:
            raise ValueError("An untrained prior cannot be frozen: set train.freeze_prior: false for explicit bootstrap")
        model.backend.freeze_prior(pretrained=prior_trained)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=train_cfg["learning_rate"],
                                  weight_decay=train_cfg.get("weight_decay", 0.01))
    step, start_epoch, start_scan = 0, 0, 0
    if resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer"])
        step = resume_payload["step"]
        torch.set_rng_state(resume_payload["rng_state"].cpu())
        random.setstate(resume_payload["python_rng_state"])
        numpy_state = resume_payload["numpy_rng_state"]
        np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32),
                             *numpy_state[2:]))
        start_epoch = resume_payload.get("next_epoch", 0)
        start_scan = resume_payload.get("next_scan", 0)
        if device.type == "cuda" and "cuda_rng_state" in resume_payload:
            torch.cuda.set_rng_state_all(resume_payload["cuda_rng_state"])
        if max_steps is not None and step >= max_steps:
            return {"checkpoint": str(resume), "steps": step, "last": None,
                    "reason": "requested_step_limit_already_reached"}
    model.train()
    roi_workers = int(train_cfg.get("roi_workers", 1))
    roi_pool = ThreadPoolExecutor(max_workers=roi_workers) if roi_workers > 1 else None
    loader = torch.utils.data.DataLoader(dataset, batch_size=None,
        num_workers=int(train_cfg.get("scan_workers", 0)), shuffle=False,
        generator=torch.Generator().manual_seed(train_cfg.get("seed", 42)),
        prefetch_factor=1 if train_cfg.get("scan_workers", 0) else None,
        persistent_workers=bool(train_cfg.get("scan_workers", 0)))
    logs = resume_payload.get("training_log", []) if resume_payload else []
    path = Path(output) / "model.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    next_epoch, next_scan = start_epoch, start_scan
    frame_lookup = {(f["scan_id"], f["label_id"]): (i, f)
                    for i, f in enumerate(canonical_frames)}

    def checkpoint_now():
        save_checkpoint(path, model, config, optimizer=optimizer, step=step, provenance=origin,
            extra={"run_config": settings, "prior_trained": prior_trained,
                   "pose_supervision_valid": any(f.get("pose_valid") for f in canonical_frames),
                   "canonical_frames": canonical_frames, "shape_codes": code_targets,
                   "training_log": logs,
                   "next_epoch": next_epoch, "next_scan": next_scan,
                   "training_status": "synthetic_check" if origin["synthetic"] else "trained"})

    for epoch in range(start_epoch, train_cfg["epochs"]):
        epoch_start = start_scan if epoch == start_epoch else 0
        for scan_index, scan in enumerate(loader):
            if scan_index < epoch_start:
                continue
            iteration_start = time.perf_counter()
            if scan["labels"] is None:
                raise ValueError("Main training requires annotated scans")
            coarse = scan.get("cached_coarse")
            if coarse is None:
                coarse = coarse_scan(scan, data_cfg)
            optimizer.zero_grad(set_to_none=True)
            with _amp(device, train_cfg.get("amp", True)):
                detection = model.encode_coarse(coarse["volume"][None].to(device),
                                                  coarse["affine"][None].to(device).float())
                coarse_loss, components, matches = detector_loss(detection, coarse["labels"],
                                          coarse["valid_mask"], coarse["affine"])
                visibility = torch.zeros_like(detection["visibility_logits"][0])
                for query, label in matches:
                    visibility[query] = visibility.new_tensor([1, int(
                        scan["metadata"]["original_truncated"].get(str(label), False))])
                coarse_loss = coarse_loss + 0.1 * F.binary_cross_entropy_with_logits(
                    detection["visibility_logits"][0].float(), visibility)
            # Separate backward passes release whole-scan activations before local
            # ROI processing. Query embeddings are detached at this training boundary.
            queries = detection["queries"].detach()
            predicted_centers = detection["centers_mm"].detach()
            coarse_loss.backward()
            del detection
            fine_matches = [m for m in matches if m[1] in
                            train_cfg.get("fine_label_ids", range(1, 29))]
            selected = fine_matches[:train_cfg.get("max_bones_per_step", 2)]
            # Rotate the selected subset so all visible identities get fine supervision.
            if fine_matches:
                offset = step % len(fine_matches)
                selected = (fine_matches[offset:] + fine_matches[:offset])[:train_cfg.get("max_bones_per_step", 2)]
            rois, supervision, chosen, anchors, class_ids, partial = [], [], [], [], [], []
            warmup, transition = train_cfg.get("roi_warmup_steps", 1000), \
                max(1, train_cfg.get("roi_transition_steps", 4000))
            predicted_fraction = min(1., max(0., (step - warmup) / transition))
            tasks = []
            for query, label in selected:
                # Even a tiny/partial bone has an observed centroid; it need not
                # satisfy the stronger prerequisites of a canonical pose fit.
                _, _, world = bone_voxels_world(scan, label)
                target_center = torch.from_numpy(world.mean(axis=0)).float()
                center = predicted_centers[0, query].cpu() if random.random() < predicted_fraction \
                    else target_center + torch.randn(3) * train_cfg.get("roi_jitter_mm", 8)
                tasks.append((query, label, center))

            def prepare_roi(task):
                query, label, center = task
                roi = crop_at_center(scan, center, size=data_cfg["local_size"],
                                     spacing_mm=data_cfg["local_spacing_mm"])
                try:
                    points = sample_supervision(roi, label, train_cfg["points_per_bone"],
                                               config.distance_scale_mm, seed=step + label)
                except ValueError:
                    # A missed ROI is still counted in detector loss. Never silently
                    # replace it with a GT-centred crop in the predicted-ROI phase.
                    return None
                return query, label, center, roi, points

            prepared = roi_pool.map(prepare_roi, tasks) if roi_pool else map(prepare_roi, tasks)
            for result in prepared:
                if result is None:
                    continue
                query, label, center, roi, points = result
                rois.append(roi)
                supervision.append(points)
                chosen.append(query)
                anchors.append(center)
                class_ids.append(label_to_class(label))
                partial.append(int(scan["metadata"]["original_truncated"].get(str(label), False)))
            fine_value = 0.
            if rois:
                q = queries[:, chosen]
                with _amp(device, train_cfg.get("amp", True)):
                    context = pack_context(model, rois, q, device)
                    state = model.reconstruct(context, q, torch.tensor([class_ids], device=device),
                                               torch.stack(anchors)[None].to(device))
                # grid_sample and physical coordinate arithmetic remain FP32.
                points = torch.stack([s["points_world"] for s in supervision])[None].to(device)
                prediction = model.field(context, state, points)
                target = torch.stack([s["target"] for s in supervision])[None].to(device)
                valid = torch.stack([s["valid"] for s in supervision])[None].to(device)
                fine_loss = field_loss(prediction["field"], target, valid & prediction["valid"],
                                       state["beta"])
                truth = torch.stack([(r["labels"] == s["label_id"]).float()
                                     for r, s in zip(rois, supervision)]).unsqueeze(1).to(device)
                observed = context["annotation_valid_mask"]
                fine_loss += 0.2 * F.binary_cross_entropy_with_logits(
                    context["mask_logits"][observed].float(), truth[observed])
                flags = torch.tensor([[[1, p] for p in partial]], device=device).float()
                fine_loss += 0.1 * F.binary_cross_entropy_with_logits(state["visibility_logits"].float(), flags)
                for bone_index, sample in enumerate(supervision):
                    fitted = frame_lookup.get((scan["metadata"]["scan_id"], sample["label_id"]))
                    if fitted is None or not fitted[1].get("pose_valid", False):
                        continue
                    code_index, frame = fitted
                    rotation_target = torch.tensor(frame["rotation"], device=device).reshape(3, 3)
                    origin_target = torch.tensor(frame["translation"], device=device)
                    weak_pose = F.mse_loss(state["rotation"][0, bone_index], rotation_target) \
                        + F.smooth_l1_loss(state["translation"][0, bone_index] / 100, origin_target / 100)
                    fine_loss += train_cfg.get("weak_pose_weight", 0.05) * weak_pose
                    if code_targets is not None:
                        fine_loss += 0.001 * F.mse_loss(state["beta"][0, bone_index].float(),
                                                        code_targets[code_index].to(device).float())
                fine_loss.backward()
                fine_value = float(fine_loss.detach())
                del context, state, prediction, fine_loss, truth, observed, points, target, valid
            if not torch.isfinite(coarse_loss) or not np.isfinite(fine_value):
                raise RuntimeError("Nonfinite training loss")
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("clip_grad", 1),
                                          error_if_nonfinite=True)
            optimizer.step()
            step += 1
            record = {"step": step, "epoch": epoch, "coarse_loss": float(coarse_loss.detach()),
                      "fine_loss": fine_value, "fine_bones": len(rois),
                      "missed_or_uncensored_rois": len(selected) - len(rois),
                      "predicted_roi_fraction": predicted_fraction,
                      "compute_and_roi_seconds":time.perf_counter()-iteration_start, **components}
            if not np.isfinite(record["coarse_loss"] + fine_value):
                raise RuntimeError("Nonfinite training loss")
            logs.append(record)
            if step % train_cfg.get("log_every_steps", 10) == 0:
                print(json.dumps(record), flush=True)
            next_epoch = epoch if scan_index + 1 < len(dataset) else epoch + 1
            next_scan = (scan_index + 1) % len(dataset)
            with (Path(output) / "train_log.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if step % train_cfg.get("checkpoint_every_steps", 100) == 0:
                checkpoint_now()
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break
    checkpoint_now()
    if roi_pool:
        roi_pool.shutdown()
    (Path(output) / "train_log.json").write_text(json.dumps(logs, indent=2))
    return {"checkpoint": str(path), "steps": step, "last": logs[-1] if logs else None}
