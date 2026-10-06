"""Four explicit stages: complete AE, CT geometry, diffusion, partial adaptation."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from scipy import ndimage
import torch
from torch.nn import functional as F

from anatoskel.config import load_config
from anatoskel.data import ScanDataset
from anatoskel.data.scans import coarse_resample, bone_voxels_world, validate_affine
from anatoskel.models.backend import sample_volume, world_to_voxel
from anatoskel.runtime import choose_device
from anatoskel.training.checkpoint import save_checkpoint, load_checkpoint
from .config import V1Config
from .domains import input_scan, synthetic_crop, prepare_bone_supervision
from .tiles import extract_tile, plan_tiles
from .network import AnatoSKELV1, instance_state
from .losses import detect_loss, geometry_loss, local_contrastive


def amp_context(device, enabled):
    return torch.autocast("cuda", dtype=torch.bfloat16) if enabled and device.type == "cuda" else nullcontext()


def global_scan(scan, config):
    """Entire visible FOV at bounded isotropic resolution, padded to Swin strides."""
    affine = validate_affine(scan["affine"])
    lengths = np.linalg.norm(affine[:3, :3], axis=0)
    extent = np.asarray(scan["volume"].shape[-3:][::-1])-1
    spacing = max(float((extent*lengths).max()/(config.global_max_size-1)),
                  float(lengths.min()))
    result = coarse_resample(scan, spacing)
    shape = result["volume"].shape[-3:]
    padded = [int(np.ceil(v/32)*32) for v in shape]
    if max(padded) > config.global_max_size:
        # Polar regridding of oblique/sheared FOV can add a voxel: recompute
        # rather than silently truncating the visible scan.
        result = coarse_resample(scan, spacing*max(padded)/config.global_max_size*1.01)
        shape = result["volume"].shape[-3:]
        padded = [int(np.ceil(v/32)*32) for v in shape]
    pad = tuple(value for old, new in zip(shape[::-1], padded[::-1]) for value in (0, new-old))
    result["volume"] = F.pad(result["volume"], pad)
    for key in ("valid_mask", "ct_valid_mask", "artificial_padding_mask"):
        result[key] = F.pad(result[key], pad)
    return result


def rasterize_annotations(target, grid):
    """GT resampling is loss-only; it does not alter the prediction inputs."""
    mapping = np.linalg.inv(target["affine"].numpy()) @ grid["affine"].detach().cpu().numpy()
    matrix, offset = mapping[:3, :3][::-1, ::-1], mapping[:3, 3][::-1]
    shape = grid["volume"].shape[-3:]
    def resample(value):
        return torch.from_numpy(ndimage.affine_transform(
            value.numpy(), matrix, offset, output_shape=shape,
            order=0, mode="constant", cval=0, prefilter=False).copy())
    return resample(target["labels"]).long(), (resample(target["valid_mask"]).bool()
                                               & grid["ct_valid_mask"])


def encode_scan(model, scan, device):
    coarse = global_scan(scan, model.config)
    context = model.encode(coarse["volume"][None].to(device),
                           coarse["affine"][None].to(device).float(),
                           coarse["ct_valid_mask"][None, None].to(device))
    return coarse, context, model.detect(context)


def prepare_instances(model, input_ct, detection, pairs, device, *, target=None,
                      warmup=False, teacher_ratio=1., jitter_mm=0., max_tiles=64):
    contexts, owners, tiles = [], [], []
    for owner, (query, label) in enumerate(pairs):
        center = detection["centers_mm"][0, query]
        bounds = detection["visible_bounds_mm"][0, query]
        if warmup and target is not None:
            # Warm-up is an explicit training-only exception. Use only the
            # currently visible GT region, never a hidden complete centroid.
            coarse_grid = global_scan(input_ct, model.config)
            labels, valid = rasterize_annotations(target, coarse_grid)
            indices = ((labels == label) & valid).nonzero().float().flip(-1)
            if len(indices):
                world = indices @ coarse_grid["affine"][:3, :3].float().T+coarse_grid["affine"][:3, 3].float()
                teacher_center = world.mean(0).to(device)+torch.randn(3, device=device)*jitter_mm
                teacher_bounds = torch.stack((world.min(0).values, world.max(0).values)).to(device)
                center = teacher_ratio*teacher_center+(1-teacher_ratio)*center
                bounds = teacher_ratio*teacher_bounds+(1-teacher_ratio)*bounds
        centers = plan_tiles(center, bounds, size=model.config.local_size,
                             spacing_mm=model.config.local_spacing_mm,
                             overlap=model.config.tile_overlap,
                             margin_mm=min(16., model.config.local_size*model.config.local_spacing_mm/8),
                             max_tiles=max_tiles)
        for tile_center in centers:
            tile = extract_tile(input_ct, tile_center, size=model.config.local_size,
                                spacing_mm=model.config.local_spacing_mm)
            if not tile["valid_mask"].any():
                continue
            context = model.encode(tile["volume"][None].to(device),
                                   tile["affine"][None].to(device),
                                   tile["valid_mask"][None, None].to(device))
            contexts.append(context)
            owners.append(owner)
            tiles.append(tile)
        if owner not in owners:
            raise ValueError("Selected instance has no observed CT tile")
    selected = torch.tensor([[0, query] for query, _ in pairs], device=device)
    state = model.prepare_instances(detection, selected, contexts,
                                    torch.tensor(owners, device=device))
    return state, contexts, owners, tiles


def _supervision(scan, label, anchor, settings, seed):
    config = V1Config.from_dict(settings["model"])
    training = settings["train"]
    return prepare_bone_supervision(scan, label, anchor,
        n_points=int(training.get("points_per_bone", 8192)),
        n_surface=int(training.get("surface_points", 4096)),
        scale_mm=config.shape_scale_mm, tau_mm=config.tsdf_truncation_mm,
        margin_mm=float(training.get("distance_margin_mm", 1.)), seed=seed)


def differential_surface_loss(model, state, contexts, supervision, memory,
                              *, shape_only=False, count=128):
    """Metric finite differences avoid unsupported grid_sample double backward."""
    device = state["z"].device
    points = supervision["surface_world"][:count].to(device)[None]
    normals = supervision["normals"][:count].to(device)
    def evaluate(world):
        if shape_only:
            u = (world-state["anchors_mm"][:, None])/model.config.shape_scale_mm
            return model.shape.decode(state["z"], u, memory=memory), torch.ones(
                world.shape[:2], device=device, dtype=torch.bool)
        result = model.field(state, contexts, world, shape_memory=memory)
        return result["sdf_mm"], result["valid"]
    value, valid = evaluate(points)
    surface = value[valid].abs().mean()/model.config.tsdf_truncation_mm if valid.any() else value.sum()*0
    gradients = []
    epsilon = .5
    for axis in range(3):
        shift = torch.zeros_like(points)
        shift[..., axis] = epsilon
        plus, vp = evaluate(points+shift)
        minus, vm = evaluate(points-shift)
        valid = valid & vp & vm
        gradients.append((plus-minus)/(2*epsilon))
    gradient = torch.stack(gradients, -1)[0]
    valid = valid[0]
    normal = (1-F.cosine_similarity(gradient[valid], normals[valid])).mean() if valid.any() else value.sum()*0
    eikonal = (gradient[valid].norm(dim=-1)-1).square().mean() if valid.any() else value.sum()*0
    return .1*surface+.05*normal+.01*eikonal


def _dense_detection_aux(detection, labels, valid, affine):
    device = detection["queries"].device
    labels, valid = labels.to(device), valid.to(device)
    context = detection["context"]
    if not valid.any():
        return detection["queries"].sum()*0
    foreground = F.binary_cross_entropy_with_logits(
        context["mask_logits"][0, 0][valid].float(), (labels[valid] > 0).float())
    shape = labels.shape
    axes = [torch.arange(n, device=device, dtype=torch.float32) for n in shape]
    xyz = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).flip(-1)
    with torch.autocast(device_type=device.type, enabled=False):
        world = xyz @ affine[:3, :3].to(device).float().T+affine[:3, 3].to(device).float()
    heatmap = torch.zeros_like(labels, dtype=torch.float32)
    for label in torch.unique(labels[valid]):
        if int(label) == 0:
            continue
        center = world[(labels == label) & valid].mean(0)
        gaussian = torch.exp(-(world-center).square().sum(-1)/(2*8.**2))
        heatmap = torch.maximum(heatmap, gaussian)
    center_loss = F.binary_cross_entropy_with_logits(
        detection["heatmap_logits"][0, 0][valid].float(), heatmap[valid])
    return foreground+center_loss


def load_v1_model(checkpoint, device="auto"):
    device = choose_device(device)
    payload = load_checkpoint(checkpoint, expected_kind="model")
    if payload["schema_version"] != 2:
        raise ValueError("v1 requires a schema-2 checkpoint; v0.1 weights cannot initialize new architecture")
    model = AnatoSKELV1(V1Config.from_dict(payload["model_config"]), load_pretrained=False).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload, device


def _eligible_pairs(model, input_ct, target, device, settings, seed):
    coarse, context, detection = encode_scan(model, input_ct, device)
    labels, valid = rasterize_annotations(target, coarse)
    detection_loss, pairs = detect_loss(detection, labels, valid, coarse["affine"])
    allowed = set(settings["train"].get("label_ids", [20, 21, 22, 23, 24, 25]))
    pairs = [(query, label) for query, label in pairs if label in allowed]
    return coarse, context, detection, pairs, detection_loss


def train_v1(config_path, manifest, output, *, stage, device="auto", init=None,
             resume=None, max_steps=None):
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps is a positive total stage-step limit")
    if init and resume:
        raise ValueError("Choose --init for next stage or --resume for the same stage")
    settings = load_config(config_path)
    config = V1Config.from_dict(settings["model"])
    training = settings["train"]
    seed = int(training.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = choose_device(device)
    dataset = ScanDataset(manifest, split="train")
    if not len(dataset):
        raise ValueError("Manifest has no training cases")
    source = None
    if init or resume:
        model, source, _ = load_v1_model(init or resume, device)
        expected, actual = config.to_dict(), model.config.to_dict()
        for values in (expected, actual):
            values.pop("pretrained_encoder", None)
        if expected != actual:
            raise ValueError("Architecture differs from the initialization checkpoint")
    else:
        model = AnatoSKELV1(config).to(device)
    parameters = model.configure_stage(stage)
    optimizer = torch.optim.AdamW(parameters, lr=float(training.get("learning_rate", 1e-4)),
                                   weight_decay=float(training.get("weight_decay", .01)))
    origin = {"manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
              "training_patients": sorted({record.patient_id for record in dataset.records}),
              "encoder_pretraining": model.encoder.pretrained_report,
              "architecture": "anatoskel_v1"}
    if source and not set(source["provenance"]["training_patients"]) <= set(origin["training_patients"]):
        raise ValueError("Initialization includes patients outside this training split")
    if source:
        origin["encoder_pretraining"] = source["provenance"].get("encoder_pretraining")
    step, start_epoch, start_scan, start_bone = 0, 0, 0, 0
    if resume:
        if source.get("stage") != stage or source["provenance"]["manifest_sha256"] != origin["manifest_sha256"]:
            raise ValueError("Resume stage or manifest differs")
        optimizer.load_state_dict(source["optimizer"])
        step = int(source["step"])
        cursor = source.get("cursor", {})
        start_epoch, start_scan, start_bone = (int(cursor.get(key, 0)) for key in ("epoch", "scan", "bone"))
        torch.set_rng_state(source["rng_state"].cpu())
        random.setstate(source["python_rng_state"])
        state = source["numpy_rng_state"]
        np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
        if device.type == "cuda" and source.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(source["cuda_rng_state"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    synthetic = training.get("software_check", False) or all(
        r.patient_id.startswith("synthetic") or "synthetic" in str(r.ct).lower()
        for r in dataset.records)
    status = "synthetic_check" if synthetic else "research_training"
    calibration_recipe = source.get("residual_calibration_recipe", {}) if source else {}
    if source and source.get("training_status") == "synthetic_check":
        status = "synthetic_check"
    if max_steps is not None and step >= max_steps:
        return {"checkpoint": str(Path(resume)), "stage": stage, "steps": step,
                "training_status": status}

    def persist(cursor):
        if step and (stage != "ct" or int(model.ct_geometry_steps) > 0):
            getattr(model, {"shape": "shape_trained", "ct": "anchor_trained",
                           "diffusion": "diffusion_trained", "partial": "adapters_trained"}[stage]).fill_(True)
        save_checkpoint(output/"model.pt", model, model.config, optimizer=optimizer,
                        step=step, kind="model", provenance=origin,
                        extra={"stage": stage, "run_config": settings,
                               "training_status": status, "cursor": cursor,
                               "residual_calibration_recipe": calibration_recipe,
                               "initialization": str(init) if init else None})

    def read_case(index):
        scan = dataset[index]
        if scan["labels"] is None:
            raise ValueError("Training requires annotated cases")
        try:
            validate_affine(scan["affine"], orthogonal=True)
        except ValueError:
            scan = coarse_resample(scan, config.local_spacing_mm)
        return scan

    if stage == "diffusion" and not bool(model.diffusion.stats_ready):
        # Reference predictor and AE are frozen BEFORE computing residual stats.
        calibration_recipe = {"seed": seed, "full_instances": 0, "crop_instances": 0,
                              "synthetic_crop_probability": float(training.get("synthetic_crop_probability", .3)),
                              "target_frame": "condition_predicted_anchor_RAS_mm"}
        with torch.no_grad():
            for index in range(len(dataset)):
                target = read_case(index)
                variants = [(input_scan(target), "full")]
                case_rng = random.Random(seed+100000+index)
                allowed = set(training.get("label_ids", [20,21,22,23,24,25]))
                present = sorted(set(int(x) for x in torch.unique(target["labels"])) & allowed)
                probability = float(training.get("synthetic_crop_probability", .3))
                if present and case_rng.random() < probability:
                    variants.append((synthetic_crop(target, seed=seed+100000+index,
                                     label_id=case_rng.choice(present)), "crop"))
                for condition_ct, variant in variants:
                    with amp_context(device, training.get("amp", True)):
                        _, _, detection, pairs, _ = _eligible_pairs(model, condition_ct, target,
                                                                  device, settings, seed+index)
                        for pair in pairs:
                            preliminary = _supervision(target, pair[1], detection["anchors_mm"][0, pair[0]], settings, seed+index)
                            if not preliminary["complete"]:
                                continue
                            state, _, _, _ = prepare_instances(model, condition_ct, detection,
                                                                [pair], device)
                            # Re-encode full GT in THIS condition-predicted frame.
                            z_gt = model.shape.encode(preliminary["surface_u"][None].to(device),
                                                       preliminary["normals"][None].to(device))
                            model.diffusion.calibrate_residuals(z_gt, state["z_anchor"],
                                complete=torch.ones(1, dtype=torch.bool, device=device), finalize=False)
                            calibration_recipe[f"{variant}_instances"] += 1
                print(json.dumps({"stage": "residual_calibration", "scan": index+1,
                                  "scans": len(dataset)}), flush=True)
        model.diffusion.finalize_statistics()
        persist({"epoch": start_epoch, "scan": start_scan, "bone": start_bone})

    log_path = output/"train.jsonl"
    applied = 0
    for epoch in range(start_epoch, int(training.get("epochs", 100))):
        for scan_index in range(start_scan if epoch == start_epoch else 0, len(dataset)):
            target = read_case(scan_index)
            labels = sorted(set(int(x) for x in torch.unique(target["labels"]))
                            & set(training.get("label_ids", [20, 21, 22, 23, 24, 25])))
            if stage == "shape":
                items = [(None, label) for label in labels]
                detection = context = None
                input_ct = input_scan(target)
                detection_loss = None
            else:
                input_ct = input_scan(target)
                case_rng = random.Random(seed+epoch*100000+scan_index)
                if stage in ("diffusion", "partial") and case_rng.random() < float(training.get("synthetic_crop_probability", .3)):
                    if labels:
                        input_ct = synthetic_crop(target, seed=seed+epoch*100000+scan_index,
                                                  label_id=case_rng.choice(labels))
                guard = nullcontext() if stage == "ct" else torch.no_grad()
                with guard, amp_context(device, training.get("amp", True)):
                    coarse, context, detection, pairs, detection_loss = _eligible_pairs(
                        model, input_ct, target, device, settings, seed+step)
                    raster_labels, raster_valid = rasterize_annotations(target, coarse)
                    detection_loss = detection_loss+_dense_detection_aux(detection, raster_labels, raster_valid, coarse["affine"])
                if stage == "ct":
                    items = [pairs[:int(training.get("max_bones_per_step", 2))]]
                else:
                    items = [[pair] for pair in pairs]
            begin = start_bone if epoch == start_epoch and scan_index == start_scan else 0
            for item_index in range(begin, len(items)):
                optimizer.zero_grad(set_to_none=True)
                diagnostics = {}
                geometry_updates = 0
                with amp_context(device, training.get("amp", True)):
                    if stage == "shape":
                        _, label = items[item_index]
                        _, _, world = bone_voxels_world(target, label)
                        anchor = torch.from_numpy(world.mean(0)).float()+torch.randn(3)*float(training.get("anchor_jitter_mm", 16.))
                        supervision = _supervision(target, label, anchor, settings, seed+step)
                        if not supervision["complete"]:
                            continue
                        z = model.shape.encode(supervision["surface_u"][None].to(device),
                                               supervision["normals"][None].to(device))
                        memory = model.shape.prepare_memory(z)
                        predicted = model.shape.decode(z, supervision["points_u"][None].to(device), memory=memory)
                        field = {"sdf_mm": predicted, "valid": torch.ones_like(predicted, dtype=torch.bool)}
                        loss, diagnostics = geometry_loss(field, supervision,
                            temperature_mm=config.occupancy_temperature_mm, tau_mm=config.tsdf_truncation_mm)
                        state = {"z": z, "anchors_mm": anchor[None].to(device)}
                        loss = loss+differential_surface_loss(model, state, [], supervision,
                            memory, shape_only=True, count=int(training.get("normal_points", 128)))
                    else:
                        pairs = items[item_index]
                        if not pairs:
                            if stage != "ct":
                                continue
                            loss = detection_loss
                        else:
                            guard = nullcontext() if stage == "ct" else torch.no_grad()
                            warmup_steps = int(training.get("roi_warmup_steps", 1000))
                            transition_steps = int(training.get("roi_transition_steps", 4000))
                            teacher_ratio = 1. if step < warmup_steps else max(
                                0., 1-(step-warmup_steps)/max(transition_steps, 1))
                            if transition_steps == 0 and step >= warmup_steps:
                                teacher_ratio = 0.
                            with guard:
                                state, contexts, owners, _ = prepare_instances(model, input_ct, detection,
                                    pairs, device, target=target,
                                    warmup=stage == "ct" and teacher_ratio > 0,
                                    teacher_ratio=teacher_ratio,
                                    jitter_mm=float(training.get("roi_jitter_mm", 8.)),
                                    max_tiles=int(training.get("max_tiles_per_bone", 64)))
                            losses = []
                            for bone_index, pair in enumerate(pairs):
                                one = instance_state(state, bone_index)
                                bone_loss = one["z_anchor"].sum()*0
                                bone_contexts = [c for c, owner in zip(contexts, owners) if owner == bone_index]
                                supervision = _supervision(target, pair[1], one["anchors_mm"][0], settings, seed+step+bone_index)
                                if stage == "diffusion" and not supervision["complete"]:
                                    continue
                                if stage in ("diffusion", "partial") and supervision["complete"]:
                                    with torch.no_grad():
                                        z_gt = model.shape.encode(supervision["surface_u"][None].to(device),
                                                                   supervision["normals"][None].to(device))
                                    v_loss, diffusion_info = model.diffusion.loss(z_gt, one["z_anchor"],
                                        one["memory"], one["memory_valid"],
                                        complete=torch.ones(1, device=device, dtype=torch.bool),
                                        adapter_replay=stage == "partial")
                                    bone_loss = bone_loss+v_loss
                                    diagnostics["v_loss"] = float(v_loss.detach())
                                    if not bool(diffusion_info["geometry_valid"][0]):
                                        losses.append(bone_loss)
                                        continue
                                    one["z"] = diffusion_info["predicted_clean_z"]
                                elif stage == "partial":
                                    one = model.sample_instances(one, seed=seed+step,
                                                                 differentiable_tail=True)
                                memory = model.shape.prepare_memory(one["z"])
                                points = supervision["points_world"][None].to(device)
                                field = model.field(one, bone_contexts, points, shape_memory=memory)
                                geometry, geometry_diagnostics = geometry_loss(field, supervision,
                                    temperature_mm=config.occupancy_temperature_mm,
                                    tau_mm=config.tsdf_truncation_mm)
                                diagnostics.update(geometry_diagnostics)
                                geometry = geometry+differential_surface_loss(model, one, bone_contexts,
                                    supervision, memory, count=int(training.get("normal_points", 128)))
                                if stage == "ct":
                                    # Only complete shapes supply latent regression targets.
                                    if supervision["complete"]:
                                        with torch.no_grad():
                                            z_gt = model.shape.encode(supervision["surface_u"][None].to(device),
                                                                       supervision["normals"][None].to(device))
                                        geometry = geometry+.05*F.smooth_l1_loss(one["z_anchor"].float(), z_gt.float())
                                    geometry = geometry+.1*field["tile_loss"]+.1*local_contrastive(
                                        model, one, bone_contexts, points, global_context=context)
                                    aux = []
                                    for c in bone_contexts:
                                        voxel = world_to_voxel(points[:, None], c["affine"])
                                        visible = sample_volume(c["valid_mask"].float(), voxel, mode="nearest")[0, 0, :, 0] > .5
                                        if visible.any():
                                            mask_logits = sample_volume(c["mask_logits"], voxel)[0, 0, :, 0]
                                            aux.append(F.binary_cross_entropy_with_logits(mask_logits[visible].float(),
                                                supervision["occupancy"].to(device)[visible]))
                                        safe = visible & supervision["sdf_valid"].to(device)
                                        if safe.any():
                                            distance = sample_volume(c["tsdf_mm"], voxel)[0, 0, :, 0]
                                            aux.append(F.smooth_l1_loss(distance[safe].float()/config.tsdf_truncation_mm,
                                                supervision["target_sdf_mm"].to(device)[safe]/config.tsdf_truncation_mm))
                                    if aux:
                                        geometry = geometry+.5*torch.stack(aux).mean()
                                    visibility = one["visibility_logits"]
                                    visibility_target = visibility.new_tensor([[1., float(supervision["complete"])]])
                                    geometry = geometry+.1*F.binary_cross_entropy_with_logits(visibility.float(), visibility_target)
                                elif stage == "partial":
                                    # Weak shape-only visible constraint gives the latent decoder
                                    # an explicit path while the CT residual head is fixed.
                                    prior_field = dict(field, sdf_mm=field["prior_mm"])
                                    prior_loss, _ = geometry_loss(prior_field, supervision,
                                        temperature_mm=config.occupancy_temperature_mm,
                                        tau_mm=config.tsdf_truncation_mm)
                                    geometry = geometry+.05*prior_loss
                                    drift = torch.stack([block.adapter.up.weight.square().mean()
                                                         +block.adapter.up.bias.square().mean()
                                                         for block in model.diffusion.blocks]).mean()
                                    geometry = geometry+1e-3*drift
                                losses.append(bone_loss+(1. if stage == "ct" else .1)*geometry)
                                if geometry_diagnostics.get("observed_points", 0):
                                    geometry_updates += 1
                            if not losses and stage != "ct":
                                continue
                            loss = (detection_loss if stage == "ct" else state["z_anchor"].sum()*0)
                            if losses:
                                loss = loss+torch.stack(losses).mean()
                            if stage == "ct" and len(pairs) > 1:
                                indices = torch.tensor([q for q, _ in pairs], device=device)
                                global_embedding = model.anchor_predictor.contrastive(detection["queries"][0, indices])
                                local_embedding = model.anchor_predictor.contrastive(state["queries"])
                                scores = global_embedding @ local_embedding.T/.1
                                identity = torch.arange(len(pairs), device=device)
                                instance_contrast = (F.cross_entropy(scores.float(), identity)
                                                    +F.cross_entropy(scores.T.float(), identity))/2
                                loss = loss+.05*instance_contrast
                                diagnostics["instance_contrastive"] = float(instance_contrast.detach())
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, float(training.get("clip_grad", 1.)),
                                                error_if_nonfinite=True)
                optimizer.step()
                if stage == "ct" and geometry_updates:
                    model.ct_geometry_steps.add_(1)
                step += 1
                applied += 1
                record = {"stage": stage, "epoch": epoch+1, "scan": scan_index+1,
                          "step": step, "loss": float(loss.detach()), **diagnostics}
                with log_path.open("a") as stream:
                    stream.write(json.dumps(record)+"\n")
                print(json.dumps(record), flush=True)
                cursor = {"epoch": epoch, "scan": scan_index, "bone": item_index+1}
                if step % int(training.get("checkpoint_every_steps", 100)) == 0:
                    persist(cursor)
                if max_steps is not None and step >= max_steps:
                    persist(cursor)
                    return {"checkpoint": str(output/"model.pt"), "stage": stage, "steps": step,
                            "training_status": status}
            start_bone = 0
        start_scan = 0
        persist({"epoch": epoch+1, "scan": 0, "bone": 0})
    if not applied:
        raise ValueError(f"No eligible optimization steps for {stage}; check complete GT/coverage and stage inputs")
    return {"checkpoint": str(output/"model.pt"), "stage": stage, "steps": step,
            "training_status": status}
