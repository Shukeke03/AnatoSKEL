"""Losses in supervised physical domains, with unknown regions ignored."""
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from anatoskel.data.scans import VERSE_IDS, label_to_class
from anatoskel.models.backend import sample_volume, world_to_voxel


def geometry_loss(output, supervision, *, temperature_mm=1., tau_mm=8.):
    predicted = output["sdf_mm"][0].float()
    valid = supervision["occupancy_valid"].to(predicted.device) & output["valid"][0]
    if not valid.any():
        return predicted.sum()*0, {"observed_points": 0}
    truth = supervision["occupancy"].to(predicted.device)
    bce = F.binary_cross_entropy_with_logits(-predicted[valid]/temperature_mm, truth[valid])
    probability = torch.sigmoid(-predicted[valid]/temperature_mm)
    dice = 1-(2*(probability*truth[valid]).sum()+1)/(probability.sum()+truth[valid].sum()+1)
    safe = valid & supervision["sdf_valid"].to(predicted.device)
    target = supervision["target_sdf_mm"].to(predicted.device)
    distance = F.smooth_l1_loss(predicted[safe]/tau_mm, target[safe]/tau_mm,
                              beta=1/tau_mm) if safe.any() else predicted.sum()*0
    return bce+dice+distance, {"observed_points": int(valid.sum()),
                              "safe_sdf_points": int(safe.sum()),
                              "occupancy_bce": float(bce.detach()),
                              "sdf": float(distance.detach())}


def detect_loss(output, labels, valid, affine, sample_points=4096):
    logits, masks = output["class_logits"][0], output["mask_logits"][0]
    labels, valid = labels.to(logits.device), valid.to(logits.device)
    ids = [int(x) for x in torch.unique(labels[valid]) if int(x) in VERSE_IDS]
    if len(ids) > len(logits):
        raise ValueError("Too few instance slots for annotated bones")
    indices = valid.flatten().nonzero().flatten()
    if not len(indices):
        return logits.sum()*0, []
    generator = torch.Generator(device=indices.device).manual_seed(42)
    pieces = [indices[torch.randperm(len(indices), device=indices.device,
                                    generator=generator)[:sample_points]]]
    centers, bounds = [], []
    for label in ids:
        zyx = ((labels == label) & valid).nonzero().float()
        xyz = zyx.flip(-1)
        world = xyz @ affine[:3, :3].to(xyz).T+affine[:3, 3].to(xyz)
        centers.append(world.mean(0))
        bounds.append(torch.stack((world.min(0).values, world.max(0).values)))
        foreground = ((labels == label) & valid).flatten().nonzero().flatten()
        pieces.append(foreground[torch.randperm(len(foreground), device=indices.device,
                                              generator=generator)[:128]])
    indices = torch.unique(torch.cat(pieces))
    sampled = masks.flatten(1)[:, indices].float()
    zero = sampled.sum()*0
    targets = torch.full((len(logits),), -100, dtype=torch.long, device=logits.device)
    # Unmatched queries are negatives only where their centres are annotated.
    voxel = world_to_voxel(output["centers_mm"][:, :, None], affine[None].to(logits.device))
    center_known = sample_volume(valid[None, None].float(), voxel,
                                mode="nearest")[0, :, 0, 0] > .5
    targets[center_known] = len(VERSE_IDS)
    if not ids:
        return (F.cross_entropy(logits.float(), targets) if (targets != -100).any() else zero), []
    truth = torch.stack([(labels.flatten()[indices] == label).float() for label in ids])
    centers, bounds = torch.stack(centers), torch.stack(bounds)
    classes = torch.tensor([label_to_class(label) for label in ids], device=logits.device)
    probability = sampled.sigmoid()
    pair_dice = 1-(2*probability@truth.T+1)/(probability.sum(1)[:, None]+truth.sum(1)+1)
    cost = (-logits.float().log_softmax(-1)[:, classes]+pair_dice
            +torch.cdist(output["centers_mm"][0].float(), centers)/10)
    rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
    rows = torch.as_tensor(rows, device=logits.device)
    columns = torch.as_tensor(columns, device=logits.device)
    targets[rows] = classes[columns]
    weights = logits.new_ones(logits.shape[-1], dtype=torch.float32)
    weights[-1] = .1
    classification = F.cross_entropy(logits.float(), targets, weight=weights)
    bce = F.binary_cross_entropy_with_logits(sampled[rows], truth[columns])
    dice = pair_dice[rows, columns].mean()
    center = F.smooth_l1_loss(output["centers_mm"][0, rows].float()/10, centers[columns]/10)
    box = F.smooth_l1_loss(output["visible_bounds_mm"][0, rows].float()/10, bounds[columns]/10)
    return classification+bce+2*dice+2*center+box, [(int(r), ids[int(c)])
                                                   for r, c in zip(rows, columns)]


def local_contrastive(model, state, contexts, points_world, global_context=None):
    """Same world points in overlapping CT tiles; repeated points are positives."""
    contexts = ([global_context] if global_context is not None else [])+contexts
    if len(contexts) < 2:
        return state["z"].sum()*0
    points_world = torch.unique(points_world[0], dim=0)[:128][None]
    embeddings, valids = [], []
    for context in contexts:
        voxel = world_to_voxel(points_world[:, None], context["affine"])
        feature = sample_volume(context["decoded"], voxel)[0, 0]
        valid = sample_volume(context["valid_mask"].float(), voxel,
                              mode="nearest")[0, 0, :, 0] > .5
        embeddings.append(F.normalize(model.local_contrast_projection(feature), dim=-1))
        valids.append(valid)
    losses = []
    for i in range(len(contexts)):
        for j in range(i+1, len(contexts)):
            valid = valids[i] & valids[j]
            if valid.sum() < 2:
                continue
            first, second = embeddings[i][valid], embeddings[j][valid]
            scores = first @ second.T/.1
            labels = torch.arange(len(first), device=first.device)
            losses.append((F.cross_entropy(scores, labels)+F.cross_entropy(scores.T, labels))/2)
    return torch.stack(losses).mean() if losses else state["z"].sum()*0
