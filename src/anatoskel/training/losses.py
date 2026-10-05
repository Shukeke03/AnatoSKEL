"""Mask-aware Hungarian assignment and observed-domain reconstruction losses."""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

VERSE_IDS = tuple(range(1, 26)) + (28,)


def detector_loss(output, labels, valid_mask, affine, sample_points=8192):
    """One scan at a time; assignment uses classes, masks and physical centers.

    Unknown/unannotated voxels must have valid_mask=False, never become background.
    """
    logits = output["class_logits"][0]
    masks = output["mask_logits"][0]
    device = logits.device
    labels, valid_mask = labels.to(device), valid_mask.to(device)
    ids = [int(x) for x in torch.unique(labels[valid_mask]) if int(x) > 0]
    if len(ids) > logits.shape[0]:
        raise ValueError("More annotated bones than instance queries")
    valid_indices = valid_mask.flatten().nonzero().flatten()
    if valid_indices.numel() == 0:
        raise ValueError("No valid annotated voxels")
    # Stratify foreground to avoid losing small bones at coarse resolution.
    choices = [valid_indices[torch.randperm(len(valid_indices), device=device)[:sample_points]]]
    for label in ids:
        foreground = ((labels == label) & valid_mask).flatten().nonzero().flatten()
        choices.append(foreground[torch.randperm(len(foreground), device=device)[:128]])
    indices = torch.unique(torch.cat(choices))
    sampled = masks.flatten(1)[:, indices]
    target = torch.stack([(labels.flatten()[indices] == label).float() for label in ids]) \
        if ids else sampled.new_empty((0, len(indices)))
    target_classes = torch.tensor([VERSE_IDS.index(x) for x in ids], device=device)
    centers = []
    for label in ids:
        zyx = ((labels == label) & valid_mask).nonzero().float().mean(0)
        xyz = zyx.flip(0)
        centers.append(affine[:3, :3].to(device).float() @ xyz + affine[:3, 3].to(device))
    centers = torch.stack(centers).float() if centers else logits.new_empty((0, 3))
    if ids:
        p = sampled.sigmoid()
        pair_dice = 1 - (2 * p @ target.T + 1) / (p.sum(1)[:, None] + target.sum(1) + 1)
        pair_bce = (F.softplus(sampled).mean(1)[:, None]
                    - sampled @ target.T / sampled.shape[1])
        cost = (-logits.log_softmax(-1)[:, target_classes] + 2 * pair_dice + pair_bce
                + torch.cdist(output["centers_mm"][0].float(), centers, p=1) / 100)
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
        rows = torch.as_tensor(rows, device=device)
        columns = torch.as_tensor(columns, device=device)
    else:
        rows = columns = torch.empty(0, dtype=torch.long, device=device)
    classes = torch.full((logits.shape[0],), logits.shape[-1] - 1, device=device,
                         dtype=torch.long)
    classes[rows] = target_classes[columns]
    weights = torch.ones(logits.shape[-1], device=device)
    weights[-1] = 0.1
    cls = F.cross_entropy(logits.float(), classes, weight=weights)
    if ids:
        selected, truth = sampled[rows], target[columns]
        bce = F.binary_cross_entropy_with_logits(selected.float(), truth)
        p = selected.float().sigmoid()
        dice = (1 - (2 * (p * truth).sum(1) + 1)
                / (p.sum(1) + truth.sum(1) + 1)).mean()
        center = F.smooth_l1_loss(output["centers_mm"][0, rows].float() / 100,
                                  centers[columns] / 100)
    else:
        bce = dice = center = logits.sum() * 0
    total = cls + 2 * dice + bce + center
    assignment = [(int(r), ids[int(c)]) for r, c in zip(rows, columns)]
    return total, {"class": float(cls.detach()), "dice": float(dice.detach()),
                   "mask_bce": float(bce.detach()), "center": float(center.detach())}, assignment


def field_loss(prediction, target, valid, beta):
    observed = valid.bool() & torch.isfinite(target)
    if not torch.isfinite(prediction[observed]).all():
        raise RuntimeError("Nonfinite field prediction in the observed supervision domain")
    if not observed.any():
        raise ValueError("No observed points available for field supervision")
    distance = F.smooth_l1_loss(prediction[observed].float(), target[observed].float(), beta=0.1)
    occupancy = F.binary_cross_entropy_with_logits(-5 * prediction[observed].float(),
                                                   (target[observed] < 0).float())
    return distance + 0.1 * occupancy + 1e-4 * beta.square().mean()
