import torch

from anatoskel.config import ModelConfig
from anatoskel.models.detector import CoarseDetector, mask_centers_mm
from anatoskel.models.encoder import (ResUNet, bounded_pool_shape, partition_windows,
                                      physical_grid, unpartition_windows)


def test_physical_grid_xyz_affine_and_stride():
    affine = torch.eye(4).unsqueeze(0)
    affine[0, :3, :3] = torch.diag(torch.tensor([-2.0, 3.0, 4.0]))
    affine[0, :3, 3] = torch.tensor([10.0, 20.0, 30.0])
    points = physical_grid((2, 2, 2), affine, stride=2)
    assert torch.equal(points[0, 0], torch.tensor([10.0, 20.0, 30.0]))
    assert torch.equal(points[0, -1], torch.tensor([6.0, 26.0, 38.0]))
    shape = bounded_pool_shape((12, 11, 8), 64)
    assert shape[0] * shape[1] * shape[2] <= 64


def test_shifted_windows_roundtrip_and_no_edge_wrap():
    value = torch.arange(5 * 6 * 9).reshape(1, 1, 5, 6, 9).float() + 1
    for shift in (0, 2):
        tokens, valid, metadata = partition_windows(value, 4, shift)
        assert torch.equal(unpartition_windows(tokens, metadata), value)
        assert valid.sum() == value.numel()
        # Left/right edges cannot be coupled through a cyclic shift.
        left = value[:, :, :, :, 0].flatten()
        right = value[:, :, :, :, -1].flatten()
        for window in tokens[..., 0]:
            has_left = torch.isin(window, left).any()
            has_right = torch.isin(window, right).any()
            assert not (has_left and has_right)


def test_mask_centroid_in_mm():
    masks = torch.zeros((1, 2, 3, 5, 7), requires_grad=True)
    affine = torch.eye(4).unsqueeze(0)
    affine[0, :3, :3] = torch.diag(torch.tensor([2.0, 3.0, 4.0]))
    affine[0, :3, 3] = torch.tensor([-5.0, 10.0, 20.0])
    centers = mask_centers_mm(masks, affine)
    assert torch.allclose(centers, torch.tensor([[[1.0, 16.0, 24.0], [1.0, 16.0, 24.0]]]))
    centers.sum().backward()
    assert torch.isfinite(masks.grad).all()


def test_mask_centroid_half_precision_uses_safe_accumulation():
    masks = torch.zeros((1, 1, 32, 32, 128), dtype=torch.float16)
    centers = mask_centers_mm(masks, torch.eye(4).unsqueeze(0))
    assert torch.allclose(centers, torch.tensor([[[63.5, 15.5, 15.5]]]))
    assert torch.isfinite(centers).all()


def test_detection_equivariant_to_arbitrary_nifti_origin():
    torch.set_num_threads(2)
    detector = CoarseDetector(ModelConfig.smoke()).eval()
    volume = torch.randn(1, 1, 17, 19, 21)
    first_affine = torch.eye(4).unsqueeze(0)
    second_affine = first_affine.clone()
    translation = torch.tensor([32.0, 16.0, 8.0])
    second_affine[0, :3, 3] = translation
    with torch.no_grad():
        first = detector(volume, first_affine)
        second = detector(volume, second_affine)
    assert torch.allclose(first["class_logits"], second["class_logits"], atol=1e-6)
    assert torch.allclose(first["mask_logits"], second["mask_logits"], atol=1e-6)
    assert torch.allclose(first["centers_mm"] + translation, second["centers_mm"], atol=1e-5)


def test_encoder_and_detector_backward_with_odd_spatial_shape():
    torch.set_num_threads(2)
    config = ModelConfig.smoke()
    config.attention_blocks = 2  # Include the shifted local-attention block.
    volume = torch.randn(1, 1, 17, 19, 21)
    affine = torch.eye(4).unsqueeze(0)
    detector = CoarseDetector(config)
    result = detector(volume, affine)
    assert result["class_logits"].shape == (1, config.num_queries, 27)
    assert result["mask_logits"].shape == (1, config.num_queries, 17, 19, 21)
    assert result["centers_mm"].shape == (1, config.num_queries, 3)
    loss = result["mask_logits"].square().mean() + result["class_logits"].square().mean()
    loss.backward()
    assert detector.query_content.weight.grad.abs().sum() > 0
    assert detector.encoder.stem.weight.grad.abs().sum() > 0
    config.gradient_checkpointing = True
    fine = ResUNet(config, fine=True)
    context = fine(volume, affine)
    assert len(context["features"]) == 5
    assert context["decoded"].shape == (1, 8, 17, 19, 21)
    context["decoded"].square().mean().backward()
    assert torch.isfinite(fine.window.blocks[1].attention.in_proj_weight.grad).all()
