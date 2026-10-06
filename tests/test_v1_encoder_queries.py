"""Geometry and gradient regressions for the v1 CT/query path."""
import torch
import pytest

from anatoskel.v1.config import V1Config
from anatoskel.v1.encoder import CTEncoder, valid_physical_center
from anatoskel.v1.queries import GeometryQueries, LatentAnchor
from anatoskel.v1.attention import physical_sample, feature_world_grid


def test_patch_merge_sampling_uses_cell_centres_with_oblique_affine():
    affine = torch.tensor([[[0., -2., 0., 11.], [1.5, 0., 0., -7.],
                            [0., 0., 3., 4.], [0., 0., 0., 1.]]])
    stride, offset = 8, 3.5
    world = feature_world_grid((3, 3, 3), affine, stride, offset)
    feature = world[..., 0].reshape(1, 1, 3, 3, 3).requires_grad_()
    # Every feature lattice centre must retrieve its own value. A missing
    # half-patch offset gives a physically different interpolation result.
    result = physical_sample(feature, world[:, None], affine, stride, offset)
    assert torch.allclose(result[0, 0, :, 0], world[0, :, 0], atol=1e-5)
    result.sum().backward()
    assert torch.allclose(feature.grad, torch.ones_like(feature), atol=1e-5)


def test_padding_does_not_move_physical_position_origin():
    valid = torch.zeros(1, 1, 32, 32, 32, dtype=torch.bool)
    valid[:, :, 3:9, 8:16, 10:20] = True
    affine = torch.eye(4)[None]
    affine[:, :3, 3] = torch.tensor([100., -20., 30.])
    center = valid_physical_center(valid, affine)
    assert torch.allclose(center, torch.tensor([[114.5, -8.5, 35.5]]))


def test_ct_queries_tiles_and_anchor_have_live_gradients():
    torch.set_num_threads(2)
    config = V1Config.smoke()
    encoder = CTEncoder(config)
    detector = GeometryQueries(config)
    anchor = LatentAnchor(config)
    volume = torch.rand(1, 1, 32, 32, 32)
    affine = torch.eye(4)[None]
    context = encoder(volume, affine, bone_volume=volume * 2 - 1)
    assert [value.shape[1] for value in context["features"]] == [12, 24, 48, 96, 192]
    assert context["offsets_vox"] == [.5, 1.5, 3.5, 7.5, 15.5]
    global_output = detector.forward_global(context)
    assert global_output["class_logits"].shape == (1, 4, 27)
    selected = torch.tensor([[0, 0], [0, 1]])
    # Repeated tile evidence produces one memory budget and fixed anchors.
    local = detector.refine_instances(global_output, selected,
                                       [context, context, context], torch.tensor([0, 0, 1]))
    assert local["memory"].shape[1] <= 2 * config.context_tokens
    assert local["memory_valid"].any(-1).all()
    assert global_output["memory_is_spatial"] is False
    assert torch.equal(local["memory"][:, :config.context_tokens], global_output["memory"].expand(2, -1, -1))
    assert torch.equal(local["anchors_mm"], global_output["anchors_mm"][0, :2])
    latent = anchor(local["queries"], local["memory"], local["memory_valid"])
    assert latent.shape == (2, 8, 16)
    loss = latent.square().mean() + global_output["heatmap_logits"].square().mean()
    loss.backward()
    for parameter in (encoder.swinViT.patch_embed.proj.weight,
                      detector.context_slots,
                      detector.local_blocks[0].level_weights.weight,
                      detector.local_blocks[0].level_content_score.weight,
                      detector.local_blocks[0].level_position_score.weight,
                      detector.point_offsets.weight, detector.proposal_query,
                      detector.proposal_score.weight, anchor.slots):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_voco_loader_requires_both_swin_and_residual_skip_coverage(tmp_path):
    encoder = CTEncoder(V1Config.smoke())
    state = {"module.backbone." + key: value.clone() for key, value in encoder.state_dict().items()
             if key.startswith(("swinViT.", "encoder1.", "encoder2.", "encoder3.", "encoder4.", "encoder10."))}
    checkpoint = tmp_path / "encoder.pt"
    torch.save({"state_dict": state}, checkpoint)
    report = encoder.load_voco_checkpoint(checkpoint)
    assert report["coverage"] == {"swin": 1., "residual_skips": 1.}
    assert report["decoder_pretrained"] is False
    bad = tmp_path / "partial.pt"
    torch.save({"state_dict": {key: value for key, value in state.items() if "swinViT" in key}}, bad)
    with pytest.raises(ValueError, match="residual_skips"):
        encoder.load_voco_checkpoint(bad)


def test_empty_ct_support_is_rejected():
    encoder = CTEncoder(V1Config.smoke())
    volume = torch.zeros(1, 1, 32, 32, 32)
    with pytest.raises(ValueError, match="observed CT"):
        encoder(volume, torch.eye(4)[None], torch.zeros_like(volume, dtype=torch.bool))


def test_duplicate_tiles_do_not_duplicate_physical_memory():
    torch.set_num_threads(2)
    config = V1Config.smoke()
    detector = GeometryQueries(config)
    context = CTEncoder(config)(torch.rand(1, 1, 32, 32, 32), torch.eye(4)[None])
    reference = torch.tensor([[15.5, 15.5, 15.5]])
    single, single_valid = detector._multi_tile_memory([context], reference)
    repeated, repeated_valid = detector._multi_tile_memory([context, context], reference)
    assert torch.equal(single_valid, repeated_valid)
    assert torch.allclose(single, repeated, atol=1e-5)


def test_extent_positive_unbounded_and_tile_scale_initialized():
    from dataclasses import replace
    config = V1Config.smoke()
    query = GeometryQueries(config)
    typed = torch.randn(1, 2, 3, config.query_dim)
    centers = torch.zeros(1, 2, 3)
    bounds = query._heads(typed, centers, centers)["visible_bounds_mm"]
    assert torch.allclose(bounds[..., 0, :], torch.full((1, 2, 3), -7.))
    assert torch.allclose(bounds[..., 1, :], torch.full((1, 2, 3), 7.))
    with torch.no_grad():
        query.extent.bias.fill_(20.)
    wide = query._heads(typed, centers, centers)["visible_bounds_mm"]
    assert wide[..., 1, :].min() > 100
    default_tile = GeometryQueries(replace(config, local_size=128))
    bounds = default_tile._heads(typed, centers, centers)["visible_bounds_mm"]
    assert torch.allclose(bounds[..., 1, :], torch.full((1, 2, 3), 25.), atol=1e-5)


def test_custom_encoder_checkpointing_preserves_forward_and_gradients():
    from dataclasses import replace
    torch.set_num_threads(2)
    config = V1Config.smoke()
    regular = CTEncoder(config).train()
    checkpointed = CTEncoder(replace(config, gradient_checkpointing=True)).train()
    checkpointed.load_state_dict(regular.state_dict())
    volume = torch.rand(1, 1, 32, 32, 32)
    affine = torch.eye(4)[None]
    outputs = [encoder(volume, affine)["decoded"] for encoder in (regular, checkpointed)]
    assert torch.allclose(outputs[0], outputs[1], atol=1e-6)
    for output in outputs:
        output.square().mean().backward()
    named = dict(regular.named_parameters())
    checked = dict(checkpointed.named_parameters())
    for name in ("swinViT.patch_embed.proj.weight", "bone_branch.0.conv1.weight", "decoder.4.conv1.weight"):
        assert named[name].grad is not None and checked[name].grad is not None
        assert torch.allclose(named[name].grad, checked[name].grad, atol=1e-6, rtol=1e-4)


def test_bfloat16_autocast_preserves_physical_sampling_coordinates():
    affine = torch.eye(4)[None]
    affine[:, :3, 3] = torch.tensor([531.37, -819.62, 150.19])
    world = feature_world_grid((3, 3, 3), affine, 8, 3.5)
    feature = torch.arange(27.).reshape(1, 1, 3, 3, 3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        sample = physical_sample(feature, world[:, None], affine, 8, 3.5)
    assert sample.dtype == torch.float32
    assert torch.allclose(sample[0, 0, :, 0], feature.flatten(), atol=5e-5)
