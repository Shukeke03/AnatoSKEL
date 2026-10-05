"""Backend checks cover geometry and gradients, not reconstruction accuracy."""
import math

import pytest
import torch

from anatoskel.config import ModelConfig
from anatoskel.models.backend import (
    AnatomicalReconstructor,
    axis_angle_to_matrix,
    canonical_to_world,
    rotation_6d_to_matrix,
    sample_volume,
    world_to_canonical,
    world_to_voxel,
)


def make_context(config, size=16):
    return {
        "features": [torch.randn(1, channel, max(1, size // (2 ** level)),
                                 max(1, size // (2 ** level)),
                                 max(1, size // (2 ** level)), requires_grad=True)
                     for level, channel in enumerate(config.fine_channels)],
        "strides": [1, 2, 4, 8, 16],
        "volume": torch.randn(1, 1, size, size, size),
        "affine": torch.eye(4).unsqueeze(0),
    }


def assert_rotation(rotation):
    identity = torch.eye(3).expand_as(rotation)
    torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation, identity,
                               atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(rotation.shape[:-2]),
                               atol=1e-5, rtol=1e-5)


def test_rotation_encodings_and_zero_update_gradients():
    encoding = torch.tensor([[1., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 0.],
                             [1., 0., 0., 2., 0., 0.]], requires_grad=True)
    rotation = rotation_6d_to_matrix(encoding)
    assert_rotation(rotation)
    rotation.sum().backward()
    assert torch.isfinite(encoding.grad).all()

    omega = torch.zeros(2, 3, requires_grad=True)
    zero_rotation = axis_angle_to_matrix(omega)
    assert_rotation(zero_rotation)
    zero_rotation[..., 0, 1].sum().backward()
    assert torch.isfinite(omega.grad).all()
    torch.testing.assert_close(omega.grad[:, 2], -torch.ones(2))
    quarter_turn = axis_angle_to_matrix(torch.tensor([[0., 0., math.pi / 2]]))
    torch.testing.assert_close(quarter_turn[0] @ torch.tensor([1., 0., 0.]),
                               torch.tensor([0., 1., 0.]), atol=1e-6, rtol=1e-6)


def test_inverse_rigid_preserves_distance():
    rotation = axis_angle_to_matrix(torch.tensor([[[0.2, -0.3, 0.5]]]))
    translation = torch.tensor([[[15., -40., 80.]]])
    canonical = torch.randn(1, 1, 16, 3) * 20
    world = canonical_to_world(canonical, rotation, translation)
    recovered = world_to_canonical(world, rotation, translation)
    torch.testing.assert_close(recovered, canonical, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.vector_norm(world[..., 1, :] - world[..., 0, :], dim=-1),
                               torch.linalg.vector_norm(canonical[..., 1, :] - canonical[..., 0, :], dim=-1))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        amp_world = canonical_to_world(canonical, rotation, translation)
        amp_recovered = world_to_canonical(amp_world, rotation, translation)
    assert amp_world.dtype == torch.float32
    torch.testing.assert_close(amp_world, world)
    torch.testing.assert_close(amp_recovered, recovered)


def test_affine_and_stride_sampling_uses_voxel_centres():
    # Reflected x axis, anisotropic spacings and translation exercise the full affine.
    affine = torch.tensor([[[-2., 0., 0., 50.], [0., 3., 0., -20.],
                            [0., 0., 4., 10.], [0., 0., 0., 1.]]])
    voxel = torch.tensor([[[[2.25, 2.5, 1.5], [4., 4., 4.]]]])
    world = torch.einsum("bij,bnpj->bnpi", affine[:, :3, :3], voxel) + affine[:, None, None, :3, 3]
    recovered = world_to_voxel(world, affine)
    torch.testing.assert_close(recovered, voxel)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        amp_voxel = world_to_voxel(world, affine)
    assert amp_voxel.dtype == torch.float32
    torch.testing.assert_close(amp_voxel, voxel)
    z, y, x = torch.meshgrid(torch.arange(8.), torch.arange(8.), torch.arange(8.), indexing="ij")
    volume = (x + 10 * y + 100 * z)[None, None]
    expected = voxel[..., 0] + 10 * voxel[..., 1] + 100 * voxel[..., 2]
    torch.testing.assert_close(sample_volume(volume, recovered).squeeze(-1), expected)
    strided = volume[..., ::2, ::2, ::2]
    torch.testing.assert_close(sample_volume(strided, recovered, stride=2).squeeze(-1), expected)
    anisotropic_strided = volume[..., ::1, ::2, ::2]
    torch.testing.assert_close(sample_volume(anisotropic_strided, recovered,
                                            stride=(1, 2, 2)).squeeze(-1), expected)


def test_backend_forward_field_and_refinement_backward():
    torch.manual_seed(4)
    config = ModelConfig.smoke()
    model = AnatomicalReconstructor(config)
    context = make_context(config)
    queries = torch.randn(1, 2, config.query_dim, requires_grad=True)
    classes = torch.tensor([[18, 19]])
    anchors = torch.tensor([[[8., 8., 7.], [8., 8., 10.]]])
    state = model(context, queries, classes, anchors, refinements=True)
    assert state["beta"].shape == (1, 2, config.beta_dim)
    assert_rotation(state["rotation"])
    # Zero-initialized refinement updates preserve the initial rigid placement.
    torch.testing.assert_close(state["translation"], anchors)
    points = anchors.unsqueeze(2) + torch.randn(1, 2, 7, 3)
    result = model.field(context, state, points)
    assert result["field"].shape == (1, 2, 7)
    assert result["valid"].all()
    assert result["residual"].count_nonzero() == 0
    loss = (result["field"] - 0.2).square().mean() + state["queries"].square().mean()
    loss.backward()
    assert torch.isfinite(queries.grad).all()
    assert model.refinement.delta[-1].weight.grad is not None
    assert torch.isfinite(model.refinement.delta[-1].weight.grad).all()
    assert torch.isfinite(model.rotation_head.weight.grad).all()
    assert model.residual.output.weight.grad.abs().sum() > 0


def test_invalid_fov_and_all_masked_refinement_are_finite():
    config = ModelConfig.smoke()
    model = AnatomicalReconstructor(config)
    context = make_context(config)
    context["valid_mask"] = torch.zeros_like(context["volume"])
    state = model(context, torch.randn(1, 1, config.query_dim), torch.tensor([[18]]),
                  torch.tensor([[[1000., 1000., 1000.]]]), refinements=True)
    for key in ("queries", "rotation", "translation", "beta"):
        assert torch.isfinite(state[key]).all()
    # A learned residual bias is ignored outside the actual observed volume.
    with torch.no_grad():
        model.residual.output.bias.fill_(2)
    result = model.field(context, state, state["translation"].unsqueeze(2))
    assert not result["valid"].any()
    torch.testing.assert_close(result["field"], result["prior"])


def test_freezing_requires_prior_and_preserves_shape_code_gradient():
    config = ModelConfig.smoke()
    model = AnatomicalReconstructor(config)
    with pytest.raises(ValueError, match="untrained"):
        model.freeze_prior()
    model.freeze_prior(pretrained=True)
    assert not any(parameter.requires_grad for parameter in model.prior.parameters())
    beta = torch.zeros(1, 1, config.beta_dim, requires_grad=True)
    values = model.prior(torch.randn(1, 1, 8, 3), beta, torch.tensor([[18]]))
    values.sum().backward()
    assert beta.grad.abs().sum() > 0


def test_refinement_mask_probabilities_select_the_same_bone_channel():
    config = ModelConfig.smoke()
    model = AnatomicalReconstructor(config)
    context = make_context(config)
    logits = torch.stack((torch.full((16, 16, 16), -2.), torch.full((16, 16, 16), 3.)), 0)
    context["mask_logits"] = logits.unsqueeze(0)
    points = torch.tensor([[[[5., 5., 5.], [6., 6., 6.]],
                            [[5., 5., 5.], [6., 6., 6.]]]])
    probabilities = model._mask_probability(context, points, bones=2)
    assert probabilities.shape == (1, 2, 2)
    torch.testing.assert_close(probabilities[0, 0], torch.full((2,), torch.sigmoid(torch.tensor(-2.))))
    torch.testing.assert_close(probabilities[0, 1], torch.full((2,), torch.sigmoid(torch.tensor(3.))))


def test_packed_per_bone_crops_use_independent_affines_and_shared_relations():
    config = ModelConfig.smoke()
    model = AnatomicalReconstructor(config)
    context = make_context(config)
    context["features"] = [torch.cat((value, value), 0) for value in context["features"]]
    context["volume"] = torch.cat((torch.zeros_like(context["volume"]),
                                    torch.ones_like(context["volume"])), 0)
    context["valid_mask"] = torch.ones_like(context["volume"])
    context["affine"] = context["affine"].repeat(2, 1, 1)
    context["affine"][1, 0, 3] = 500
    context["bone_batch_shape"] = (1, 2)
    anchors = torch.tensor([[[8., 8., 8.], [508., 8., 8.]]])
    queries = torch.randn(1, 2, config.query_dim, requires_grad=True)
    state = model(context, queries, torch.tensor([[18, 19]]), anchors, refinements=True)
    points = anchors.unsqueeze(2).expand(-1, -1, 4, -1)
    result = model.field(context, state, points)
    assert result["valid"].all()
    torch.testing.assert_close(result["hu"][0, 0, :, 0], torch.zeros(4))
    torch.testing.assert_close(result["hu"][0, 1, :, 0], torch.ones(4))
    assert result["features"].shape == (1, 2, 4, 4 * config.sample_channels)
    (result["field"].square().mean() + state["queries"].square().mean()).backward()
    assert torch.isfinite(queries.grad).all()
    misplaced = points.clone()
    misplaced[0, 1, :, 0] = 8
    invalid = model.field(context, state, misplaced)
    assert not invalid["valid"][0, 1].any()


def test_low_precision_grid_sampling_promotes_and_preserves_gradients():
    for dtype in (torch.bfloat16, torch.float16):
        volume = torch.randn(1, 2, 4, 4, 4, dtype=dtype, requires_grad=True)
        points = torch.tensor([[[[1.5, 2., 1.]]]], requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            sampled = sample_volume(volume, points)
        assert sampled.dtype == torch.float32
        sampled.square().sum().backward()
        assert torch.isfinite(volume.grad).all()
        assert volume.grad.abs().sum() > 0
        assert torch.isfinite(points.grad).all()


def test_cpu_bf16_packed_fine_encoder_field_and_backward():
    from anatoskel.models.network import AnatoSKEL

    config = ModelConfig.smoke()
    model = AnatoSKEL(config)
    volume = torch.randn(2, 1, 32, 32, 32, requires_grad=True)
    affine = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    affine[1, 0, 3] = 50
    queries = torch.randn(1, 2, config.query_dim, requires_grad=True)
    anchors = torch.tensor([[[16., 16., 16.], [66., 16., 16.]]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        context = model.encode_fine(volume, affine)
        context["bone_batch_shape"] = (1, 2)
        context["valid_mask"] = torch.ones_like(volume, dtype=torch.bool)
        context["mask_logits"] = model.fine_masks(context, queries)
        context = model.backend.prepare_context(context)
        state = model.reconstruct(context, queries, torch.tensor([[18, 19]]), anchors)
    assert context["anatomical_features"][0].dtype == torch.bfloat16
    assert state["rotation"].dtype == torch.float32
    assert state["translation"].dtype == torch.float32
    # Main training intentionally queries cached AMP features outside autocast.
    points = anchors.unsqueeze(2) + torch.randn(1, 2, 8, 3)
    samples = model.field(context, state, points)
    assert samples["features"].dtype == torch.float32
    assert samples["field"].dtype == torch.float32
    loss = samples["field"].square().mean() + context["mask_logits"].float().square().mean()
    loss.backward()
    assert torch.isfinite(volume.grad).all()
    assert torch.isfinite(queries.grad).all()
    assert torch.isfinite(model.backend.refinement.delta[-1].weight.grad).all()
