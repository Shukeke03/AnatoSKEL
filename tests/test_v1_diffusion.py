"""Geometry gradients, partial-target safety, and residual diffusion numerics."""
import pytest
import torch

from anatoskel.v1.config import V1Config
from anatoskel.v1.diffusion import ResidualDiffusion, cosine_coefficients, v_to_clean, v_to_noise
from anatoskel.v1.shape import ShapeAutoencoder


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tensors(config, batch=2):
    return (
        torch.randn(batch, config.latent_tokens, config.latent_dim),
        torch.randn(batch, config.latent_tokens, config.latent_dim),
        torch.randn(batch, 5, config.query_dim),
    )


def calibrated(config):
    model = ResidualDiffusion(config)
    target, anchor, condition = tensors(config)
    model.calibrate_residuals(target, anchor, complete=torch.ones(len(target), dtype=torch.bool))
    return model, target, anchor, condition


def test_v_parameterization_recovers_clean_and_noise_and_cosine_schedule():
    alpha, sigma = cosine_coefficients(1000)
    assert torch.all(alpha[1:] <= alpha[:-1])
    assert torch.allclose(alpha.square() + sigma.square(), torch.ones_like(alpha), atol=1e-6)
    assert alpha[-1] < 1e-3
    clean, noise = torch.randn(3, 4, 8), torch.randn(3, 4, 8)
    a = alpha[torch.tensor([0, 300, 999])][:, None, None]
    s = sigma[torch.tensor([0, 300, 999])][:, None, None]
    noisy, velocity = a * clean + s * noise, a * noise - s * clean
    assert torch.allclose(v_to_clean(noisy, velocity, a, s), clean, atol=5e-7)
    assert torch.allclose(v_to_noise(noisy, velocity, a, s), noise, atol=5e-7)


def test_complete_bank_statistics_accumulate_freeze_and_restore():
    config = V1Config.smoke()
    model = ResidualDiffusion(config)
    target, anchor, condition = tensors(config, batch=4)
    with pytest.raises(RuntimeError, match="not calibrated"):
        model.loss(target, anchor, condition)
    with pytest.raises(ValueError, match="complete"):
        model.calibrate_residuals(target, anchor, complete=torch.tensor([True, True, False, True]))
    model.calibrate_residuals(target[:2], anchor[:2], finalize=False)
    assert not model.stats_ready
    model.calibrate_residuals(target[2:], anchor[2:])
    expected = (target - anchor).flatten(0, 1)
    assert torch.allclose(model.residual_mean, expected.mean(0), atol=1e-6)
    assert torch.allclose(model.residual_scale, expected.std(0, correction=0), atol=1e-6)
    with pytest.raises(RuntimeError, match="frozen"):
        model.calibrate_residuals(target, anchor)
    restored = ResidualDiffusion(config)
    restored.load_state_dict(model.state_dict())
    assert restored.stats_ready
    assert torch.equal(restored.residual_scale, model.residual_scale)
    model.calibrate_residuals(target + 1, anchor, reset=True)
    assert torch.allclose(model.residual_mean, expected.mean(0) + 1, atol=1e-6)


def test_frozen_decoder_keeps_latent_gradients_and_reuses_memory():
    config = V1Config.smoke()
    shape = ShapeAutoencoder(config).freeze()
    z = torch.randn(2, config.latent_tokens, config.latent_dim, requires_grad=True)
    points = torch.randn(2, 23, 3)
    memory = shape.prepare_memory(z)
    first = shape.decode(z, points, memory=memory, chunk_size=7)
    second = shape.decode(z, points, memory=memory, chunk_size=100)
    assert torch.allclose(first, second, atol=2e-7)
    first.square().mean().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in shape.parameters())


def test_masked_surface_padding_has_no_effect_on_encoded_complete_shape():
    config = V1Config.smoke()
    shape = ShapeAutoencoder(config).eval()
    points, normals = torch.randn(2, 9, 3), torch.randn(2, 9, 3)
    valid = torch.ones(2, 9, dtype=torch.bool)
    valid[:, -3:] = False
    expected = shape.encode(points, normals, valid)
    points[:, -3:] = float("nan")
    normals[:, -3:] = float("nan")
    actual = shape.encode(points, normals, valid)
    assert torch.allclose(actual, expected)
    with pytest.raises(ValueError, match="reliable"):
        shape.encode(points, normals, torch.zeros_like(valid))


def test_diffusion_loss_min_snr_and_no_full_target_for_partial_case():
    config = V1Config.smoke()
    model, target, anchor, condition = calibrated(config)
    loss, info = model.loss(target, anchor, condition, complete=torch.ones(2, dtype=torch.bool))
    assert torch.isfinite(loss)
    assert info["predicted_clean_z"].shape == target.shape
    assert torch.allclose(info["weight"], info["snr"].clamp_max(5) / (info["snr"] + 1))
    loss.backward()
    assert model.output_projection.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="complete"):
        model.loss(target, anchor, condition, complete=torch.tensor([True, False]))


def test_condition_masked_values_are_ignored_and_empty_observation_rejected():
    config = V1Config.smoke()
    model, _, anchor, condition = calibrated(config)
    # Make the initialized model genuinely condition dependent, as after training.
    with torch.no_grad():
        model.output_projection.weight.normal_(std=0.02)
        for block in model.blocks:
            block.modulation[-1].bias.fill_(0.1)
    valid = torch.ones(condition.shape[:2], dtype=torch.bool)
    valid[:, -2:] = False
    expected = model.sample(anchor, condition, valid, seed=52)
    condition[:, -2:] = float("nan")
    actual = model.sample(anchor, condition, valid, seed=52)
    assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match="unobserved"):
        model.sample(anchor, condition, torch.zeros_like(valid))


def test_ddim_is_seeded_without_changing_global_rng_and_has_clean_endpoint():
    config = V1Config.smoke()
    model, _, anchor, condition = calibrated(config)
    state = torch.random.get_rng_state()
    first = model.sample(anchor, condition, seed=123)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.equal(first, model.sample(anchor, condition, seed=123))
    assert not torch.equal(first, model.sample(anchor, condition, seed=124))
    assert not first.requires_grad
    # Exact velocity for a known target must recover that target at final alpha=1.
    residual_clean = torch.randn_like(anchor)

    def oracle(noisy, reference, time, ct, valid=None, adapt=False):
        alpha, sigma = model._coefficients(time, noisy.dtype)
        return (alpha * noisy - residual_clean) / sigma

    model.predict_velocity = oracle
    result = model.sample(anchor, condition)
    expected = anchor + model.residual_mean + model.residual_scale * residual_clean
    assert torch.allclose(result, expected, atol=2e-6)


def test_partial_sampling_tail_preserves_decoder_gradient_to_only_adapters():
    config = V1Config.smoke()
    model, _, anchor, condition = calibrated(config)
    shape = ShapeAutoencoder(config).freeze()
    with torch.no_grad():
        # A trained base has nonzero output Jacobians; all-zero initialization
        # intentionally delays hidden-layer learning during the very first update.
        model.output_projection.weight.normal_(std=0.02)
        for block in model.blocks:
            block.modulation[-1].bias.fill_(0.1)
    model.train_adapters()
    calls = []
    original = model.predict_velocity

    def record(*args, **kwargs):
        calls.append((torch.is_grad_enabled(), kwargs["adapt"]))
        return original(*args, **kwargs)

    model.predict_velocity = record
    z = model.sample(anchor, condition, adapt=True, differentiable_tail=True)
    assert calls == [(False, False), (False, False), (True, True), (True, True)]
    assert z.requires_grad
    observed_sdf = shape.decode(z, torch.randn(2, 11, 3))
    observed_sdf.square().mean().backward()
    adapter_parameters = [(name, parameter) for name, parameter in model.named_parameters()
                          if ".adapter." in name]
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for _, parameter in adapter_parameters)
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if ".adapter." not in name)
    assert all(parameter.grad is None for parameter in shape.parameters())
    loss, info = model.loss(torch.randn_like(anchor), anchor, condition, adapter_replay=True)
    assert info["time"].max() <= model.sampling_times()[-config.adapter_tail_steps]
    loss.backward()


def test_checkpointed_frozen_shape_matches_gradients_and_reduces_saved_activations():
    import copy

    config = V1Config.smoke()
    baseline = ShapeAutoencoder(config).freeze()
    checkpointed = copy.deepcopy(baseline)
    checkpointed.config.gradient_checkpointing = True
    z = torch.randn(2, config.latent_tokens, config.latent_dim, requires_grad=True)
    second_z = z.detach().clone().requires_grad_()
    points = torch.randn(2, 89, 3)

    def evaluate(model, latent):
        saved_sizes = []
        parameter_storage = {parameter.untyped_storage().data_ptr() for parameter in model.parameters()}

        def pack(tensor):
            # Exclude references to existing frozen weights; count saved data
            # activations rather than pretending those weights are new memory.
            if tensor.untyped_storage().data_ptr() not in parameter_storage:
                saved_sizes.append(tensor.numel())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            memory = model.prepare_memory(latent)
            result = model.decode(latent, points, memory=memory, chunk_size=17)
        result.square().mean().backward()
        return result, sum(saved_sizes)

    original, original_saved = evaluate(baseline, z)
    actual, checkpoint_saved = evaluate(checkpointed, second_z)
    assert torch.allclose(actual, original, atol=1e-7)
    assert torch.allclose(second_z.grad, z.grad, atol=2e-7)
    assert checkpoint_saved < original_saved/2
    assert all(parameter.grad is None for parameter in checkpointed.parameters())


def test_checkpointed_tail_updates_adapters_through_frozen_base_and_decoder():
    config = V1Config.smoke()
    config.gradient_checkpointing = True
    model, _, anchor, condition = calibrated(config)
    with torch.no_grad():
        model.output_projection.weight.normal_(std=0.02)
        for block in model.blocks:
            block.modulation[-1].bias.fill_(0.1)
    model.train_adapters()
    shape = ShapeAutoencoder(config).freeze()
    z = model.sample(anchor, condition, adapt=True, differentiable_tail=True)
    value = shape.decode(z, torch.randn(2, 21, 3), chunk_size=9)
    value.square().mean().backward()
    assert any(block.adapter.up.weight.grad is not None
               and block.adapter.up.weight.grad.abs().sum() > 0 for block in model.blocks)
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if ".adapter." not in name)
    assert all(parameter.grad is None for parameter in shape.parameters())
