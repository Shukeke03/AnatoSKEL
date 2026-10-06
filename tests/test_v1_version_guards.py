"""A changed shape/reference model cannot silently reuse old diffusion stats."""
import torch

from anatoskel.v1 import AnatoSKELV1, V1Config


def initialized():
    model = AnatoSKELV1(V1Config.smoke())
    for name in ("shape_trained", "anchor_trained", "diffusion_trained", "adapters_trained"):
        getattr(model, name).fill_(True)
    model.ct_geometry_steps.fill_(5)
    z = torch.randn(2, 8, 16)
    model.diffusion.calibrate_residuals(z, torch.zeros_like(z))
    return model


def test_updated_ct_reference_invalidates_statistics_and_downstream_readiness():
    model = initialized()
    model.configure_stage("ct")
    assert model.shape_trained
    assert not model.diffusion_trained and not model.adapters_trained
    assert not model.diffusion.stats_ready
    assert not any(p.requires_grad for p in model.shape.parameters())
    assert any(p.requires_grad for p in model.anchor_predictor.parameters())
    model.configure_stage("shape")
    assert not model.anchor_trained and model.ct_geometry_steps == 0


def test_new_diffusion_base_disables_and_resets_old_partial_adapters():
    model = initialized()
    with torch.no_grad():
        for block in model.diffusion.blocks:
            block.adapter.up.weight.fill_(.1)
            block.adapter.up.bias.fill_(.2)
    model.configure_stage("diffusion")
    assert model.diffusion.stats_ready  # Same fixed CT reference.
    assert not model.adapters_trained
    for block in model.diffusion.blocks:
        assert torch.count_nonzero(block.adapter.up.weight) == 0
        assert torch.count_nonzero(block.adapter.up.bias) == 0
        assert not any(p.requires_grad for p in block.adapter.parameters())
