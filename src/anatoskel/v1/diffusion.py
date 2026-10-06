"""Conditional latent-residual v-diffusion with an observation-adaptation tail.

A full latent target is required for the ordinary diffusion objective. Real
scan-truncated instances use ``sample(..., adapt=True, differentiable_tail=True)``
and a separate observation-space loss; they never produce a synthetic v-target.
"""
from __future__ import annotations

from contextlib import nullcontext
import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .config import V1Config
from .shape import SwiGLU


def cosine_coefficients(steps: int) -> tuple[Tensor, Tensor]:
    """Discrete cosine alpha/sigma; the last training state is nearly pure noise."""
    if steps < 2:
        raise ValueError("Diffusion needs at least two training steps")
    time = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
    cumulative = torch.cos((time + 0.008) / 1.008 * math.pi / 2).square()
    cumulative = (cumulative / cumulative[0])[1:].clamp(1e-8, 1.0)
    return cumulative.sqrt().float(), (1.0 - cumulative).sqrt().float()


def v_to_clean(noisy: Tensor, velocity: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
    return alpha * noisy - sigma * velocity


def v_to_noise(noisy: Tensor, velocity: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
    return sigma * noisy + alpha * velocity


class TimeEmbedding(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        half = width // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half) / max(half - 1, 1))
        self.register_buffer("frequencies", frequencies)
        self.width = width
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, time: Tensor) -> Tensor:
        phase = time.float()[:, None] * self.frequencies[None]
        embedding = torch.cat((phase.cos(), phase.sin()), -1)
        if embedding.shape[-1] < self.width:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return self.mlp(embedding)


class TailAdapter(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, hidden)
        self.up = nn.Linear(hidden, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, value: Tensor) -> Tensor:
        return self.up(torch.nn.functional.silu(self.down(self.norm(value))))


class DiTBlock(nn.Module):
    def __init__(self, config: V1Config):
        super().__init__()
        width = config.query_dim
        self.self_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.cross_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.ffn_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.self_attention = nn.MultiheadAttention(width, config.heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, config.heads, batch_first=True)
        self.ffn = SwiGLU(width, config.ffn_hidden)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 9 * width))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.adapter = TailAdapter(width, config.adapter_hidden)

    @staticmethod
    def modulate(value: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return value * (1 + scale[:, None]) + shift[:, None]

    def forward(self, value: Tensor, time_embedding: Tensor, condition: Tensor,
                invalid: Tensor | None, adapt: bool = False) -> Tensor:
        shift_s, scale_s, gate_s, shift_c, scale_c, gate_c, shift_f, scale_f, gate_f = (
            self.modulation(time_embedding).chunk(9, dim=-1)
        )
        query = self.modulate(self.self_norm(value), shift_s, scale_s)
        value = value + gate_s[:, None] * self.self_attention(query, query, query,
                                                            need_weights=False)[0]
        query = self.modulate(self.cross_norm(value), shift_c, scale_c)
        value = value + gate_c[:, None] * self.cross_attention(
            query, condition, condition, key_padding_mask=invalid, need_weights=False,
        )[0]
        if adapt:
            value = value + self.adapter(value)
        value = value + gate_f[:, None] * self.ffn(
            self.modulate(self.ffn_norm(value), shift_f, scale_f)
        )
        return value


class ResidualDiffusion(nn.Module):
    """One shared bone latent, cosine v-prediction, deterministic DDIM sampling.

    Residual statistics are checkpoint buffers. They must be explicitly calibrated
    from complete targets after the initial CT latent predictor has been frozen.
    Statistics cannot change silently once finalized.
    """

    def __init__(self, config: V1Config):
        super().__init__()
        self.config = config
        width = config.query_dim
        self.input_projection = nn.Linear(2 * config.latent_dim, width)
        self.slot_embedding = nn.Parameter(torch.randn(config.latent_tokens, width) * 0.02)
        self.condition_norm = nn.LayerNorm(width)
        self.time_embedding = TimeEmbedding(width)
        self.blocks = nn.ModuleList([DiTBlock(config) for _ in range(config.diffusion_layers)])
        self.output_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.output_projection = nn.Linear(width, config.latent_dim)
        nn.init.zeros_(self.output_modulation[-1].weight)
        nn.init.zeros_(self.output_modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        alpha, sigma = cosine_coefficients(config.diffusion_train_steps)
        self.register_buffer("alpha", alpha)
        self.register_buffer("sigma", sigma)
        self.register_buffer("residual_mean", torch.zeros(config.latent_dim))
        self.register_buffer("residual_scale", torch.ones(config.latent_dim))
        self.register_buffer("stats_ready", torch.tensor(False))
        self.register_buffer("stats_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("stats_sum", torch.zeros(config.latent_dim, dtype=torch.float64))
        self.register_buffer("stats_square_sum", torch.zeros(config.latent_dim, dtype=torch.float64))
        self.adapter_training = False
        self.train_main()

    def _check_latent(self, value: Tensor):
        expected = (self.config.latent_tokens, self.config.latent_dim)
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(f"Expected latent [B,{expected[0]},{expected[1]}]")

    @staticmethod
    def _require_complete(complete: Tensor | None, batch: int):
        if complete is None:
            return  # Supplying a full latent target is an explicit caller assertion.
        if complete.shape != (batch,) or not complete.bool().all():
            raise ValueError("Full latent diffusion targets require complete reliable shapes")

    def _require_stats(self):
        if not bool(self.stats_ready.item()):
            raise RuntimeError("Residual statistics are not calibrated from complete shape targets")

    @torch.no_grad()
    def reset_statistics(self):
        self.stats_ready.fill_(False)
        self.stats_count.zero_()
        self.stats_sum.zero_()
        self.stats_square_sum.zero_()
        self.residual_mean.zero_()
        self.residual_scale.fill_(1)

    @torch.no_grad()
    def calibrate_residuals(self, z_gt: Tensor, z_anchor: Tensor,
                            complete: Tensor | None = None, *, reset: bool = False,
                            finalize: bool = True):
        """Accumulate a complete-shape bank, optionally across several batches.

        Pass ``finalize=False`` for preceding batches and ``finalize=True`` on the
        last one. ``reset=True`` explicitly invalidates previous statistics.
        ``z_anchor`` must come from the fixed reference predictor in the same frame.
        """
        self._check_latent(z_gt)
        self._check_latent(z_anchor)
        if z_gt.shape != z_anchor.shape:
            raise ValueError("Target and reference latent shapes differ")
        self._require_complete(complete, z_gt.shape[0])
        if bool(self.stats_ready.item()) and not reset:
            raise RuntimeError("Residual statistics are frozen; use reset=True to recalibrate")
        if reset:
            self.reset_statistics()
        residual = (z_gt.detach().float() - z_anchor.detach().float()).reshape(
            -1, self.config.latent_dim,
        )
        if residual.shape[0] == 0 or not torch.isfinite(residual).all():
            raise ValueError("Residual bank must be nonempty and finite")
        residual = residual.to(device=self.stats_sum.device, dtype=self.stats_sum.dtype)
        self.stats_sum.add_(residual.sum(0))
        self.stats_square_sum.add_(residual.square().sum(0))
        self.stats_count.add_(residual.shape[0])
        if finalize:
            self.finalize_statistics()
        return self

    @torch.no_grad()
    def finalize_statistics(self):
        if bool(self.stats_ready.item()):
            raise RuntimeError("Residual statistics have already been finalized")
        count = int(self.stats_count.item())
        if count < 2:
            raise ValueError("At least two residual tokens are needed for calibration")
        mean = self.stats_sum / count
        variance = (self.stats_square_sum / count - mean.square()).clamp_min(0)
        self.residual_mean.copy_(mean.to(self.residual_mean))
        self.residual_scale.copy_(variance.sqrt().clamp_min(1e-3).to(self.residual_scale))
        self.stats_ready.fill_(True)

    def train_main(self):
        """Enable base v-model training while keeping the unused adapters frozen."""
        self.requires_grad_(True)
        for block in self.blocks:
            block.adapter.requires_grad_(False)
        self.adapter_training = False
        return self

    def train_adapters(self):
        """Freeze base DiT parameters without suppressing its tail input gradients."""
        self.requires_grad_(False)
        for block in self.blocks:
            block.adapter.requires_grad_(True)
        self.adapter_training = True
        self.train()
        return self

    def _condition(self, condition: Tensor, valid: Tensor | None,
                   batch: int) -> tuple[Tensor, Tensor | None]:
        if condition.ndim != 3 or condition.shape[0] != batch or (
            condition.shape[-1] != self.config.query_dim
        ):
            raise ValueError("CT condition must be [B,K,query_dim]")
        if condition.shape[1] == 0:
            raise ValueError("At least one CT/context condition token is required")
        invalid = None
        if valid is not None:
            if valid.shape != condition.shape[:2]:
                raise ValueError("CT valid mask must have shape [B,K]")
            valid = valid.to(device=condition.device, dtype=torch.bool)
            if not valid.any(1).all():
                raise ValueError("An entirely unobserved instance cannot be reconstructed as observed")
            condition = torch.where(valid[..., None], condition, torch.zeros_like(condition))
            invalid = ~valid
        return self.condition_norm(condition), invalid

    def predict_velocity(self, noisy: Tensor, z_anchor: Tensor, time: Tensor,
                         condition: Tensor, valid: Tensor | None = None,
                         adapt: bool = False) -> Tensor:
        self._check_latent(noisy)
        if noisy.shape != z_anchor.shape:
            raise ValueError("Noisy residual and reference latent shapes differ")
        if time.shape != (noisy.shape[0],):
            raise ValueError("Diffusion time must have shape [B]")
        condition, invalid = self._condition(condition, valid, noisy.shape[0])
        tokens = self.input_projection(torch.cat((noisy, z_anchor), -1)) + self.slot_embedding[None]
        time_embedding = self.time_embedding(time)
        for block in self.blocks:
            if (self.config.gradient_checkpointing and torch.is_grad_enabled()
                    and (self.training or tokens.requires_grad)):
                # Preserve the trainable adapter path through a frozen base. A
                # reentrant checkpoint would lose gradients in the first tail
                # block when its inputs and base weights are all frozen.
                tokens = checkpoint(block, tokens, time_embedding, condition, invalid,
                                    adapt, use_reentrant=False)
            else:
                tokens = block(tokens, time_embedding, condition, invalid, adapt)
        shift, scale = self.output_modulation(time_embedding).chunk(2, dim=-1)
        tokens = self.output_norm(tokens) * (1 + scale[:, None]) + shift[:, None]
        return self.output_projection(tokens)

    def _coefficients(self, time: Tensor, dtype) -> tuple[Tensor, Tensor]:
        return self.alpha[time].to(dtype)[:, None, None], self.sigma[time].to(dtype)[:, None, None]

    def sampling_times(self, device=None) -> Tensor:
        count = self.config.diffusion_sample_steps
        if not 1 <= count <= self.config.diffusion_train_steps:
            raise ValueError("DDIM steps must be between 1 and the number of training times")
        return torch.linspace(self.config.diffusion_train_steps - 1, 0, count,
                              device=device or self.alpha.device).round().long()

    def loss(self, z_gt: Tensor, z_anchor: Tensor, condition: Tensor,
             valid: Tensor | None = None, complete: Tensor | None = None,
             adapter_replay: bool = False) -> tuple[Tensor, dict[str, Tensor]]:
        self._require_stats()
        self._check_latent(z_gt)
        if z_gt.shape != z_anchor.shape:
            raise ValueError("Target and reference latent shapes differ")
        self._require_complete(complete, z_gt.shape[0])
        if adapter_replay and not self.adapter_training:
            raise RuntimeError("Call train_adapters before adapter replay")
        # The target/reference define a fixed residual distribution. Neither may
        # move in response to the diffusion objective.
        z_gt, z_anchor = z_gt.detach(), z_anchor.detach()
        residual = ((z_gt.float() - z_anchor.float() - self.residual_mean)
                    / self.residual_scale)
        max_time = self.config.diffusion_train_steps - 1
        if adapter_replay:
            max_time = int(self.sampling_times()[-self.config.adapter_tail_steps].item())
        time = torch.randint(max_time + 1, (z_gt.shape[0],), device=z_gt.device)
        alpha, sigma = self._coefficients(time, residual.dtype)
        noise = torch.randn_like(residual)
        noisy = alpha * residual + sigma * noise
        target_velocity = alpha * noise - sigma * residual
        predicted_velocity = self.predict_velocity(noisy, z_anchor, time, condition, valid,
                                                    adapt=adapter_replay)
        snr = (alpha.flatten().square() / sigma.flatten().square().clamp_min(1e-12))
        # Min-SNR weighting for v-prediction, not the epsilon-prediction formula.
        weight = snr.clamp_max(5.0) / (snr + 1.0)
        per_instance = (predicted_velocity.float() - target_velocity).square().mean(dim=(1, 2))
        loss = (weight * per_instance).mean()
        clean_residual = v_to_clean(noisy, predicted_velocity.float(), alpha, sigma)
        clean_z = z_anchor.float() + self.residual_mean + self.residual_scale * clean_residual
        return loss, {
            "predicted_clean_z": clean_z,
            "time": time,
            "snr": snr,
            "geometry_valid": snr >= 1.0,
            "weight": weight,
            "per_instance": per_instance.detach(),
        }

    def sample(self, z_anchor: Tensor, condition: Tensor, valid: Tensor | None = None,
               seed: int = 42, adapt: bool = False,
               differentiable_tail: bool = False) -> Tensor:
        self._require_stats()
        self._check_latent(z_anchor)
        if differentiable_tail and (not adapt or not self.adapter_training):
            raise RuntimeError("Differentiable sampling tail requires train_adapters and adapt=True")
        # Fixed reference; no partial GT or posterior target is accepted here.
        z_anchor = z_anchor.detach()
        self._condition(condition, valid, z_anchor.shape[0])  # Validate before starting the trajectory.
        generator_device = z_anchor.device if z_anchor.device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        residual = torch.randn(z_anchor.shape, generator=generator,
                               device=generator_device, dtype=torch.float32).to(z_anchor.device)
        times = self.sampling_times(z_anchor.device)
        tail_start = len(times) - self.config.adapter_tail_steps
        for index, step in enumerate(times):
            is_tail = index >= tail_start
            track = differentiable_tail and is_tail
            with nullcontext() if track else torch.no_grad():
                time = step.expand(z_anchor.shape[0])
                alpha, sigma = self._coefficients(time, residual.dtype)
                velocity = self.predict_velocity(residual, z_anchor, time, condition, valid,
                                                  adapt=adapt and is_tail).float()
                clean = v_to_clean(residual, velocity, alpha, sigma)
                noise = v_to_noise(residual, velocity, alpha, sigma)
                if index + 1 < len(times):
                    next_time = times[index + 1].expand(z_anchor.shape[0])
                    next_alpha, next_sigma = self._coefficients(next_time, residual.dtype)
                    residual = next_alpha * clean + next_sigma * noise
                else:
                    residual = clean  # The endpoint has alpha=1, sigma=0.
        with nullcontext() if differentiable_tail else torch.no_grad():
            return z_anchor.float() + self.residual_mean + self.residual_scale * residual
