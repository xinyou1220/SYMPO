import torch

from .advantage import masked_mean
from .kl import compute_delta, kl_penalty


def policy_loss(new_logp, old_logp, advantages, action_mask, clip_eps: float = 0.2,
                ref_logp=None, kl_loss_coef: float = 0.0, kl_estimator: str = "k2"):
    """PPO-clip surrogate (no critic) plus an optional KL loss. Returns (loss, stats)."""
    action_mask = action_mask.to(new_logp.dtype)
    ratio = torch.exp((new_logp - old_logp) * action_mask)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    pg_loss = masked_mean(-torch.min(ratio * advantages, clipped * advantages), action_mask)

    kl_loss = torch.zeros((), device=new_logp.device, dtype=new_logp.dtype)
    if ref_logp is not None and kl_loss_coef > 0:
        kl_loss = masked_mean(kl_penalty(compute_delta(new_logp, ref_logp), kl_estimator), action_mask)

    with torch.no_grad():
        clip_frac = ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).to(ratio.dtype)
        stats = {
            "pg_loss": pg_loss.item(),
            "kl_loss": kl_loss.item(),
            "ratio_mean": masked_mean(ratio, action_mask).item(),
            "clip_frac": masked_mean(clip_frac, action_mask).item(),
        }
    return pg_loss + kl_loss_coef * kl_loss, stats
