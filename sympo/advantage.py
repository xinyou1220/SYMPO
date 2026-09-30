"""REINFORCE++ advantage estimators (arXiv:2501.03262).

reinforce          : k1 KL folded into the per-token reward (Eq.4) -> return-to-go -> global norm (Eq.5)
reinforce_baseline : reward minus group mean (Eq.6) -> global norm (Eq.7); KL is a separate loss (Eq.8)

Global normalization z-scores over every response token in the batch, not per group.
Shapes: rewards (B,), log-probs and action_mask (B,T).
"""
import torch

from .kl import compute_delta, k1


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def global_normalize(adv: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = masked_mean(adv, mask)
    std = masked_mean((adv - mean).pow(2), mask).clamp_min(0.0).sqrt()
    return ((adv - mean) / (std + eps)) * mask


def _last_token_index(action_mask: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(action_mask.size(1), device=action_mask.device).unsqueeze(0)
    return torch.where(action_mask > 0, idx, torch.full_like(idx, -1)).max(dim=1).values.clamp_min(0)


def _reward_to_go(per_token_reward: torch.Tensor, mask: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(per_token_reward)
    running = torch.zeros_like(per_token_reward[:, 0])
    for t in reversed(range(per_token_reward.size(1))):
        running = per_token_reward[:, t] + gamma * running * mask[:, t]
        returns[:, t] = running
    return returns * mask


def advantage_reinforce(rewards, old_logp, ref_logp, action_mask,
                        kl_coef: float = 0.001, gamma: float = 1.0, eps: float = 1e-6,
                        global_norm: bool = True):
    """Returns (advantages, returns). The sequence reward is placed on the last response token."""
    action_mask = action_mask.to(old_logp.dtype)
    per_token = -kl_coef * k1(compute_delta(old_logp, ref_logp)) * action_mask
    rows = torch.arange(rewards.size(0), device=rewards.device)
    per_token[rows, _last_token_index(action_mask)] += rewards.to(per_token.dtype)
    returns = _reward_to_go(per_token, action_mask, gamma)
    adv = global_normalize(returns, action_mask, eps) if global_norm else returns
    return adv.detach(), returns.detach()


def advantage_reinforce_baseline(rewards, group_id, action_mask,
                                 eps: float = 1e-6, global_norm: bool = True):
    """Returns (advantages, baseline-subtracted rewards)."""
    rewards = rewards.float()
    action_mask = action_mask.to(rewards.dtype)
    shaped = rewards.clone()
    for gid in torch.unique(group_id):
        m = group_id == gid
        shaped[m] = rewards[m] - rewards[m].mean()
    adv = shaped.unsqueeze(1) * action_mask
    if global_norm:
        adv = global_normalize(adv, action_mask, eps)
    return adv.detach(), shaped.detach()
