"""Model-agnostic REINFORCE++ update step."""
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Optional

import torch

from .advantage import advantage_reinforce, advantage_reinforce_baseline
from .config import ReinforcePPConfig
from .loss import policy_loss


@dataclass
class RolloutBatch:
    """action_mask, old_logp, ref_logp: (B,T). rewards, group_id: (B,).

    ``extra`` holds whatever ``logprob_fn`` needs (input ids, attention mask, ...).
    """
    action_mask: torch.Tensor
    old_logp: torch.Tensor
    rewards: torch.Tensor
    ref_logp: Optional[torch.Tensor] = None
    group_id: Optional[torch.Tensor] = None
    extra: Optional[dict] = None

    def __len__(self):
        return self.rewards.size(0)

    def select(self, idx: torch.Tensor) -> "RolloutBatch":
        """Select rows. Tensors and lists in ``extra`` with batch size B are sliced as well."""
        n = len(self)

        def take(v):
            if torch.is_tensor(v) and v.dim() > 0 and v.size(0) == n:
                return v[idx.to(v.device)]
            if isinstance(v, (list, tuple)) and len(v) == n:
                return [v[i] for i in idx.tolist()]
            return v

        return replace(
            self,
            action_mask=take(self.action_mask), old_logp=take(self.old_logp), rewards=take(self.rewards),
            ref_logp=take(self.ref_logp), group_id=take(self.group_id),
            extra={k: take(v) for k, v in (self.extra or {}).items()},
        )


class ReinforcePP:
    def __init__(self, params: Iterable[torch.nn.Parameter], optimizer: torch.optim.Optimizer,
                 cfg: Optional[ReinforcePPConfig] = None):
        self.params = [p for p in params if p.requires_grad]
        self.opt = optimizer
        self.cfg = cfg or ReinforcePPConfig()

    def compute_advantages(self, batch: RolloutBatch) -> torch.Tensor:
        c = self.cfg
        if c.estimator == "reinforce":
            if batch.ref_logp is None:
                raise ValueError("estimator='reinforce' requires batch.ref_logp")
            adv, _ = advantage_reinforce(
                batch.rewards, batch.old_logp, batch.ref_logp, batch.action_mask,
                kl_coef=c.kl_coef, gamma=c.gamma, eps=c.norm_eps, global_norm=c.global_norm,
            )
        elif c.estimator == "reinforce_baseline":
            if batch.group_id is None:
                raise ValueError("estimator='reinforce_baseline' requires batch.group_id")
            adv, _ = advantage_reinforce_baseline(
                batch.rewards, batch.group_id, batch.action_mask, eps=c.norm_eps, global_norm=c.global_norm,
            )
        else:
            raise ValueError(f"unknown estimator: {c.estimator}")
        return adv

    def update(self, batch: RolloutBatch, logprob_fn: Callable[[RolloutBatch], torch.Tensor]) -> dict:
        """Run one REINFORCE++ update.

        ``logprob_fn(batch)`` returns the current policy's (B,T) log-probs with gradients.
        Advantages are normalized over the full batch before micro-batching.
        """
        c = self.cfg
        batch = replace(batch, extra=dict(batch.extra or {}))
        batch.extra["_advantages"] = self.compute_advantages(batch)
        use_kl_loss = c.estimator == "reinforce_baseline" and batch.ref_logp is not None and c.kl_loss_coef > 0

        n = len(batch)
        mb = c.micro_batch_size if c.micro_batch_size > 0 else n
        total_tokens = batch.action_mask.sum().clamp_min(1.0)
        stats = {}
        for _ in range(c.ppo_epochs):
            self.opt.zero_grad(set_to_none=True)
            agg = {}
            for start in range(0, n, mb):
                sub = batch.select(torch.arange(start, min(start + mb, n)))
                weight = (sub.action_mask.sum() / total_tokens).item()
                loss, s = policy_loss(
                    logprob_fn(sub), sub.old_logp.detach(), sub.extra["_advantages"], sub.action_mask,
                    clip_eps=c.clip_eps,
                    ref_logp=sub.ref_logp.detach() if use_kl_loss else None,
                    kl_loss_coef=c.kl_loss_coef if use_kl_loss else 0.0,
                    kl_estimator=c.kl_loss_estimator,
                )
                (loss * weight).backward()
                s["loss"] = loss.item()
                for k, v in s.items():
                    agg[k] = agg.get(k, 0.0) + v * weight
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params, c.grad_clip)
            self.opt.step()
            stats = {**agg, "grad_norm": float(grad_norm)}
        stats["reward_mean"] = batch.rewards.float().mean().item()
        return stats
