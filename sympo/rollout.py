"""Pairwise preference rollouts with balanced swap and group sampling.

Every pair is rolled out in both presentation orders and each (pair, order) is sampled K times.
Each (pair, order) forms one group for the group-mean baseline.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch

from .trainer import RolloutBatch


@dataclass
class PreferencePair:
    """``label`` is the index of the preferred response (0 or 1), or -1 if unknown."""
    prompt: str
    response_0: str
    response_1: str
    label: int = -1


@dataclass
class PreferenceSample:
    """One presentation of a pair. ``orientation`` 1 means the two responses are swapped."""
    prompt: str
    response_a: str
    response_b: str
    pair_index: int
    orientation: int
    group_id: int
    label: int


@dataclass
class PreferenceRolloutConfig:
    group_k: int = 4
    balanced_swap: bool = True


@dataclass
class Generation:
    """What ``policy.generate`` returns. ``decision`` is 0 for A, 1 for B, -1 if unparsable."""
    action_mask: torch.Tensor
    old_logp: torch.Tensor
    decision: torch.Tensor
    extra: Optional[dict] = None


def expand_pairs(pairs: Sequence[PreferencePair], group_k: int = 1,
                 balanced_swap: bool = True) -> List[PreferenceSample]:
    orientations = (0, 1) if balanced_swap else (0,)
    samples, group = [], 0
    for i, p in enumerate(pairs):
        for o in orientations:
            a, b = (p.response_0, p.response_1) if o == 0 else (p.response_1, p.response_0)
            samples.extend(PreferenceSample(p.prompt, a, b, i, o, group, p.label) for _ in range(group_k))
            group += 1
    return samples


def correctness_reward(decision: torch.Tensor, orientation: torch.Tensor, label: torch.Tensor):
    """+1 if the preferred response is chosen, -1 otherwise (including unparsable), 0 if unlabeled.

    Returns (reward, chosen, correct, valid). ``chosen`` = decision XOR orientation.
    """
    valid = decision >= 0
    chosen = decision.clamp_min(0) ^ orientation
    labeled = label >= 0
    correct = (chosen == label) & labeled & valid
    reward = torch.where(correct, 1.0, -1.0) * labeled.float()
    return reward, chosen, correct, valid


def build_preference_rollout(policy, pairs: Sequence[PreferencePair],
                             cfg: Optional[PreferenceRolloutConfig] = None,
                             reward_fn: Optional[Callable] = None):
    """Sample decisions for ``pairs`` and return ``(RolloutBatch, stats)``.

    ``policy.generate(samples)`` must return a :class:`Generation`. If ``policy`` has
    ``ref_logprobs(batch)``, reference log-probs are attached. ``reward_fn(samples, decision)``
    replaces the default correctness reward and must return a (B,) tensor.
    """
    cfg = cfg or PreferenceRolloutConfig()
    samples = expand_pairs(pairs, cfg.group_k, cfg.balanced_swap)
    gen = policy.generate(samples)
    device = gen.decision.device

    def col(name):
        return torch.tensor([getattr(s, name) for s in samples], device=device)

    orientation, label = col("orientation"), col("label")
    reward, _, correct, valid = correctness_reward(gen.decision, orientation, label)
    if reward_fn is not None:
        reward = reward_fn(samples, gen.decision).to(device)

    batch = RolloutBatch(
        action_mask=gen.action_mask, old_logp=gen.old_logp, rewards=reward,
        group_id=col("group_id"), extra=dict(gen.extra or {}),
    )
    if hasattr(policy, "ref_logprobs"):
        batch.ref_logp = policy.ref_logprobs(batch)

    labeled = label >= 0
    stats = {
        "accuracy": correct[labeled].float().mean().item() if labeled.any() else 0.0,
        "valid_rate": valid.float().mean().item(),
        "slot_a_rate": (gen.decision[valid] == 0).float().mean().item() if valid.any() else 0.0,
    }
    return batch, stats
