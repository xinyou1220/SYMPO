"""KL estimators (REINFORCE++ Appendix B.1), with delta = log pi_theta - log pi_ref.

k1 is used inside the reward; k2 is the recommended loss; k3 is the GRPO estimator.
"""
import torch


def compute_delta(logp_theta: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    return logp_theta - logp_ref


def k1(delta: torch.Tensor) -> torch.Tensor:
    return delta


def k2(delta: torch.Tensor) -> torch.Tensor:
    return 0.5 * delta.pow(2)


def k3(delta: torch.Tensor) -> torch.Tensor:
    return torch.exp(-delta) - 1.0 + delta


ESTIMATORS = {"k1": k1, "k2": k2, "k3": k3}


def kl_penalty(delta: torch.Tensor, name: str) -> torch.Tensor:
    if name not in ESTIMATORS:
        raise ValueError(f"unknown KL estimator: {name}")
    return ESTIMATORS[name](delta)
