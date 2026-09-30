from dataclasses import dataclass


@dataclass
class ReinforcePPConfig:
    # "reinforce" or "reinforce_baseline"
    estimator: str = "reinforce_baseline"
    gamma: float = 1.0
    global_norm: bool = True
    norm_eps: float = 1e-6

    kl_coef: float = 0.001
    kl_loss_coef: float = 0.01
    kl_loss_estimator: str = "k2"

    clip_eps: float = 0.2
    ppo_epochs: int = 1
    micro_batch_size: int = 0  # 0: no micro-batching
    grad_clip: float = 1.0
