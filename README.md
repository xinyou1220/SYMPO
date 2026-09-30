# SymPO

Reinforcement learning algorithm from **SymPO: Consistency by Construction via Permutation Symmetry
for Preference Optimization** (MRAC '26, ACM Multimedia 2026 workshop).

This release contains only the RL algorithm: balanced swap rollouts for pairwise preference
decisions and a critic-free REINFORCE++ update. It is model-agnostic and depends only on PyTorch.
You supply the model, the prompt format, the sampling code and, if needed, your own reward.

## Background: MER-Prefer (MER 2026)

The algorithm was developed for MER-Prefer, the preference-judging task of the MER 2026 challenge.
Each sample is a video clip (frames, audio, ASR transcript) and two candidate emotion analyses
$(c_0, c_1)$. The model must predict which one human annotators preferred. Labels are binary,
$g \in \{0, 1\}$, and $g = -1$ marks pairs without a label. The metric is two-class weighted F1:

$$
\text{WAF-2} = \sum_{c \in \{0,1\}} \frac{n_c}{n} \cdot \frac{2 P_c R_c}{P_c + R_c}
$$

Two properties of the task drive the algorithm design:

- The judge is biased toward one presentation slot (above 80% zero-shot), and RL fine-tuning can
  amplify this, because the label prior can be learned as a slot preference.
- Multimodal rollouts are expensive, so only $K = 4$ samples per prompt are affordable. At this
  size, per-group reward normalization is unstable.

## Algorithm

### Notation

The orientation $o \in \{0, 1\}$ sets the presentation order:

$$
(u, v) = \begin{cases} (c_0, c_1), & o = 0 \\ (c_1, c_0), & o = 1 \end{cases}
$$

The policy outputs a slot decision $d \in \{0, 1, \bot\}$ (A, B, unparsable). The chosen candidate is
$c = d \oplus o$.

### 1. Balanced swap rollouts

Every pair is placed in the same batch in both orientations, and each orientation is sampled $K$
times by the current policy. With $P$ pairs per step the batch has $B = 2PK$ responses. Each
(pair, orientation) is one group; groups never mix orientations.

The default reward is symmetric correctness:

$$
r = \begin{cases} +1, & g \ge 0,\ c = g \\ -1, & g \ge 0,\ c \ne g \text{ or } d = \bot \\ 0, & g < 0 \end{cases}
$$

Why this matters: a positional policy (one that picks the same slot regardless of content) wins
once and loses once for every pair, so its expected reward is exactly $0$. A policy with per-pair
accuracy $a$ earns $2a - 1$. The label prior therefore gives no expected gradient toward a slot
preference, and the only way to gain reward is to use the content.

Code: `expand_pairs`, `correctness_reward` and `build_preference_rollout` in `sympo/rollout.py`.
A custom reward can be passed with `reward_fn`.

### 2. Advantage: group-mean baseline with global normalization

Rewards are centered within each group of $K$ samples, copied to every response token ($M$ is the
response mask), and normalized once over the whole batch:

$$
\tilde R_i = R_i - \frac{1}{K} \sum_{j \in G(i)} R_j, \qquad
A_{i,t} = M_{i,t} \, \frac{\tilde R_i M_{i,t} - \mu_B}{\sigma_B + \varepsilon}
$$

$\mu_B$ and $\sigma_B$ are the mean and standard deviation over all response tokens in the batch.
There is no per-group standard deviation. If the $K$ decisions in a group are i.i.d.
Bernoulli($p$), all rewards in the group are equal with probability $p^K + (1-p)^K \ge 2^{1-K}$.
That is at least 1/8 for $K = 4$ and approaches 1 as the policy converges. When it happens,
per-group z-scoring divides by zero variance. A single global scale fails only when the whole batch
is degenerate. It also keeps reward differences between groups on the same scale.

Code: `advantage_reinforce_baseline` in `sympo/advantage.py`.

### 3. Policy loss: PPO clip with k2 KL

No value network and no entropy bonus. With
$\varrho_{i,t} = \exp(\log \pi_\theta - \log \pi_{\theta_{\text{old}}})$ and
$\delta_{i,t} = \log \pi_\theta - \log \pi_{\text{ref}}$:

$$
L = -\min\big(\varrho_{i,t} A_{i,t},\ \mathrm{clip}(\varrho_{i,t}, 1 \pm \epsilon) A_{i,t}\big)
+ \beta \cdot \tfrac{1}{2} \delta_{i,t}^2
$$

averaged over response tokens, with $\epsilon = 0.2$ and $\beta = 0.01$. As a loss, the k2 term has
the same expected gradient as $\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$. With LoRA, the
reference policy is the same model with the adapter disabled, so no second copy of the weights is
needed.

Code: `policy_loss` in `sympo/loss.py`, `ReinforcePP.update` in `sympo/trainer.py`.

### Summary of changes to standard RL for pairwise judges

| | Typical setup | This algorithm |
|---|---|---|
| Rollout order | one fixed order per pair | both orders of every pair in the same batch |
| Group | per prompt | per (pair, orientation) |
| Reward sign | 1 / 0 | +1 / -1, so positional policies earn exactly 0 |
| Normalization | per-group mean and std (GRPO) | per-group mean, global std (REINFORCE++) |
| Critic | value network (PPO) | none |
| KL | k1 or k3 | k2 as a loss |

## Effect on the challenge

Setting in the paper: Qwen2.5-Omni-7B thinker with LoRA, 4 pairs x 2 orientations x 4 samples per
step, 100 steps, learning rate $5 \times 10^{-5}$, three seeds. Test WAF-2 is from the official
scoring (379 pairs). The symmetry metrics are measured on a held-out internal half (539 pairs), in
each orientation separately, before any averaging over the two orders. In the paper, test scores
of all trained models use the same inference rule, which averages the prediction over both orders.

| | Test WAF-2 | Swap consistency ↑ | Position bias ↓ |
|---|---|---|---|
| Qwen2.5-Omni zero-shot | 78.74 | 49.91 | 25.05 |
| Naive RL (one order, no balanced swap) | 90.24 ± 1.47 | 49.35 ± 17.58 | 25.08 ± 9.06 |
| RL with balanced swap (this algorithm) | 89.89 ± 0.15 | 65.99 ± 14.75 | 16.39 ± 7.65 |

Swap consistency is the share of pairs that get the same answer in both orders. Position bias is
$|\Pr[d = 1] - \tfrac12|$ over both orders, in %.

- The two RL variants reach the same test score, but naive RL collapses to always choosing the same
  slot in all three seeds. The final score does not show this. Balanced swap raises swap
  consistency and lowers position bias at the same score.
- The test score also varied less across seeds with balanced swap (s.d. 0.15 vs 1.47).
- In the paper's single-run ablations, removing balanced swap lowered test WAF-2 by 1.84, and
  replacing global normalization with per-group normalization lowered it by 0.52.

## Effect on pairwise preference learning

- Position bias is a property of the training signal, not only of the model. If a pair is shown in
  one order, RL can reach the label prior through a slot preference. Presenting both orders removes
  that gradient in expectation, for any pairwise judge: reward models, LLM-as-a-judge, A/B
  evaluation.
- Accuracy alone can hide order dependence. Report swap consistency and position bias as well.
- With small groups ($K \le 4$), per-group std normalization often sees groups with identical
  rewards. A group-mean baseline with one global scale keeps the update well defined.
- Balanced swap reduces positional incentive but does not remove all order dependence. It sets the
  expected reward of a positional policy to zero, but the search can still reach one.

## Install

```bash
pip install -e .
```

## Files

```
sympo/rollout.py     balanced swap, group sampling, correctness reward
sympo/advantage.py   group-mean baseline and global normalization (plus the k1 REINFORCE++ variant)
sympo/loss.py        PPO clipped loss with KL loss
sympo/trainer.py     ReinforcePP.update(), RolloutBatch
sympo/kl.py          k1, k2, k3 KL estimators
sympo/config.py      ReinforcePPConfig
tests/test_sympo.py
```

## Usage

The policy object implements:

- `generate(samples)`: returns a `Generation` with the response mask `(B, T)`, sampling-time
  log-probs `(B, T)`, a decision per sample (0 = A, 1 = B, -1 = unparsable) and any tensors needed
  later in `extra`. Each sample has `prompt`, `response_a` and `response_b`.
- `logprobs(batch)`: current per-token log-probs `(B, T)` with gradients.
- `ref_logprobs(batch)` (optional): reference-model log-probs.

```python
import torch
from sympo import (PreferencePair, PreferenceRolloutConfig, ReinforcePP, ReinforcePPConfig,
                   build_preference_rollout)

pairs = [PreferencePair(prompt, response_0, response_1, label=g), ...]

params = [p for p in model.parameters() if p.requires_grad]
algo = ReinforcePP(params, torch.optim.Adam(params, lr=5e-5),
                   ReinforcePPConfig(estimator="reinforce_baseline", kl_loss_coef=0.01,
                                     micro_batch_size=8))
rollout_cfg = PreferenceRolloutConfig(group_k=4, balanced_swap=True)

for step in range(num_steps):
    batch, stats = build_preference_rollout(policy, sample(pairs, 4), rollout_cfg)
    algo.update(batch, policy.logprobs)
```

The update also works outside the pairwise setting. Build a `RolloutBatch` with `action_mask`,
`old_logp`, `rewards`, `group_id` and `ref_logp`, then call `algo.update(batch, logprob_fn)`.
With `micro_batch_size` set, gradients are accumulated over micro-batches while advantages are
still normalized over the full batch. `estimator="reinforce"` selects the other REINFORCE++
variant (k1 KL inside the reward, return-to-go, global normalization).

## Tests

```bash
python tests/test_sympo.py
```

## Citation

```bibtex
@inproceedings{chen2026sympo,
  title     = {SymPO: Consistency by Construction via Permutation Symmetry for Preference Optimization},
  author    = {Chen, Meng-Ching and Song, Xin-You and Chen, Sheng-Bi and Lee, Chia-Yen},
  booktitle = {Proceedings of the 4th International Workshop on Multimodal, Generative and Responsible Affective Computing (MRAC '26)},
  year      = {2026},
  doi       = {10.1145/3840474.3840539}
}
```

The update follows REINFORCE++ (Hu, Liu, Xu and Shen, arXiv:2501.03262).

## Contributors

- Meng-Ching Chen
- Xin-You Song
- Sheng-Bi Chen

## License

MIT
