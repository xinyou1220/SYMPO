import torch

from sympo import (Generation, PreferencePair, ReinforcePP, ReinforcePPConfig, RolloutBatch,
                   advantage_reinforce, advantage_reinforce_baseline, build_preference_rollout,
                   correctness_reward, expand_pairs, global_normalize)


def test_global_normalize_ignores_masked_tokens():
    x = torch.tensor([[1.0, 2.0, 99.0], [3.0, 4.0, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    out = global_normalize(x, mask)
    valid = out[mask.bool()]
    assert torch.allclose(valid.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(valid.std(unbiased=False), torch.tensor(1.0), atol=1e-4)
    assert torch.all(out[~mask.bool()] == 0)


def test_baseline_subtracts_group_mean():
    rewards = torch.tensor([1.0, 0.0, 5.0, 5.0])
    group = torch.tensor([0, 0, 1, 1])
    mask = torch.ones(4, 2)
    _, shaped = advantage_reinforce_baseline(rewards, group, mask, global_norm=False)
    assert torch.allclose(shaped, torch.tensor([0.5, -0.5, 0.0, 0.0]))


def test_reinforce_places_reward_on_last_token():
    rewards = torch.tensor([1.0, 2.0])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    zeros = torch.zeros(2, 3)
    _, returns = advantage_reinforce(rewards, zeros, zeros, mask, kl_coef=0.0, global_norm=False)
    assert torch.allclose(returns, torch.tensor([[1.0, 1.0, 0.0], [2.0, 2.0, 2.0]]))


def test_reinforce_kl_penalty_accumulates():
    rewards = torch.tensor([0.0])
    mask = torch.ones(1, 2)
    old = torch.tensor([[0.0, 0.0]])
    ref = torch.tensor([[-1.0, -1.0]])
    _, returns = advantage_reinforce(rewards, old, ref, mask, kl_coef=0.5, global_norm=False)
    assert torch.allclose(returns, torch.tensor([[-1.0, -0.5]]))


def test_balanced_swap_and_groups():
    pairs = [PreferencePair("q", "good", "bad", label=0), PreferencePair("q2", "x", "y", label=1)]
    s = expand_pairs(pairs, group_k=3)
    assert len(s) == 12
    assert [x.group_id for x in s] == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert (s[0].response_a, s[3].response_a) == ("good", "bad")
    assert [x.orientation for x in s[:6]] == [0, 0, 0, 1, 1, 1]


def test_correctness_reward():
    decision = torch.tensor([0, 1, 1, -1, 0])
    orientation = torch.tensor([0, 0, 1, 1, 0])
    label = torch.tensor([0, 0, 0, 0, -1])
    reward, chosen, correct, valid = correctness_reward(decision, orientation, label)
    assert chosen.tolist() == [0, 1, 0, 1, 0]
    assert valid.tolist() == [True, True, True, False, True]
    assert reward.tolist() == [1.0, -1.0, 1.0, -1.0, 0.0]


def test_positional_policy_gets_zero_expected_reward():
    pairs = [PreferencePair("q", "x", "y", label=g) for g in (0, 1)]
    s = expand_pairs(pairs, group_k=1)
    orientation = torch.tensor([x.orientation for x in s])
    label = torch.tensor([x.label for x in s])
    for slot in (0, 1):
        reward, *_ = correctness_reward(torch.full((len(s),), slot), orientation, label)
        assert reward.sum().item() == 0


def test_rollout_to_update_learns_content_not_position():
    torch.manual_seed(0)
    pairs = [PreferencePair(f"q{i}", f"good{i}", f"bad{i}", label=0) for i in range(8)]
    bias = torch.nn.Parameter(torch.tensor(2.0))
    skill = torch.nn.Parameter(torch.tensor(0.0))

    class StubPolicy:
        def _logp_b(self, extra):
            logit = bias - skill * extra["good_is_a"]
            return torch.nn.functional.logsigmoid(torch.stack([-logit, logit], 1))

        def logprobs(self, batch):
            return self._logp_b(batch.extra).gather(1, batch.extra["d"].unsqueeze(1))

        def generate(self, samples):
            good_is_a = torch.tensor([1.0 if x.response_a.startswith("good") else -1.0 for x in samples])
            extra = {"good_is_a": good_is_a}
            with torch.no_grad():
                d = torch.distributions.Categorical(logits=self._logp_b(extra)).sample()
            extra["d"] = d
            with torch.no_grad():
                old = self._logp_b(extra).gather(1, d.unsqueeze(1))
            return Generation(torch.ones(len(samples), 1), old, d, extra)

    policy = StubPolicy()
    algo = ReinforcePP([bias, skill], torch.optim.Adam([bias, skill], lr=0.05), ReinforcePPConfig())
    for _ in range(150):
        batch, stats = build_preference_rollout(policy, pairs)
        algo.update(batch, policy.logprobs)
    assert stats["accuracy"] > 0.9
    assert abs(bias.item()) < abs(skill.item())


def _grads(micro_batch_size):
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(6, 4))
    ids = torch.randint(0, 4, (6, 3))
    mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=torch.float)

    def logprob_fn(b):
        logp = torch.log_softmax(w[b.extra["row"]], dim=-1)
        return logp.gather(-1, b.extra["ids"])

    rows = torch.arange(6)
    with torch.no_grad():
        old = logprob_fn(RolloutBatch(mask, mask, mask[:, 0], extra={"row": rows, "ids": ids}))
    batch = RolloutBatch(mask, old, torch.tensor([1.0, 0.0, 1.0, 0.0, 0.5, 0.2]), ref_logp=old - 0.1,
                         group_id=torch.tensor([0, 0, 1, 1, 2, 2]), extra={"row": rows, "ids": ids})
    opt = torch.optim.SGD([w], lr=0.0)
    ReinforcePP([w], opt, ReinforcePPConfig(micro_batch_size=micro_batch_size, grad_clip=1e9)).update(batch, logprob_fn)
    return w.grad.clone()


def test_micro_batching_matches_full_batch():
    assert torch.allclose(_grads(0), _grads(2), atol=1e-6)


def test_update_improves_reward_on_bandit():
    torch.manual_seed(0)
    logits = torch.nn.Parameter(torch.zeros(4))
    algo = ReinforcePP([logits], torch.optim.Adam([logits], lr=0.1), ReinforcePPConfig(kl_loss_coef=0.0))
    for _ in range(100):
        actions = torch.distributions.Categorical(logits=logits.detach()).sample((32,))
        with torch.no_grad():
            old = torch.log_softmax(logits, -1)[actions].unsqueeze(1)
        batch = RolloutBatch(torch.ones(32, 1), old, (actions == 2).float(),
                             group_id=torch.zeros(32, dtype=torch.long), extra={"a": actions})
        algo.update(batch, lambda b: torch.log_softmax(logits, -1)[b.extra["a"]].unsqueeze(1))
    assert torch.softmax(logits, -1)[2] > 0.9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
