"""Frozen base-policy loader (2.1). The recovery study treats the ACT baseline as a FROZEN black box
and reads competence signals off it. Local path uses a schema-only stub; the real ACT lazily imports
torch on AICR. Fence: imports `recovery` only."""

import pytest

from recovery.policy.base_policy import (
    BasePolicy,
    FrozenACTPolicy,
    StubBasePolicy,
)


def test_stub_base_policy_is_frozen_and_deterministic():
    pol = StubBasePolicy(action_dim=7, latent_dim=8)
    obs = {"proprioception": [0.1] * 7}
    a1 = pol.predict(obs)
    a2 = pol.predict(obs)
    assert len(a1) == 7
    assert a1 == a2                      # deterministic (frozen, inference only)
    assert pol.frozen is True


def test_stub_exposes_latent_and_action_ensemble_for_competence():
    pol = StubBasePolicy(action_dim=7, latent_dim=8, ensemble_size=5)
    obs = {"proprioception": [0.2] * 7}
    z = pol.latent(obs)
    ens = pol.action_ensemble(obs)
    assert len(z) == 8
    assert len(ens) == 5                 # a small ensemble of action heads
    assert all(len(a) == 7 for a in ens)


def test_stub_is_a_base_policy_structurally():
    assert isinstance(StubBasePolicy(), BasePolicy)


def test_frozen_act_policy_is_lazy_and_not_callable_locally():
    # The real ACT loader must not import torch at construction (keeps the module import torch-free),
    # and its inference path is AICR-only, so calling it locally raises a clear NotImplementedError.
    pol = FrozenACTPolicy(checkpoint="/scratch/does/not/exist")
    assert pol.frozen is True
    with pytest.raises(NotImplementedError):
        pol.predict({"proprioception": [0.0] * 7})
