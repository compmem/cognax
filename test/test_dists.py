import jax.random as random

import numpy as np
import numpyro
import pytest
from collections import namedtuple
from cognax.decisions import TRDM, WFPT, WFPTNormalDrift

numpyro.enable_x64(True)


TestDist = namedtuple("TestDist", ["dist", "valid_params"])

DISTS = [
    TestDist(
        TRDM,
        [np.full((3,), 0.5), np.full((3,), 1.0), np.full((3,), 1.0), np.array(0.14)],
    ),
    TestDist(WFPT, [0.5, 1.0, 0.5, 0.25]),
    TestDist(WFPTNormalDrift, [0.5, 1.0, 1.0, 0.5, 0.25]),
]


def get_choice_RTs(n_choice, dt, max_RT):
    """choice_RTs with RTs from 0 -> max_RTs for each choice"""
    t_range = np.arange(0, max_RT + dt, dt)
    choices = np.repeat(np.arange(n_choice), repeats=len(t_range))
    RTs = np.concatenate([t_range] * n_choice)
    choice_RTs = np.vstack([choices, RTs]).T

    return choice_RTs


@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("batch_shape", [(), (2,), (1, 2), (4, 2)])
def test_dist_broadcast_value(dist, batch_shape):
    choice_RTs = np.broadcast_to(np.array([0, 0.5]), (*batch_shape, 2))

    assert dist.dist(*dist.valid_params).log_prob(value=choice_RTs).shape == batch_shape


def test_wfpt_integrate_to_one():
    dt = 0.0001
    choice_RTs = get_choice_RTs(n_choice=2, dt=dt, max_RT=10)

    wfpt = WFPT(v=0.2, a=1.0, w=0.5, t0=0.25)

    probs = np.exp(wfpt.log_prob(value=choice_RTs))
    integral = np.sum(probs * dt)

    assert np.isclose(integral, 1, atol=0.01)


@pytest.mark.parametrize(
    "timer_args",
    [
        {"v_timer": None, "alpha_timer": None, "sigma_timer": None},
        {"v_timer": 0.2, "alpha_timer": 0.4, "sigma_timer": 0.3},
    ],
)
def test_trdm_integrate_to_one(timer_args):
    dt = 0.0001
    choice_RTs = get_choice_RTs(n_choice=3, dt=dt, max_RT=10)

    trdm = TRDM(
        v=np.full((3,), 0.5),
        alpha=np.full((3,), 1.0),
        sigma=np.full((3,), 1.0),
        t0=np.array(0.14),
        **timer_args,
    )

    probs = np.exp(trdm.log_prob(value=choice_RTs))
    integral = np.sum(probs * dt)

    assert np.isclose(integral, 1, atol=0.01)
