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


@pytest.mark.parametrize("v, alpha, sigma", [(0.5, 1.0, 1.0), (3.0, 1.0, 0.5)])
def test_trdm_inverse_gaussian_matches_scipy(v, alpha, sigma):
    """first passage log density / log survival agree with scipy, including far tails"""
    from scipy.stats import invgauss

    from cognax.decisions.trdm import cum_log_p_not_choice, log_p_choice

    mu, lam = alpha / v, alpha**2 / sigma**2
    ref = invgauss(mu / lam, scale=lam)
    x = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 50.0, 200.0, 1000.0])

    np.testing.assert_allclose(
        log_p_choice(x, v, sigma, alpha), ref.logpdf(x), rtol=1e-6, atol=1e-10
    )
    np.testing.assert_allclose(
        cum_log_p_not_choice(x, v, sigma, alpha), ref.logsf(x), rtol=1e-6, atol=1e-10
    )


@pytest.mark.parametrize(
    "timer_args",
    [
        {"v_timer": None, "alpha_timer": None, "sigma_timer": None},
        {"v_timer": 0.2, "alpha_timer": 0.4, "sigma_timer": 0.3},
    ],
)
def test_trdm_nonresponse_completes_probability(timer_args):
    """P(response before deadline) + P(nonresponse at deadline) = 1"""
    dt, t0, deadline = 0.0001, 0.14, 1.0

    trdm = TRDM(
        v=np.full((3,), 0.5),
        alpha=np.full((3,), 1.0),
        sigma=np.full((3,), 1.0),
        t0=np.array(t0),
        **timer_args,
    )

    t_range = np.arange(t0 + dt, deadline, dt)
    choice_RTs = np.vstack(
        [np.repeat(np.arange(3), len(t_range)), np.tile(t_range, 3)]
    ).T
    p_response = np.sum(np.exp(trdm.log_prob(value=choice_RTs)) * dt)
    p_nonresponse = np.exp(trdm.log_prob(value=np.array([-1, deadline])))

    assert np.isclose(p_response + p_nonresponse, 1, atol=0.01)


TIMER_ARGS = [
    {"v_timer": None, "alpha_timer": None, "sigma_timer": None},
    {"v_timer": 0.2, "alpha_timer": 0.4, "sigma_timer": 0.3},
]


V = np.full((3,), 0.5)


def make_trdm(timer_args, v=V, **kwargs):
    return TRDM(
        v=v,
        alpha=np.ones_like(v),
        sigma=np.ones_like(v),
        t0=np.array(0.14),
        **timer_args,
        **kwargs,
    )


@pytest.mark.parametrize("timer_args", TIMER_ARGS)
def test_trdm_samples_within_deadline(timer_args):
    deadline = 1.0
    samples = make_trdm(timer_args, deadline=deadline).sample(
        random.PRNGKey(0), (5000,)
    )
    choices, RTs = samples[:, 0], samples[:, 1]
    responded = choices != -1

    assert np.any(~responded)
    assert np.all(np.isin(choices, [-1, 0, 1, 2]))
    assert np.all((RTs[responded] > 0.14) & (RTs[responded] <= deadline))
    assert np.all(RTs[~responded] == deadline)


@pytest.mark.parametrize("timer_args", TIMER_ARGS)
def test_trdm_probs_and_samples_match_log_prob(timer_args):
    """probs = (p(choice 0), ..., p(choice n-1), p(nonresponse)), from the integrated
    log_prob, and sampled choice rates match them"""
    dt, deadline, n = 0.001, 1.0, 20000
    trdm = make_trdm(timer_args, deadline=deadline, dt=dt)

    fine_dt = 0.0001
    t_range = np.arange(0.14 + fine_dt, deadline, fine_dt)
    expected = [
        np.sum(
            np.exp(trdm.log_prob(np.stack([np.full_like(t_range, c), t_range], -1)))
            * fine_dt
        )
        for c in range(3)
    ] + [np.exp(trdm.log_prob(np.array([-1, deadline])))]
    np.testing.assert_allclose(trdm.probs, expected, atol=0.005)

    samples = trdm.sample(random.PRNGKey(1), (n,))
    observed = [np.mean(samples[:, 0] == c) for c in [0, 1, 2, -1]]
    np.testing.assert_allclose(
        observed, trdm.probs, atol=4 * np.sqrt(np.max(trdm.probs) / n)
    )


def test_wfpt_samples_within_deadline():
    deadline = 1.0
    samples = WFPT(v=0.0, a=3.0, w=0.5, t0=0.25, deadline=deadline).sample(
        random.PRNGKey(0), (5000,)
    )

    assert np.all(np.isin(samples[:, 0], [0, 1]))
    assert np.all((samples[:, 1] > 0.25) & (samples[:, 1] <= deadline))


def test_no_tail_mass_on_last_grid_point():
    """mass beyond the grid isn't dumped on the last grid point as fake responses"""
    n = 20000
    wfpt = WFPT(v=0.0, a=3.0, w=0.5, t0=0.25)
    RTs = wfpt.sample(random.PRNGKey(0), (n,))[:, 1]

    # last bin's density mass, renormalized over responses before the 5.0 deadline
    fine_dt = 0.0001
    choice_RTs = get_choice_RTs(n_choice=2, dt=fine_dt, max_RT=5.0)
    p_before_deadline = np.sum(np.exp(wfpt.log_prob(choice_RTs)) * fine_dt)
    p_last = (
        np.sum(np.exp(wfpt.log_prob(np.array([[0, 5.0], [1, 5.0]])))) * wfpt.dt
    ) / p_before_deadline

    observed = np.mean(RTs == RTs.max())
    assert np.isclose(observed, p_last, atol=4 * np.sqrt(p_last / n))


def test_trdm_probs_batched():
    v = np.array([[0.5, 0.5, 0.5], [1.0, 0.5, 0.2]])
    probs = make_trdm(TIMER_ARGS[0], v=v, deadline=1.0).probs

    assert probs.shape == (2, 4)
    for i in range(2):
        np.testing.assert_allclose(
            probs[i], make_trdm(TIMER_ARGS[0], v=v[i], deadline=1.0).probs
        )
