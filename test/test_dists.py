import jax
import jax.numpy as jnp
import jax.random as random

import numpy as np
import numpyro
import pytest
import scipy.stats
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


# (dist with nonresponse and deadline = 1.0, t0, choices)
NONRESPONSE_DISTS = [
    *[
        (make_trdm(timer_args, dt=0.001, deadline=1.0), 0.14, [0, 1, 2])
        for timer_args in TIMER_ARGS
    ],
    (WFPT(v=0.5, a=1.5, w=0.5, t0=0.25, dt=0.001, deadline=1.0), 0.25, [0, 1]),
]


@pytest.mark.parametrize("dist, t0, choices", NONRESPONSE_DISTS)
def test_samples_within_deadline(dist, t0, choices):
    deadline = 1.0
    samples = dist.sample(random.PRNGKey(0), (5000,))
    sampled_choices, RTs = samples[:, 0], samples[:, 1]
    responded = sampled_choices != -1

    assert np.any(~responded)
    assert np.all(np.isin(sampled_choices, [-1, *choices]))
    assert np.all((RTs[responded] > t0) & (RTs[responded] <= deadline))
    assert np.all(RTs[~responded] == deadline)


@pytest.mark.parametrize("dist, t0, choices", NONRESPONSE_DISTS)
def test_probs_and_samples_match_log_prob(dist, t0, choices):
    """probs = (p(choice 0), ..., p(choice n-1), p(nonresponse)), from the integrated
    log_prob, and sampled choice rates match them"""
    deadline, n = 1.0, 20000

    fine_dt = 0.0001
    t_range = np.arange(t0 + fine_dt, deadline, fine_dt)
    expected = [
        np.sum(
            np.exp(dist.log_prob(np.stack([np.full_like(t_range, c), t_range], -1)))
            * fine_dt
        )
        for c in choices
    ] + [np.exp(dist.log_prob(np.array([-1, deadline])))]
    np.testing.assert_allclose(dist.probs, expected, atol=0.005)

    samples = dist.sample(random.PRNGKey(1), (n,))
    observed = [np.mean(samples[:, 0] == c) for c in [*choices, -1]]
    np.testing.assert_allclose(
        observed, dist.probs, atol=4 * np.sqrt(np.max(dist.probs) / n)
    )


@pytest.mark.parametrize(
    "dist, n_choice, deadline",
    [
        *[(make_trdm(timer_args), 3, 1.0) for timer_args in TIMER_ARGS],
        (WFPT(v=0.5, a=1.0, w=0.5, t0=0.25), 2, 1.5),
        # asymmetric: catches sign errors per boundary
        (WFPT(v=1.0, a=1.5, w=0.3, t0=0.2), 2, 2.0),
        # (deadline - t0) / a**2 small: slow series convergence
        (WFPT(v=0.3, a=2.0, w=0.6, t0=0.25), 2, 0.35),
    ],
    ids=["trdm", "trdm_timer", "wfpt", "wfpt_asymmetric", "wfpt_short_deadline"],
)
def test_nonresponse_completes_probability(dist, n_choice, deadline):
    """P(response before deadline) + P(nonresponse at deadline) = 1"""
    dt = 0.00001

    t_range = np.arange(dist.t0 + dt, deadline, dt)
    choice_RTs = np.vstack(
        [np.repeat(np.arange(n_choice), len(t_range)), np.tile(t_range, n_choice)]
    ).T
    p_response = np.sum(np.exp(dist.log_prob(value=choice_RTs)) * dt)
    p_nonresponse = np.exp(dist.log_prob(value=np.array([-1, deadline])))

    assert np.isclose(p_response + p_nonresponse, 1, atol=5e-4)


def test_wfpt_log_survival_finite_in_tail():
    """fast drift, long deadline: log P(nonresponse) is tiny but finite, and matches
    the leading (k = 1) term of the large-time series"""
    v, a, w, deadline = 3.0, 1.0, 0.5, 10.0
    lam = v**2 / 2 + np.pi**2 / (2 * a**2)
    leading = (
        np.log(np.pi / a**2 * np.sin(np.pi * w) / lam)
        - lam * deadline
        + np.logaddexp(-v * a * w, v * a * (1 - w))
    )

    log_p = WFPT(v=v, a=a, w=w, t0=0.0).log_prob(np.array([-1, deadline]))

    assert np.isclose(log_p, leading, rtol=1e-6)


def test_no_tail_mass_on_last_grid_point():
    """mass beyond the grid isn't dumped on the last grid point as fake responses
    (WFPTNormalDrift has no nonresponse, so it renormalizes before the deadline)"""
    n = 20000
    dist = WFPTNormalDrift(v_loc=0.0, v_scale=0.5, a=3.0, w=0.5, t0=0.25)
    RTs = dist.sample(random.PRNGKey(0), (n,))[:, 1]

    # last bin's density mass, renormalized over responses before the 5.0 deadline
    fine_dt = 0.0001
    choice_RTs = get_choice_RTs(n_choice=2, dt=fine_dt, max_RT=5.0)
    p_before_deadline = np.sum(np.exp(dist.log_prob(choice_RTs)) * fine_dt)
    p_last = (
        np.sum(np.exp(dist.log_prob(np.array([[0, 5.0], [1, 5.0]])))) * dist.dt
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


def test_trdm_drift_is_real():
    """accumulators may drift away from their threshold (v <= 0), so v is unconstrained"""
    assert np.all(TRDM.arg_constraints["v"](np.array([-1.0, 0.0, 0.5])))


def make_single_accumulator_trdm(v, alpha, sigma):
    return TRDM(
        v=np.array([v]),
        alpha=np.array([alpha]),
        sigma=np.array([sigma]),
        t0=np.array(0.0),
        deadline=50.0,
        validate_args=False,  # test_trdm_drift_is_real covers the constraint
    )


@pytest.mark.parametrize(
    "v, reference",
    [
        # positive drift: first-passage time ~ InverseGaussian(alpha / v, alpha**2 / sigma**2)
        (
            0.7,
            lambda alpha, sigma: scipy.stats.invgauss(
                mu=(alpha / 0.7) / (alpha / sigma) ** 2, scale=(alpha / sigma) ** 2
            ),
        ),
        # zero drift: first-passage time ~ Levy(0, alpha**2 / sigma**2)
        (0.0, lambda alpha, sigma: scipy.stats.levy(scale=(alpha / sigma) ** 2)),
    ],
    ids=["positive_drift", "zero_drift"],
)
def test_trdm_single_accumulator_matches_scipy(v, reference):
    """with one accumulator and no timer, log_prob of a response is the first-passage
    log density, and log_prob of a nonresponse is the log survival"""
    alpha, sigma = 1.2, 0.8
    t = np.array([0.2, 1.0, 3.0])
    trdm = make_single_accumulator_trdm(v, alpha, sigma)

    log_dens = trdm.log_prob(np.stack([np.zeros_like(t), t], -1))
    log_surv = trdm.log_prob(np.stack([np.full_like(t, -1), t], -1))

    np.testing.assert_allclose(log_dens, reference(alpha, sigma).logpdf(t))
    np.testing.assert_allclose(log_surv, reference(alpha, sigma).logsf(t))


def test_trdm_negative_drift_may_never_finish():
    """with v < 0 the accumulator never reaches alpha with probability
    1 - exp(2 v alpha / sigma**2), so that much mass is left as nonresponse"""
    v, alpha, sigma = -0.5, 1.2, 0.8
    trdm = make_single_accumulator_trdm(v, alpha, sigma)

    p_nonresponse = np.exp(trdm.log_prob(np.array([-1, trdm.deadline])))

    np.testing.assert_allclose(
        p_nonresponse, 1 - np.exp(2 * v * alpha / sigma**2), rtol=1e-6
    )


def trdm_from_flat(p, timer):
    timer_args = {"v_timer": p[3], "alpha_timer": 0.4, "sigma_timer": 0.3}
    return TRDM(
        v=jnp.stack([p[0], 0.5, 0.5]),
        alpha=jnp.full((3,), p[1]),
        sigma=jnp.ones(3),
        t0=p[2],
        deadline=1.0,
        **(timer_args if timer else TIMER_ARGS[0]),
    )


@pytest.mark.parametrize(
    "make_dist, params, choices",
    [
        (lambda p: WFPT(*p, deadline=1.0), [0.5, 1.5, 0.4, 0.25], [0, 1, -1]),
        # w = 0.5: even-k survival terms have zero weight in logsumexp
        (lambda p: WFPT(*p, deadline=1.0), [0.5, 1.5, 0.5, 0.25], [0, 1, -1]),
        (lambda p: trdm_from_flat(p, timer=False), [0.8, 1.0, 0.14], [0, 2, -1]),
        (lambda p: trdm_from_flat(p, timer=True), [0.8, 1.0, 0.14, 0.2], [0, 2, -1]),
    ],
    ids=["wfpt", "wfpt_w_half", "trdm", "trdm_timer"],
)
def test_log_prob_grads_match_finite_differences(make_dist, params, choices):
    """gradients of log_prob are finite and correct for responses and nonresponses,
    i.e. the unused branch of the nonresponse `jnp.where` doesn't leak NaNs"""
    params = jnp.array(params)
    x = jnp.array(
        [[c, 1.0 if c == -1 else 0.6 + 0.1 * i] for i, c in enumerate(choices)]
    )

    def log_p(p):
        return make_dist(p).log_prob(x)

    grads = jax.jacrev(log_p)(params)
    eps = 1e-6
    finite_diffs = jnp.stack(
        [
            (log_p(params.at[i].add(eps)) - log_p(params.at[i].add(-eps))) / (2 * eps)
            for i in range(len(params))
        ],
        -1,
    )

    assert np.all(np.isfinite(grads))
    np.testing.assert_allclose(grads, finite_diffs, rtol=1e-5, atol=1e-7)
