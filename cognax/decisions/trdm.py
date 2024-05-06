from functools import partial

import jax.numpy as jnp
from jax import lax

from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes
from tensorflow_probability.substrates.jax import distributions as tfd

from cognax.decisions.discrete_choice_rt import DiscreteChoiceRT
from cognax.util import vmap_n


def log_p_choice(x, v, sigma, alpha):
    """
    log probability that a choice with params (v, sigma, alpha) was selected at exactly t=x.

    Args:
        x: RTs. (0, inf)
        v: drift rate
        sigma: diffusion coefficient
        alpha: decision threshold
    """
    return tfd.InverseGaussian(alpha / v, alpha**2 / sigma**2).log_prob(x)


def cum_log_p_not_choice(x, v, sigma, alpha):
    """
    log probability that a choice with params (v, sigma, alpha) was *not* selected from t=0 to t=x.

    Args:
        x: RTs. (0, inf)
        v: drift rate
        sigma: diffusion coefficient
        alpha: decision threshold
    """
    return jnp.log(
        1 - tfd.InverseGaussian(alpha / v, alpha**2 / sigma**2).cdf(x)
    )  # XXX guard in case this is 0?


def trdm_log_dens(
    timer, choice, RT, v, alpha, sigma, v_timer, alpha_timer, sigma_timer
):
    """
    Compute the TRDM log density for a choice with a given response time.

    Args:
        timer (bool): If set to True, adds the timer term to the density
        choice (int): choice index
        RT (float): response time
        v (`(n_choice,)` array_like): drift rates for each choice
        alpha (`(n_choice,)` array_like): boundary for each choice
        sigma (`(n_choice,)` array_like): diffusion coefficient for each choice
        v_timer (float): timer drift rate
        alpha_timer (float): timer boundary
        sigma_timer (float): timer diffusion coefficient
    """
    n_choice = v.shape[0]

    log_p_selected_choice = log_p_choice(RT, v[choice], sigma[choice], alpha[choice])
    cum_log_p_not_choices = cum_log_p_not_choice(RT, v, sigma, alpha)
    cum_log_p_others_inactive = jnp.sum(cum_log_p_not_choices.at[choice].set(0.0))
    cum_log_p_none_active = jnp.sum(cum_log_p_not_choices)

    # --- process non-response ---
    nonresponse = choice == -1
    log_p_selected_choice = jnp.where(nonresponse, 0.0, log_p_selected_choice)
    cum_log_p_others_inactive = jnp.where(
        nonresponse, cum_log_p_none_active, cum_log_p_others_inactive
    )
    # ---

    if timer:
        log_p_timer_activated = log_p_choice(RT, v_timer, sigma_timer, alpha_timer)
        cum_log_p_not_timer = cum_log_p_not_choice(
            RT, v_timer, sigma_timer, alpha_timer
        )
        log_p_choice_at_timer = -jnp.log(n_choice)  # = 1 / n_choice

        # TODO:
        # log_p_ahead: probability that the selected choice was ahead of all the other choices at time t
        # log_p_choice_at_timer = rho * log_p_ahead + (1 - rho) * (1. / n_choice)

        # p(choice_i, t) *
        # p(other choices not activated, 0 -> t) *
        # p(timer not activated, 0 -> t))
        log_dens_choice = (
            log_p_selected_choice + cum_log_p_others_inactive + cum_log_p_not_timer
        )

        # p(choice_i, moment of timer activation) *
        # p(timer activated) *
        # p(no choices activated, 0 -> t)
        log_dens_timer = (
            log_p_choice_at_timer + log_p_timer_activated + cum_log_p_none_active
        )

        return jnp.logaddexp(log_dens_choice, log_dens_timer)
    else:
        # p(choice_i, t) *
        # p(other choices not activated, 0 -> t)
        log_dens_choice = log_p_selected_choice + cum_log_p_others_inactive

        return log_dens_choice


class TRDM(DiscreteChoiceRT):
    """
    Timed Racing Diffusion Model.

    Input to `log_prob` should be a `(..., 2)` array_like of choice-RTs. The first
    event dimension should contain choice indeces in {0, ..., n}, and the second event dimension
    should contain the response-time for that choice.

    This distribution *does* handle nonresponse. Nonresponse choices should be coded as `-1`.

    **References:**

    1. https://psycnet.apa.org/record/2021-17581-001

    Args:
        v (`(..., n_choice)` array_like): drift rates for each choice
        alpha (`(..., n_choice)` array_like): boundary for each choice
        sigma (`(..., n_choice)` array_like): diffusion coefficient for each choice
        t0 (array_like): non-decision time
        v_timer (array_like, optional): timer drift rate. Defaults to None.
        alpha_timer (array_like, optional): timer boundary. Defaults to None.
        sigma_timer (array_like, optional): timer diffusion coefficient. Defaults to None.
    """

    arg_constraints = {
        "v": constraints.positive,
        "alpha": constraints.positive,
        "sigma": constraints.positive,
        "t0": constraints.nonnegative,
        "v_timer": constraints.real,
        "alpha_timer": constraints.positive,
        "sigma_timer": constraints.positive,
    }

    def log_prob(self, value):
        choices, RTs = value[..., 0].astype(int), value[..., 1] - self.t0

        eval_batch_shape = jnp.broadcast_shapes(RTs.shape, self.batch_shape)

        v, alpha, sigma = [
            jnp.broadcast_to(param, (*eval_batch_shape, self.n_choice))
            for param in (self.v, self.alpha, self.sigma)
        ]

        v_timer, alpha_timer, sigma_timer, choices, RTs = [
            jnp.broadcast_to(param, eval_batch_shape)
            for param in (
                self.v_timer,
                self.alpha_timer,
                self.sigma_timer,
                choices,
                RTs,
            )
        ]

        log_p = vmap_n(
            partial(trdm_log_dens, self.timer),
            n_times=RTs.ndim,
            choice=choices,
            RT=RTs,
            v=v,
            alpha=alpha,
            sigma=sigma,
            v_timer=v_timer,
            alpha_timer=alpha_timer,
            sigma_timer=sigma_timer,
        )

        return jnp.where(RTs > 0, log_p, -jnp.inf)

    def __init__(
        self,
        v,
        alpha,
        sigma,
        t0,
        v_timer=None,
        alpha_timer=None,
        sigma_timer=None,
        validate_args=None,
    ):
        timer_params = (v_timer, alpha_timer, sigma_timer)

        if all([param is None for param in timer_params]):
            self.timer = False
            v_timer, alpha_timer, sigma_timer = jnp.nan, jnp.nan, jnp.nan
        elif all([param is not None for param in timer_params]):
            self.timer = True
        else:
            raise ValueError(
                "Either all of `v_timer`, `alpha_timer`, `sigma_timer`"
                "must be specified, or all must be `None`"
            )
        n_choice = v.shape[-1]

        assert all(
            param.shape[-1] == n_choice for param in (v, alpha, sigma)
        ), "All choice params must have consistent rightmost (choice) dimension"

        (self.v, self.alpha, self.sigma, t0, v_timer, alpha_timer, sigma_timer) = (
            promote_shapes(v, alpha, sigma, t0, v_timer, alpha_timer, sigma_timer)
        )

        self.t0, self.v_timer, self.alpha_timer, self.sigma_timer = (
            t0[..., 0],
            v_timer[..., 0],
            alpha_timer[..., 0],
            sigma_timer[..., 0],
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(self.v)[:-1],
            jnp.shape(self.alpha)[:-1],
            jnp.shape(self.sigma)[:-1],
            jnp.shape(self.t0),
            jnp.shape(self.v_timer),
            jnp.shape(self.alpha_timer),
            jnp.shape(self.sigma_timer),
        )

        super(TRDM, self).__init__(
            n_choice=n_choice, batch_shape=batch_shape, validate_args=validate_args
        )
