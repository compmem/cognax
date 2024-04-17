from functools import partial

import jax.numpy as jnp
import jax.random as random

from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import lazy_property

from cognax.util import vmap_n


def all_choice_rts(t0, n_choice, dt=0.01, rel_max_time=5.0):
    t_range = t0 + jnp.arange(0, rel_max_time + dt, dt)

    choices = jnp.repeat(jnp.arange(n_choice), repeats=len(t_range))
    rts = jnp.concatenate([t_range] * n_choice)
    return jnp.vstack([choices, rts]).T


def icdf_sample(key, vals, probs, sample_shape=()):
    """
    Given an array of values spanning the support of a distribution
    and an associated array of probs associated with each value (adding up to 1),
    sample from the distribution using inverse transform sampling.

    Args:
        vals: array of shape `(batch_shape, event_shape)`
        probs: array of shape `(batch_shape,)
    """
    cdf = jnp.concatenate([(probs).cumsum(), jnp.array([1.0])])

    rand_nums = random.uniform(key, sample_shape)
    inds = jnp.argmax(cdf[..., jnp.newaxis] > rand_nums.flatten(), axis=0)

    return vals[inds].reshape((*sample_shape, -1))


class _DiscreteChoiceRTConstraint(constraints._SingletonConstraint):
    event_dim = 1

    def __call__(self, x):
        return (x[..., 0] % 1 == 0) & (x[..., 1] > 0)

    def feasible_like(self, prototype):
        return jnp.ones_like(prototype)


class DiscreteChoiceRT(Distribution):
    """
    Base class for discrete choice-rt distributions.

    Input to `log_prob` should be a `(..., 2)` array_like of choice-RTs. The first
    event dimension should contain choice indeces in {0, ..., self.n_choice}, and the
    second event dimension should contain the response-time for that choice.
    """

    support = _DiscreteChoiceRTConstraint()

    def __init__(self, n_choice, batch_shape=(), *, validate_args=None):
        self._n_choice = n_choice

        super().__init__(
            batch_shape=batch_shape, event_shape=(2,), validate_args=validate_args
        )

    @property
    def n_choice(self):
        return self._n_choice

    # @property
    # def t0(self):
    #     """non-decision time"""
    #     return self.t0

    @lazy_property
    def probs(self, dt=0.01, rel_max_time=5.0):
        """
        The probability of selecting each choice index. By default, we compute this
        by marginalizing the rt distribution at each choice (discretizing continous time).

        XXX: We don't include the probability of a non-response by default.
        """
        n_batch_dims = len(self.batch_shape)

        get_all_choice_rts = partial(
            all_choice_rts, n_choice=self.n_choice, dt=dt, rel_max_time=rel_max_time
        )
        choice_rts = vmap_n(
            get_all_choice_rts,
            n_times=n_batch_dims,
            t0=jnp.broadcast_to(self.t0, self.batch_shape),
        )

        probs_each_dt = jnp.exp(self.log_prob(jnp.moveaxis(choice_rts, -2, 0))) * dt
        probs_each_dt = jnp.moveaxis(probs_each_dt, 0, -1)

        probs = jnp.zeros((*self.batch_shape, self.n_choice))

        for choice in range(self.n_choice):
            active_probs = jnp.where(choice_rts[..., 0] == choice, probs_each_dt, 0.0)
            probs = probs.at[..., choice].set(jnp.sum(active_probs, axis=-1))

        return probs / jnp.sum(probs, axis=-1)

    def sample(self, key, sample_shape=(), dt=0.01, rel_max_time=5.0):
        """
        The default sampler uses approximate inverse-transform sampling by discretizing
        continuous time into discrete chunks. The approximation error can be reduced by
        decreasing `dt` and increasing `rel_max_time` at the expense of increased memory usage.

        XXX: We don't sample any non-response

        Args:
            key: jax.random.PRNGKey
            sample_shape (tuple, optional):
            dt (float, optional): step size to discretize continuous time with. Defaults to .01.
            rel_max_time (float, optional): Amount of time after nondecision time
                to compute probabilities for. Defaults to 5.0.
        """

        n_batch_dims = len(self.batch_shape)

        get_all_choice_rts = partial(
            all_choice_rts, n_choice=self.n_choice, dt=dt, rel_max_time=rel_max_time
        )
        icdf_sampler = partial(icdf_sample, sample_shape=sample_shape)

        choice_rts = vmap_n(
            get_all_choice_rts,
            n_times=n_batch_dims,
            t0=jnp.broadcast_to(self.t0, self.batch_shape),
        )

        # reshape (*batch_shape, sample_bins, 2) to (sample_bins, *batch_shape, 2)
        probs = jnp.exp(self.log_prob(jnp.moveaxis(choice_rts, -2, 0)) * dt)
        # reshape (sample_bins, *batch_shape) to (*batch_shape, sample_bins)
        probs = jnp.moveaxis(probs, 0, -1)

        samps = vmap_n(
            icdf_sampler,
            n_batch_dims,
            random.split(key, self.batch_shape),
            choice_rts,
            probs,
        )

        # reshape (*batch_shape, *sample_shape, 2) to (*sample_shape, *batch_shape, 2)
        return jnp.moveaxis(
            samps, tuple(range(n_batch_dims)), tuple(range(-1 - n_batch_dims, -1))
        )
