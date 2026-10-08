from functools import partial

import jax.numpy as jnp
import jax.random as random

from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import lazy_property

from cognax.util import vmap_n


def all_choice_rts(n_choice, dt=0.01, deadline=5.0):
    """absolute (choice, rt) grid ending exactly at the deadline, and its bin width"""
    n_bins = round(deadline / dt)
    t_range = jnp.linspace(deadline / n_bins, deadline, n_bins)

    choices = jnp.repeat(jnp.arange(n_choice), repeats=n_bins)
    rts = jnp.tile(t_range, n_choice)
    return jnp.stack([choices, rts], axis=-1), deadline / n_bins


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

    # XXX this is a weak form of choice rt-constraint. We should actually
    # ensure rts > t0 and choices are in {valid_choices}

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
    pytree_aux_fields = ("n_choice", "dt", "deadline")
    # whether log_prob scores (-1, deadline) as a nonresponse
    has_nonresponse = False

    def __init__(
        self,
        n_choice,
        dt=0.01,
        deadline=5.0,
        batch_shape=(),
        *,
        validate_args=None,
    ):
        self.n_choice = n_choice
        self.dt = dt
        self.deadline = deadline

        super().__init__(
            batch_shape=batch_shape, event_shape=(2,), validate_args=validate_args
        )

    @lazy_property
    def _grid(self):
        """
        `(choice_rts, probs)`: the discretized (choice, rt) grid up to `deadline` (plus
        `(-1, deadline)` if the distribution handles nonresponse), and the probability
        of each grid point, shape `(*batch_shape, n_grid)`. Bins before t0 get 0
        probability. Without nonresponse, probabilities are renormalized over responses
        before the deadline.
        """
        n_batch_dims = len(self.batch_shape)
        choice_rts, bin_width = all_choice_rts(self.n_choice, self.dt, self.deadline)
        n_response_bins = choice_rts.shape[0]
        if self.has_nonresponse:
            choice_rts = jnp.concatenate([choice_rts, jnp.array([[-1, self.deadline]])])

        # (n_grid, *batch_shape)
        probs = jnp.exp(self.log_prob(choice_rts.reshape(-1, *[1] * n_batch_dims, 2)))
        # response bins are densities; the nonresponse bin is already a probability
        probs = probs.at[:n_response_bins].multiply(bin_width)
        # reshape (n_grid, *batch_shape) to (*batch_shape, n_grid)
        probs = jnp.moveaxis(probs, 0, -1)
        return choice_rts, probs / probs.sum(-1, keepdims=True)

    @lazy_property
    def probs(self):
        """
        The probability of each choice index, `(*batch_shape, n_choice)`, computed by
        marginalizing the rt distribution up to `deadline`. If the distribution handles
        nonresponse, the last column is the nonresponse probability
        (`(*batch_shape, n_choice + 1)`).
        """
        _, probs = self._grid
        n_response_bins = self.n_choice * round(self.deadline / self.dt)
        choice_probs = (
            probs[..., :n_response_bins]
            .reshape(*self.batch_shape, self.n_choice, -1)
            .sum(-1)
        )
        return jnp.concatenate([choice_probs, probs[..., n_response_bins:]], -1)

    def sample(self, key, sample_shape=()):
        """
        The default sampler uses approximate inverse-transform sampling by discretizing
        continuous time into discrete chunks (see `_grid`). The approximation error can
        be reduced by decreasing `dt` at the expense of increased memory usage. If the
        distribution handles nonresponse, `(-1, deadline)` is sampled for nonresponses.

        Args:
            key: jax.random.PRNGKey
            sample_shape (tuple, optional):
        """
        n_batch_dims = len(self.batch_shape)
        icdf_sampler = partial(icdf_sample, sample_shape=sample_shape)

        choice_rts, probs = self._grid
        choice_rts = jnp.broadcast_to(
            choice_rts, (*self.batch_shape, *choice_rts.shape)
        )

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
