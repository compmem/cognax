import jax.numpy as jnp
from jax import lax

from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes

from cognax.decisions.discrete_choice_rt import DiscreteChoiceRT
from cognax.util import vmap_n


def use_fast_expansion(x):
    """
    For each element in `x`, return `True` if the fast-RT expansion is more efficient
    than the slow-RT expansion.

    Args:
        x: RTs. (0, inf).
    """
    err = 1e-7  # error tolerance; XXX: if accuracy is low, tune this

    # determine number of terms needed for small-t expansion
    _a = 2 * jnp.sqrt(2 * jnp.pi * x) * err < 1.0
    _b = 2 + jnp.sqrt(-2 * x * jnp.log(2 * jnp.sqrt(2 * jnp.pi * x) * err))
    _c = jnp.sqrt(x) + 1
    _d = jnp.max(jnp.stack([_b, _c]), axis=0)
    ks = _a * _d + (1 - _a) * 2

    # determine number of terms needed for large-t expansion
    _a = jnp.pi * x * err < 1
    _b = 1.0 / (jnp.pi * jnp.sqrt(x))
    _c = jnp.sqrt(-2 * jnp.log(jnp.pi * x * err) / (jnp.pi**2 * x))
    _d = jnp.max(jnp.stack([_b, _c]), axis=0)
    kl = _a * _d + (1 - _a) * _b

    # select the most accurate expansion for a fixed number of terms
    return ks < kl


def fnorm_fast(x, w):
    """
    Density function for lower-bound first-passage times with drift rate set to 0 and
    upper bound set to 1, calculated using the fast-RT expansion.

    Args:
        x: RTs. (0, inf).
        w: Normalized decision starting point. (0, 1).
    """
    K = 10  # number of terms to evaluate; XXX: if accuracy is low, tune this

    # calculated using the "log-sum-exp trick" to reduce under/overflows
    k = jnp.arange(K) - jnp.floor(K / 2)
    y = w + 2 * k.reshape((-1, 1))
    r = -jnp.power(y, 2) / 2 / x
    c = jnp.max(r, axis=0)
    p = jnp.exp(c + jnp.log(jnp.sum(y * jnp.exp(r - c), axis=0)))
    p = p / jnp.sqrt(2 * jnp.pi * jnp.power(x, 3))

    return p


def fnorm_slow(x, w):
    """
    Density function for lower-bound first-passage times with drift rate set to 0 and
    upper bound set to 1, calculated using the slow-RT expansion.

    Args:
        x: RTs. (0, inf).
        w: Normalized decision starting point. (0, 1).
    """
    K = 10  # number of terms to evaluate; XXX: if accuracy is low, tune this

    # calculated the better way
    k = jnp.arange(1, K + 1).reshape((-1, 1))
    y = k * jnp.sin(k * jnp.pi * w)
    r = -jnp.power(k, 2) * jnp.power(jnp.pi, 2) * x / 2

    # -- without log-sum-exp --
    p = jnp.sum(y * jnp.exp(r), axis=0) * jnp.pi
    # -- with logsumexp to reduce under/overflows --
    # c = jnp.max(r, axis=0)
    # p = jnp.exp(c + jnp.log(jnp.sum(y * jnp.exp(r - c), axis=0))) * jnp.pi

    return p


def fnorm(x, w):
    """
    Density function for lower-bound first-passage times with drift rate set to 0 and
    upper bound set to 1, selecting the most efficient expansion per element in `x`.

    Args:
        x: RTs. (0, inf).
        w: Normalized decision starting point. (0, 1).
    """
    y = jnp.abs(x)

    densities = jnp.where(
        use_fast_expansion(y), fnorm_fast(y, w), fnorm_slow(y, w)
    ).squeeze()

    return jnp.where(x > 0, densities, 0.0)


class WFPT(DiscreteChoiceRT):
    """
    Weiner First Passage Time with the Navarro-Fuss parameterization:

    - absorption boundaries at `[0, a]`,
    - diffusion coefficient: `1`
    - `w`: the proportion of the distance to `a` the particle starts at (e.g. `w = .5` -> the particle starts halfway to `a`)

    Input to `log_prob` should be a `(..., 2)` array_like of choice-RTs. The first
    event dimension should contain choice indeces in {0, 1}, and the second event dimension
    should contain the response-time for that choice.

    This distribution does *not* handle nonresponse. To exclude missing observations
    from the likelihood, use `WFPT(...).mask`.

    **References:**

    1. https://compcogscisydney.org/publications/NavarroFuss2009.pdf
    2. https://gist.github.com/sammosummo/c1be633a74937efaca5215da776f194b#file-ddm_in_aesara-py-L116

    Args:
        v (array_like): drift rate
        a (array_like): upper absorption boundary
        w (array_like): relative starting point
        t0 (array_like): nondecision time
    """

    arg_constraints = {
        "v": constraints.real,
        "a": constraints.positive,
        "w": constraints.unit_interval,
        "t0": constraints.positive,
    }

    def log_prob(self, value):
        choices = value[..., 0]
        RTs = value[..., 1]

        v = jnp.where(choices == 1, -self.v, self.v)
        w = jnp.where(choices == 1, 1 - self.w, self.w)

        RTs, w = jnp.broadcast_arrays(RTs, w)
        p = vmap_n(fnorm, n_times=RTs.ndim, x=(RTs - self.t0) / self.a**2, w=w)

        return jnp.log(p) - (
            (v * self.a * w) + 0.5 * jnp.square(v) * RTs + jnp.log(self.a**2)
        )

    def __init__(self, v, a, w, t0, *, validate_args=None):
        self.v, self.a, self.w, self.t0 = promote_shapes(v, a, w, t0)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(v), jnp.shape(a), jnp.shape(w), jnp.shape(t0)
        )
        super(WFPT, self).__init__(
            n_choice=2, batch_shape=batch_shape, validate_args=validate_args
        )


class WFPTNormalDrift(DiscreteChoiceRT):
    """
    WFPT with normally distributed trial-level variability in drift rate.

    v ~ Normal(v_loc, v_scale)

    Args:
        v_loc (array_like): mean drift rate
        v_scale (array_like): drift rate standard deviation
        a (array_like): upper absorption boundary
        w (array_like): relative starting point
        t0 (array_like): nondecision time
    """

    arg_constraints = {
        "v_loc": constraints.real,
        "v_scale": constraints.positive,
        "a": constraints.positive,
        "w": constraints.unit_interval,
        "t0": constraints.positive,
    }

    def log_prob(self, value):
        choices = value[..., 0]
        RTs = value[..., 1]

        v_loc = jnp.where(choices == 1, -self.v_loc, self.v_loc)
        w = jnp.where(choices == 1, 1 - self.w, self.w)

        RTs, w = jnp.broadcast_arrays(RTs, w)
        p = vmap_n(fnorm, n_times=RTs.ndim, x=(RTs - self.t0) / self.a**2, w=w)

        return jnp.log(
            jnp.exp(
                jnp.log(p)
                + (
                    (self.a * w * self.v_scale) ** 2
                    - 2 * self.a * v_loc * w
                    - (v_loc**2) * RTs
                )
                / (2 * (self.v_scale**2) * RTs + 2)
            )
            / jnp.sqrt((self.v_scale**2) * RTs + 1)
            / (self.a**2)
        )

    def __init__(self, v_loc, v_scale, a, w, t0, *, validate_args=None):
        self.v_loc, self.v_scale, self.a, self.w, self.t0 = promote_shapes(
            v_loc, v_scale, a, w, t0
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(v_loc),
            jnp.shape(v_scale),
            jnp.shape(a),
            jnp.shape(w),
            jnp.shape(t0),
        )
        super(WFPTNormalDrift, self).__init__(
            n_choice=2, batch_shape=batch_shape, validate_args=validate_args
        )
