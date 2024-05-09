# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
import inspect

import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy.stats as osp

import jax
from jax import lax
import jax.numpy as jnp
import jax.random as random

import numpyro.distributions as dist
from numpyro.distributions import constraints, transforms
from numpyro.distributions.util import multinomial, signed_stick_breaking_tril

from cognax.decisions.discrete_choice_rt import DiscreteChoiceRT
from cognax.decisions.wfpt import WFPT, WFPTNormalDrift


class T(namedtuple("TestCase", ["jax_dist", "params"])):
    def __new__(cls, jax_dist, *params):
        return super(cls, T).__new__(cls, jax_dist, params)


T0 = 0.3
DISCRETE_CHOICE_RT = [
    T(WFPT, 0.3, 1.0, 0.5, T0),
    T(WFPTNormalDrift, 0.3, 0.1, 1.0, 0.5, T0),
]


def gen_values_within_bounds(constraint, size, key=random.PRNGKey(11)):
    eps = 1e-6

    if constraint is constraints.boolean:
        return random.bernoulli(key, shape=size)
    elif isinstance(constraint, constraints.greater_than):
        return jnp.exp(random.normal(key, size)) + constraint.lower_bound + eps
    elif isinstance(constraint, constraints.integer_interval):
        lower_bound = jnp.broadcast_to(constraint.lower_bound, size)
        upper_bound = jnp.broadcast_to(constraint.upper_bound, size)
        return random.randint(key, size, lower_bound, upper_bound + 1)
    elif isinstance(constraint, constraints.integer_greater_than):
        return constraint.lower_bound + random.poisson(key, np.array(5), shape=size)
    elif isinstance(constraint, constraints.interval):
        lower_bound = jnp.broadcast_to(constraint.lower_bound, size)
        upper_bound = jnp.broadcast_to(constraint.upper_bound, size)
        return random.uniform(key, size, minval=lower_bound, maxval=upper_bound)
    elif constraint in (constraints.real, constraints.real_vector):
        return random.normal(key, size)
    elif constraint is constraints.simplex:
        return osp.dirichlet.rvs(alpha=jnp.ones((size[-1],)), size=size[:-1])
    elif isinstance(constraint, constraints.multinomial):
        n = size[-1]
        return multinomial(
            key, p=jnp.ones((n,)) / n, n=constraint.upper_bound, shape=size[:-1]
        )
    elif constraint is constraints.corr_cholesky:
        return signed_stick_breaking_tril(
            random.uniform(
                key, size[:-2] + (size[-1] * (size[-1] - 1) // 2,), minval=-1, maxval=1
            )
        )
    elif constraint is constraints.corr_matrix:
        cholesky = signed_stick_breaking_tril(
            random.uniform(
                key, size[:-2] + (size[-1] * (size[-1] - 1) // 2,), minval=-1, maxval=1
            )
        )
        return jnp.matmul(cholesky, jnp.swapaxes(cholesky, -2, -1))
    elif constraint is constraints.lower_cholesky:
        return jnp.tril(random.uniform(key, size))
    elif constraint is constraints.positive_definite:
        x = random.normal(key, size)
        return jnp.matmul(x, jnp.swapaxes(x, -2, -1))
    elif constraint is constraints.ordered_vector:
        x = jnp.cumsum(random.exponential(key, size), -1)
        return x - random.normal(key, size[:-1] + (1,))
    elif isinstance(constraint, constraints.independent):
        return gen_values_within_bounds(constraint.base_constraint, size, key)
    elif constraint is constraints.sphere:
        x = random.normal(key, size)
        return x / jnp.linalg.norm(x, axis=-1)
    elif constraint is constraints.l1_ball:
        key1, key2 = random.split(key)
        sign = random.bernoulli(key1)
        bounds = [0, (-1) ** sign * 0.5]
        return random.uniform(key, size, float, *sorted(bounds))
    elif isinstance(constraint, constraints.zero_sum):
        x = random.normal(key, size)
        zero_sum_axes = tuple(i for i in range(-constraint.event_dim, 0))
        for axis in zero_sum_axes:
            x -= x.mean(axis)
        return x
    elif constraint is DiscreteChoiceRT.support:
        choices = jnp.zeros(size)[..., None]
        rts = random.uniform(key, size, minval=T0, maxval=5.0)[..., None]
        return jnp.concatenate([choices, rts], axis=-1)
    else:
        raise NotImplementedError("{} not implemented.".format(constraint))


def gen_values_outside_bounds(constraint, size, key=random.PRNGKey(11)):
    if constraint is constraints.boolean:
        return random.bernoulli(key, shape=size) - 2
    elif isinstance(constraint, constraints.greater_than):
        return constraint.lower_bound - jnp.exp(random.normal(key, size))
    elif isinstance(constraint, constraints.integer_interval):
        lower_bound = jnp.broadcast_to(constraint.lower_bound, size)
        return random.randint(key, size, lower_bound - 1, lower_bound)
    elif isinstance(constraint, constraints.integer_greater_than):
        return constraint.lower_bound - random.poisson(key, np.array(5), shape=size)
    elif isinstance(constraint, constraints.interval):
        upper_bound = jnp.broadcast_to(constraint.upper_bound, size)
        return random.uniform(key, size, minval=upper_bound, maxval=upper_bound + 1.0)
    elif constraint in [constraints.real, constraints.real_vector]:
        return lax.full(size, np.nan)
    elif constraint is constraints.simplex:
        return osp.dirichlet.rvs(alpha=jnp.ones((size[-1],)), size=size[:-1]) + 1e-2
    elif isinstance(constraint, constraints.multinomial):
        n = size[-1]
        return (
            multinomial(
                key, p=jnp.ones((n,)) / n, n=constraint.upper_bound, shape=size[:-1]
            )
            + 1
        )
    elif constraint is constraints.corr_cholesky:
        return (
            signed_stick_breaking_tril(
                random.uniform(
                    key,
                    size[:-2] + (size[-1] * (size[-1] - 1) // 2,),
                    minval=-1,
                    maxval=1,
                )
            )
            + 1e-2
        )
    elif constraint is constraints.corr_matrix:
        cholesky = 1e-2 + signed_stick_breaking_tril(
            random.uniform(
                key, size[:-2] + (size[-1] * (size[-1] - 1) // 2,), minval=-1, maxval=1
            )
        )
        return jnp.matmul(cholesky, jnp.swapaxes(cholesky, -2, -1))
    elif constraint is constraints.lower_cholesky:
        return random.uniform(key, size)
    elif constraint is constraints.positive_definite:
        return random.normal(key, size)
    elif constraint is constraints.ordered_vector:
        x = jnp.cumsum(random.exponential(key, size), -1)
        return x[..., ::-1]
    elif isinstance(constraint, constraints.independent):
        return gen_values_outside_bounds(constraint.base_constraint, size, key)
    elif constraint is constraints.sphere:
        x = random.normal(key, size)
        x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        return 2 * x
    elif constraint is constraints.l1_ball:
        key1, key2 = random.split(key)
        sign = random.bernoulli(key1)
        bounds = [(-1) ** sign * 1.1, (-1) ** sign * 2]
        return random.uniform(key, size, float, *sorted(bounds))
    elif isinstance(constraint, constraints.zero_sum):
        x = random.normal(key, size)
        return x
    elif constraint is DiscreteChoiceRT.support:
        choices = jnp.zeros(size)[..., None]
        rts = random.uniform(key, size, minval=-3, maxval=-1)[..., None]
        return jnp.concatenate([choices, rts], axis=-1)
    else:
        raise NotImplementedError("{} not implemented.".format(constraint))


@pytest.mark.parametrize("jax_dist_cls, params", DISCRETE_CHOICE_RT)
@pytest.mark.parametrize("prepend_shape", [(), (2,), (2, 3)])
def test_dist_shape(jax_dist_cls, params, prepend_shape):
    jax_dist = jax_dist_cls(*params)
    rng_key = random.PRNGKey(0)
    expected_shape = prepend_shape + jax_dist.batch_shape + jax_dist.event_shape
    samples = jax_dist.sample(key=rng_key, sample_shape=prepend_shape)

    assert isinstance(samples, jnp.ndarray)
    assert jnp.shape(samples) == expected_shape


dist.ZeroSumNormal


@pytest.mark.parametrize("jax_dist, params", DISCRETE_CHOICE_RT)
def test_jit_log_likelihood(jax_dist, params):
    rng_key = random.PRNGKey(0)
    samples = jax_dist(*params).sample(key=rng_key, sample_shape=(2, 3))

    def log_likelihood(*params):
        return jax_dist(*params).log_prob(samples)

    expected = log_likelihood(*params)
    actual = jax.jit(log_likelihood)(*params)
    assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("jax_dist, params", DISCRETE_CHOICE_RT)
def test_independent_shape(jax_dist, params):
    d = jax_dist(*params)
    batch_shape, event_shape = d.batch_shape, d.event_shape
    shape = batch_shape + event_shape
    for i in range(len(batch_shape)):
        indep = dist.Independent(d, reinterpreted_batch_ndims=i)
        sample = indep.sample(random.PRNGKey(0))
        event_boundary = len(shape) - len(event_shape) - i
        assert indep.batch_shape == shape[:event_boundary]
        assert indep.event_shape == shape[event_boundary:]
        assert jnp.shape(indep.log_prob(sample)) == shape[:event_boundary]


@pytest.mark.parametrize("jax_dist, params", DISCRETE_CHOICE_RT)
def test_log_prob_gradient(jax_dist, params):
    rng_key = random.PRNGKey(0)
    value = jax_dist(*params).sample(rng_key)

    def fn(*args):
        return jnp.sum(jax_dist(*args).log_prob(value))

    eps = 1e-3
    for i in range(len(params)):
        if jax_dist is dist.ZeroSumNormal and i != 0:  # XXX special case for trdm
            # skip taking grad w.r.t. event_shape
            continue
        if params[i] is None or jnp.result_type(params[i]) in (jnp.int32, jnp.int64):
            continue
        actual_grad = jax.grad(fn, i)(*params)
        args_lhs = [p if j != i else p - eps for j, p in enumerate(params)]
        args_rhs = [p if j != i else p + eps for j, p in enumerate(params)]
        fn_lhs = fn(*args_lhs)
        fn_rhs = fn(*args_rhs)
        # finite diff approximation
        expected_grad = (fn_rhs - fn_lhs) / (2.0 * eps)
        assert jnp.shape(actual_grad) == jnp.shape(params[i])
        assert_allclose(jnp.sum(actual_grad), expected_grad, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("jax_dist, params", DISCRETE_CHOICE_RT)
@pytest.mark.parametrize("prepend_shape", [(), (2,), (2, 3)])
def test_distribution_constraints(jax_dist, params, prepend_shape):
    dist_args = [p for p in inspect.getfullargspec(jax_dist.__init__)[0][1:]]

    valid_params, oob_params = list(params), list(params)
    key = random.PRNGKey(1)
    dependent_constraint = False
    for i in range(len(params)):
        if params[i] is None:
            oob_params[i] = None
            valid_params[i] = None
            continue
        constraint = jax_dist.arg_constraints[dist_args[i]]
        if isinstance(constraint, constraints._Dependent):
            dependent_constraint = True
            break
        key, key_gen = random.split(key)
        oob_params[i] = gen_values_outside_bounds(
            constraint, jnp.shape(params[i]), key_gen
        )
        valid_params[i] = gen_values_within_bounds(
            constraint, jnp.shape(params[i]), key_gen
        )
    assert jax_dist(*oob_params)

    # Invalid parameter values throw ValueError
    if not dependent_constraint:
        with pytest.raises(ValueError):
            jax_dist(*oob_params, validate_args=True)

        with pytest.raises(ValueError):
            # test error raised under jit omnistaging
            oob_params = jax.device_get(oob_params)

            def dist_gen_fn():
                d = jax_dist(*oob_params, validate_args=True)
                return d

            jax.jit(dist_gen_fn)()

    d = jax_dist(*valid_params, validate_args=True)

    # Out of support samples throw ValueError
    oob_samples = gen_values_outside_bounds(
        d.support, size=prepend_shape + d.batch_shape + d.event_shape
    )
    with pytest.warns(UserWarning, match="Out-of-support"):
        d.log_prob(oob_samples)

    with pytest.warns(UserWarning, match="Out-of-support"):
        # test warning work under jit omnistaging
        oob_samples = jax.device_get(oob_samples)
        valid_params = jax.device_get(valid_params)

        def log_prob_fn():
            d = jax_dist(*valid_params, validate_args=True)
            return d.log_prob(oob_samples)

        jax.jit(log_prob_fn)()


########################################
# Tests for constraints and transforms #
########################################
@pytest.mark.parametrize("jax_dist, params", DISCRETE_CHOICE_RT)
@pytest.mark.parametrize("prepend_shape", [(), (2, 3)])
@pytest.mark.parametrize("sample_shape", [(), (4,)])
def test_expand(jax_dist, params, prepend_shape, sample_shape):
    jax_dist = jax_dist(*params)
    new_batch_shape = prepend_shape + jax_dist.batch_shape
    expanded_dist = jax_dist.expand(new_batch_shape)
    rng_key = random.PRNGKey(0)
    samples = expanded_dist.sample(rng_key, sample_shape)
    assert expanded_dist.batch_shape == new_batch_shape
    assert jnp.shape(samples) == sample_shape + new_batch_shape + jax_dist.event_shape
    assert expanded_dist.log_prob(samples).shape == sample_shape + new_batch_shape
    # test expand of expand
    assert (
        expanded_dist.expand((3,) + new_batch_shape).batch_shape
        == (3,) + new_batch_shape
    )
    # test expand error
    if prepend_shape:
        with pytest.raises(ValueError, match="Cannot broadcast distribution of shape"):
            assert expanded_dist.expand((3,) + jax_dist.batch_shape)


@pytest.mark.parametrize("jax_dist, params", DISCRETE_CHOICE_RT)
def test_dist_pytree(jax_dist, params):
    def f(x):
        return jax_dist(*params)

    jax.jit(f)(0)  # this test for flatten/unflatten
    lax.map(f, np.ones(3))  # this test for compatibility w.r.t. scan
    # Test that parameters do not change after flattening.
    expected_dist = f(0)
    actual_dist = jax.jit(f)(0)
    for name in expected_dist.arg_constraints:
        expected_arg = getattr(expected_dist, name)
        actual_arg = getattr(actual_dist, name)
        assert actual_arg is not None, f"arg {name} is None"
        if np.issubdtype(np.asarray(expected_arg).dtype, np.number):
            assert_allclose(actual_arg, expected_arg)
        else:
            assert (
                actual_arg.shape == expected_arg.shape
                and actual_arg.dtype == expected_arg.dtype
            )
    expected_sample = expected_dist.sample(random.PRNGKey(0))
    actual_sample = actual_dist.sample(random.PRNGKey(0))
    expected_log_prob = expected_dist.log_prob(expected_sample)
    actual_log_prob = actual_dist.log_prob(actual_sample)
    assert_allclose(actual_sample, expected_sample, rtol=1e-6)
    assert_allclose(actual_log_prob, expected_log_prob, rtol=2e-6)
