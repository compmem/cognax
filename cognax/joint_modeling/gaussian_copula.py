import jax
import jax.numpy as jnp

import numpyro.distributions.constraints as constraints
from numpyro.distributions import MultivariateNormal, Normal, Distribution
from numpyro.distributions.util import clamp_probs, lazy_property, validate_sample
from numpyro.util import is_prng_key

import cognax.joint_modeling.collection as collection

class GaussianCopula(Distribution):
    """Like NumPyro's Gaussian Copula, but couples different marginal distributions."""
    
    arg_constraints = {
        "correlation_matrix": constraints.corr_matrix,
        "correlation_cholesky": constraints.corr_cholesky,
    }
    reparametrized_params = [
        "correlation_matrix",
        "correlation_cholesky",
    ]

    pytree_data_fields = ("marginal_dists", "marginal_slices", "base_dist")

    def __init__(
        self,
        marginal_dists,
        correlation_matrix=None,
        correlation_cholesky=None,
        *,
        validate_args=None,
    ):
        for marginal_dist in marginal_dists:
            if len(marginal_dist.event_shape) > 0:
                raise ValueError(
                    "`marginal_dist` needs to be a univariate distribution."
                )

        self.marginal_dists = marginal_dists
        self.base_dist = MultivariateNormal(
            covariance_matrix=correlation_matrix,
            scale_tril=correlation_cholesky,
        )
        
        batch_shapes = []
        marginal_slices = []  # slices of the MultivariateNormal that correspond to each marginal
        slice_start = 0

        for marginal_dist in self.marginal_dists:
            batch_shapes.append(marginal_dist.batch_shape[:-1])

            slice_size = (
                1 if marginal_dist.batch_shape == () else marginal_dist.batch_shape[-1]
            )
            marginal_slices.append((slice_start, slice_start + slice_size))
            slice_start += slice_size

        self.marginal_slices = marginal_slices

        assert (
            self.marginal_slices[-1][1] == self.base_dist.event_shape[0]
        ), "marginal dists did not match dimension of correlation matrix"

        super(GaussianCopula, self).__init__(
            batch_shape=jax.lax.broadcast_shapes(
                *batch_shapes, self.base_dist.batch_shape
            ),
            event_shape=self.base_dist.event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)

        shape = sample_shape + self.batch_shape
        normal_samples = self.base_dist.expand(shape).sample(key)

        cdf = Normal().cdf(normal_samples) # compute the cdf for all not normal

        for marginal_slice, marginal_dist in zip(
            self.marginal_slices, self.marginal_dists
        ):
            cdf_subset = jax.lax.slice_in_dim(
                cdf, *marginal_slice, axis=-1
            )
            normal_samples = normal_samples.at[..., marginal_slice[0]:marginal_slice[1]].set(marginal_dist.icdf(cdf_subset))

        return normal_samples

    @validate_sample
    def log_prob(self, value):
        marginal_lps = jnp.zeros_like(value)
        probs = jnp.zeros_like(value)

        for marginal_slice, marginal_dist in zip(self.marginal_slices, self.marginal_dists):
            value_subset = jax.lax.slice_in_dim(
                value, *marginal_slice, axis=-1
            )
            marginal_lps = marginal_lps.at[..., marginal_slice[0]:marginal_slice[1]].set(marginal_dist.log_prob(value_subset))
            probs = probs.at[..., marginal_slice[0]:marginal_slice[1]].set(marginal_dist.cdf(value_subset))

        quantiles = Normal().icdf(clamp_probs(probs)) 

        copula_lp = (
            self.base_dist.log_prob(quantiles)
            + 0.5 * (quantiles**2).sum(-1)
            + 0.5 * jnp.log(2 * jnp.pi) * quantiles.shape[-1]
        )

        return copula_lp + marginal_lps.sum(axis=-1)

    @lazy_property
    def mean(self):
        marginal_means = []

        for marginal_slice, marginal_dist in zip(self.marginal_slices, self.marginal_dists):
            slice_size = marginal_slice[1] - marginal_slice[0]
            bcast_shape = (*self.batch_shape, slice_size)             
            marginal_means.append(jnp.broadcast_to(marginal_dist.mean, bcast_shape))            

        return jnp.concatenate(marginal_means, axis=-1)

    @lazy_property
    def variance(self):
        marginal_variances = []

        for marginal_slice, marginal_dist in zip(self.marginal_slices, self.marginal_dists):
            slice_size = marginal_slice[1] - marginal_slice[0]
            bcast_shape = (*self.batch_shape, slice_size)             
            marginal_variances.append(jnp.broadcast_to(marginal_dist.variance, bcast_shape))            

        return jnp.concatenate(marginal_variances, axis=-1)

    @property
    def support(self):
        base_constraints = [marginal_dist.support 
                            for marginal_dist in self.marginal_dists]
        
        return collection.constraint_collection(base_constraints, self.marginal_slices)

    @lazy_property
    def correlation_matrix(self):
        return self.base_dist.covariance_matrix

    @lazy_property
    def correlation_cholesky(self):
        return self.base_dist.scale_tril