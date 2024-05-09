import jax
import jax.numpy as jnp

import numpyro.distributions.constraints as constraints
from numpyro.distributions.constraints import Constraint
from numpyro.distributions.transforms import Transform, biject_to
from numpyro.distributions import MultivariateNormal, Normal, Distribution
from numpyro.distributions.util import clamp_probs, lazy_property, validate_sample
from numpyro.util import is_prng_key


class _ConstraintCollection(Constraint):
    """
    Applies a collection of constraints to the rightmost dimension of an array.
    Each constraint in `base_constraints[i]` is applied to `arr[..., slices[i][0]: slices[i][1]]`.
    Slices must be contiguous and non-overlapping.
    """

    event_dim = 1

    def __init__(self, base_constraints, slices):
        assert len(base_constraints) == len(slices)

        prev_slice_end = 0
        for base_constraint, slice in zip(base_constraints, slices):
            assert isinstance(base_constraint, Constraint)
            assert isinstance(slice, tuple)
            assert slice[0] == prev_slice_end
            prev_slice_end = slice[1]

        self.base_constraints = base_constraints
        self.slices = slices
        super().__init__()

    def __call__(self, value):
        results = jnp.zeros_like(value)

        for slice, base_constraint in zip(self.slices, self.base_constraints):
            value_subset = jax.lax.slice_in_dim(value, *slice, axis=-1)
            results = results.at[..., slice[0] : slice[1]].set(
                base_constraint(value_subset)
            )
        return results.all(-1)

    def feasible_like(self, prototype):
        feasible_likes = jnp.zeros_like(prototype)

        for slice, base_constraint in zip(self.slices, self.base_constraints):
            prototype_subset = jax.lax.slice_in_dim(prototype, *slice, axis=-1)
            feasible_likes = feasible_likes.at[..., slice[0] : slice[1]].set(
                base_constraint(prototype_subset)
            )
        return feasible_likes

    def __eq__(self, other):
        if not isinstance(other, _ConstraintCollection) or self.slices != other.slices:
            return False
        constraints_match = True
        for constraint, other_constraint in zip(
            self.base_constraints, other.base_constraints
        ):
            constraints_match = constraints_match & (constraint == other_constraint)
        return constraints_match

    def tree_flatten(self):
        return (self.base_constraints,), (
            ("base_constraints",),
            {"slices": self.slices},
        )


class CollectionTransform(Transform):
    """
    Applies a collection of transforms to the rightmost dimension of an array.
    Each transform is applied to `arr[..., slices[i][0]: slices[i][1]]`.
    Slices must be contiguous and non-overlapping. Transformations must be shape preserving.
    """

    def __init__(self, base_transforms, slices):
        self.base_transforms = base_transforms
        self.slices = slices

    @property
    def domain(self):
        return constraints.real

    @property
    def codomain(self):
        base_constraints = [transform.codomain for transform in self.base_transforms]
        return _ConstraintCollection(base_constraints, self.slices)

    def __call__(self, x):
        y = jnp.zeros_like(x)
        for slice, transform in zip(self.slices, self.base_transforms):
            x_subset = jax.lax.slice_in_dim(x, *slice, axis=-1)
            y = y.at[..., slice[0] : slice[1]].set(transform(x_subset))
        return y

    def _inverse(self, y):
        x = jnp.zeros_like(y)
        for slice, transform in zip(self.slices, self.base_transforms):
            y_subset = jax.lax.slice_in_dim(y, *slice, axis=-1)
            x = x.at[..., slice[0] : slice[1]].set(transform._inverse(y_subset))
        return x

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        jacs = jnp.zeros(x.shape[:-1])

        for i, (slice, transform) in enumerate(zip(self.slices, self.base_transforms)):
            x_subset = jax.lax.slice_in_dim(x, *slice, axis=-1)
            y_subset = None if y is None else jax.lax.slice_in_dim(y, *slice, axis=-1)
            inter = None if intermediates is None else intermediates[i]
            jacs += transform.log_abs_det_jacobian(x_subset, y_subset, inter).sum(-1)

        return jacs

    def call_with_intermediates(self, x):
        intermediates = []
        y = jnp.zeros_like(x)
        for slice, transform in zip(self.slices, self.base_transforms):
            x_subset = jax.lax.slice_in_dim(x, *slice, axis=-1)
            y_subset, inter = transform.call_with_intermediates(x_subset)
            y = y.at[..., slice[0] : slice[1]].set(y_subset)
            intermediates.append(inter)
        return y, intermediates

    def __eq__(self, other):
        if not isinstance(other, CollectionTransform) or self.slices != other.slices:
            return False
        transforms_match = True
        for transform, other_transform in zip(
            self.base_transforms, other.base_transforms
        ):
            transforms_match = transforms_match & (transform == other_transform)
        return transforms_match

    def tree_flatten(self):
        return (self.base_transforms,), (("base_transforms",), {"slices": self.slices})


@biject_to.register(_ConstraintCollection)
def _transform_collection_to_valid(constraint):
    base_transforms = [
        biject_to(constraint) for constraint in constraint.base_constraints
    ]
    return CollectionTransform(base_transforms, constraint.slices)


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

    pytree_data_fields = ("marginal_dists", "base_dist")
    pytree_aux_fields = "marginal_slices"

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

        cdf = Normal().cdf(normal_samples)  # compute the cdf for all not normal

        for marginal_slice, marginal_dist in zip(
            self.marginal_slices, self.marginal_dists
        ):
            cdf_subset = jax.lax.slice_in_dim(cdf, *marginal_slice, axis=-1)
            normal_samples = normal_samples.at[
                ..., marginal_slice[0] : marginal_slice[1]
            ].set(marginal_dist.icdf(cdf_subset))

        return normal_samples

    @validate_sample
    def log_prob(self, value):
        marginal_lps = jnp.zeros_like(value)
        probs = jnp.zeros_like(value)

        for marginal_slice, marginal_dist in zip(
            self.marginal_slices, self.marginal_dists
        ):
            value_subset = jax.lax.slice_in_dim(value, *marginal_slice, axis=-1)
            marginal_lps = marginal_lps.at[
                ..., marginal_slice[0] : marginal_slice[1]
            ].set(marginal_dist.log_prob(value_subset))
            probs = probs.at[..., marginal_slice[0] : marginal_slice[1]].set(
                marginal_dist.cdf(value_subset)
            )

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

        for marginal_slice, marginal_dist in zip(
            self.marginal_slices, self.marginal_dists
        ):
            slice_size = marginal_slice[1] - marginal_slice[0]
            bcast_shape = (*self.batch_shape, slice_size)
            marginal_means.append(jnp.broadcast_to(marginal_dist.mean, bcast_shape))

        return jnp.concatenate(marginal_means, axis=-1)

    @lazy_property
    def variance(self):
        marginal_variances = []

        for marginal_slice, marginal_dist in zip(
            self.marginal_slices, self.marginal_dists
        ):
            slice_size = marginal_slice[1] - marginal_slice[0]
            bcast_shape = (*self.batch_shape, slice_size)
            marginal_variances.append(
                jnp.broadcast_to(marginal_dist.variance, bcast_shape)
            )

        return jnp.concatenate(marginal_variances, axis=-1)

    @property
    def support(self):
        base_constraints = [
            marginal_dist.support for marginal_dist in self.marginal_dists
        ]

        return _ConstraintCollection(base_constraints, self.marginal_slices)

    @lazy_property
    def correlation_matrix(self):
        return self.base_dist.covariance_matrix

    @lazy_property
    def correlation_cholesky(self):
        return self.base_dist.scale_tril
