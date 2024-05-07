import jax
import jax.numpy as jnp
from numpyro.distributions import constraints
from numpyro.distributions.constraints import Constraint
from numpyro.distributions.transforms import Transform, biject_to


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


constraint_collection = _ConstraintCollection


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
