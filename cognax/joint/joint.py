import numpy as np
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import biject_to


# XXX: this will be added to numpyro.distributions.utils in https://github.com/pyro-ppl/numpyro/pull/1779
def assert_one_of(**kwargs):
    """
    Assert that exactly one of the keyword arguments is not None.
    """
    specified = [key for key, value in kwargs.items() if value is not None]
    if len(specified) != 1:
        raise ValueError(
            f"Exactly one of {list(kwargs)} must be specified; got {specified}."
        )


def assert_valid_corr_indeces(cov_n_dim, indices):
    """
    Valid correlation indices are unique and only specify the lower triangular.
    """
    cov_prototype = np.zeros((cov_n_dim, cov_n_dim))
    for idx in indices:
        cov_prototype[idx] += 1

    assert np.all(cov_prototype[np.tril_indices(cov_n_dim, k=-1)] == 1)
    assert np.all(cov_prototype[np.triu_indices(cov_n_dim, k=1)] == 0)


def constrain_params(unconstrained_params, constraints_dict):
    """
    Constrain params, add associated factor term to model,
    and register constrained params as deterministic sites.

    This is a useful pattern when drawing correlated latent parameters with different supports
    and is a computationally cheaper alternative to `cognax.joint.GaussianCopula`.

    Ex:

    ```python
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from cognax.decisions import WFPT

    def model():
        joint_dist = dist.MultivariateNormal(jnp.zeros(4), jnp.eye(4))

        unconstrained_samples = numpyro.sample('unconstrained_joint', joint_dist)
        unconstrained_samples = {'v': pred_unconstrained[..., 0],
                                 'a': pred_unconstrained[..., 1],
                                 'w': pred_unconstrained[..., 2],
                                 't0': pred_unconstrained[..., 3]}

        constrained_params = constrain_params(unconstrained_samples, WFPT.arg_constraints)
        numpyro.sample("obs_choice_rt", WFPT(**constrained_params), obs=...)
    ```

    Args:
        unconstrained_params: dictionary mapping param names to unconstrained values
        constraints: dictionary mapping param names to constraints

    Returns:
        dictionary of constrained params
    """
    params = {}

    for name, unconstrained_value in unconstrained_params.items():
        transform = biject_to(constraints_dict[name])
        value = transform(unconstrained_value)
        params[name] = numpyro.deterministic(name, value)

        if numpyro.get_mask() is True:
            numpyro.factor(
                f"{name}_factor",
                -transform.log_abs_det_jacobian(unconstrained_value, value),
            )

    return params


def make_sample_structured_cov(cov_n_dim, corr_priors, indexers=None, masks=None):
    """
    Create a function to sample from a structured covariance matrix.

    Args:
        cov_n_dim (int): number of features in the covariance matrix
        corr_priors (list): list of prior Distributions on the correlation for each indexer/mask.
        indexers (list, optional): list of indexers that map each prior to coordinates in the covariance matrix.
        masks (list, optional): list of boolean arrays with shape `(cov_n_dim, cov_n_dim)` containing True if
            that element in `masks[i]` should have prior `corr_priors[i]`

    ```python
    import jax.numpy as jnp
    import numpyro.distributions as dist

    # --- with indexers ---

    cov_n_dim = 4
    indexers = [jnp.tril_indices(cov_n_dim, k=-2), jnp.diag_indices(cov_n_dim, k=-1)]
    corr_priors = [dist.Uniform(-1, 1), dist.Beta(1, 1)]

    sample_cov = make_sample_structured_cov(cov_n_dim, corr_priors, indexers=indexers)

    # --- with mask ---

    cov_prototype = jnp.ones((4, 4), dtype=jnp.bool_)
    masks = [jnp.tril(cov_prototype, k=-2), jnp.diag(cov_prototype, k=-1)]
    corr_priors = [dist.Uniform(-1, 1), dist.Beta(1, 1)]

    sample_cov = make_sample_structured_cov(4, corr_priors, masks=masks)
    ```

    Returns:
        Callable function to sample a structured covariance matrix
    """

    assert_one_of(indexers=indexers, masks=masks)

    if masks:
        assert all([mask.shape == (cov_n_dim, cov_n_dim) for mask in masks])
        indexers = [jnp.nonzero(mask) for mask in masks]

    assert len(corr_priors) == len(indexers)
    assert all([isinstance(prior, dist.Distribution) for prior in corr_priors])
    assert all(
        [
            jnp.broadcast_shapes(prior.batch_shape, (len(idx),))
            for prior, idx in zip(corr_priors, indexers)
        ]
    )
    assert_valid_corr_indeces(cov_n_dim, indexers)

    def sample_structured_cov(scale):
        """
        Sample from a structured covariance matrix.

        Args:
            scale: array-like of shape `(cov_n_dim,)` containing the variance
                for each variable
        """
        cov = scale[:, jnp.newaxis] * scale

        for i, (prior, idx) in enumerate(zip(corr_priors, indexers)):
            with numpyro.plate(f"_corr_plate_{i}", len(idx[0])):
                params = numpyro.sample(f"_corr_{i}", prior)
                cov = cov.at[idx].multiply(params, unique_indices=True)

        cov = cov.at[jnp.triu_indices_from(cov, k=1)].set(
            cov[jnp.tril_indices_from(cov, k=-1)]
        )

        return numpyro.deterministic("cov", cov)

    return sample_structured_cov
