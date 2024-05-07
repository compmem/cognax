import numpyro
from numpyro.distributions.transforms import biject_to


def constrain_params(unconstrained_params, constraints_dict):
    """
    Constrain params, add associated factor term to model,
    and register constrained params as deterministic sites.

    This is a useful pattern when drawing correlated latent parameters with different supports
    and is a computationally cheaper alternative to `cognax.joint_modeling.GaussianCopula`.

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
