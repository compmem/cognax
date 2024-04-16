import jax.numpy as jnp
import numpyro


def biased_threshold(bias, base_threshold, n_choices=2):
    """
    Bias a decision threshold to favor some choices over others.

    If there is `n_choices * base_threshold` total threshold to be distributed over all decisions,
    `(1 - bias_i)` describes the threshold proportion allocated to `choice_i`.

    Thus, `bias = jnp.ones(n_choices)/n_choices` describes an unbiased threshold.

    Args:
        bias: an `(n_choices,)` array lying on the unit simplex
        base_threshold: a scalar describing the decision threshold for each choice
            assuming choices are unbiased
        n_choices (int): the total number of choices

    Returns:
        `(n_choices,)` array containing decision thresholds for each choice
    """
    assert bias.shape == (n_choices,)
    return n_choices * base_threshold * (1 - bias)


def advantage_drift_rates(
    diff_weight, sum_weight, unscaled_base_drift, left_coherence, right_coherence
):
    """
    Advantage drift rates (2-choice) for a single subject with n trials.
    Records `base_drift` as a deterministic site.

    Args:
        diff_weight: (0, inf)
        sum_weight: (0, inf)
        unscaled_base_drift: scalar containing the amount of base drift to add
            after guaranteeing nonnegative drift rates. (0, inf)
        left_coherence: `(n_trials,)` array
        right_coherence: `(n_trials,)` array

    Returns:
        `(n_trials, 2)` array containing drift rates for each choice
    """
    strength_sum = sum_weight * (left_coherence + right_coherence)
    left_drift = diff_weight * (left_coherence - right_coherence) + strength_sum
    right_drift = diff_weight * (right_coherence - left_coherence) + strength_sum

    # guarantee drift rates to be nonnegative
    base_drift = jnp.abs(
        jnp.min(jnp.array([jnp.min(left_drift), jnp.min(right_drift), 0]))
    )
    base_drift = numpyro.deterministic("base_drift", base_drift + unscaled_base_drift)

    drift_rates = jnp.stack([left_drift + base_drift, right_drift + base_drift], axis=1)

    return drift_rates


def RCS(choice_rts, correct_responses, mask):
    """rate correct score (RCS): number of correct responses per second of effort

    Args:
        choice_rts: `(..., 2)` or `(..., 2)` array
            containing choice RTs.
        correct_responses: `(...,)` array containing the
            correct response at each trial.
        mask: `(...,)` bool array where True denotes
            that a trial should be included in the calculation.
    """
    choices, rts = choice_rts[..., 0], choice_rts[..., 1]

    n_correct = jnp.sum(jnp.where(mask, choices == correct_responses, 0))
    total_rt = jnp.sum(jnp.where(mask, rts, 0))

    return n_correct / total_rt
