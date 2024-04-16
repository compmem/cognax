import numpy as np
import matplotlib.pyplot as plt


def rts_by_condition(rts, conditions):
    """
    Histogram of RTs disaggregated by each condition.

    Args:
        rts: `(n_trials,)` array containing rts
        conditions: `(n_trials,)` array describing the condition for each trial
    """
    unique_conditions = np.unique(conditions)

    fig, axs = plt.subplots(nrows=len(unique_conditions), figsize=(5, 10), sharex=True)

    for ax, cond in zip(axs, unique_conditions):
        ax.hist(rts[conditions == cond])
        ax.set_xlabel(f"cond: {cond}")

    return fig, axs


def rts_by_correct_condition(rts, correct, conditions):
    """
    Histogram of RTs disaggregated by each condition and whether or not the subject answered correctly.

    Args:
        rts: `(n_trials,)` array containing rts
        correct: `(n_trials,)` array denoting whether or not the choice was correct
        conditions: `(n_trials,)` array describing the condition for each trial
    """
    unique_conditions = np.unique(conditions)

    fig, axs = plt.subplots(
        nrows=len(unique_conditions), ncols=2, figsize=(5, 10), sharex=True, sharey=True
    )

    for ax, cond in zip(axs, unique_conditions):
        this_cond = conditions == cond
        correct_rts = rts[correct & this_cond]
        incorrect_rts = rts[~correct & this_cond]

        ax[0].hist(correct_rts)
        ax[1].hist(incorrect_rts)
        ax[0].set_xlabel(f"cond: {cond}")
        ax[1].set_xlabel(f"cond: {cond}")

    axs[0, 0].set_title("Correct")
    axs[0, 1].set_title("Incorrect")

    return fig, axs
