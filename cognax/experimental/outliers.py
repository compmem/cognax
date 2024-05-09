# ported from RunDEMC: https://github.com/compmem/RunDEMC/blob/32dd11be22bee0fb5778b0dfdef52bc84312c55c/RunDEMC/density.py#L521
import scipy
import numpy as np


# Box Cox transformation utils
def boxcox(x, lambdax, shift=0.0):
    """
    Performs a box-cox transformation to data vector X.
    WARNING: elements of X should be all positive!
    """
    if np.any(x + shift <= 0):
        raise ValueError("Nonpositive value(s) in X vector")
    return (
        np.log(x + shift)
        if np.abs(lambdax) < 1.0e-5
        else ((x + shift) ** lambdax - 1.0) / lambdax
    )


def boxcox_loglike(x, lambdax, shift=0.0):
    """
    Computes the log-likelihood function for a transformed vector Xtransform.
    """
    n = len(x)
    try:
        xb = boxcox(x, lambdax, shift)
    except ValueError:
        # outside range
        return -np.inf
    S2 = (lambdax - 1.0) * np.log(x + shift).sum()
    S = np.sum((xb - xb.mean()) ** 2)
    S1 = (-n / 2.0) * np.log(S / n)
    return float(S2 + S1)


def best_boxcox_lambdax(x, lambdax=0, shift=None, verbose=False, min_shift=None):
    if shift is None:

        def to_min(lambdax, *args):
            # return the neg so maximize log like
            return -boxcox_loglike(x, lambdax)
    else:
        # picking a starting shift
        shift = np.max([shift, -x.min() + 1.0e-5])

        def to_min(lambdax, *args):
            # return the neg so maximize log like
            if min_shift is not None and lambdax[1] < min_shift:
                # this value is not allowed
                log_like = -np.inf
            else:
                # calc the log like
                log_like = boxcox_loglike(x, lambdax[0], shift=lambdax[1])
            return -log_like

    # run the minimization
    if verbose:
        disp = 1
    else:
        disp = 0
    if shift is None:
        # just optimize lambdax
        best_lambdax = scipy.optimize.fmin(to_min, [lambdax], disp=disp)
    else:
        # optimize both lambdax and the shift
        best_lambdax = scipy.optimize.fmin(to_min, [lambdax, shift], disp=disp)

    return best_lambdax


def find_good_ind(dat, sd=3.0, verbose=True, max_iter=100, shift_start=0.0):
    """Return good indices excluding outliers.

    Identifies outliers by ensuring normality with a Box--Cox transformation
    and applying upper and lower bounds based on standard deviation. It applies
    the same transformation and threshold repeatedly until no items are removed.

    Note, the Box--Cox transformation only works on positive values, but will shift
    values with a second paramter as needed.

    Parameters
    ----------
    dat : array of positive continuous data
        Array of positive input values.

    sd : float, optional
        Standard deviation threshold to be applied on each iteration.
        Default = 3.0

    verbose : boolean
        Whether to print out useful information.

    max_iter : int
        Maximum number of iterations to apply the normalization and
        thresholding process.
        Default = 100

    shift_start : float or None
        Whether to apply the shift in the Box--Cox transform. Set to None
        for no shift or to some starting value for the function minimization.
        NB: You should likely set this to None if you are culling based on
        reaction times b/c the shift option can sometimes give rise to removing
        trials on the front edge that should not be removed.
        Default = 0.0

    Returns
    -------
    ind : boolean index
        Boolean index of good indices (i.e., those that passed the
        iterative thresholding process). You can then apply this
        index to the data you passed in to cull the bad items.
    """
    # initialize the ind to all good
    ind = np.ones(len(dat), dtype=np.bool_)
    tot_removed = 0

    for i in range(max_iter):
        # convert data with BoxCox, making use of the shift
        if shift_start is None:
            shift = 0.0
            lambdax = best_boxcox_lambdax(dat[ind], shift=shift_start, verbose=True)
        else:
            lambdax, shift = best_boxcox_lambdax(
                dat[ind], shift=shift_start, verbose=True
            )
        bcdat = boxcox(dat[ind], lambdax=lambdax, shift=shift)

        # determine mean and sdt
        bmean = bcdat.mean()
        bstd = bcdat.std()

        # identify good range based on sd
        xmin = bmean - bstd * sd
        xmax = bmean + bstd * sd

        # pick good ind
        gind = (bcdat > xmin) & (bcdat < xmax)
        removed = len(gind) - gind.sum()
        if removed == 0:
            break

        # apply those removed (this is the index trick)
        ind[ind] = gind
        tot_removed += removed
        if verbose:
            print(
                "Removed %d out of %d, lambdax = %f, shift = %f"
                % (removed, len(ind), lambdax, shift)
            )

    # eventually add in better warning about hitting max
    if i == max_iter:
        print("Warning: Max iterations hit, which likely should not happen.")
    if verbose:
        print(
            "Removed %d in total out of %d in %d iterations."
            % (tot_removed, len(ind), i)
        )
        print("Final lambdax = %f, shift = %f" % (lambdax, shift))
    return ind
