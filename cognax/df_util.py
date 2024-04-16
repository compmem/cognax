from itertools import product

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like


def map_cat_to_idx(cats):
    """
    Get a dictionary mapping categorical values to a unique index.

    Args:
        cats: an iterable filled with hashable values

    Returns:
        a dictionary mapping categorical values to a unique index
    """
    cat_to_idx = {}
    idx = 0
    for cat in cats:
        if cat_to_idx.get(cat) is None:
            cat_to_idx[cat] = idx
            idx += 1

    return cat_to_idx


def expand_df_to_arrs(df, expand_to, value_vars=None, cast_obj_to_float=True):
    """
    Args:
        df (pd.DataFrame)
        expand_to (dict): {column to expand df with: unique values the column can take on}
        value_vars (list_like): Column(s) to extract after expanding. If not specified, extracts all
            columns not included in `expand_to`. Defaults to None.
        cast_obj_to_float: Whether or not to try to cast object types to floats. Defaults to True.

    Returns:
        a tuple containing (expanded_vals, expanded_shape_names)

        expanded: a dictionary mapping each value_var to an array
        expanded_shape_names: a list of names for each expanded value dimension

    ```python
        >>> df = pd.DataFrame({'subj': ['subj1', 'subj1', 'subj2', 'subj2', 'subj2'],
        ...                    'trial': [0, 1, 0, 1, 2],
        ...                    'value': [10, 11, 20, 21, 22]})
        >>> df
            subj  trial  value
        0  subj1      0     10
        1  subj1      1     11
        2  subj2      0     20
        3  subj2      1     21
        4  subj2      2     22

        >>> mapper = {'subj': ['subj1', 'subj2'], 'trial': range(3)}
        >>> expanded_df, batch_cols = expand_df_to_arrs(df, mapper, 'value')
        >>> expanded_df['value'].shape
        (2, 3)
        >>> batch_cols
        ['subj', 'trial']
    ```
    """
    batch_cols, batch_vals = list(expand_to.keys()), expand_to.values()
    expand_shape = tuple(len(value) for value in batch_vals) + (-1,)
    df_columns = set(df.columns)

    # assert all(batch_col in df_columns for batch_col in batch_cols),
    # "one or more `expand_to` keys is an invalid column in `df`"

    if value_vars is None:
        value_vars = df_columns.difference(batch_cols)
    if not is_list_like(value_vars):
        value_vars = [value_vars]

    expand_df = pd.DataFrame(data=product(*batch_vals), columns=batch_cols)
    expanded = pd.merge(expand_df, df, on=batch_cols, how="left")
    expanded = expanded.sort_values(by=batch_cols, ascending=True)

    expanded = {
        var: expanded[var].to_numpy().reshape(expand_shape).squeeze(-1)
        for var in value_vars
    }

    if cast_obj_to_float:
        for var, value in expanded.items():
            if np.issubdtype(value.dtype, np.object_):
                try:
                    expanded[var] = value.astype(np.float32)
                except ValueError:
                    print(
                        f"Couldn't cast {var} to `np.float32`. Does this column contain strings?"
                    )

    return expanded, batch_cols


def contract_arrs_to_df(expanded, expand_to):
    """
    Inverse of expand_df_to_arrs.

    Args:
        expanded (dict): a dictionary mapping each value_var to an array
        expand_to (dict): {column to expand df with: unique values the column can take on}

    Returns:
        the original pd.DataFrame
    """
    batch_cols = expand_to.keys()
    expanded_df = pd.DataFrame(product(*expand_to.values()), columns=batch_cols)

    for col, value in expanded.items():
        expanded_df[col] = value.flatten()

    return expanded_df.dropna()


def all_not_nan(arrs):
    """
    Given a list/dictionary/tuple of arrays of the same shape, return a boolean array
    indicating whether all values are present (elementwise).

    Useful for masking `np.nan` values produced during `df_expand_batch`.
    """
    if isinstance(arrs, dict):
        arrs = list(arrs.values())

    # np.isnan() is only valid for float types, but we will probably have object types
    for i, arr in enumerate(arrs):
        if np.issubdtype(arr.dtype, np.object_):
            arr_shape = arr.shape
            arrs[i] = np.array(
                [
                    np.nan if isinstance(x, float) and np.isnan(x) else 0.0
                    for x in arr.flatten()
                ]
            ).reshape(arr_shape)

    return np.logical_and.reduce(~np.isnan(arrs), axis=0)
