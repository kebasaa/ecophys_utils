# Outlier removal functions
#--------------------------
import pandas as pd
import numpy as np
from typing import Tuple, Union

def remove_outliers_by_regression(df: pd.DataFrame, x_col: str, y_col: str, n_std: float = 2, ddof: int = 0, na_action: str = 'keep', silent: bool = False) -> Tuple[pd.DataFrame, Tuple[float, float]]:
    """
    Remove rows whose vertical residual from linear fit y ~ x is > n_std * std(resid).

    Parameters
    ----------
    df : pandas.DataFrame
    x_col, y_col : str
        Column names for x and y.
    n_std : float
        Threshold in standard deviations.
    ddof : int
        Degrees of freedom for std calculation.
    na_action : {'keep', 'drop_either', 'drop_both'}
        'keep'       : retain rows where either x or y is NaN (default).
        'drop_either': drop rows where either x or y is NaN (dropna how='any').
        'drop_both'  : drop rows where both x and y are NaN (dropna how='all').
    silent : bool
        If False, print summary statistics.

    Returns
    -------
    filtered_df : pandas.DataFrame
        DataFrame after removing outliers (and applying na_action).
    (slope, intercept) : tuple of floats
        Fitted slope and intercept (np.nan if fit not possible).
    """
    if na_action not in ('keep', 'drop_either', 'drop_both'):
        raise ValueError("na_action must be one of 'keep', 'drop_either', 'drop_both'")

    df = df.copy()
    # numeric conversions
    x_num = pd.to_numeric(df[x_col], errors='coerce')
    y_num = pd.to_numeric(df[y_col], errors='coerce')

    # indices that will be used for fitting (both numeric & non-NaN)
    paired_idx = df.index[x_num.notna() & y_num.notna()]
    total_pairs = len(paired_idx)

    # indices kept by the chosen na_action (before outlier removal)
    if na_action == 'keep':
        kept_by_na_action = df.index  # keep all rows
    elif na_action == 'drop_either':
        kept_by_na_action = df.dropna(subset=[x_col, y_col], how='any').index
    else:  # 'drop_both'
        kept_by_na_action = df.dropna(subset=[x_col, y_col], how='all').index

    # If not enough paired points to fit a line, no outlier-removal; just apply na_action
    if total_pairs < 2:
        filtered_df = df.loc[kept_by_na_action].copy()
        if(not silent):
            print(f"Not enough paired (non-NaN) rows to fit a line: {total_pairs} found.")
            print(f"Applied na_action='{na_action}'. Returned {len(filtered_df)} / {len(df)} rows.")
        return filtered_df, (np.nan, np.nan)

    # fit linear regression y = m*x + b on paired rows
    x = x_num.loc[paired_idx].values
    y = y_num.loc[paired_idx].values
    slope, intercept = np.polyfit(x, y, 1)

    # residuals aligned to paired_idx
    y_pred = slope * x_num.loc[paired_idx] + intercept
    resid = (y_num.loc[paired_idx] - y_pred)
    resid_std = resid.std(ddof=ddof)

    # degenerate case: no residual spread -> nothing removed (but na_action still applied)
    if resid_std == 0 or np.isnan(resid_std):
        filtered_df = df.loc[kept_by_na_action].copy()
        if(not silent):
            print("Residual standard deviation is zero/NaN (degenerate). No outlier removal performed.")
            print(f"Applied na_action='{na_action}'. Returned {len(filtered_df)} / {len(df)} rows.")
        return filtered_df, (slope, intercept)

    # determine inliers/outliers among paired rows
    inlier_mask = resid.abs() <= (n_std * resid_std)
    inlier_idx = paired_idx[inlier_mask]
    outlier_idx = paired_idx[~inlier_mask]

    # final kept indices: all inlier paired rows plus any rows that the na_action kept but were not paired
    non_paired_but_kept = set(kept_by_na_action) - set(paired_idx)
    final_kept_idx = set(inlier_idx) | non_paired_but_kept

    filtered_df = df.loc[sorted(final_kept_idx)].copy()

    # stats
    removed_outliers = len(outlier_idx)
    removed_due_to_na = len(df.index) - len(kept_by_na_action)
    total_removed = len(df) - len(filtered_df)
    pct_outliers = (removed_outliers / total_pairs * 100) if total_pairs > 0 else 0.0
    pct_total = (total_removed / len(df) * 100) if len(df) > 0 else 0.0

    if(not silent):
        print(f"Fitted line on {total_pairs} paired rows: slope={slope:.6g}, intercept={intercept:.6g}")
        print(f"Removed {removed_outliers} / {total_pairs} paired rows as outliers ({pct_outliers:.2f}%).")
        print(f"Removed {removed_due_to_na} rows where either column is NaN (na_action='{na_action}').")
        print(f"Total removed: {total_removed} rows out of {len(df)} ({pct_total:.2f}%).")

    return filtered_df, (slope, intercept)