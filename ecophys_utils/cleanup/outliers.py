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
    
def rolling_tod_median(df, col, timestamp_col='timestamp', window_days=14, max_missing=7):
    """
    Rolling median by time-of-day across previous N days, keeping diurnal trends
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a 'timestamp' column and the data column.
    col : str
        Name of column to compute rolling median on.
    window_days : int, default 14
        Number of days in rolling window.
    max_missing : int, default 0
        Maximum allowed NaNs inside the rolling window.
        
    Returns
    -------
    pandas.Series aligned to df.index
    """
    
    # Work on copy of required columns only
    data = df[[timestamp_col, col]].copy()
    data = data.sort_values(timestamp_col)

    # Extract date and time-of-day
    data['_date'] = data[timestamp_col].dt.date
    data['_tod'] = data[timestamp_col].dt.time

    def compute_group(group):
        group = group.sort_values('_date')

        # Rolling object on daily rows (one per day per ToD assumed)
        rolling_obj = group[col].rolling(window=window_days, min_periods=1)

        med = rolling_obj.median()
        count_valid = rolling_obj.count()

        # Total window size at each step
        window_size = rolling_obj.apply(lambda x: len(x), raw=False)

        # Missing values in window
        missing = window_size - count_valid

        med[missing > max_missing] = np.nan

        return med

    result = (
        data
        .groupby('_tod', group_keys=False)
        .apply(compute_group)
    )

    # Return aligned to original dataframe index
    return result.reindex(data.index).sort_index()

def rolling_tod_mad(
    df,
    col,
    window_days=14,
    max_missing=0,
    threshold=3,
    silent=False
):
    """
    Time-of-day stratified rolling MAD outlier filter, keeping diurnal trends

    This function computes a robust, median-based anomaly metric for a
    half-hourly (or generally sub-daily) time series by grouping observations
    by time-of-day and applying a rolling window across successive days.
    Within each time-of-day group, a rolling median and median absolute
    deviation (MAD) are calculated, and a robust z-score is derived as:

        z_robust = 0.67448975 * (x - median) / MAD

    where 0.67448975 equals Φ⁻¹(0.75), making MAD / 0.67448975 asymptotically
    equivalent to the standard deviation under normality.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at least:
        - 'timestamp' (datetime64-like)
        - the data column specified by `col`

    col : str
        Name of the column for which the rolling MAD-based filtering
        should be performed.

    window_days : int, default=14
        Number of consecutive days included in each rolling window.
        The window operates independently within each time-of-day group.

    max_missing : int, default=0
        Maximum number of missing values permitted within a rolling
        window. If the number of NaNs exceeds this threshold, the
        resulting statistic for that timestamp is set to NaN.

    threshold : float or None, default=3
        Robust z-score threshold for outlier removal
        If not None, observations with |z_robust| > threshold are set to NaN
        and the filtered data column is returned. (e.g., 3 ≈ 3σ equivalent,
        recommended for flux data). 
        If None, the robust z-score series is returned instead.

    Returns
    -------
    pandas.Series
        If threshold is not None:
            Filtered version of `col` with outliers replaced by NaN.
        If threshold is None:
            Robust z-score series aligned to the original index.

    Notes
    -----
    - The procedure preserves diurnal structure by restricting comparisons
      to identical times of day (e.g., all 11:00 observations are evaluated
      only against other 11:00 observations).
    - MAD provides a robust scale estimator that is substantially less
      sensitive to extreme values than the standard deviation.
    - Division-by-zero cases (MAD = 0) are handled by returning NaN.
    - Assumes one observation per day per time-of-day slot.
    """

    data = df[['timestamp', col]].copy()
    data = data.sort_values('timestamp')

    data['_date'] = data['timestamp'].dt.date
    data['_tod'] = data['timestamp'].dt.time

    CONST = 0.67448975  # Phi^-1(0.75)

    def compute_group(group):
        group = group.sort_values('_date')

        values = group[col]

        roll = values.rolling(window_days, min_periods=1)

        median = roll.median()

        # absolute deviation from rolling median
        abs_dev = (values - median).abs()

        # rolling MAD
        mad = abs_dev.rolling(window_days, min_periods=1).median()

        # count missing enforcement
        count_valid = roll.count()
        window_size = roll.apply(lambda x: len(x), raw=False)
        missing = window_size - count_valid

        median[missing > max_missing] = np.nan
        mad[missing > max_missing] = np.nan

        # robust z-score
        robust_z = CONST * (values - median) / mad
        robust_z[mad == 0] = np.nan

        return robust_z

    z = (
        data
        .groupby('_tod', group_keys=False)
        .apply(compute_group)
    )

    z = z.reindex(data.index).sort_index()

    # Return filtered data when threshold holds a value
    if threshold is not None:
        # Stats
        n = len(data.loc[(~data[col].isna())].index)
        n_bad = len(data.loc[z.abs() > threshold, col].index)
        if(not silent):
            print('    - Removing', str(np.round(n_bad/n*100, 2)) + '% flagged', col, 'data')
        # Set to nan
        data.loc[z.abs() > threshold, col] = np.nan
        return data[col]

    return z # Returns values of the difference when threshold is None