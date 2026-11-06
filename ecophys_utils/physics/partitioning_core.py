# Partitioning: Core partitioning functions
#---------------------------------
import os
import numpy as np
from typing import Union, List, Optional
import pandas as pd
from scipy import stats
from .partitioning_respiration import fit_E0, fit_E0_short_term, estimate_R_ref_moving_window_overlapping, interpolate_R_ref
from ..utils.data_processing import create_season_northern_hemisphere, create_season_southern_hemisphere

def calculate_nee(co2_flux: Union[float, np.ndarray], storage_flux: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate net ecosystem exchange (NEE).

    Parameters
    ----------
    co2_flux : float or numpy.ndarray
        CO2 flux.
    storage_flux : float or numpy.ndarray
        Storage flux.

    Returns
    -------
    float or numpy.ndarray
        Net ecosystem exchange.
    """
    nee = co2_flux + storage_flux
    return(nee)
    
def calculate_gpp(nee: Union[float, np.ndarray], reco: Union[float, np.ndarray],
                  day_night: Optional[Union[np.ndarray, pd.Series]] = None,
                  clip_negative: bool = True) -> Union[float, np.ndarray]:
    """
    Calculate gross primary production (GPP).

    Parameters
    ----------
    nee : float or numpy.ndarray
        Net ecosystem exchange.
    reco : float or numpy.ndarray
        Ecosystem respiration.
    day_night : numpy.ndarray or pandas.Series or None, optional
        Day (1) / night (0) indicator, e.g. the output of is_day(). When given, GPP is
        forced to exactly zero at night. PASS THIS: see the note below on why omitting it
        biases the result. Default is None, for backwards compatibility.
    clip_negative : bool, optional
        Set negative GPP to zero. Default is True.

    Returns
    -------
    float or numpy.ndarray
        Gross primary production, zero at night when day_night is supplied, and with
        negative values set to zero unless clip_negative is False.

    See Also
    --------
    calculate_nee : Calculate net ecosystem exchange.
    ecophys_utils.physics.partitioning_day_night.is_day : Produces the day_night column.
    ecophys_utils.physics.partitioning_daytime.partition_lasslop : Daytime partitioning,
        which applies the equivalent guard on PAR instead.

    Notes
    -----
    GPP is calculated as respiration minus NEE. Negative GPP values are set to zero.

    WHY day_night MATTERS. There is no photosynthesis in the dark, so GPP is zero at
    night by definition. Without the mask, night-time GPP evaluates to
    `reco_modelled - nee_measured`, which is a zero-centred residual rather than a flux -
    and `clip_negative` then rectifies it, keeping the positive half and discarding the
    negative half. Rectifying symmetric noise produces a strictly positive mean. At
    Jonkershoek that came to +0.91 umol m-2 s-1 over every night-time half-hour, or
    +162 to +177 g C m-2 yr-1 of GPP that never occurred. NEE and Reco are unaffected;
    the error lands entirely on GPP and on the Reco/GPP ratio derived from it.
    """
    gpp = reco - nee
    if day_night is not None:
        # Zero the night, but only where the inputs were present. Setting GPP to 0 for a
        # night half-hour with no NEE measurement would be defensible - GPP is zero at
        # night by definition, not by observation - but it would report data coverage
        # that does not exist. The annual sum is unaffected either way, because every
        # observed night value is 0, so keep the coverage statistics honest instead.
        gpp = np.where((np.asarray(day_night) == 0) & np.isfinite(gpp), 0.0, gpp)
    if clip_negative:
        gpp = np.where(gpp < 0, 0, gpp)
    return(gpp)

def calculate_uStar_threshold_reichstein(df: pd.DataFrame, Tair_col: str = 'TA_1_1_1', dn_col: str = 'day_night', uStar_col: str = 'u*', nee_col: str = 'nee', filter_threshold: float = 0.99, use_night_only: bool = True, min_uStar_threshold: float = 0.01, na_uStar_threshold: float = 0.4, threshold_if_none_found: bool = False) -> float:
    """
    Calculate u* threshold using Reichstein et al. (2005) method.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing temperature, day/night, u*, and NEE data.
    Tair_col : str, optional
        Column name for air temperature. Default is 'TA_1_1_1'.
    dn_col : str, optional
        Column name for day/night indicator. Default is 'day_night'.
    uStar_col : str, optional
        Column name for u*. Default is 'u*'.
    nee_col : str, optional
        Column name for NEE. Default is 'nee'.
    filter_threshold : float, optional
        Threshold percentage for NEE comparison. Default is 0.99.
    use_night_only : bool, optional
        Whether to use night-time data only. Default is True.
    min_uStar_threshold : float, optional
        Minimum u* threshold. Default is 0.01.
    na_uStar_threshold : float, optional
        Threshold if none found. Default is 0.4.
    threshold_if_none_found : bool, optional
        Whether to insert threshold if none found. Default is False.

    Returns
    -------
    float
        Calculated u* threshold.

    See Also
    --------
    uStar_filtering_wrapper : Apply u* filtering to data.
    """
    # Exclude rows where Tair is NA.
    mask = ~df[Tair_col].isna()
    temp = df.loc[mask].copy()

    # Optionally use only night-time data
    if use_night_only:
        temp = temp.loc[temp[dn_col] == 0].copy()
        
    # If no data after filtering -> return according to threshold_if_none_found
    if temp.shape[0] == 0:
        return na_uStar_threshold if threshold_if_none_found else np.nan

    # Create 6 temperature classes based on quantiles.
    temp['Tair_class'] = pd.qcut(temp[Tair_col], 6, labels=False)

    # Within each Tair class, create 20 uStar classes.
    # We use groupby on Tair_class together with a lambda to avoid an explicit loop.
    temp['uStar_class'] = temp.groupby('Tair_class')[uStar_col].transform(
        lambda x: pd.qcut(x, 20, labels=False, duplicates='drop')
    )

    # Helper function for group processing:
    # For each temperature class (group), perform the linear regression between Tair and uStar.
    # If the absolute r-value is below 0.4, calculate the threshold by comparing mean NEE in 
    # each uStar bin versus the combination of all higher uStar bins.
    def compute_uStar_threshold(group):
        # Run linear regression on the entire group.
        res = stats.linregress(group[Tair_col], group[uStar_col])
        # If the correlation is too strong, do not compute a threshold.
        if abs(res.rvalue) >= 0.4:
            return np.nan

        # Initialise
        current_threshold = np.nan
        # Get sorted unique uStar_class values.
        # We drop NaN in case 'duplicates' were dropped during qcut.
        sorted_uStar_class = sorted(group['uStar_class'].dropna().unique())

        # Loop over each uStar_class except the highest (because it has no "higher" classes).
        for current_uStar_class in sorted_uStar_class[:-1]:
            # Create a mask for the current and higher classes.
            mask_current = group['uStar_class'] == current_uStar_class
            mask_higher  = group['uStar_class'] > current_uStar_class

            # Calculate mean NEE for current and combined higher uStar classes.
            F_current = group.loc[mask_current, nee_col].mean()
            F_higher  = group.loc[mask_higher, nee_col].mean()

            # Check the filter condition; if true, take the mean uStar of the higher classes.
            if F_current >= filter_threshold * F_higher:
                current_threshold = group.loc[mask_current, uStar_col].mean()
                # This stops the for loop so that the lowest threshold that meets
                # the criterion is used.
                break
    
        # In strict mode, you may optionally set a threshold if no value met the condition.
        if pd.isna(current_threshold) and threshold_if_none_found:
            current_threshold = na_uStar_threshold
        if(current_threshold < min_uStar_threshold):
            current_threshold = min_uStar_threshold

        # Enforce the minimum allowable threshold.
        if not pd.isna(current_threshold):
            current_threshold = max(current_threshold, min_uStar_threshold)
        return current_threshold

    # Apply across Tair classes:
    # Use groupby to process each Tair_class.
    thresholds_by_Tair = temp.groupby('Tair_class').apply(compute_uStar_threshold)

    # Compute the final uStar threshold as the median of thresholds across classes.
    uStar_threshold = thresholds_by_Tair.median()

    return(uStar_threshold)
    
def create_seasonal_uStar_threshold_list(df: pd.DataFrame, groupby: List[str] = ['year', 'season'], Tair_col: str = 'TA_1_1_1', dn_col: str = 'day_night', uStar_col: str = 'u*', nee_col: str = 'nee') -> pd.DataFrame:
    """
    Create seasonal u* threshold list.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing data.
    groupby : list of str, optional
        Columns to group by. Default is ['year', 'season'].
    Tair_col : str, optional
        Column name for air temperature. Default is 'TA_1_1_1'.
    dn_col : str, optional
        Column name for day/night indicator. Default is 'day_night'.
    uStar_col : str, optional
        Column name for u*. Default is 'u*'.
    nee_col : str, optional
        Column name for NEE. Default is 'nee'.

    Returns
    -------
    pandas.DataFrame
        DataFrame with seasonal thresholds.
    """
    groups = df.groupby(groupby)
    grouped_thresholds = groups.apply(
        lambda group: calculate_uStar_threshold_reichstein(group,
                                             Tair_col=Tair_col,
                                             dn_col=dn_col,
                                             uStar_col=uStar_col,
                                             nee_col=nee_col,
                                             filter_threshold=0.99,
                                             use_night_only=True,
                                             min_uStar_threshold=0.01,
                                             na_uStar_threshold=0.4,
                                             threshold_if_none_found=False)
    )
    thresholds_df = grouped_thresholds.reset_index()

    # Count rows of data for each group, for relevant columns only
    # Take the maximum number of rows and then some % of those as a limit
    relevant_cols = [Tair_col, dn_col, uStar_col, nee_col]
    grouped_rowcounts = groups.apply(lambda grp: grp[relevant_cols].dropna().shape[0])\
                              .reset_index(name='non_na_row_count')
    
    # Create pandas dataframe
    thresholds_df.columns = groupby + ['uStar_threshold']
    thresholds_df = thresholds_df.merge(grouped_rowcounts, on=groupby)

    return(thresholds_df)

def bootstrap_uStar_threshold(df: pd.DataFrame, groupby: List[str] = ['year', 'season'], n_boot: int = 200, percentiles: List[float] = [5, 50, 95], seed: int = 42, Tair_col: str = 'TA_1_1_1', dn_col: str = 'day_night', uStar_col: str = 'u*', nee_col: str = 'nee') -> pd.DataFrame:
    """
    Bootstrap the seasonal u* threshold (Papale et al. 2006).

    The night-time records of each group are resampled with replacement and the threshold
    is re-estimated with the same settings as create_seasonal_uStar_threshold_list().

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing data.
    groupby : list of str, optional
        Columns to group by. Default is ['year', 'season'].
    n_boot : int, optional
        Number of bootstrap samples per group. Default is 200.
    percentiles : list of float, optional
        Percentiles of the bootstrap distribution to report. Default is [5, 50, 95].
    seed : int, optional
        Seed of the random number generator. Default is 42.
    Tair_col, dn_col, uStar_col, nee_col : str, optional
        Column names, as in create_seasonal_uStar_threshold_list().

    Returns
    -------
    pandas.DataFrame
        One row per group with columns 'uStar_p<percentile>' and 'n_valid_boot', the
        number of bootstrap samples that returned a threshold.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for key, group in df.groupby(groupby):
        night = group.loc[group[dn_col] == 0, [Tair_col, dn_col, uStar_col, nee_col]].dropna()
        values = []
        for _ in range(n_boot if len(night) > 0 else 0):
            sample = night.iloc[rng.integers(0, len(night), len(night))]
            values.append(calculate_uStar_threshold_reichstein(sample,
                                         Tair_col=Tair_col,
                                         dn_col=dn_col,
                                         uStar_col=uStar_col,
                                         nee_col=nee_col,
                                         filter_threshold=0.99,
                                         use_night_only=True,
                                         min_uStar_threshold=0.01,
                                         na_uStar_threshold=0.4,
                                         threshold_if_none_found=False))
        values = np.array(values, dtype=float)
        row = dict(zip(groupby, key if isinstance(key, tuple) else (key,)))
        for p in percentiles:
            row['uStar_p%g' % p] = np.nanpercentile(values, p) if np.isfinite(values).any() else np.nan
        row['n_valid_boot'] = int(np.isfinite(values).sum())
        rows.append(row)
    return(pd.DataFrame(rows))

def calculate_overall_uStar_threshold(thresholds_df: pd.DataFrame, missing_fraction: float = 1.0, use_mean: bool = True) -> float:
    """
    Calculate overall u* threshold from seasonal thresholds.

    Parameters
    ----------
    thresholds_df : pandas.DataFrame
        DataFrame with seasonal thresholds.
    missing_fraction : float, optional
        Fraction of max row count to filter. Default is 1.0.
    use_mean : bool, optional
        Whether to use mean or max. Default is True.

    Returns
    -------
    float
        Overall u* threshold.
    """
    max_rowcount = thresholds_df['non_na_row_count'].max()
    if(use_mean):
        threshold = thresholds_df.loc[thresholds_df['non_na_row_count'] > max_rowcount*missing_fraction,'uStar_threshold'].mean()
    else:
        threshold = thresholds_df.loc[thresholds_df['non_na_row_count'] > max_rowcount*missing_fraction,'uStar_threshold'].max()
    return(threshold)
    
def uStar_filtering_wrapper(temp: pd.DataFrame, timestamp_col: str = 'timestamp', hemisphere: str = 'north', apply_to_cols: List[str] = ['nee','co2_flux','co2_strg','h2o_flux','H','LE'], silent: bool = False) -> pd.DataFrame:
    """
    Wrapper for u* filtering.

    Parameters
    ----------
    temp : pandas.DataFrame
        DataFrame to filter.
    timestamp_col : str, optional
        Column name for timestamps. Default is 'timestamp'.
    hemisphere : str, optional
        Hemisphere ('north' or 'south'). Default is 'north'.
    apply_to_cols : list of str, optional
        Columns to apply filter to. Default is ['nee','co2_flux','co2_strg','h2o_flux','H','LE'].
    silent : bool, optional
        Whether to suppress output. Default is False.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame.
    """
    temp = temp.copy()
    if(not silent):
        print('Calculating u* filter threshold, following Reichstein et al. (2005):')

    # Add year variable
    temp['year']  = temp[timestamp_col].dt.year
    temp['month'] = temp[timestamp_col].dt.month
    # Create season
    if(hemisphere == 'north'):
        temp['season'] = create_season_northern_hemisphere(temp[timestamp_col])
    elif(hemisphere == 'south'):
        temp['season'] = create_season_southern_hemisphere(temp[timestamp_col])
    else:
        print('Error: Choose north or south as a hemisphere!')
        return

    # Calculate seasonal thresholds
    thresholds_df  = create_seasonal_uStar_threshold_list(temp, groupby=['year', 'season'])
    if(not silent):
        print('  - u* thresholds, by season:')
        from IPython.display import display
        display(thresholds_df)

    # Calculate overall threshold
    threshold_overall  = calculate_overall_uStar_threshold(thresholds_df, missing_fraction=0.75, use_mean=True)
    if(not silent):
        print('  - Final u* threshold:  ', threshold_overall)

    # Apply thresholds, remove NEE, LE, H
    if(not silent):
        print('  - Applying to:', apply_to_cols)
    for col in apply_to_cols:
        temp[col + '_f'] = temp[col]
        temp.loc[(temp['u*'] <= threshold_overall), col + '_f'] = np.nan

    if(not silent):
        print('Done applying u* filter...')
    return(temp)
    
def partitioning_reichstein_wrapper(temp: pd.DataFrame, timestamp_col: str = 'timestamp', dn_col: str = 'day_night', Tair_col: str = 'Tair', nee_col: str = 'nee_f', grouping_col: str = 'year', E0_method: str = 'short_term', window_days: int = 15, shift_days: int = 5) -> pd.DataFrame:
    """
    Wrapper for Reichstein partitioning method.

    Parameters
    ----------
    temp : pandas.DataFrame
        DataFrame with data.
    timestamp_col : str, optional
        Column name for timestamps. Default is 'timestamp'.
    dn_col : str, optional
        Column name for day/night indicator. Default is 'day_night'.
    Tair_col : str, optional
        Column name for air temperature. Default is 'Tair'.
    nee_col : str, optional
        Column name for u*-filtered NEE. Default is 'nee_f'.
    grouping_col : str, optional
        Column to group by; one E0 is estimated per group. Default is 'year'.
    E0_method : {'short_term', 'annual'}, optional
        'short_term' (default): E0 from short windows, as in Reichstein et al. (2005), see
        fit_E0_short_term(). 'annual': one fit over all night-time data of the group, see
        fit_E0(); this inflates E0 where R_ref follows the seasonal cycle of temperature.
    window_days, shift_days : int, optional
        Window length and shift (days) for R_ref. Default 15 and 5. REddyProc uses 7 and 4.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the timestamp, 'E0' and the interpolated 'R_ref', sorted by time.

    See Also
    --------
    fit_E0_short_term : Fit E0 in short-term windows.
    fit_E0 : Fit E0 over all data.
    estimate_R_ref_moving_window_overlapping : Estimate R_ref.
    interpolate_R_ref : Interpolate R_ref values.
    calculate_uStar_threshold_reichstein : Calculate u* threshold.
    """
    if E0_method not in ('short_term', 'annual'):
        raise ValueError(f"E0_method must be 'short_term' or 'annual', got {E0_method!r}")
    # interpolate_R_ref() returns its values in time order, so the data must be in time
    # order for them to land on the right rows
    temp = temp.sort_values(timestamp_col).reset_index(drop=True)

    # Step 1: Estimate E0 for each large (e.g. year) group
    # 1a) Define a wrapper function to use with .apply()
    def apply_fit_E0(group):
        if E0_method == 'short_term':
            E0_fit, _ = fit_E0_short_term(group, dn_col=dn_col, Tair_col=Tair_col,
                                          nee_col=nee_col, timestamp_col=timestamp_col)
        else:
            _, E0_fit = fit_E0(group, dn_col=dn_col, Tair_col=Tair_col, nee_col=nee_col)
        return pd.Series({
            'E0': E0_fit,
            'n_rows': len(group)
        })

    # 1b) Apply the function to each year group
    result_E0 = temp.groupby(grouping_col).apply(apply_fit_E0).reset_index()

    # 1c) Merge back into temp, under the name that estimate_R_ref reads below
    temp = temp.merge(result_E0[[grouping_col, 'E0']], on=grouping_col, how='left')

    # Step 2: Estimate R_ref for each smaller group in moving windows
    # 2a) Define the wrapper function
    def apply_estimate_R_ref(group):
        # Call the function with the appropriate columns and parameters
        R_ref_df = estimate_R_ref_moving_window_overlapping(
            group,
            dn_col=dn_col,
            Tair_col=Tair_col,
            nee_col=nee_col,
            timestamp_col=timestamp_col,
            E0_col='E0',
            window_days=window_days,
            shift_days=shift_days
        )
        R_ref_df[grouping_col] = group.name
        return(R_ref_df)

    # 2b) Group by 'year' and apply the wrapper function
    result_R_ref = temp.groupby(grouping_col).apply(apply_estimate_R_ref).reset_index(drop=True)

    # Step 3: Interpolate R_ref across the year (or other grouping variable)
    temp['R_ref'] = np.nan
    for year, group in temp.groupby(grouping_col):
        R_ref_group = result_R_ref.loc[result_R_ref[grouping_col] == year]
        temp.loc[group.index, 'R_ref'] = interpolate_R_ref(group, R_ref_group, timestamp_col=timestamp_col)

    return(temp[[timestamp_col, 'E0', 'R_ref']])