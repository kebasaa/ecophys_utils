# Partitioning: Core partitioning functions
#---------------------------------
import os
import warnings
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

# Rule sets for the u* threshold detection. Exposed as a module constant so the settings
# are auditable and overridable rather than buried in the function body.
#   'reichstein': as published in Reichstein et al. (2005): 6 temperature classes, a u*
#                 class is the threshold when its NEE reaches 99 % of the mean NEE of all
#                 higher u* classes, correlation limit 0.4.
#   'reddyproc':  as in REddyProc (usControlUstarEst, usControlUstarSubsetting), the
#                 reference implementation used for FLUXNET: 7 temperature classes, a u*
#                 class and the next one must both reach 95 % of the mean NEE of the
#                 following 10 classes, at least 3 classes must follow, correlation limit
#                 0.5, temperature classes whose lowest u* class already averages more
#                 than 0.2 are skipped, and the threshold must be found in at least 20 % of
#                 the temperature classes.
USTAR_RULES = {
    'reichstein': {'filter_threshold': 0.99, 'n_Tair_classes': 6, 'n_plateau_bins': None,
                   'n_consecutive_bins': 1, 'min_plateau_bins': 1, 'corr_threshold': 0.4,
                   'first_uStar_bin_max': None, 'min_valid_Tair_class_fraction': 0.0},
    'reddyproc':  {'filter_threshold': 0.95, 'n_Tair_classes': 7, 'n_plateau_bins': 10,
                   'n_consecutive_bins': 2, 'min_plateau_bins': 3, 'corr_threshold': 0.5,
                   'first_uStar_bin_max': 0.2, 'min_valid_Tair_class_fraction': 0.2},
}

def calculate_uStar_threshold_reichstein(df: pd.DataFrame, Tair_col: str = 'TA_1_1_1', dn_col: str = 'day_night', uStar_col: str = 'u*', nee_col: str = 'nee', rules: str = 'reichstein', filter_threshold: Optional[float] = None, use_night_only: bool = True, min_uStar_threshold: float = 0.01, na_uStar_threshold: float = 0.4, threshold_if_none_found: bool = False, n_Tair_classes: Optional[int] = None, n_uStar_classes: int = 20, n_plateau_bins: Optional[int] = None, n_consecutive_bins: Optional[int] = None, min_plateau_bins: Optional[int] = None, corr_threshold: Optional[float] = None, first_uStar_bin_max: Optional[float] = None, min_valid_Tair_class_fraction: Optional[float] = None, min_records: Optional[int] = None, min_records_per_Tair_class: Optional[int] = None) -> float:
    """
    Calculate u* threshold using Reichstein et al. (2005) method.

    Night-time data are split into temperature classes, each class into u* classes of
    equal size. In every temperature class the threshold is the mean u* of the first u*
    class whose NEE reaches `filter_threshold` of the mean NEE of the u* classes above it.
    The threshold of the data set is the median across temperature classes.

    The settings come from the rule set `rules` (see USTAR_RULES); any of them given
    explicitly overrides the rule set. The 'reichstein' rules (default) are those of the
    paper, the 'reddyproc' rules those of REddyProc.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing temperature, day/night, u*, and NEE data, normally one season.
    Tair_col : str, optional
        Column name for air temperature. Default is 'TA_1_1_1'.
    dn_col : str, optional
        Column name for day/night indicator. Default is 'day_night'.
    uStar_col : str, optional
        Column name for u*. Default is 'u*'.
    nee_col : str, optional
        Column name for NEE. Default is 'nee'.
    rules : {'reichstein', 'reddyproc'}, optional
        Rule set providing the defaults of the settings below. Default 'reichstein'.
    filter_threshold : float or None, optional
        Fraction of the mean NEE of the higher u* classes that a class must reach
        ('reichstein': 0.99, 'reddyproc': 0.95).
    use_night_only : bool, optional
        Whether to use night-time data only. Default is True.
    min_uStar_threshold : float, optional
        Minimum u* threshold. Default is 0.01.
    na_uStar_threshold : float, optional
        Threshold used if none is found and `threshold_if_none_found` is True. Default 0.4.
    threshold_if_none_found : bool, optional
        Whether to insert `na_uStar_threshold` where no threshold is found. Default False.
    n_Tair_classes : int or None, optional
        Number of temperature classes ('reichstein': 6, 'reddyproc': 7).
    n_uStar_classes : int, optional
        Number of u* classes per temperature class. Default is 20.
    n_plateau_bins : int or None, optional
        Number of higher u* classes forming the plateau; None means all higher classes
        ('reichstein': None, 'reddyproc': 10).
    n_consecutive_bins : int or None, optional
        Number of consecutive u* classes that must reach the plateau, the first of which
        is the threshold ('reichstein': 1, 'reddyproc': 2).
    min_plateau_bins : int or None, optional
        Min. number of u* classes that must lie above the threshold class
        ('reichstein': 1, 'reddyproc': 3).
    corr_threshold : float or None, optional
        Temperature classes in which |r(Tair, u*)| reaches this value give no threshold,
        because NEE would then vary with temperature along u* ('reichstein': 0.4,
        'reddyproc': 0.5).
    first_uStar_bin_max : float or None, optional
        Temperature classes whose lowest u* class already averages more than this give no
        threshold, as the low-turbulence regime is not sampled. None disables the check
        ('reichstein': None, 'reddyproc': 0.2).
    min_valid_Tair_class_fraction : float or None, optional
        Min. fraction of temperature classes that must give a threshold
        ('reichstein': 0, 'reddyproc': 0.2).
    min_records : int or None, optional
        Min. number of valid records; fewer give NaN. Default None, no minimum
        (REddyProc: 160 per season).
    min_records_per_Tair_class : int or None, optional
        Min. number of valid records per temperature class; fewer give NaN. Default None,
        no minimum (REddyProc: 100).

    Returns
    -------
    float
        Calculated u* threshold.

    See Also
    --------
    uStar_filtering_wrapper : Apply u* filtering to data.
    """
    if rules not in USTAR_RULES:
        raise ValueError(f"rules must be one of {list(USTAR_RULES)}, got {rules!r}")
    # Settings not given explicitly come from the rule set
    preset = USTAR_RULES[rules]
    if filter_threshold is None: filter_threshold = preset['filter_threshold']
    if n_Tair_classes is None: n_Tair_classes = preset['n_Tair_classes']
    if n_plateau_bins is None: n_plateau_bins = preset['n_plateau_bins']
    if n_consecutive_bins is None: n_consecutive_bins = preset['n_consecutive_bins']
    if min_plateau_bins is None: min_plateau_bins = preset['min_plateau_bins']
    if corr_threshold is None: corr_threshold = preset['corr_threshold']
    if first_uStar_bin_max is None: first_uStar_bin_max = preset['first_uStar_bin_max']
    if min_valid_Tair_class_fraction is None: min_valid_Tair_class_fraction = preset['min_valid_Tair_class_fraction']
    # Every consecutive class needs at least one class above it to compare against
    min_plateau_bins = max(min_plateau_bins, n_consecutive_bins)

    def none_found():
        return na_uStar_threshold if threshold_if_none_found else np.nan

    # Valid records: temperature, u* and NEE all present
    temp = df.loc[df[[Tair_col, uStar_col, nee_col]].notna().all(axis=1)]

    # Optionally use only night-time data
    if use_night_only:
        temp = temp.loc[temp[dn_col] == 0]
    temp = temp[[Tair_col, uStar_col, nee_col]].copy()

    # If no or too few data after filtering -> return according to threshold_if_none_found
    if temp.shape[0] == 0:
        return none_found()
    if (min_records is not None) and (len(temp) < min_records):
        return none_found()
    if (min_records_per_Tair_class is not None) and (len(temp) / n_Tair_classes < min_records_per_Tair_class):
        return none_found()

    # Create temperature classes of equal size (quantiles)
    temp['Tair_class'] = pd.qcut(temp[Tair_col], n_Tair_classes, labels=False, duplicates='drop')

    # Helper function for group processing: the threshold of one temperature class
    def compute_uStar_threshold(group):
        # If temperature and u* are correlated, NEE changes along u* partly because of
        # temperature, and the plateau cannot be attributed to turbulence
        if len(group) < 2:
            return np.nan
        with np.errstate(invalid='ignore', divide='ignore'):
            r = np.corrcoef(group[Tair_col], group[uStar_col])[0, 1]
        if not np.isfinite(r) or abs(r) >= corr_threshold:
            return np.nan

        # u* classes of equal size, and their mean u* and NEE
        uStar_class = pd.qcut(group[uStar_col], n_uStar_classes, labels=False, duplicates='drop')
        bins = group.groupby(uStar_class)[[uStar_col, nee_col]].mean().sort_index()
        uStar_mean = bins[uStar_col].to_numpy()
        F = bins[nee_col].to_numpy()

        # The lowest u* class is already well mixed: the threshold is below the data
        if (first_uStar_bin_max is not None) and (uStar_mean[0] > first_uStar_bin_max):
            return np.nan

        # Mean NEE of the plateau above u* class j: all higher classes, or the next
        # n_plateau_bins of them
        def F_plateau(j):
            end = None if n_plateau_bins is None else j + 1 + n_plateau_bins
            return np.nanmean(F[j + 1:end])

        # Initialise
        current_threshold = np.nan
        # Search upwards for the first class that, together with the following
        # n_consecutive_bins - 1 classes, reaches the plateau. This stops at the lowest u*
        # that meets the criterion.
        for i in range(len(F) - min_plateau_bins):
            if all(F[i + k] >= filter_threshold * F_plateau(i + k) for k in range(n_consecutive_bins)):
                current_threshold = uStar_mean[i]
                break

        # In strict mode, you may optionally set a threshold if no value met the condition.
        if pd.isna(current_threshold) and threshold_if_none_found:
            current_threshold = na_uStar_threshold

        # Enforce the minimum allowable threshold.
        if not pd.isna(current_threshold):
            current_threshold = max(current_threshold, min_uStar_threshold)
        return current_threshold

    # Apply across Tair classes
    thresholds_by_Tair = np.array([compute_uStar_threshold(g) for _, g in temp.groupby('Tair_class')],
                                  dtype=float)

    # A threshold found in no, or only a few, temperature classes is not representative
    n_found = np.isfinite(thresholds_by_Tair).sum()
    if (n_found == 0) or (n_found < min_valid_Tair_class_fraction * len(thresholds_by_Tair)):
        return np.nan

    # Compute the final uStar threshold as the median of thresholds across classes.
    uStar_threshold = np.nanmedian(thresholds_by_Tair)

    return(uStar_threshold)

def create_seasonal_uStar_threshold_list(df: pd.DataFrame, groupby: List[str] = ['year', 'season'], Tair_col: str = 'TA_1_1_1', dn_col: str = 'day_night', uStar_col: str = 'u*', nee_col: str = 'nee', rules: str = 'reichstein', min_records: Optional[int] = None, min_records_per_Tair_class: Optional[int] = None) -> pd.DataFrame:
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
    rules : {'reichstein', 'reddyproc'}, optional
        Rule set of the threshold detection, see USTAR_RULES. Default 'reichstein'.
    min_records, min_records_per_Tair_class : int or None, optional
        Min. number of valid records per group and per temperature class. Default None,
        see calculate_uStar_threshold_reichstein().

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
                                             rules=rules,
                                             use_night_only=True,
                                             min_records=min_records,
                                             min_records_per_Tair_class=min_records_per_Tair_class)
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

def bootstrap_uStar_threshold(df: pd.DataFrame, groupby: List[str] = ['year', 'season'], n_boot: int = 200, percentiles: List[float] = [5, 50, 95], seed: int = 42, Tair_col: str = 'TA_1_1_1', dn_col: str = 'day_night', uStar_col: str = 'u*', nee_col: str = 'nee', rules: str = 'reichstein', min_records: Optional[int] = None, min_records_per_Tair_class: Optional[int] = None) -> pd.DataFrame:
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
    rules, min_records, min_records_per_Tair_class : optional
        As in create_seasonal_uStar_threshold_list().

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
                                         rules=rules,
                                         use_night_only=True,
                                         min_records=min_records,
                                         min_records_per_Tair_class=min_records_per_Tair_class))
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
    
def uStar_filtering_wrapper(temp: pd.DataFrame, timestamp_col: str = 'timestamp', hemisphere: str = 'north', apply_to_cols: List[str] = ['nee','co2_flux','co2_strg','h2o_flux','H','LE'], silent: bool = False, uStar_aggregation: str = 'season', uStar_rules: str = 'reichstein', min_records: Optional[int] = None, min_records_per_Tair_class: Optional[int] = None) -> pd.DataFrame:
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
    uStar_aggregation : {'season', 'overall'}, optional
        'season' (default): every season of every year is filtered with its own threshold;
        seasons without one use the mean across seasons. 'overall': one threshold, the
        mean across seasons, for the whole record. REddyProc offers both
        (aggregationMode 'season' and 'single'), and by default uses the maximum across
        the seasons of each year.
    uStar_rules : {'reichstein', 'reddyproc'}, optional
        Rule set of the threshold detection within a season: as published in Reichstein
        et al. (2005), the default, or as in REddyProc. See USTAR_RULES.
    min_records, min_records_per_Tair_class : int or None, optional
        Min. number of valid records per season and per temperature class for a seasonal
        threshold. Default None, see calculate_uStar_threshold_reichstein().

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame, with the '_f' columns and the applied 'uStar_threshold'.
    """
    if uStar_aggregation not in ('season', 'overall'):
        raise ValueError(f"uStar_aggregation must be 'season' or 'overall', got {uStar_aggregation!r}")
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
    thresholds_df  = create_seasonal_uStar_threshold_list(temp, groupby=['year', 'season'],
                                                          rules=uStar_rules,
                                                          min_records=min_records,
                                                          min_records_per_Tair_class=min_records_per_Tair_class)

    # Calculate overall threshold: the threshold of the record, and the fallback for
    # seasons without their own
    threshold_overall  = calculate_overall_uStar_threshold(thresholds_df, missing_fraction=0.75, use_mean=True)
    if uStar_aggregation == 'season':
        thresholds_df['uStar_threshold_applied'] = thresholds_df['uStar_threshold'].fillna(threshold_overall)
    else:
        thresholds_df['uStar_threshold_applied'] = threshold_overall
    if(not silent):
        print('  - u* thresholds, by season:')
        from IPython.display import display
        display(thresholds_df)
        print('  - Mean u* threshold across seasons:  ', threshold_overall)
        print('  - Applied:', 'per season (mean where a season has none)'
              if uStar_aggregation == 'season' else 'mean across seasons')

    if pd.isna(threshold_overall) and thresholds_df['uStar_threshold'].isna().all():
        warnings.warn('No u* threshold could be estimated for any season; no data are filtered.')

    # Look up the threshold of every record's season
    lookup = thresholds_df.set_index(['year', 'season'])['uStar_threshold_applied']
    temp['uStar_threshold'] = lookup.reindex(pd.MultiIndex.from_frame(temp[['year', 'season']])).to_numpy()

    # Apply thresholds, remove NEE, LE, H
    if(not silent):
        print('  - Applying to:', apply_to_cols)
    for col in apply_to_cols:
        temp[col + '_f'] = temp[col]
        temp.loc[(temp['u*'] <= temp['uStar_threshold']), col + '_f'] = np.nan

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