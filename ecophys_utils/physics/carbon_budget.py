# Carbon budget functions: defensible annual sums of NEE / GPP / Reco
#--------------------------------------------------------------------
#
# Annual carbon sums from eddy covariance data are dangerous to compute naively.
# u*-filtering removes ONLY night-time data, so the gaps left behind are not a
# random sample of the flux: they are almost entirely periods of positive NEE
# (respiration). Summing such a series while treating gaps as zero therefore does
# not add noise, it deletes one sign of the flux and inflates the apparent carbon
# gain, potentially by several t C ha-1 yr-1.
#
# This module implements the "mean diurnal course" (MDC) approach of
# Falge et al. (2001), Agric. For. Meteorol. 107:43-69, "Gap filling strategies for
# defensible annual sums of net ecosystem exchange". Observations are stratified by
# time of year and time of day, and each stratum ("bin") contributes its own mean,
# weighted by the number of days the bin represents. Night-time bins draw their mean
# from the surviving high-turbulence observations, which is precisely the estimate
# u*-filtering is designed to produce, so respiration enters the sum at full weight.
#
# NOTE ON THE ASSUMPTION: binned averaging is implicitly mean-imputation within a
# bin. It assumes missing half-hours resemble observed ones *within the same time of
# year and time of day*. Not producing gap-filled columns does not avoid that
# assumption; it scopes it narrowly and makes it testable. Compare against
# gapfill_mds() (see gapfilling.py) for an independent estimate that conditions on
# meteorology instead of on time.

import calendar
import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .energy_budget import total_annual_periods

# Molar mass of carbon (g mol-1). Fluxes are in umol m-2 s-1, so the conversion to
# g C m-2 per averaging period is: flux * 1e-6 * C_MOLAR_MASS * period_seconds
C_MOLAR_MASS = 12.011


def umol_m2_s1_to_g_m2(flux: np.ndarray, period_secs: float) -> np.ndarray:
    """
    Convert a CO2 flux to a carbon mass per averaging period.

    Parameters
    ----------
    flux : float or numpy.ndarray
        Flux (umol C m-2 s-1).
    period_secs : float
        Length of the averaging period (s).

    Returns
    -------
    float or numpy.ndarray
        Carbon mass per averaging period (g C m-2).
    """
    return(flux * 1e-6 * C_MOLAR_MASS * period_secs)


def _check_regular_grid(temp: pd.DataFrame, timestamp_col: str, period_mins: int) -> None:
    """
    Warn if timestamps are not on a regular grid.

    Bin occupancy statistics ('n_expected', 'coverage.perc') count rows, so they are
    only meaningful when every averaging period is present as a row (including the
    ones that are entirely NaN). Run complete_timestamps() first.
    """
    diffs = temp[timestamp_col].diff().dropna().unique()
    expected = pd.Timedelta(minutes=period_mins)
    if len(diffs) != 1 or diffs[0] != expected:
        warnings.warn(
            f"Timestamps are not on a regular {period_mins}-minute grid. Bin coverage "
            "statistics will be wrong. Run complete_timestamps() before binning."
        )


def _assign_time_bins(temp: pd.DataFrame, timestamp_col: str, time_bin: str,
                      period_mins: int = 30) -> pd.DataFrame:
    """
    Add the two stratification columns used by all MDC functions.

    Adds
    ----
    bin_time : int
        Time-of-year bin. For time_bin='month' this is the month (1-12). For
        time_bin='halfmonth' it is a sortable code (month*2-1 for days 1-15,
        month*2 for the rest), i.e. 1-24.
    bin_hh : int
        Time-of-day bin, one per averaging period: 0-47 for half-hourly data
        (0 = 00:00, 47 = 23:30).
    """
    temp = temp.copy()
    month = temp[timestamp_col].dt.month
    day = temp[timestamp_col].dt.day

    if time_bin == 'month':
        temp['bin_time'] = month
    elif time_bin == 'halfmonth':
        temp['bin_time'] = np.where(day <= 15, month * 2 - 1, month * 2)
    else:
        raise ValueError(f"time_bin must be 'month' or 'halfmonth', got {time_bin!r}")

    temp['bin_hh'] = (temp[timestamp_col].dt.hour * 60 + temp[timestamp_col].dt.minute) // period_mins
    return(temp)


def _bin_n_days(bin_time: int, time_bin: str, reference_year: int) -> int:
    """
    Number of days a time-of-year bin represents in the composite year.

    The composite year is built on `reference_year`, so choosing a non-leap
    reference year gives a 365-day annual sum and a leap year gives 366.
    """
    if time_bin == 'month':
        return(calendar.monthrange(reference_year, bin_time)[1])
    # halfmonth: odd codes are days 1-15, even codes are the remainder of the month
    month = (bin_time + 1) // 2
    month_len = calendar.monthrange(reference_year, month)[1]
    return(15 if bin_time % 2 == 1 else month_len - 15)


def mdc_annual_sum(temp: pd.DataFrame, flux_col: str = 'nee_f', timestamp_col: str = 'timestamp',
                   group_cols: Sequence[str] = (), time_bin: str = 'month',
                   min_n: int = 5, period_mins: int = 30, reference_year: int = 2021,
                   dn_col: Optional[str] = 'day_night',
                   fallback_reco_col: Optional[str] = None) -> pd.DataFrame:
    """
    Stratify a flux into mean-diurnal-course bins and convert each bin to carbon mass.

    Implements Falge et al. (2001). Observations are binned by
    `group_cols x time-of-year x time-of-day`; each bin contributes
    `mean * n_days * period * molar mass`, so no gap-filled half-hourly series is
    produced and gaps are never treated as zero.

    Parameters
    ----------
    temp : pandas.DataFrame
        Half-hourly data on a regular, complete timestamp grid (see complete_timestamps).
    flux_col : str, optional
        Column to aggregate, in umol m-2 s-1. Default is 'nee_f'.
    timestamp_col : str, optional
        Timestamp column. Default is 'timestamp'.
    group_cols : sequence of str, optional
        Extra grouping columns, applied on top of the time bins. Default is () (one
        composite year from the whole record). Use ('year',) for a per-calendar-year
        budget, or e.g. ('site',) or ('site', 'period') to keep blocks apart.
        Every group gets the full set of time bins, so a group covering only part of the
        year has empty bins and therefore a NaN total.
    time_bin : {'month', 'halfmonth'}, optional
        Width of the time-of-year stratum. Default is 'month'. Narrower bins track
        seasonality more closely but hold fewer observations; re-running with
        'halfmonth' is the natural sensitivity test.
    min_n : int, optional
        Bins with fewer than this many observations are flagged as sparse. Default is 5.
    period_mins : int, optional
        Averaging period in minutes. Default is 30.
    reference_year : int, optional
        Year defining the composite year's month lengths. Default is 2021 (non-leap,
        so the composite year is 365 days).
    dn_col : str or None, optional
        Day/night indicator (1 = day, 0 = night), used to classify bins and to route
        the sparse-bin fallback. Default is 'day_night'.
    fallback_reco_col : str or None, optional
        Modelled respiration column (e.g. 'reco_umol_m2_s1' from Lloyd-Taylor, which is
        gap-free because it is driven by air temperature rather than by NEE). If given,
        sparse *night* bins take their mean from this column, since at night NEE == Reco.
        Sparse *day* bins are interpolated across adjacent time-of-year bins at the same
        time of day; where that finds nothing, the sparse bin keeps its own mean.
        Default is None (sparse bins are flagged but left as-is).

    Returns
    -------
    pandas.DataFrame
        One row per bin, including bins with no rows at all, with 'mean', 'sd', 'n',
        'n_expected', 'coverage.perc', 'se', 'n_days', 'carbon.g_m2', 'carbon_se.g_m2',
        'is_night', 'sparse' and 'fill_method' ('measured', 'reco_model',
        'interpolated', or 'none' for bins without an estimate).

    See Also
    --------
    mdc_total : Reduce the bins to an annual total per group.
    mdc_cumulative : Expand the bins into a cumulative within-year series for plotting.
    """
    group_cols = list(group_cols)
    period_secs = period_mins * 60.0

    _check_regular_grid(temp, timestamp_col, period_mins)
    temp = _assign_time_bins(temp, timestamp_col, time_bin, period_mins)

    bin_cols = group_cols + ['bin_time', 'bin_hh']

    agg = {flux_col: ['mean', 'std', 'count', 'size']}
    if dn_col is not None and dn_col in temp.columns:
        agg[dn_col] = ['mean']
    if fallback_reco_col is not None and fallback_reco_col in temp.columns:
        agg[fallback_reco_col] = ['mean', 'std', 'count']

    bins = temp.groupby(bin_cols, observed=True, dropna=False).agg(agg)
    bins.columns = ['_'.join(col).strip('_') for col in bins.columns.values]
    bins = bins.reset_index()

    bins = bins.rename(columns={
        f'{flux_col}_mean': 'mean',
        f'{flux_col}_std': 'sd',
        f'{flux_col}_count': 'n',
        f'{flux_col}_size': 'n_expected',
    })

    # Every expected bin must exist. Otherwise a month with no rows at all would simply be
    # missing from the table, and from every total built on it, without being counted as
    # empty.
    n_time = 12 if time_bin == 'month' else 24
    full = pd.MultiIndex.from_product([range(1, n_time + 1), range(int(24 * 60 / period_mins))],
                                      names=['bin_time', 'bin_hh']).to_frame(index=False)
    if group_cols:
        full = bins[group_cols].drop_duplicates().merge(full, how='cross')
    bins = full.merge(bins, on=bin_cols, how='left')
    bins['n'] = bins['n'].fillna(0).astype(int)
    bins['n_expected'] = bins['n_expected'].fillna(0).astype(int)

    # A bin is a night bin if the majority of its half-hours are night-time. Bins that
    # straddle sunrise/sunset are classified by the majority; the fallback below is only
    # applied to sparse bins, so a marginal classification has limited consequences.
    if f'{dn_col}_mean' in bins.columns:
        bins['is_night'] = (bins[f'{dn_col}_mean'] < 0.5).where(bins[f'{dn_col}_mean'].notna())
        bins = bins.drop(columns=[f'{dn_col}_mean'])
    else:
        bins['is_night'] = np.nan

    bins['coverage.perc'] = 100.0 * bins['n'] / bins['n_expected']
    bins['sparse'] = bins['n'] < min_n
    bins['fill_method'] = 'measured'

    # ---- sparse-bin fallback -------------------------------------------------------
    # Never zero-fill: that is the bug this module exists to prevent. Sparse bins are
    # either filled from an independent model or flagged and left as NaN.
    if fallback_reco_col is not None and f'{fallback_reco_col}_mean' in bins.columns:
        # Night: NEE == Reco, so the (gap-free) modelled respiration is a direct estimate
        use_reco = bins['sparse'] & (bins['is_night'] == True) & bins[f'{fallback_reco_col}_mean'].notna()
        bins.loc[use_reco, 'mean'] = bins.loc[use_reco, f'{fallback_reco_col}_mean']
        bins.loc[use_reco, 'sd'] = bins.loc[use_reco, f'{fallback_reco_col}_std']
        bins.loc[use_reco, 'n'] = bins.loc[use_reco, f'{fallback_reco_col}_count']
        bins.loc[use_reco, 'fill_method'] = 'reco_model'

        # Day: interpolate along the seasonal axis at fixed time of day
        bins = bins.sort_values(group_cols + ['bin_hh', 'bin_time'])
        needs_interp = bins['sparse'] & (bins['fill_method'] == 'measured')
        if needs_interp.any():
            measured_mean = bins['mean'].copy()
            bins.loc[needs_interp, 'mean'] = np.nan
            interpolated = (
                bins.groupby(group_cols + ['bin_hh'], observed=True, dropna=False)['mean']
                    .transform(lambda s: s.interpolate(limit_direction='both'))
            )
            filled = needs_interp & interpolated.notna()
            bins.loc[filled, 'mean'] = interpolated[filled]
            bins.loc[filled, 'fill_method'] = 'interpolated'
            # Where there was nothing to interpolate from, a sparse mean beats none
            keep = needs_interp & ~filled
            bins.loc[keep, 'mean'] = measured_mean[keep]

    bins = bins.drop(columns=[c for c in bins.columns
                              if fallback_reco_col is not None and c.startswith(f'{fallback_reco_col}_')])

    still_empty = bins['mean'].isna()
    bins.loc[still_empty, 'fill_method'] = 'none'
    if still_empty.any():
        warnings.warn(
            f"{int(still_empty.sum())} of {len(bins)} bins have no estimate, so any total "
            "that includes them is NaN (never a partial sum). Widen time_bin, lower "
            "min_n, or supply fallback_reco_col."
        )

    # ---- convert to carbon mass ----------------------------------------------------
    bins['n_days'] = bins['bin_time'].apply(lambda b: _bin_n_days(int(b), time_bin, reference_year))
    # Standard error of the bin mean. Bins sampled many times are trusted more; this is
    # what makes the uncertainty respond to coverage rather than to record length.
    bins['se'] = bins['sd'] / np.sqrt(bins['n'].where(bins['n'] > 0))
    # The same estimated mean is reused for every day in the bin, so its error is fully
    # correlated across those days: the bin total scales linearly, not as sqrt(n_days).
    bins['carbon.g_m2'] = umol_m2_s1_to_g_m2(bins['mean'], period_secs) * bins['n_days']
    bins['carbon_se.g_m2'] = umol_m2_s1_to_g_m2(bins['se'], period_secs) * bins['n_days']

    ordered = group_cols + ['bin_time', 'bin_hh', 'mean', 'sd', 'se', 'n', 'n_expected',
                            'coverage.perc', 'n_days', 'carbon.g_m2', 'carbon_se.g_m2',
                            'is_night', 'sparse', 'fill_method']
    bins = bins[[c for c in ordered if c in bins.columns]]
    return(bins.sort_values(group_cols + ['bin_time', 'bin_hh']).reset_index(drop=True))


def mdc_total(bins: pd.DataFrame, group_cols: Sequence[str] = ()) -> pd.DataFrame:
    """
    Reduce MDC bins to an annual total per group.

    Bin errors are treated as independent of each other and combined in quadrature.
    Within a bin the error is already accumulated linearly by mdc_annual_sum, because
    one estimated mean is reused for every day the bin represents.

    A group with any empty bin has a NaN total: summing only the bins that have data would
    quietly leave out part of the year. That partial sum is still reported, under its own
    name, as 'carbon_available.g_m2'.

    Parameters
    ----------
    bins : pandas.DataFrame
        Output of mdc_annual_sum().
    group_cols : sequence of str, optional
        Grouping columns used when the bins were built. Default is ().

    Returns
    -------
    pandas.DataFrame
        One row per group: 'carbon.g_m2', 'carbon_available.g_m2', 'carbon_se.g_m2',
        'carbon.t_ha' (= g m-2 / 100), 'carbon_available.t_ha', 'carbon_se.t_ha',
        'n_bins', 'n_sparse_bins', 'n_empty_bins' and 'mean_coverage.perc'.
    """
    group_cols = list(group_cols)
    no_groups = not group_cols
    if no_groups:
        bins = bins.assign(_all=0)
        group_cols = ['_all']

    grouped = bins.groupby(group_cols, observed=True, dropna=False)
    available = grouped['carbon.g_m2'].sum(min_count=1)
    n_empty = grouped['carbon.g_m2'].apply(lambda s: int(s.isna().sum()))
    out = pd.DataFrame({
        'carbon.g_m2': available.where(n_empty == 0),
        'carbon_available.g_m2': available,
        'carbon_se.g_m2': np.sqrt(grouped['carbon_se.g_m2'].apply(lambda s: np.nansum(s.values ** 2))),
        'n_bins': grouped.size(),
        'n_sparse_bins': grouped['sparse'].sum(),
        'n_empty_bins': n_empty,
        'mean_coverage.perc': grouped['coverage.perc'].mean(),
    }).reset_index()
    for col in ('n_bins', 'n_sparse_bins', 'n_empty_bins'):
        out[col] = out[col].astype(int)

    # 1 g m-2 == 0.01 t ha-1
    out['carbon.t_ha'] = out['carbon.g_m2'] * 0.01
    out['carbon_available.t_ha'] = out['carbon_available.g_m2'] * 0.01
    out['carbon_se.t_ha'] = out['carbon_se.g_m2'] * 0.01
    if no_groups:
        out = out.drop(columns=['_all'])
    return(out)


def mdc_cumulative(bins: pd.DataFrame, group_cols: Sequence[str] = (),
                   time_bin: str = 'month', reference_year: int = 2021,
                   period_mins: int = 30) -> pd.DataFrame:
    """
    Expand MDC bins into a cumulative within-year carbon series for plotting.

    Builds a complete composite year at the native averaging period, assigns each
    timestamp the mean of its bin, and accumulates. The resulting curve retains the
    diurnal sawtooth within each time-of-year bin and steps at bin boundaries.

    The uncertainty band is evaluated at bin-completion granularity and interpolated in
    between: within a time-of-year bin the same estimated means are reused day after day
    (so the error grows linearly), while separate bins are combined in quadrature.

    Parameters
    ----------
    bins : pandas.DataFrame
        Output of mdc_annual_sum().
    group_cols : sequence of str, optional
        Grouping columns used when the bins were built. Default is ().
    time_bin : {'month', 'halfmonth'}, optional
        Must match what was passed to mdc_annual_sum(). Default is 'month'.
    reference_year : int, optional
        Year used to lay out the composite timeline. Default is 2021.
    period_mins : int, optional
        Averaging period in minutes. Default is 30.

    Returns
    -------
    pandas.DataFrame
        Columns: group_cols, 'timestamp', 'carbon.g_m2' (per period),
        'cum_carbon.g_m2', 'cum_carbon_se.g_m2', 'cum_carbon.t_ha', 'cum_carbon_se.t_ha'.
        The cumulative sum is NaN from the first empty bin onwards.
    """
    group_cols = list(group_cols)
    no_groups = not group_cols
    if no_groups:
        bins = bins.assign(_all=0)
        group_cols = ['_all']
    period_secs = period_mins * 60.0

    idx = pd.date_range(start=f'{reference_year}-01-01 00:00',
                        end=f'{reference_year}-12-31 23:59',
                        freq=f'{period_mins}min')
    skeleton = pd.DataFrame({'timestamp': idx})
    skeleton = _assign_time_bins(skeleton, 'timestamp', time_bin, period_mins)

    groups = bins[group_cols].drop_duplicates()
    skeleton = skeleton.merge(groups, how='cross')

    out = skeleton.merge(bins[group_cols + ['bin_time', 'bin_hh', 'mean']],
                         on=group_cols + ['bin_time', 'bin_hh'], how='left')
    out['carbon.g_m2'] = umol_m2_s1_to_g_m2(out['mean'], period_secs)
    out = out.sort_values(group_cols + ['timestamp'])

    # cumsum() steps over NaN, which would treat an empty bin as zero; stop the curve instead
    out['cum_carbon.g_m2'] = out.groupby(group_cols, observed=True)['carbon.g_m2'].cumsum()
    after_gap = (out.assign(_na=out['carbon.g_m2'].isna().astype(int))
                    .groupby(group_cols, observed=True)['_na'].cummax().astype(bool))
    out.loc[after_gap, 'cum_carbon.g_m2'] = np.nan

    # Uncertainty at bin-completion granularity, then interpolated onto the timeline
    bin_var = (bins.groupby(group_cols + ['bin_time'], observed=True, dropna=False)['carbon_se.g_m2']
                   .apply(lambda s: np.nansum(s.values ** 2)).reset_index(name='bin_var'))
    bin_var['cum_var'] = bin_var.groupby(group_cols, observed=True)['bin_var'].cumsum()
    bin_var['cum_carbon_se.g_m2'] = np.sqrt(bin_var['cum_var'])

    out = out.merge(bin_var[group_cols + ['bin_time', 'cum_carbon_se.g_m2']],
                    on=group_cols + ['bin_time'], how='left')
    out['cum_carbon_se.g_m2'] = (
        out.groupby(group_cols, observed=True)['cum_carbon_se.g_m2']
           .transform(lambda s: s.interpolate(limit_direction='both'))
    )

    out['cum_carbon.t_ha'] = out['cum_carbon.g_m2'] * 0.01
    out['cum_carbon_se.t_ha'] = out['cum_carbon_se.g_m2'] * 0.01

    keep = ([] if no_groups else group_cols) + ['timestamp', 'carbon.g_m2', 'cum_carbon.g_m2',
                                                'cum_carbon_se.g_m2', 'cum_carbon.t_ha', 'cum_carbon_se.t_ha']
    return(out[keep].reset_index(drop=True))


def _annual_flux_sum(group: pd.DataFrame, flux_col: str, method: str, timestamp_col: str,
                     time_bin: str, min_n: int, period_mins: int, dn_col: Optional[str],
                     fallback_reco_col: Optional[str], sd_col: Optional[str] = None
                     ) -> Tuple[float, float, float, int, int]:
    """
    Sum one flux over one year by the requested method.

    Returns (carbon_g_m2, carbon_se_random, carbon_se_systematic, n_sparse_bins,
    n_empty_bins). For method='mds' the series is expected to be complete, so the sum is
    direct; any residual NaN is reported as an empty count. Either way, the total is NaN
    whenever something is empty, rather than a sum that silently treats it as zero.

    The two error terms bracket the truth. The random term combines per-point errors in
    quadrature, which is right if they are independent; the systematic term adds them
    linearly, which is right if they are perfectly correlated. Half-hourly flux errors
    are autocorrelated, so the honest annual uncertainty lies between the two.
    """
    period_secs = period_mins * 60.0

    if method == 'mds':
        n_missing = int(group[flux_col].isna().sum())
        if n_missing:
            warnings.warn(
                f"{n_missing} missing values remain in '{flux_col}', so its annual total is "
                "NaN. Gap-fill first (gapfill_mds) or use method='mdc'."
            )
        total = np.nan if n_missing else umol_m2_s1_to_g_m2(group[flux_col].sum(min_count=1), period_secs)
        se_rand = se_syst = np.nan
        if sd_col is not None and sd_col in group.columns:
            sd = group[sd_col].values
            se_rand = umol_m2_s1_to_g_m2(np.sqrt(np.nansum(sd ** 2)), period_secs)
            se_syst = umol_m2_s1_to_g_m2(np.nansum(np.abs(sd)), period_secs)
        return(total, se_rand, se_syst, 0, n_missing)

    year_bins = mdc_annual_sum(
        group, flux_col=flux_col, timestamp_col=timestamp_col, group_cols=(),
        time_bin=time_bin, min_n=min_n, period_mins=period_mins,
        reference_year=int(group[timestamp_col].dt.year.iloc[0]),
        dn_col=dn_col, fallback_reco_col=fallback_reco_col,
    )
    n_empty = int(year_bins['carbon.g_m2'].isna().sum())
    total = np.nan if n_empty else year_bins['carbon.g_m2'].sum(min_count=1)
    se_rand = np.sqrt(np.nansum(year_bins['carbon_se.g_m2'].values ** 2))
    se_syst = np.nansum(np.abs(year_bins['carbon_se.g_m2'].values))
    return(total, se_rand, se_syst, int(year_bins['sparse'].sum()), n_empty)


def annual_carbon_budget(temp: pd.DataFrame, method: str = 'mdc', nee_col: str = 'nee_f',
                         reco_col: str = 'reco_umol_m2_s1', timestamp_col: str = 'timestamp',
                         year_col: str = 'year', dn_col: str = 'day_night',
                         time_bin: str = 'month', min_n: int = 5, period_mins: int = 30,
                         fallback_reco_col: Optional[str] = None,
                         sd_col: Optional[str] = None, qc_col: Optional[str] = None,
                         measured_col: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate annual carbon budget statistics per calendar year.

    Deliberately mirrors annual_energy_budget() in shape, so the two tables read the same
    way and the 'mdc' and 'mds' variants can be diffed row for row.

    NEE and Reco are summed by the same method so that GPP = Reco - NEE stays internally
    consistent.

    IMPORTANT: annual_energy_budget() scales its residual by overall data coverage
    (res_sum * nb_halfhours / n_obs). That idiom must NOT be reused here. It assumes gaps
    are distributed like the observed data, which is false for u*-filtered CO2 fluxes
    because the gaps are overwhelmingly night-time. Day/night-stratified binning is the
    fix; a single global scalar is not.

    Parameters
    ----------
    temp : pandas.DataFrame
        Half-hourly data on a regular, complete timestamp grid.
    method : {'mdc', 'mds'}, optional
        'mdc' bins the measured data (Falge et al. 2001) and never gap-fills.
        'mds' sums an already gap-filled series (see gapfill_mds); pass the filled column
        names via nee_col/reco_col. Default is 'mdc'.
    nee_col, reco_col : str, optional
        NEE and modelled respiration columns, in umol m-2 s-1.
    timestamp_col, year_col, dn_col : str, optional
        Timestamp, calendar year and day/night columns.
    time_bin : {'month', 'halfmonth'}, optional
        Time-of-year stratum width for method='mdc'. Default is 'month'.
    min_n : int, optional
        Sparse-bin threshold for method='mdc'. Default is 5.
    period_mins : int, optional
        Averaging period in minutes. Default is 30.
    fallback_reco_col : str or None, optional
        Passed through to mdc_annual_sum() for sparse night bins.
    sd_col : str or None, optional
        Per-point uncertainty column for method='mds' (e.g. 'nee_f_fill_sd' from
        gapfill_mds). Used to report the random and systematic error bounds.
    qc_col : str or None, optional
        Fill-quality column for method='mds' (e.g. 'nee_f_fill_qc'). When given,
        'gapfilled.perc' is reported.
    measured_col : str or None, optional
        Column holding the MEASURED flux, used for the coverage statistics. Defaults to
        `nee_col`. Set this when `nee_col` points at a gap-filled series, otherwise
        coverage is reported as 100 % and the very thing that caused the original bug -
        a night-biased gap distribution - becomes invisible again.

    Returns
    -------
    pandas.DataFrame
        One row per year: 'method', 'n_obs', 'data_coverage.perc', 'night_coverage.perc',
        'gapfilled.perc', 'sparse_bins', 'empty_bins', 'NEE', 'NEE_se_random',
        'NEE_se_syst', 'Reco', 'GPP', 'NEP' (all g C m-2 yr-1) and 'Reco_GPP_ratio'.
        The sums are NaN for a year with empty bins or unfilled values ('empty_bins').

    See Also
    --------
    annual_energy_budget : The energy-balance equivalent, same table shape.
    mdc_annual_sum : The binning that method='mdc' is built on.
    """
    if method not in ('mdc', 'mds'):
        raise ValueError(f"method must be 'mdc' or 'mds', got {method!r}")

    if measured_col is None:
        measured_col = nee_col

    results = []

    for year, group in temp.groupby(year_col):
        nb_periods = total_annual_periods(int(year), averaging_period_mins=period_mins)
        n_obs = int(group[measured_col].notna().sum())
        if n_obs == 0:
            continue

        night = group.loc[group[dn_col] == 0] if dn_col in group.columns else group.iloc[0:0]
        night_coverage = (100.0 * night[measured_col].notna().sum() / len(night)) if len(night) else np.nan

        gapfilled = np.nan
        if qc_col is not None and qc_col in group.columns:
            gapfilled = 100.0 * group[qc_col].isin([1, 2, 3]).sum() / len(group)

        nee, nee_se_rand, nee_se_syst, sparse, empty = _annual_flux_sum(
            group, nee_col, method, timestamp_col, time_bin, min_n, period_mins,
            dn_col, fallback_reco_col, sd_col,
        )
        reco, _, _, _, _ = _annual_flux_sum(
            group, reco_col, method, timestamp_col, time_bin, min_n, period_mins,
            dn_col, None, None,
        )

        # NEE is negative for uptake, so GPP = Reco - NEE
        gpp = reco - nee if (reco is not None and np.isfinite(reco) and np.isfinite(nee)) else np.nan

        results.append({
            'year': year,
            'method': method,
            'n_obs': n_obs,
            'data_coverage.perc': 100.0 * n_obs / nb_periods,
            'night_coverage.perc': night_coverage,
            'gapfilled.perc': gapfilled,
            'sparse_bins': sparse,
            'empty_bins': empty,
            'NEE': nee,          # [g C m-2 yr-1], negative = net uptake
            'NEE_se_random': nee_se_rand,
            'NEE_se_syst': nee_se_syst,
            'Reco': reco,        # [g C m-2 yr-1]
            'GPP': gpp,          # [g C m-2 yr-1]
            'NEP': -nee if np.isfinite(nee) else np.nan,
            'Reco_GPP_ratio': reco / gpp if (np.isfinite(gpp) and gpp != 0) else np.nan,
        })

    return(pd.DataFrame(results))
