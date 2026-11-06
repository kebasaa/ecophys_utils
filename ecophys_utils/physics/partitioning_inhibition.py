# Partitioning: light inhibition of daytime respiration (Keenan et al. 2019)
#--------------------------------------------------------------------------
#
# Implements the respiration-inhibition correction of the night-time partitioning described
# by Keenan et al. (2019), Nat. Ecol. Evol. 3:407-415, "Widespread inhibition of daytime
# ecosystem respiration".
#
# WHY THIS EXISTS: night-time partitioning (Reichstein et al. 2005, see
# partitioning_respiration.py) fits Lloyd-Taylor respiration to night-time NEE and
# extrapolates it through the day with temperature. Leaf respiration is however inhibited
# in the light (the Kok effect), so daytime Reco is overestimated - and because GPP is
# taken as the residual GPP = Reco - NEE, every unit of that overestimate lands on GPP as
# well. Across FLUXNET, Keenan et al. found the night-time method overestimates annual
# Reco by 11.4 % and annual GPP by 7.0 %, and growing-season daytime Reco by 17.4 %.
#
# Model. The reference respiration is estimated twice, with the SAME temperature
# sensitivity E0:
#
#   RrefN  from night-time NEE (the existing night-time R_ref)
#   RrefD  from daytime NEE, as the respiration intercept (gamma) of the Lasslop et al.
#          (2010) light-response fit, see partitioning_daytime.py
#
#   Reco_day   = LT(T, RrefD, E0)       daytime respiration, inhibited
#   Reco_night = LT(T, RrefN, E0)       unchanged
#   GPP        = Reco - NEE             as in the night-time method
#   I          = (RrefN - RrefD) / RrefN   inhibition, a diagnostic only
#
# Because both share E0 and Lloyd-Taylor is linear in R_ref,
#
#   LT(T, RrefD, E0) = Reco_NT * RrefD / RrefN = Reco_NT * (1 - I)
#
# so the correction is applied as a factor on an existing Reco column. That keeps the
# half-hours whose Reco was gap-filled (e.g. by MDS where air temperature was missing),
# for which the formula above would give NaN.
#
# Correspondence with REddyProc. This follows the modified night-time partitioning in
# Keenan's REddyProc fork (sMRFluxPartition) and the paper's post-processing
# (inhibitionPaperCode): RrefD from 4-day windows every 2 days, linearly interpolated
# between window centres and held constant beyond the first and last window, no clipping
# of the fluxes, and inhibition summarised as monthly medians of the per-window values.
# Three details of the fork are deliberately NOT copied, because they are wrong:
#   1. The fork fits RrefD with the daytime method's smoothed E0 but applies it with the
#      night-time method's constant E0. Here RrefD is fitted with the same E0 column that
#      produced RrefN, which the paper describes and the factor above requires.
#   2. The fork uses three different day/night definitions (Rg <= 4, Rg == 0, Rg > 10 or
#      potential radiation > 0). Here the day/night column is used throughout.
#   3. The fork rejects windows only when RrefD is much higher than RrefN
#      (Rb > 2 RrefN), but never when it is much lower. That one-sided rule selects for
#      positive inhibition and is not used.
#
# THE INHIBITION IS AN ESTIMATE, NOT AN OBSERVATION. RrefD is an extrapolation of the
# light-response curve to zero light and inherits all of the uncertainty of that fit.
# Keep the uncorrected night-time partitioning, write the corrected fluxes to separate
# columns, and read the per-window table before trusting the correction.
#
# I ALSO ABSORBS ANY ERROR IN E0. Daytime is warmer than night, so an E0 that is too high
# makes the night-time model overestimate daytime Reco, and the daytime fit compensates
# with a lower RrefD - which reads as inhibition. An E0 fitted over a whole year of
# nights is inflated wherever R_ref follows the same seasonal cycle as temperature; use
# fit_E0_short_term(), which estimates E0 in short windows as Reichstein et al. (2005)
# do for this reason. In a synthetic
# year with seasonal R_ref and no inhibition at all, a whole-year E0 of 314 K instead of
# the true 200 K produced I = 0.20; with the correct E0 the method returned I = 0.01, and
# I = 0.16 for a true 0.15. fit_E0_short_term() gave 213 K on the same data, and I = 0.03.

import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .partitioning_core import calculate_gpp
from .partitioning_daytime import partition_lasslop, LASSLOP_BOUNDS
from .gapfilling import _as_list

# Filters applied to the per-window inhibition before it is summarised, as in the
# post-processing of Keenan et al. (2019): reference respiration must lie in
# (0, R_ref_max] umol m-2 s-1, |I| must not exceed I_abs_max (2.0 = 200 %), and windows
# with little photosynthesis are dropped, because RrefD is barely identifiable there. A
# window is "low GPP" when the 95th percentile of its GPP is below
# P05 + low_gpp_fraction * (P95 - P05) of those percentiles across its group. Exposed as a
# module constant so the filters are auditable and overridable.
INHIBITION_FILTERS = {
    'R_ref_max': 40.0,
    'I_abs_max': 2.0,
    'low_gpp_fraction': 0.1,
}

# A bounded fit that ends on a bound has not found an optimum. REddyProc rejects windows
# with Rb < 0 or beta >= 250; with curve_fit bounds those cases pin to the bound instead.
_BOUND_RTOL = 1e-3


def calculate_respiration_inhibition(R_ref_night: Union[float, np.ndarray],
                                     R_ref_day: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Inhibition of daytime respiration (Keenan et al. 2019).

    Parameters
    ----------
    R_ref_night : float or numpy.ndarray
        Reference respiration from night-time data (umol m-2 s-1).
    R_ref_day : float or numpy.ndarray
        Reference respiration from daytime data, with the same E0 (umol m-2 s-1).

    Returns
    -------
    float or numpy.ndarray
        I = (R_ref_night - R_ref_day) / R_ref_night, as a fraction (Keenan et al. report
        it in percent). Negative values (daytime respiration above night-time) are kept, as
        in the paper. NaN where R_ref_night <= 0 or either input is missing.

    See Also
    --------
    apply_respiration_inhibition : Applies the corresponding correction to Reco.
    """
    R_ref_night = np.asarray(R_ref_night, dtype=float)
    R_ref_day = np.asarray(R_ref_day, dtype=float)
    # A ratio to a non-positive night-time R_ref has no meaning as an inhibition
    ok = np.isfinite(R_ref_night) & np.isfinite(R_ref_day) & (R_ref_night > 0)
    inhib = np.divide(R_ref_night - R_ref_day, R_ref_night,
                      out=np.full(np.broadcast(R_ref_night, R_ref_day).shape, np.nan),
                      where=ok)
    if inhib.ndim == 0:
        return(float(inhib))
    return(inhib)


def estimate_R_ref_day(temp: pd.DataFrame, nee_col: str = 'nee_f', par_col: Optional[str] = None,
                       sw_col: str = 'SW_IN_1_1_1', sw_to_par: float = 2.1,
                       tair_col: str = 'Tair', vpd_col: str = 'VPD', E0_col: str = 'E0',
                       R_ref_night_col: Optional[str] = 'R_ref', dn_col: str = 'day_night',
                       timestamp_col: str = 'timestamp',
                       group_col: Optional[Union[str, Sequence[str]]] = None,
                       window_days: int = 4, step_days: int = 2, min_points: int = 10,
                       vpd0: Optional[float] = None, vpd_unit: str = 'hPa',
                       out_col: str = 'R_ref_day') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate daytime reference respiration (RrefD) from the light-response curve.

    Fits the Lasslop et al. (2010) model to daytime NEE in moving windows with E0 held
    fixed, takes its respiration intercept as RrefD (Keenan et al. 2019), and interpolates
    it linearly between window centres onto every half-hour.

    ADDS one column; never modifies existing ones.

    Parameters
    ----------
    temp : pandas.DataFrame
        Half-hourly data.
    nee_col : str, optional
        Measured NEE (umol m-2 s-1), u*-filtered, NOT gap-filled. Default is 'nee_f'.
    par_col : str or None, optional
        PAR column (umol m-2 s-1). If None, PAR is derived as `sw_to_par * sw_col`. The
        light unit does not affect RrefD, only the quantum yield.
    sw_col : str, optional
        Incoming shortwave (W m-2), used when `par_col` is None. Default 'SW_IN_1_1_1'.
    sw_to_par : float, optional
        SW -> PAR conversion (umol J-1). Default 2.1.
    tair_col : str, optional
        Air temperature (degC). Must be the temperature used for the night-time fit.
        Default is 'Tair'.
    vpd_col : str, optional
        VPD column, in `vpd_unit`. Default is 'VPD'.
    E0_col : str, optional
        Temperature sensitivity (K) of the night-time fit. MUST be the E0 that produced
        `R_ref_night_col`, otherwise RrefD and RrefN are not comparable. Default 'E0'.
    R_ref_night_col : str or None, optional
        Interpolated night-time R_ref (umol m-2 s-1), used for the per-window inhibition.
        If None, the inhibition columns of the window table are NaN. Default 'R_ref'.
    dn_col, timestamp_col : str, optional
        Day/night (1/0) and timestamp columns.
    group_col : str, sequence of str or None, optional
        Column, or list of columns, whose groups no window may span. Use the grouping of
        the E0 fit, e.g. ['year', 'period'] when a year is split by a disturbance.
    window_days, step_days : int, optional
        Window width and step. Defaults 4 and 2, as in REddyProc and Lasslop et al. The
        paper's text says the night-time window lengths, but the code it ran used 4 days;
        longer windows would let phenological change in the light response leak into RrefD.
    min_points : int, optional
        Minimum daytime records per window. Default is 10, as in REddyProc.
    vpd0 : float or None, optional
        VPD threshold of the light-response curve, see partition_lasslop.
    vpd_unit : {'hPa', 'kPa', 'Pa'}, optional
        Unit of `vpd_col`. Default is 'hPa'.
    out_col : str, optional
        Name of the RrefD column. Default 'R_ref_day'.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        The input frame with `out_col` added; and the per-window table of
        partition_lasslop, where 'status' is 'at_bound' for rejected fits, with the added
        columns 'R_ref_night', 'inhibition', 'gpp_p95', 'low_gpp' and 'valid_inhibition'.

    See Also
    --------
    apply_respiration_inhibition : Applies RrefD to the night-time Reco.
    summarise_inhibition : Monthly inhibition from the window table.
    ecophys_utils.physics.partitioning_daytime.partition_lasslop : The underlying fit.

    Notes
    -----
    REddyProc additionally rejects windows with beta > 100 whose standard error exceeds
    beta. fit_lasslop_window does not return parameter errors, so that check is not made.
    """
    temp = temp.copy()
    # The daytime fit is done by partition_lasslop: its gamma * LT(T, 1, E0) is
    # Lloyd-Taylor respiration with R_ref = gamma, so gamma is RrefD. Its GPP is only
    # needed to flag low-GPP windows.
    dt, win_df = partition_lasslop(temp, nee_col=nee_col, par_col=par_col, sw_col=sw_col,
                                   sw_to_par=sw_to_par, tair_col=tair_col, vpd_col=vpd_col,
                                   E0_col=E0_col, dn_col=dn_col, timestamp_col=timestamp_col,
                                   group_col=group_col, window_days=window_days,
                                   step_days=step_days, min_points=min_points, vpd0=vpd0,
                                   vpd_unit=vpd_unit, suffix='_dt')

    # Reject fits that ended on a bound
    at_bound = ((win_df['gamma'] <= LASSLOP_BOUNDS['gamma'][1] * _BOUND_RTOL)
                | (win_df['beta0'] >= LASSLOP_BOUNDS['beta0'][1] * (1 - _BOUND_RTOL)))
    win_df.loc[(win_df['status'] == 'ok') & at_bound, 'status'] = 'at_bound'
    usable = win_df['status'] == 'ok'

    # Same grouping as partition_lasslop, so that group_idx refers to the same groups
    group_cols = _as_list(group_col)
    dt = dt.sort_values(timestamp_col)
    groups = ([(None, dt)] if not group_cols else
              list(dt.groupby(group_cols[0] if len(group_cols) == 1 else group_cols, observed=True)))

    win_df['R_ref_night'] = np.nan
    win_df['gpp_p95'] = np.nan
    r_ref_day = pd.Series(np.nan, index=dt.index)
    for gi, (label, g) in enumerate(groups):
        in_group = win_df['group_idx'] == gi
        ts = g[timestamp_col].astype('int64').to_numpy()

        # 95th percentile of the daytime-method GPP in each window
        for wi in win_df.index[in_group]:
            s = win_df.at[wi, 'window_start']
            w = g.loc[(g[timestamp_col] >= s) & (g[timestamp_col] < s + pd.Timedelta(days=window_days)),
                      'gpp_dt'].dropna()
            if not w.empty:
                win_df.at[wi, 'gpp_p95'] = np.percentile(w, 95)

        # Night-time R_ref at the window centres, to compare like with like
        if R_ref_night_col is not None:
            rn = g[R_ref_night_col].to_numpy(dtype=float)
            ok = np.isfinite(rn)
            if ok.any():
                win_df.loc[in_group, 'R_ref_night'] = np.interp(
                    win_df.loc[in_group, 'window_mid'].astype('int64').to_numpy(), ts[ok], rn[ok])

        # Interpolate RrefD between the centres of the usable windows. np.interp holds the
        # end values constant, as REddyProc does (approx(..., rule = 2)).
        wf = win_df[in_group & usable].sort_values('window_mid')
        if wf.empty:
            continue
        r_ref_day.loc[g.index] = np.interp(ts, wf['window_mid'].astype('int64').to_numpy(),
                                           wf['gamma'].to_numpy())

    win_df['inhibition'] = calculate_respiration_inhibition(win_df['R_ref_night'].to_numpy(),
                                                            win_df['gamma'].to_numpy())

    # Low-GPP windows, relative to the spread of window GPP within each group
    win_df['low_gpp'] = False
    for gi in win_df['group_idx'].unique():
        in_group = win_df['group_idx'] == gi
        p = win_df.loc[in_group, 'gpp_p95'].dropna()
        if p.empty:
            continue
        p05, p95 = np.percentile(p, [5, 95])
        limit = p05 + INHIBITION_FILTERS['low_gpp_fraction'] * (p95 - p05)
        win_df.loc[in_group, 'low_gpp'] = ~(win_df.loc[in_group, 'gpp_p95'] >= limit)

    r_max = INHIBITION_FILTERS['R_ref_max']
    win_df['valid_inhibition'] = (usable
                                  & (win_df['R_ref_night'] > 0) & (win_df['R_ref_night'] <= r_max)
                                  & (win_df['gamma'] > 0) & (win_df['gamma'] <= r_max)
                                  & (win_df['inhibition'].abs() <= INHIBITION_FILTERS['I_abs_max'])
                                  & ~win_df['low_gpp'])

    n_ok = int(usable.sum())
    if n_ok > 0 and n_ok < 0.5 * len(win_df):
        warnings.warn(f'Only {100 * n_ok / len(win_df):.0f}% of windows gave a usable daytime '
                      'R_ref; RrefD is interpolated over long stretches.')
    if r_ref_day.isna().all():
        warnings.warn('No daytime R_ref could be estimated; returning a NaN column.')

    temp[out_col] = r_ref_day.reindex(temp.index)
    return(temp, win_df)


def apply_respiration_inhibition(temp: pd.DataFrame, reco_col: str, nee_col: str,
                                 R_ref_night_col: str = 'R_ref', R_ref_day_col: str = 'R_ref_day',
                                 dn_col: str = 'day_night', clip_negative: bool = True,
                                 suffix: str = '_inhib') -> pd.DataFrame:
    """
    Correct night-time partitioned Reco and GPP for daytime respiration inhibition.

    In daylight, Reco is scaled by R_ref_day / R_ref_night, which equals Lloyd-Taylor
    respiration with the daytime reference respiration (Keenan et al. 2019). Night-time
    Reco is unchanged. GPP is recomputed as Reco - NEE.

    ADDS columns; never modifies existing ones.

    Parameters
    ----------
    temp : pandas.DataFrame
        Half-hourly data.
    reco_col : str
        Night-time partitioned Reco (umol m-2 s-1), may be gap-filled.
    nee_col : str
        NEE (umol m-2 s-1) from which GPP is computed, normally gap-filled.
    R_ref_night_col : str, optional
        Night-time R_ref that produced `reco_col`. Default 'R_ref'.
    R_ref_day_col : str, optional
        Daytime R_ref from estimate_R_ref_day(), fitted with the same E0. Default 'R_ref_day'.
    dn_col : str, optional
        Day (1) / night (0) column. Default 'day_night'.
    clip_negative : bool, optional
        Set negative GPP to zero, as in calculate_gpp(). Default is True.
    suffix : str, optional
        Suffix for the generated columns. Default '_inhib'.

    Returns
    -------
    pandas.DataFrame
        The input frame with 'reco<suffix>', 'gpp<suffix>' and 'inhibition<suffix>' (the
        inhibition fraction applied; 0 at night) added.

    See Also
    --------
    estimate_R_ref_day : Produces the R_ref_day column.
    calculate_gpp : GPP from NEE and Reco.

    Notes
    -----
    The factor R_ref_day / R_ref_night equals LT(T, R_ref_day, E0) / LT(T, R_ref_night, E0)
    only if both were obtained with the SAME E0, and if `reco_col` was modelled from
    `R_ref_night_col`. Pass the columns of one night-time run, and the E0 of that run to
    estimate_R_ref_day().

    Where daytime R_ref is missing, daytime Reco and GPP become NaN rather than falling
    back to the uncorrected values: silently mixing corrected and uncorrected half-hours
    would make the size of the correction unknowable.
    """
    temp = temp.copy()
    day = (temp[dn_col] == 1).to_numpy()
    inhib = calculate_respiration_inhibition(temp[R_ref_night_col].to_numpy(dtype=float),
                                             temp[R_ref_day_col].to_numpy(dtype=float))
    reco = temp[reco_col].to_numpy(dtype=float)

    reco_inhib = np.where(day, reco * (1.0 - inhib), reco)

    n_lost = int((day & np.isfinite(reco) & ~np.isfinite(inhib)).sum())
    if n_lost > 0:
        warnings.warn(f'{n_lost} daytime half-hours have Reco but no usable R_ref ratio; '
                      'their corrected Reco and GPP are NaN.')

    temp[f'inhibition{suffix}'] = np.where(day, inhib, 0.0)
    temp[f'reco{suffix}'] = reco_inhib
    temp[f'gpp{suffix}'] = calculate_gpp(temp[nee_col].to_numpy(dtype=float), reco_inhib,
                                         day_night=temp[dn_col], clip_negative=clip_negative)
    return(temp)


def summarise_inhibition(win_df: pd.DataFrame, by_group: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Monthly inhibition of daytime respiration, as reported by Keenan et al. (2019).

    Uses only the windows that pass the filters (column 'valid_inhibition').

    Parameters
    ----------
    win_df : pandas.DataFrame
        Window table returned by estimate_R_ref_day().
    by_group : bool, optional
        Summarise each group of estimate_R_ref_day() separately. Default is True.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Monthly table: median inhibition of the valid windows per year and month
        ('inhibition_median', fraction) and their number ('n_windows'). Climatology: mean
        of those monthly medians per calendar month ('inhibition_mean') and the number of
        years ('n_years').

    See Also
    --------
    estimate_R_ref_day : Produces the window table.
    """
    w = win_df.loc[win_df['valid_inhibition'].astype(bool)].copy()
    w['year'] = w['window_mid'].dt.year
    w['month'] = w['window_mid'].dt.month
    # Groups are keyed by position, because a label of None never compares equal to itself
    keys = (['group_idx'] if by_group else [])

    monthly = (w.groupby(keys + ['year', 'month'])
                .agg(group=('group', 'first'),
                     inhibition_median=('inhibition', 'median'),
                     n_windows=('inhibition', 'size'))
                .reset_index())
    climatology = (monthly.groupby(keys + ['month'])
                          .agg(group=('group', 'first'),
                               inhibition_mean=('inhibition_median', 'mean'),
                               n_years=('inhibition_median', 'size'))
                          .reset_index())
    if not by_group:
        monthly = monthly.drop(columns=['group'])
        climatology = climatology.drop(columns=['group'])
    return(monthly, climatology)
