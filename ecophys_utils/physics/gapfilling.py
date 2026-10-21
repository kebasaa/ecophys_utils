# Flux gap-filling: Marginal Distribution Sampling (MDS)
#-------------------------------------------------------
#
# Implements the look-up-table gap-filling of Reichstein et al. (2005),
# Glob. Change Biol. 11:1424-1439. A missing half-hour is estimated from the mean of
# observed half-hours nearby in time that experienced similar meteorological
# conditions (incoming shortwave radiation, air temperature, vapour pressure deficit).
# Where no meteorologically similar period can be found, the method falls back to the
# mean diurnal course, progressively widening the search window and degrading the
# quality flag as it goes.
#
# This is the FLUXNET-standard method and exists here as an INDEPENDENT CROSS-CHECK on
# the mean-diurnal-course annual sums in carbon_budget.py. The two condition on
# different things - MDS on meteorology, MDC on time of year and time of day - so
# agreement between them is evidence that a term (in particular night-time
# respiration) is real rather than an artefact of one aggregation scheme.
#
# CAVEAT WHEN COMPARING THE TWO: if most gaps end up filled at qc 3 (the widest,
# mean-diurnal-course fallback passes), MDS has degenerated into something close to MDC
# and the two methods are no longer independent. Always read the qc breakdown from
# gapfill_report() before treating their agreement as corroboration.

import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Search ladder: (kind, window in days, quality flag).
#   'drivers' - require SW_IN, Tair and VPD all within tolerance
#   'sw'      - require SW_IN within tolerance only
#   'mdc'     - mean diurnal course: same time of day +-1 h, within +-window days
# Exposed as a module constant so the hierarchy is auditable and overridable rather
# than buried in the function body.
MDS_PASSES: List[Tuple[str, int, int]] = (
    [('drivers', 7, 1), ('drivers', 14, 1), ('sw', 7, 1), ('mdc', 0, 1), ('mdc', 1, 2)]
    + [('drivers', w, 2 if w <= 28 else 3) for w in range(21, 141, 7)]
    + [('sw', w, 2 if w <= 14 else 3) for w in range(14, 141, 7)]
    + [('mdc', w, 2 if w <= 5 else 3) for w in list(range(2, 8)) + list(range(14, 211, 7))]
)

# Default VPD tolerance (Reichstein et al. use 5 hPa) expressed per unit
_VPD_TOL = {'hpa': 5.0, 'kpa': 0.5, 'pa': 500.0}
# Rough plausibility range of a typical median VPD, used only to catch a mislabelled unit
_VPD_PLAUSIBLE_MEDIAN = {'hpa': (0.5, 60.0), 'kpa': (0.05, 6.0), 'pa': (50.0, 6000.0)}


def _as_list(cols: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """One column name, a sequence of them, or None, as a list."""
    if cols is None:
        return []
    if isinstance(cols, str):
        return [cols]
    return list(cols)


def _sw_tolerance(sw: np.ndarray) -> np.ndarray:
    """
    Radiation tolerance of Reichstein et al. (2005): 20 % of the value, but never
    below 20 W m-2 and never above 50 W m-2.
    """
    return(np.clip(0.2 * np.abs(sw), 20.0, 50.0))


def _check_vpd_unit(vpd: pd.Series, vpd_unit: str) -> None:
    """Warn if the VPD values look inconsistent with the declared unit."""
    med = np.nanmedian(vpd.values.astype(float))
    if not np.isfinite(med):
        return
    lo, hi = _VPD_PLAUSIBLE_MEDIAN[vpd_unit.lower()]
    if not (lo <= med <= hi):
        warnings.warn(
            f"Median VPD is {med:.3g}, which looks inconsistent with vpd_unit='{vpd_unit}'. "
            "The look-up tolerance would then be wrong by orders of magnitude. Check the "
            "units of the VPD column before trusting the fill."
        )


def _check_regular_grid(ts: pd.Series, period_mins: int, label: str) -> None:
    """
    Warn if timestamps are not on a regular grid.

    The look-up windows are expressed as row offsets, so a missing row silently shifts
    the window in time. Run complete_timestamps() first.
    """
    diffs = ts.diff().dropna().unique()
    if len(diffs) != 1 or diffs[0] != pd.Timedelta(minutes=period_mins):
        warnings.warn(
            f"Timestamps in group {label!r} are not on a regular {period_mins}-minute grid. "
            "Look-up windows are row-based and will be misaligned. Run complete_timestamps()."
        )


def _fill_one_group(flux: np.ndarray, sw: np.ndarray, tair: np.ndarray, vpd: np.ndarray,
                    steps_per_day: int, vpd_tol: float, tair_tol: float,
                    min_n: int, passes: Sequence[Tuple[str, int, int]]
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the MDS ladder over one contiguous group.

    Returns (filled, qc, sd, window_days). Measured values keep qc 0 and sd NaN.
    Values that no pass could estimate stay NaN with qc -1, and are never zeroed.
    """
    n = len(flux)
    filled = flux.copy()
    qc = np.where(np.isnan(flux), -1, 0).astype(float)
    sd = np.full(n, np.nan)
    win = np.full(n, np.nan)

    observed = ~np.isnan(flux)
    sw_tol = _sw_tolerance(sw)
    one_hour = max(1, steps_per_day // 24)

    for kind, window_days, flag in passes:
        todo = np.flatnonzero(np.isnan(filled))
        if todo.size == 0:
            break

        half = int(window_days * steps_per_day)

        for i in todo:
            if kind == 'mdc':
                # Same time of day, +-1 hour, within +-window_days
                offsets = np.arange(-window_days, window_days + 1) * steps_per_day
                cand = (i + offsets[:, None] + np.arange(-one_hour, one_hour + 1)[None, :]).ravel()
                cand = cand[(cand >= 0) & (cand < n)]
                cand = cand[observed[cand]]
                if cand.size < min_n:
                    continue
                vals = flux[cand]
            else:
                lo, hi = max(0, i - half), min(n, i + half + 1)
                sl = slice(lo, hi)
                mask = observed[sl] & (np.abs(sw[sl] - sw[i]) <= sw_tol[i])
                if kind == 'drivers':
                    mask &= (np.abs(tair[sl] - tair[i]) <= tair_tol)
                    mask &= (np.abs(vpd[sl] - vpd[i]) <= vpd_tol)
                # A driver that is itself missing at the target cannot be matched
                if np.isnan(sw[i]) or (kind == 'drivers' and (np.isnan(tair[i]) or np.isnan(vpd[i]))):
                    continue
                if mask.sum() < min_n:
                    continue
                vals = flux[sl][mask]

            filled[i] = np.mean(vals)
            # SD of the look-up population: the standard MDS per-point uncertainty
            sd[i] = np.std(vals, ddof=1) if vals.size > 1 else np.nan
            qc[i] = flag
            win[i] = window_days

    return(filled, qc, sd, win)


def gapfill_mds(temp: pd.DataFrame, flux_col: str = 'nee_f', sw_col: str = 'SW_IN_1_1_1',
                tair_col: str = 'Tair', vpd_col: str = 'VPD', timestamp_col: str = 'timestamp',
                group_col: Optional[Union[str, Sequence[str]]] = None, period_mins: int = 30,
                tair_tol: float = 2.5, vpd_tol: Optional[float] = None, vpd_unit: str = 'hPa',
                min_n: int = 2, passes: Optional[Sequence[Tuple[str, int, int]]] = None,
                suffix: str = '_fill') -> pd.DataFrame:
    """
    Gap-fill a flux by Marginal Distribution Sampling (Reichstein et al. 2005).

    Adds columns rather than overwriting `flux_col`, so downstream analyses that must
    use measured data only (light-response and Michaelis-Menten curve fitting, in
    particular, where filling would be circular) can keep using the original column.

    Parameters
    ----------
    temp : pandas.DataFrame
        Half-hourly data on a regular, complete timestamp grid (see complete_timestamps).
    flux_col : str, optional
        Flux to fill, in umol m-2 s-1. Default is 'nee_f'.
    sw_col, tair_col, vpd_col : str, optional
        Meteorological driver columns: incoming shortwave (W m-2), air temperature (degC)
        and vapour pressure deficit (see vpd_unit).
    timestamp_col : str, optional
        Timestamp column. Default is 'timestamp'.
    group_col : str, sequence of str or None, optional
        Column, or list of columns, defining blocks that look-up windows must never
        cross, e.g. 'site' or ['site', 'period']. Default is None (one block). THIS
        MATTERS when the record spans a change in the ecosystem or the setup (a
        disturbance, a management change, a new sensor): filling gaps on one side from
        observations on the other would manufacture exactly the difference such a
        comparison sets out to test, so group by it unless the record is genuinely
        homogeneous.
    period_mins : int, optional
        Averaging period in minutes. Default is 30.
    tair_tol : float, optional
        Air temperature tolerance (degC). Default is 2.5.
    vpd_tol : float or None, optional
        VPD tolerance. If None (default), taken from `vpd_unit`: 5 hPa / 0.5 kPa / 500 Pa.
    vpd_unit : {'hPa', 'kPa', 'Pa'}, optional
        Unit of `vpd_col`. Default is 'hPa'. The values are sanity-checked against this
        and a warning is raised if they look inconsistent.
    min_n : int, optional
        Minimum number of look-up values required to accept an estimate. Default is 2.
    passes : sequence of (kind, window_days, qc) or None, optional
        Search ladder. Default is MDS_PASSES.
    suffix : str, optional
        Suffix for the generated columns. Default is '_fill'.

    Returns
    -------
    pandas.DataFrame
        Copy of `temp` with four extra columns:
        '<flux><suffix>'         filled series (measured value where available)
        '<flux><suffix>_qc'      0 = measured, 1-3 = fill quality, -1 = could not fill
        '<flux><suffix>_sd'      SD of the look-up population (per-point uncertainty)
        '<flux><suffix>_window'  window width in days that produced the estimate

    Notes
    -----
    Runs one Python-level loop per remaining gap per pass, so expect order-of-minutes
    runtime on a multi-year half-hourly record. It is a one-off processing step.

    Values that no pass could estimate remain NaN and are flagged qc -1. They are never
    set to zero: for a u*-filtered CO2 flux the gaps are overwhelmingly night-time, so
    zeroing them deletes respiration and inflates the annual carbon gain.

    See Also
    --------
    gapfill_report : Fill-fraction diagnostics by quality class and day/night.
    ecophys_utils.physics.carbon_budget.mdc_annual_sum : Independent annual sum that
        needs no gap-filling; use it to cross-check sums built on this output.
    """
    if vpd_unit.lower() not in _VPD_TOL:
        raise ValueError(f"vpd_unit must be one of {list(_VPD_TOL)}, got {vpd_unit!r}")
    if vpd_tol is None:
        vpd_tol = _VPD_TOL[vpd_unit.lower()]
    if passes is None:
        passes = MDS_PASSES

    group_cols = _as_list(group_col)
    for col in [flux_col, sw_col, tair_col, vpd_col, timestamp_col] + group_cols:
        if col not in temp.columns:
            raise KeyError(f"Column {col!r} not found in the dataframe")

    _check_vpd_unit(temp[vpd_col], vpd_unit)

    temp = temp.sort_values(timestamp_col).copy()
    steps_per_day = int(24 * 60 / period_mins)

    fill_col = f'{flux_col}{suffix}'
    temp[fill_col] = np.nan
    temp[f'{fill_col}_qc'] = np.nan
    temp[f'{fill_col}_sd'] = np.nan
    temp[f'{fill_col}_window'] = np.nan

    # A single column is grouped by name so that labels stay scalars rather than 1-tuples
    groups = ([(None, temp)] if not group_cols else
              list(temp.groupby(group_cols[0] if len(group_cols) == 1 else group_cols, observed=True)))

    for label, group in groups:
        _check_regular_grid(group[timestamp_col], period_mins, label)

        if group[flux_col].notna().sum() == 0:
            warnings.warn(f"Group {label!r} has no observed {flux_col}; nothing to fill from.")
            temp.loc[group.index, f'{fill_col}_qc'] = -1
            continue

        filled, qc, sd, win = _fill_one_group(
            group[flux_col].to_numpy(dtype=float),
            group[sw_col].to_numpy(dtype=float),
            group[tair_col].to_numpy(dtype=float),
            group[vpd_col].to_numpy(dtype=float),
            steps_per_day, vpd_tol, tair_tol, min_n, passes,
        )

        temp.loc[group.index, fill_col] = filled
        temp.loc[group.index, f'{fill_col}_qc'] = qc
        temp.loc[group.index, f'{fill_col}_sd'] = sd
        temp.loc[group.index, f'{fill_col}_window'] = win

    n_unfilled = int((temp[f'{fill_col}_qc'] == -1).sum())
    if n_unfilled:
        warnings.warn(
            f"{n_unfilled} half-hours could not be filled and remain NaN (qc -1). They must "
            "not be summed as zero; use carbon_budget.annual_carbon_budget, which reports "
            "them as 'empty_bins' rather than silently dropping them."
        )

    return(temp)


def gapfill_report(temp: pd.DataFrame, flux_col: str = 'nee_f', suffix: str = '_fill',
                   group_col: Optional[Union[str, Sequence[str]]] = None, dn_col: str = 'day_night',
                   year_col: Optional[str] = 'year') -> pd.DataFrame:
    """
    Summarise how much of a flux was measured versus filled, and at what quality.

    This is the diagnostic that makes a night-biased gap distribution visible. For a
    u*-filtered CO2 flux, expect night-time retention to be far lower than daytime; that
    asymmetry is the reason a naive annual sum overestimates carbon uptake.

    Parameters
    ----------
    temp : pandas.DataFrame
        Output of gapfill_mds().
    flux_col : str, optional
        Original flux column name. Default is 'nee_f'.
    suffix : str, optional
        Suffix used when filling. Default is '_fill'.
    group_col : str, sequence of str or None, optional
        Extra grouping column, or list of columns, e.g. 'site'. Default is None.
    dn_col : str, optional
        Day/night indicator (1 = day, 0 = night). Default is 'day_night'.
    year_col : str or None, optional
        Year column to break the report down by. Default is 'year'.

    Returns
    -------
    pandas.DataFrame
        Rows per group x year x day/night ('day_or_night'), with the number of periods,
        the measured fraction, and the fraction filled at each quality class (including
        qc -1, i.e. not fillable).
    """
    qc_col = f'{flux_col}{suffix}_qc'
    if qc_col not in temp.columns:
        raise KeyError(f"Column {qc_col!r} not found; run gapfill_mds() first")

    keys = [c for c in _as_list(group_col) + _as_list(year_col) if c in temp.columns]
    if dn_col in temp.columns:
        temp = temp.copy()
        temp['day_or_night'] = np.where(temp[dn_col] == 1, 'day', 'night')
        keys = keys + ['day_or_night']

    rows = []
    grouped = temp.groupby(keys, observed=True, dropna=False) if keys else [((), temp)]
    for label, group in grouped:
        label = label if isinstance(label, tuple) else (label,)
        n = len(group)
        row = dict(zip(keys, label))
        row['n_periods'] = n
        row['measured.perc'] = 100.0 * (group[qc_col] == 0).sum() / n
        for flag in (1, 2, 3):
            row[f'filled_qc{flag}.perc'] = 100.0 * (group[qc_col] == flag).sum() / n
        row['unfillable.perc'] = 100.0 * (group[qc_col] == -1).sum() / n
        rows.append(row)

    return(pd.DataFrame(rows))
