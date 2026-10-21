# Partitioning: daytime method (Lasslop et al. 2010)
#---------------------------------------------------
#
# Implements the daytime flux partitioning of Lasslop et al. (2010),
# Glob. Change Biol. 16:187-208.
#
# WHY THIS EXISTS: night-time partitioning (Reichstein et al. 2005, implemented in
# partitioning_respiration.py) fits respiration to night data and extrapolates it through
# the day using temperature, then takes GPP as the residual GPP = Reco - NEE. That makes
# it structurally unable to check itself: any bias in the daytime respiration
# extrapolation flows straight into GPP, and nothing in the method can detect it.
#
# The daytime method fits a light-response curve and a respiration term jointly to
# DAYTIME NEE, so its Reco/GPP split is derived from independent information.
#
# THIS IS A DIAGNOSTIC, NOT A REPLACEMENT. It is intended to be run alongside the
# night-time method and compared against it. Disagreement localises a problem; it does not
# by itself establish which method is right. Write its output to separate columns and keep
# the night-time partitioning as the primary product.
#
# Model (NEE negative for uptake):
#
#     NEE = -(alpha * PAR * beta) / (alpha * PAR + beta)  +  gamma * LT(T, 1, E0)
#            \___________ GPP, rectangular hyperbola ___/     \___ Reco, Lloyd-Taylor __/
#
#   alpha  canopy quantum yield          [umol CO2 / umol photons]
#   beta   light-saturated assimilation  [umol m-2 s-1], reduced at high VPD
#   gamma  respiration at T_ref          [umol m-2 s-1]  (= R_ref)
#   E0     temperature sensitivity       [K] - held FIXED from the night-time fit, as in
#          Lasslop et al., so the two methods differ in their DAYTIME treatment rather
#          than in temperature sensitivity.
#
# The VPD limitation is the part that matters at a summer-drought site: beta declines
# exponentially above VPD0, which is what stops the fit attributing drought-suppressed
# uptake to respiration instead.

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .partitioning_respiration import lloyd_taylor
from .gapfilling import _check_vpd_unit

# VPD threshold above which beta declines (Lasslop et al. use 10 hPa), per unit
_VPD0 = {'hpa': 10.0, 'kpa': 1.0, 'pa': 1000.0}

# Parameter bounds. alpha's upper bound is the theoretical maximum quantum yield
# (~0.125 mol CO2 per mol photons); the rest are generous but keep the optimiser away
# from unphysical corners.
LASSLOP_BOUNDS = {
    'alpha': (0.0, 0.125),      # umol CO2 / umol photons
    'beta0': (0.0, 250.0),      # umol m-2 s-1
    'k':     (0.0, 2.0),        # 1 / VPD-unit
    'gamma': (0.0, 50.0),       # umol m-2 s-1
}


def lasslop_nee(drivers: Tuple[np.ndarray, np.ndarray, np.ndarray],
                alpha: float, beta0: float, k: float, gamma: float,
                E0: float, vpd0: float = 10.0,
                T_ref: float = 15.0, T0: float = -46.02) -> np.ndarray:
    """
    Lasslop et al. (2010) daytime NEE model.

    Parameters
    ----------
    drivers : tuple of numpy.ndarray
        (PAR, T, VPD). PAR in umol m-2 s-1, T in degC, VPD in the same unit as `vpd0`.
    alpha : float
        Canopy quantum yield (umol CO2 / umol photons).
    beta0 : float
        Light-saturated assimilation at or below `vpd0` (umol m-2 s-1).
    k : float
        Exponential VPD sensitivity of beta (1 / VPD-unit).
    gamma : float
        Respiration at `T_ref` (umol m-2 s-1).
    E0 : float
        Temperature sensitivity (K). Held fixed, normally from the night-time fit.
    vpd0 : float, optional
        VPD threshold above which beta declines. Default is 10.0 (hPa; = 1 kPa).
    T_ref, T0 : float, optional
        Lloyd-Taylor reference and lower-bound temperatures (degC).

    Returns
    -------
    numpy.ndarray
        Modelled NEE (umol m-2 s-1), negative for net uptake.

    See Also
    --------
    lloyd_taylor : the respiration term.
    partition_lasslop : windowed fitting driver.
    """
    par, T, vpd = drivers

    # VPD limitation of light-saturated assimilation
    beta = np.where(vpd > vpd0, beta0 * np.exp(-k * (vpd - vpd0)), beta0)

    # Rectangular hyperbola; guard the denominator where PAR and beta are both ~0
    denom = alpha * par + beta
    gpp = np.divide(alpha * par * beta, denom, out=np.zeros_like(par, dtype=float),
                    where=denom > 0)

    reco = gamma * lloyd_taylor(T, 1.0, E0, T_ref=T_ref, T0=T0)
    return(-gpp + reco)


def fit_lasslop_window(par: np.ndarray, T: np.ndarray, vpd: np.ndarray, nee: np.ndarray,
                       E0: float, vpd0: float = 10.0, min_points: int = 30
                       ) -> Optional[dict]:
    """
    Fit the Lasslop model to one window of daytime data.

    Returns None when the window cannot be fitted (too few points, no temperature range,
    or the optimiser fails) rather than returning a meaningless best-effort fit.

    Parameters
    ----------
    par, T, vpd, nee : numpy.ndarray
        Daytime driver and flux arrays for the window.
    E0 : float
        Temperature sensitivity, held fixed.
    vpd0 : float, optional
        VPD threshold, in the units of `vpd`. Default is 10.0 (hPa).
    min_points : int, optional
        Minimum valid records required. Default is 30.

    Returns
    -------
    dict or None
        Fitted 'alpha', 'beta0', 'k', 'gamma', plus 'n', 'rmse' and 'status'.
    """
    ok = np.isfinite(par) & np.isfinite(T) & np.isfinite(vpd) & np.isfinite(nee)
    if not np.isfinite(E0):
        return(None)
    par, T, vpd, nee = par[ok], T[ok], vpd[ok], nee[ok]

    if len(nee) < min_points:
        return(None)
    # A light-response curve needs a spread of light levels to be identifiable
    if np.ptp(par) < 100:
        return(None)

    lo = [LASSLOP_BOUNDS[p][0] for p in ('alpha', 'beta0', 'k', 'gamma')]
    hi = [LASSLOP_BOUNDS[p][1] for p in ('alpha', 'beta0', 'k', 'gamma')]
    # Start from data-informed guesses: beta0 near the observed maximum uptake, gamma near
    # the smallest (least negative) fluxes. k starts at 1/vpd0, so the guess means the same
    # thing whatever the VPD unit.
    p0 = [0.02,
          float(np.clip(-np.nanpercentile(nee, 5), 1.0, 200.0)),
          float(np.clip(1.0 / vpd0, 1e-6, hi[2] * 0.9)),
          float(np.clip(np.nanpercentile(nee, 95), 0.1, 20.0))]

    def _model(x, alpha, beta0, k, gamma):
        return lasslop_nee(x, alpha, beta0, k, gamma, E0=E0, vpd0=vpd0)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, _ = curve_fit(_model, (par, T, vpd), nee, p0=p0,
                                bounds=(lo, hi), maxfev=10000)
    except Exception:
        return(None)

    resid = nee - _model((par, T, vpd), *popt)
    return({'alpha': popt[0], 'beta0': popt[1], 'k': popt[2], 'gamma': popt[3],
            'n': len(nee), 'rmse': float(np.sqrt(np.mean(resid ** 2))), 'status': 'ok'})


def partition_lasslop(temp: pd.DataFrame, nee_col: str = 'nee_f', par_col: Optional[str] = None,
                      sw_col: str = 'SW_IN_1_1_1', sw_to_par: float = 2.1,
                      tair_col: str = 'Tair', vpd_col: str = 'VPD', E0_col: str = 'E0',
                      dn_col: str = 'day_night', timestamp_col: str = 'timestamp',
                      group_col: Optional[str] = None, window_days: int = 4,
                      step_days: int = 2, min_points: int = 30, vpd0: Optional[float] = None,
                      vpd_unit: str = 'hPa',
                      suffix: str = '_lasslop') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partition NEE into Reco and GPP by the daytime method of Lasslop et al. (2010).

    Fits the light-response + respiration model to daytime data in overlapping windows,
    interpolates the parameters to every half-hour, and evaluates the model over the whole
    record (at night PAR is 0, so GPP is 0 and modelled NEE reduces to Reco).

    ADDS columns; never modifies existing ones. Keep the night-time partitioning as the
    primary product and use this only for comparison.

    Parameters
    ----------
    temp : pandas.DataFrame
        Half-hourly data.
    nee_col : str, optional
        Measured NEE (umol m-2 s-1), u*-filtered. Default is 'nee_f'.
    par_col : str or None, optional
        PAR column (umol m-2 s-1). If None, PAR is derived as `sw_to_par * sw_col`.
    sw_col : str, optional
        Incoming shortwave (W m-2), used when `par_col` is None. Default 'SW_IN_1_1_1'.
    sw_to_par : float, optional
        SW -> PAR conversion (umol J-1). Default 2.1, a common approximation for broadband
        incoming shortwave.
    tair_col, vpd_col, E0_col, dn_col, timestamp_col : str, optional
        Driver, temperature-sensitivity, day/night and timestamp columns.
    group_col : str or None, optional
        Windows never span a group boundary. Default None (one block). Pass e.g. 'fire'
        for a record spanning a disturbance - the same guard as in gapfill_mds, for the
        same reason.
    window_days, step_days : int, optional
        Overlapping window width and step. Defaults 4 and 2, as in Lasslop et al.
    min_points : int, optional
        Minimum daytime records per window. Default is 30.
    vpd0 : float or None, optional
        VPD threshold for beta limitation, in the units of `vpd_col`. If None (default),
        taken from `vpd_unit` as the Lasslop et al. value: 10 hPa / 1 kPa / 1000 Pa.
    vpd_unit : {'hPa', 'kPa', 'Pa'}, optional
        Unit of `vpd_col`. Default is 'hPa', as in gapfill_mds. The values are
        sanity-checked against it and a warning is raised if they look inconsistent.
    suffix : str, optional
        Suffix for the generated columns. Default '_lasslop'.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        The input frame with 'reco<suffix>', 'gpp<suffix>', 'nee<suffix>' and the
        interpolated parameter columns added; and the per-window fit table.

    See Also
    --------
    ecophys_utils.physics.partitioning_respiration : the night-time (Reichstein) method
        this is meant to be compared against.
    """
    if vpd_unit.lower() not in _VPD0:
        raise ValueError(f"vpd_unit must be one of {list(_VPD0)}, got {vpd_unit!r}")
    if vpd0 is None:
        vpd0 = _VPD0[vpd_unit.lower()]
    _check_vpd_unit(temp[vpd_col], vpd_unit)

    temp = temp.sort_values(timestamp_col).copy()

    if par_col is None:
        temp['_par'] = temp[sw_col] * sw_to_par
        par_col = '_par'

    groups = [(None, temp)] if group_col is None else list(temp.groupby(group_col, observed=True))

    # Windows are matched back to their group by position, because a label of None (no
    # grouping) never compares equal to itself in a pandas column
    windows = []
    for gi, (label, g) in enumerate(groups):
        t0, t1 = g[timestamp_col].min(), g[timestamp_col].max()
        starts = pd.date_range(t0, t1, freq=f'{step_days}D')
        for s in starts:
            e = s + pd.Timedelta(days=window_days)
            w = g[(g[timestamp_col] >= s) & (g[timestamp_col] < e) & (g[dn_col] == 1)]
            if w.empty:
                continue
            E0 = w[E0_col].median()
            fit = fit_lasslop_window(w[par_col].to_numpy(dtype=float),
                                     w[tair_col].to_numpy(dtype=float),
                                     w[vpd_col].to_numpy(dtype=float),
                                     w[nee_col].to_numpy(dtype=float),
                                     E0=E0, vpd0=vpd0, min_points=min_points)
            row = {'group': label, 'group_idx': gi, 'window_start': s,
                   'window_mid': s + pd.Timedelta(days=window_days / 2), 'E0': E0}
            row.update(fit if fit is not None else
                       {'alpha': np.nan, 'beta0': np.nan, 'k': np.nan, 'gamma': np.nan,
                        'n': len(w), 'rmse': np.nan, 'status': 'failed'})
            windows.append(row)

    win_df = pd.DataFrame(windows, columns=['group', 'group_idx', 'window_start', 'window_mid', 'E0',
                                            'alpha', 'beta0', 'k', 'gamma', 'n', 'rmse', 'status'])
    n_ok = int((win_df['status'] == 'ok').sum())
    if n_ok == 0:
        warnings.warn('No Lasslop window could be fitted; returning NaN columns.')
    else:
        frac = 100 * n_ok / len(win_df)
        if frac < 50:
            warnings.warn(f'Only {frac:.0f}% of Lasslop windows converged; treat the '
                          'comparison with caution.')

    # Interpolate parameters onto the half-hourly timeline, within groups
    par_names = ['alpha', 'beta0', 'k', 'gamma']
    for p in par_names + ['nee', 'reco', 'gpp']:
        temp[f'{p}{suffix}'] = np.nan

    for gi, (label, g) in enumerate(groups):
        wf = win_df[win_df['group_idx'] == gi].dropna(subset=['alpha']).sort_values('window_mid')
        if wf.empty:
            continue
        idx = g.index
        interp = {}
        for p in par_names:
            interp[p] = np.interp(g[timestamp_col].astype('int64').to_numpy(),
                                  wf['window_mid'].astype('int64').to_numpy(),
                                  wf[p].to_numpy())
            temp.loc[idx, f'{p}{suffix}'] = interp[p]

        drivers = (g[par_col].to_numpy(dtype=float),
                   g[tair_col].to_numpy(dtype=float),
                   g[vpd_col].to_numpy(dtype=float))
        E0 = g[E0_col].to_numpy(dtype=float)

        reco = interp['gamma'] * lloyd_taylor(drivers[1], 1.0, E0)
        beta = np.where(drivers[2] > vpd0,
                        interp['beta0'] * np.exp(-interp['k'] * (drivers[2] - vpd0)),
                        interp['beta0'])
        denom = interp['alpha'] * drivers[0] + beta
        gpp = np.divide(interp['alpha'] * drivers[0] * beta, denom,
                        out=np.zeros_like(drivers[0], dtype=float), where=denom > 0)

        # No photosynthesis without light, whatever the interpolated parameters say. Missing
        # light, or missing VPD in daylight, is unknown rather than zero.
        gpp = np.where(drivers[0] > 0, gpp, 0.0)
        gpp = np.where(np.isnan(drivers[0]) | (np.isnan(drivers[2]) & (drivers[0] > 0)), np.nan, gpp)

        temp.loc[idx, f'reco{suffix}'] = reco
        temp.loc[idx, f'gpp{suffix}'] = gpp
        temp.loc[idx, f'nee{suffix}'] = -gpp + reco

    if par_col == '_par':
        temp = temp.drop(columns=['_par'])
    return(temp, win_df)
