# Radiation: solar geometry and reconstruction of incoming shortwave from net radiation
#--------------------------------------------------------------------------------------
#
# Some sites measure net radiation continuously but incoming shortwave only for part of
# the record. Anything that needs a light driver - light-response curves, daytime flux
# partitioning, meteorology-conditioned gap-filling - is then unusable outside that
# window, even though Rn is available throughout.
#
# reconstruct_swin() models SWin from Rn plus the diurnal/seasonal shape of the day. It
# is a Python port of the GAM used in the Jonkershoek analysis:
#
#     gam(SW_IN ~ s(NETRAD) + te(normalized_hour, doy), method='REML', select=TRUE)
#
# implemented with penalised B-spline bases and a linear least-squares solve, so it needs
# only numpy/scipy - no mgcv, statsmodels, sklearn or solar-position package.
#
# NO CEILING IS IMPOSED on the prediction. Cloud-edge lensing can drive surface irradiance
# above the clear-sky value and briefly towards the extraterrestrial one, so clipping
# would remove real behaviour rather than errors. potential_radiation() is provided so
# callers can REPORT how far predictions stray beyond clear-sky - useful for spotting a
# drifted Rn relationship - but nothing here acts on it.
#
# EXTRAPOLATION WARNING: the model can only be fitted where SWin exists. Applying it to
# other periods assumes the Rn -> SWin relationship is stationary. It is not, if surface
# albedo or longwave emission changed in between (after a fire, say, or a canopy change).
# Validation on held-out data from the fitting period is a LOWER BOUND on the error
# elsewhere, never an estimate of it.

import warnings
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from .partitioning_day_night import localize_timestamps

SOLAR_CONSTANT = 1361.0        # W m-2


# ----------------------------------------------------------------- solar geometry -----

def potential_radiation(timestamps: pd.Series, lat: float, lon: float, tz: Optional[str] = None,
                        transmissivity: float = 0.75, offset_mins: float = 0.0,
                        time_ref: str = 'standard') -> pd.DataFrame:
    """
    Extraterrestrial and clear-sky shortwave irradiance on a horizontal surface.

    Uses the Spencer (1971) series for declination and the equation of time. Intended as a
    DIAGNOSTIC reference, not as a constraint - see the module docstring.

    Parameters
    ----------
    timestamps : pandas.Series
        Naive timestamps, recorded as described by `time_ref`.
    lat, lon : float
        Latitude and longitude in decimal degrees; north and east positive.
    tz : str or None, optional
        IANA timezone of the site (e.g. 'Europe/Zurich'), used to convert to UTC. Not
        needed for time_ref='utc'.
    transmissivity : float, optional
        Bulk clear-sky atmospheric transmissivity. Default 0.75.
    offset_mins : float, optional
        Shift applied before computing solar position, for records timestamped at the END
        of an averaging period (e.g. -15 for half-hourly means). Default 0.
    time_ref : {'standard', 'local', 'utc'}, optional
        How the timestamps were recorded: local standard time without daylight saving
        (default), local time with daylight saving, or UTC. See localize_timestamps().

    Returns
    -------
    pandas.DataFrame
        'sw_extraterrestrial' and 'sw_clearsky' (W m-2), 'solar_zenith_deg', and
        'solar_azimuth_deg' measured clockwise from north (0 = N, 90 = E, 180 = S, 270 = W).
    """
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    if offset_mins:
        ts = ts + pd.Timedelta(minutes=offset_mins)

    # To UTC. Many flux sites log in standard time year-round, hence the default. Local
    # times that are ambiguous or nonexistent at a DST change become NaT and give NaN
    # rather than raising, since a DST edge case must not abort a whole record.
    try:
        utc = localize_timestamps(ts, tz, time_ref).dt.tz_convert('UTC')
    except ValueError:  # Invalid time_ref, or tz missing
        raise
    except Exception as exc:
        warnings.warn(f"Could not localise timestamps to {tz!r} ({exc}); assuming UTC.")
        utc = ts.dt.tz_localize('UTC')

    doy = utc.dt.dayofyear.to_numpy(dtype=float)
    hour_utc = (utc.dt.hour + utc.dt.minute / 60.0 + utc.dt.second / 3600.0).to_numpy(dtype=float)

    gamma = 2.0 * np.pi * (doy - 1.0) / 365.0

    # Earth-Sun distance correction
    e0 = (1.00011 + 0.034221 * np.cos(gamma) + 0.00128 * np.sin(gamma)
          + 0.000719 * np.cos(2 * gamma) + 0.000077 * np.sin(2 * gamma))

    # Solar declination [rad]
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))

    # Equation of time [minutes]
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                    - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))

    # True solar time from UTC: 4 min per degree of longitude, plus the equation of time
    solar_hour = hour_utc + (4.0 * lon + eot) / 60.0
    hour_angle = np.deg2rad(15.0 * (solar_hour - 12.0))

    phi = np.deg2rad(lat)
    cos_zenith = (np.sin(phi) * np.sin(decl)
                  + np.cos(phi) * np.cos(decl) * np.cos(hour_angle))
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)

    daylight = np.maximum(cos_zenith, 0.0)
    extra = SOLAR_CONSTANT * e0 * daylight

    # Azimuth clockwise from north. Needed to resolve terrain obstruction by direction:
    # a horizon profile is a function of azimuth, whereas any morning/afternoon split is an
    # arbitrary slice that mixes directions differently across seasons.
    elev = np.arcsin(cos_zenith)
    cos_elev = np.cos(elev)
    with np.errstate(invalid='ignore', divide='ignore'):
        sin_az = -np.sin(hour_angle) * np.cos(decl) / cos_elev
        cos_az = (np.sin(decl) - np.sin(phi) * np.sin(elev)) / (np.cos(phi) * cos_elev)
    az = np.rad2deg(np.arctan2(np.clip(sin_az, -1, 1), np.clip(cos_az, -1, 1))) % 360.0

    return(pd.DataFrame({
        'sw_extraterrestrial': extra,
        'sw_clearsky': transmissivity * extra,
        'solar_zenith_deg': np.rad2deg(np.arccos(cos_zenith)),
        'solar_azimuth_deg': az,
    }, index=pd.Series(timestamps).index))


def normalized_daylight_hour(temp: pd.DataFrame, timestamp_col: str = 'timestamp',
                             dn_col: str = 'day_night') -> pd.Series:
    """
    Position within the daylight period: 0 at sunrise, 1 at sunset, NaN at night.

    Sunrise and sunset are taken as the first and last daytime record of each calendar
    day, exactly as the original R implementation does, so no solar-position library is
    required and the definition matches whatever `dn_col` already encodes.

    Parameters
    ----------
    temp : pandas.DataFrame
        Data with a timestamp and a day/night indicator (1 = day, 0 = night).
    timestamp_col, dn_col : str, optional
        Column names.

    Returns
    -------
    pandas.Series
        Normalised hour, aligned to `temp.index`.
    """
    ts = pd.to_datetime(temp[timestamp_col])
    date = ts.dt.normalize()
    is_day = temp[dn_col] == 1

    sun = (pd.DataFrame({'date': date, 'ts': ts})[is_day.to_numpy()]
           .groupby('date')['ts'].agg(sunrise='min', sunset='max'))

    sr = date.map(sun['sunrise'])
    ss = date.map(sun['sunset'])
    daylight_secs = (ss - sr).dt.total_seconds()
    elapsed = (ts - sr).dt.total_seconds()

    out = elapsed / daylight_secs.where(daylight_secs > 0)
    return(out.where(is_day))


# ------------------------------------------------------------------ terrain horizon ----

def fit_horizon_profile(temp: pd.DataFrame, rad_col: str = 'SW_IN_1_1_1',
                        extra_col: str = 'sw_extraterrestrial',
                        azimuth_col: str = 'solar_azimuth_deg',
                        zenith_col: str = 'solar_zenith_deg',
                        az_bin: float = 15.0, elev_bin: float = 2.0,
                        max_horizon: float = 35.0, min_n: int = 15,
                        smooth_bins: int = 3, quantile: float = 0.95,
                        subtract_night_baseline: bool = False,
                        timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Estimate the terrain horizon elevation as a function of solar azimuth.

    In each azimuth sector the normalised radiation rises with solar elevation in three
    parts: a low plateau where the direct beam is blocked and only diffuse sky remains, a
    transition at the horizon, and an open-sky plateau above it.

    The difficulty is that radiation also falls at low sun WITHOUT any terrain, because the
    atmospheric path lengthens. Absolute thresholds therefore cannot separate terrain from
    air mass. Instead each azimuth is compared against an OPEN-SKY REFERENCE built from the
    least-obstructed azimuths at the same elevation, so only *directional* deficits are
    attributed to terrain.

    TERRAIN VERSUS CLOUD - the distinction this function turns on. A ridge blocks the direct
    beam on EVERY day, so the *upper envelope* of clearness stays at the diffuse level below
    the horizon. Persistent directional cloud (valley fog, orographic cloud on one flank)
    lowers the TYPICAL clearness just as much, but clear days still punch through, so the
    envelope recovers. Fitting on the median therefore mistakes a cloudy aspect for a
    mountain; fitting on a high `quantile` does not. At Jonkershoek the median suggested a
    36 deg eastern horizon while the maximum reached 0.65 clearness underneath it - morning
    cloud, not rock.

    Parameters
    ----------
    temp : pandas.DataFrame
        Data with radiation, solar azimuth and zenith. Run potential_radiation() first.
    rad_col : str, optional
        Radiation used to detect obstruction. **Use a measured shortwave column.**

        Net radiation is tempting because it usually has far better coverage, but it is NOT
        a clearness index: the longwave loss makes Rn/extraterrestrial negative at low sun.
        Removing a night-time baseline (`subtract_night_baseline`) is not enough either,
        because the daytime longwave loss grows as the surface warms, leaving a morning
        deficit that masquerades as an eastern ridge. Tried at Jonkershoek, the Rn route
        returned a 30 deg eastern horizon against 10-12 deg from measured SWin, and failed
        the pre/post-fire agreement test (34 vs 24 deg) - correctly, because it was tracking
        surface properties. Kept as an option only for sites with a genuinely small or
        stable longwave term.
    extra_col : str, optional
        Extraterrestrial irradiance used to normalise. Default 'sw_extraterrestrial'.
    az_bin, elev_bin : float, optional
        Azimuth and elevation bin widths in degrees.
    max_horizon : float, optional
        Largest horizon elevation considered. Default 35 deg.
    min_n : int, optional
        Minimum records per (azimuth, elevation) cell. Default 20.
    smooth_bins : int, optional
        Rolling-median width (in azimuth bins) applied to the fitted profile. A horizon is
        a physical surface and must be smooth; smoothing across AZIMUTH is legitimate in a
        way that interpolating across day-of-year is not.
    quantile : float, optional
        Upper quantile of clearness used per cell. Default 0.95 - see the note above on
        terrain versus cloud. Lower it towards 0.5 only to deliberately capture typical
        rather than best-case conditions, which is NOT a horizon.
    subtract_night_baseline : bool, optional
        Estimate the night-time level of `rad_col` per day and subtract it before forming
        the ratio. Default True. For net radiation this removes the longwave loss, leaving
        a quantity proportional to net SHORTWAVE. The proportionality constant (1 - albedo)
        is unknown and changes if the surface changes, but it CANCELS: every azimuth is
        judged against an open-sky reference at the same elevation, so only the ratio
        matters. That is what allows the horizon to be fitted from a long Rn record instead
        of a short SWin one - and hence allows the pre/post-fire agreement check, which is
        the strongest evidence that the profile is geometry rather than surface.
        Set False when `rad_col` is already a pure shortwave measurement.
    timestamp_col : str, optional
        Used to group by day when estimating the night-time baseline.

    Returns
    -------
    pandas.DataFrame
        One row per azimuth bin: 'azimuth', 'horizon_elev_deg', 'diffuse_fraction',
        'open_level', 'shaded_level', 'n'.

    Notes
    -----
    Terrain does not burn. Fitting this separately for periods with different surface
    properties and requiring the results to agree is a strong check that the profile is
    geometry rather than an artefact of albedo or emissivity.
    """
    cols = [rad_col, extra_col, azimuth_col, zenith_col]
    if subtract_night_baseline and timestamp_col in temp.columns:
        cols = cols + [timestamp_col]
    d = temp[cols].dropna().copy()
    d['elev'] = 90.0 - d[zenith_col]

    # Remove the longwave offset, so what remains is proportional to net shortwave.
    # Estimated per day from periods when the sun is down, then interpolated across days
    # that have no night-time record.
    if subtract_night_baseline and timestamp_col in d.columns:
        night = d[d[extra_col] <= 0]
        day_key = d[timestamp_col].dt.normalize()
        base = night.groupby(night[timestamp_col].dt.normalize())[rad_col].median()
        base = base.reindex(np.sort(day_key.unique())).interpolate(limit_direction='both')
        d['_base'] = day_key.map(base).fillna(base.median() if len(base) else 0.0)
        d[rad_col] = d[rad_col] - d['_base']

    d = d[(d['elev'] > 0) & (d[extra_col] > 20)]
    if d.empty:
        raise ValueError('No daytime records with radiation and solar geometry')

    d['r'] = d[rad_col] / d[extra_col]
    d['az_bin'] = (d[azimuth_col] // az_bin) * az_bin
    d['el_bin'] = (d['elev'] // elev_bin) * elev_bin

    # Upper envelope per cell, NOT the median: only terrain caps the best-case radiation.
    cell = (d.groupby(['az_bin', 'el_bin'])['r']
              .agg(r=lambda s: s.quantile(quantile), r_median='median', n='size')
              .reset_index())
    cell = cell[cell['n'] >= min_n]
    if cell.empty:
        raise ValueError('No azimuth/elevation cell has enough records; widen the bins')

    # Open-sky reference per elevation: the upper envelope across azimuths. Using a high
    # quantile rather than the maximum keeps one noisy sector from defining the reference.
    ref = cell.groupby('el_bin')['r'].quantile(0.9).rename('r_open')
    cell = cell.join(ref, on='el_bin')

    rows = []
    for az, g in cell.groupby('az_bin'):
        g = g.sort_values('el_bin')
        open_level = g.loc[g['el_bin'] > max_horizon, 'r'].median()
        if not np.isfinite(open_level):
            open_level = g['r'].max()

        # Obstructed where this azimuth falls well below the open-sky reference
        obstructed = g[(g['el_bin'] <= max_horizon) & (g['r'] < 0.5 * g['r_open'])]
        horizon = float(obstructed['el_bin'].max() + elev_bin) if len(obstructed) else 0.0

        shaded = g.loc[g['el_bin'] < horizon, 'r']
        shaded_level = float(shaded.median()) if len(shaded) else np.nan

        # Diffuse fraction: what survives when the direct beam is blocked. Taken from the
        # data rather than assumed, so it is consistent with this site and sensor.
        if np.isfinite(shaded_level) and open_level > 0:
            diffuse = float(np.clip(shaded_level / open_level, 0.0, 1.0))
        else:
            diffuse = np.nan

        # Typical-condition deficit, for comparison. A large gap between this and the
        # horizon means the aspect is CLOUDY rather than obstructed - worth seeing, and not
        # something to correct with a geometric shading term.
        typ = g.loc[g['el_bin'] <= max_horizon, 'r_median'].median()

        rows.append({'azimuth': az, 'horizon_elev_deg': horizon, 'diffuse_fraction': diffuse,
                     'open_level': open_level, 'shaded_level': shaded_level,
                     'median_level_lowsun': typ, 'n': int(g['n'].sum())})

    prof = pd.DataFrame(rows).sort_values('azimuth').reset_index(drop=True)

    # Smooth around the compass; a horizon cannot jump between adjacent sectors
    if smooth_bins > 1 and len(prof) >= smooth_bins:
        k = len(prof)
        tripled = pd.concat([prof['horizon_elev_deg']] * 3, ignore_index=True)
        sm = tripled.rolling(smooth_bins, center=True, min_periods=1).median()
        prof['horizon_elev_deg'] = sm.iloc[k:2 * k].to_numpy()

    prof['diffuse_fraction'] = prof['diffuse_fraction'].fillna(
        prof['diffuse_fraction'].median() if prof['diffuse_fraction'].notna().any() else 0.15)
    return(prof)


def terrain_clearsky(temp: pd.DataFrame, horizon: pd.DataFrame,
                     clearsky_col: str = 'sw_clearsky',
                     azimuth_col: str = 'solar_azimuth_deg',
                     zenith_col: str = 'solar_zenith_deg',
                     out_col: str = 'sw_clearsky_terrain') -> pd.DataFrame:
    """
    Clear-sky irradiance with the direct beam removed where terrain blocks the sun.

    Shade is not darkness: when the sun drops behind a ridge the direct beam is lost but
    diffuse sky radiation continues, so the diffuse fraction is retained. That fraction
    comes from the horizon fit, where it was measured as the level surviving in already
    shaded conditions.

    Parameters
    ----------
    temp : pandas.DataFrame
        Data with solar geometry and clear-sky irradiance.
    horizon : pandas.DataFrame
        Output of fit_horizon_profile().
    clearsky_col, azimuth_col, zenith_col, out_col : str, optional
        Column names.

    Returns
    -------
    pandas.DataFrame
        Copy of `temp` with `out_col` and 'terrain_shaded' (bool) added.
    """
    out = temp.copy()
    az_bin = float(np.diff(np.sort(horizon['azimuth'].unique())).min())

    key = (out[azimuth_col] // az_bin) * az_bin
    h = key.map(horizon.set_index('azimuth')['horizon_elev_deg'])
    # A DEM-derived profile carries no diffuse fraction (it knows geometry, not optics), so
    # fall back to a typical clear-sky diffuse share.
    if 'diffuse_fraction' in horizon.columns:
        f = key.map(horizon.set_index('azimuth')['diffuse_fraction'])
    else:
        f = pd.Series(0.18, index=out.index)
        horizon = horizon.assign(diffuse_fraction=0.18)
    # Sectors the sun never reaches have no fitted value; treat as unobstructed
    h = h.fillna(0.0)
    f = f.fillna(float(horizon['diffuse_fraction'].median()))

    elev = 90.0 - out[zenith_col]
    shaded = (elev <= h) & (elev > 0)

    out[out_col] = np.where(shaded, out[clearsky_col] * f, out[clearsky_col])
    out['terrain_shaded'] = shaded
    return(out)


# --------------------------------------------------------- penalised spline machinery --

def _knots(x: np.ndarray, n_basis: int, degree: int) -> np.ndarray:
    """Clamped knot vector spanning the data range, giving `n_basis` B-spline functions."""
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError('Cannot build a spline basis: predictor has no finite range')
    n_interior = n_basis - degree - 1
    interior = np.linspace(lo, hi, n_interior + 2)[1:-1]
    return(np.concatenate([np.repeat(lo, degree + 1), interior, np.repeat(hi, degree + 1)]))


def _design(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """B-spline design matrix, with values clamped into the basis interval."""
    xc = np.clip(x, knots[degree], knots[-degree - 1])
    return(np.asarray(BSpline.design_matrix(xc, knots, degree, extrapolate=False).todense()))


def _diff_penalty(n: int, order: int = 2) -> np.ndarray:
    """Second-difference roughness penalty S = D'D."""
    D = np.diff(np.eye(n), n=order, axis=0)
    return(D.T @ D)


def _row_kron(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise Kronecker product, i.e. the tensor-product basis."""
    n, p = A.shape
    q = B.shape[1]
    return((A[:, :, None] * B[:, None, :]).reshape(n, p * q))


def _fit_penalised(X: np.ndarray, y: np.ndarray, penalties: Sequence[np.ndarray],
                   lambda_grid: Sequence[float]) -> Tuple[np.ndarray, dict]:
    """
    Penalised least squares with one smoothing parameter per penalty block, chosen by GCV.

    Stands in for mgcv's REML selection. The grid is coarse on purpose: the fit is used as
    a driver reconstruction, not for inference on the smooth terms themselves.
    """
    n, p = X.shape
    XtX = X.T @ X
    Xty = X.T @ y

    best = None
    for lam in np.ndindex(*[len(lambda_grid)] * len(penalties)):
        lams = [lambda_grid[i] for i in lam]
        P = np.zeros((p, p))
        for l, S in zip(lams, penalties):
            P += l * S
        A = XtX + P
        # A tiny ridge keeps the solve conditioned. B-spline bases for several predictors
        # are close to collinear (they all contain a near-constant direction), and without
        # this the trace below can come back negative - an impossible edf that then feeds
        # straight into GCV and corrupts the smoothing choice.
        A = A + 1e-8 * np.trace(A) / p * np.eye(p)
        try:
            beta = np.linalg.solve(A, Xty)
            edf = float(np.trace(np.linalg.solve(A, XtX)))
        except np.linalg.LinAlgError:
            continue
        # edf must lie between 1 and the number of columns; anything else means the solve
        # was ill-conditioned and this smoothing parameter cannot be scored.
        if not np.isfinite(edf) or edf < 0.5 or edf > p + 1e-6:
            continue
        rss = float(np.sum((y - X @ beta) ** 2))
        if not np.isfinite(rss):
            continue
        denom = (n - edf) ** 2
        if denom <= 0:
            continue
        gcv = n * rss / denom
        if best is None or gcv < best[0]:
            best = (gcv, beta, lams, edf, rss)

    if best is None:
        raise RuntimeError('Penalised fit failed for every smoothing parameter tried')
    gcv, beta, lams, edf, rss = best

    # A smoothing parameter chosen at the edge of the grid means the optimum may lie
    # outside it, so the reported fit is grid-limited rather than GCV-optimal. Harmless if
    # held-out performance is good, misleading if taken as an optimum.
    edge = [l for l in lams if l in (lambda_grid[0], lambda_grid[-1])]
    if edge:
        warnings.warn(
            f'GCV selected a smoothing parameter at the edge of the search grid '
            f'({edge}); the optimum may lie outside [{lambda_grid[0]:g}, '
            f'{lambda_grid[-1]:g}]. Check held-out error before relying on the fit.'
        )

    return(beta, {'gcv': gcv, 'lambdas': lams, 'edf': edf,
                  'rmse': float(np.sqrt(rss / n)), 'n': n,
                  'lambda_at_grid_edge': bool(edge)})


# ------------------------------------------------------------------- reconstruction ----

def reconstruct_swin(temp: pd.DataFrame, sw_col: str = 'SW_IN_1_1_1',
                     rn_col: str = 'NETRAD_1_1_1', timestamp_col: str = 'timestamp',
                     dn_col: str = 'day_night', out_col: str = 'swin_reconstructed',
                     n_basis_rn: int = 10, n_basis_hour: int = 8, n_basis_doy: int = 8,
                     degree: int = 3, lambda_grid: Sequence[float] = (1e-3, 1e-1, 1e1, 1e3, 1e5),
                     fit_mask: Optional[pd.Series] = None, method: str = 'clearness',
                     pot_col: str = 'sw_clearsky',
                     zenith_col: str = 'solar_zenith_deg',
                     min_pot: float = 20.0,
                     extra_cols: Sequence[str] = ()) -> Tuple[pd.DataFrame, dict]:
    """
    Reconstruct incoming shortwave radiation from net radiation and time of day/year.

    Two methods are available, and the choice matters most for months with no measured SWin:

    - ``method='clearness'`` (default) models the clearness index against ``Rn / SW_pot`` and
      solar elevation, taking the seasonal and diurnal cycle from solar geometry rather than
      fitting it.
    - ``method='gam'`` ports `gam(SW_IN ~ s(NETRAD) + te(normalized_hour, doy))`.

    WHY THE DEFAULT IS 'clearness'. The two differ in what they must extrapolate. At the
    Jonkershoek site SWin was measured only June-December, and the unmeasured months are
    mirror images of measured ones in SOLAR GEOMETRY - January's sun reaches 79.0 deg against
    December's 79.4 - so 100 % of them fall inside the clearness model's predictor range.
    They fall inside 0 % of the GAM's, whose ``te(hour, doy)`` surface simply has no data for
    those days of year. Reconstructing them is interpolation for one method and extrapolation
    for the other, and no amount of holdout testing changes that.

    TERRAIN. `pot_col` defaults to plain clear-sky radiation. A terrain-shaded version can be
    supplied via `terrain_clearsky()`, but at this site that was measured and rejected: it
    changed mean reconstructed SWin by 0.37 % while doubling the error against measured SWin
    (RMSE 60.5 vs 32.8 W m-2), because a hard direct-beam block over-corrects. Re-measure
    before re-enabling it.

    The result goes in its OWN column and `sw_col` is never modified, so the two can be
    compared wherever both exist. The reconstruction is produced for the fitting period
    too, which is what makes that comparison possible on every measured record rather than
    only on a held-out subset.

    Parameters
    ----------
    temp : pandas.DataFrame
        Half-hourly data.
    sw_col : str, optional
        Measured incoming shortwave (W m-2). Read only, never written.
    rn_col : str, optional
        Net radiation (W m-2) - the predictor available throughout.
    timestamp_col, dn_col : str, optional
        Timestamp and day/night (1 = day, 0 = night) columns.
    out_col : str, optional
        Name of the reconstructed column. Default 'swin_reconstructed'.
    n_basis_rn, n_basis_hour, n_basis_doy : int, optional
        Basis sizes for the Rn smooth and the two tensor-product margins.
    degree : int, optional
        B-spline degree. Default 3 (cubic).
    lambda_grid : sequence of float, optional
        Smoothing parameters searched by GCV, per penalty block.
    fit_mask : pandas.Series or None, optional
        Extra boolean restricting which rows are used for FITTING (prediction still covers
        everything). Use for held-out validation.

    Returns
    -------
    (pandas.DataFrame, dict)
        Copy of `temp` with `out_col` added, and a diagnostics dict with the fit summary.

    Notes
    -----
    Night-time values are set to 0 and negative predictions to 0, matching the original R
    implementation. No upper limit is applied - see the module docstring on cloud lensing.
    """
    out = temp.copy()

    if method == 'clearness':
        return(_reconstruct_clearness(out, sw_col=sw_col, rn_col=rn_col, pot_col=pot_col,
                                      zenith_col=zenith_col, dn_col=dn_col, out_col=out_col,
                                      degree=degree, lambda_grid=lambda_grid,
                                      fit_mask=fit_mask, min_pot=min_pot,
                                      extra_cols=extra_cols))
    if method != 'gam':
        raise ValueError(f"method must be 'gam' or 'clearness', got {method!r}")

    nh = normalized_daylight_hour(out, timestamp_col=timestamp_col, dn_col=dn_col)
    doy = pd.to_datetime(out[timestamp_col]).dt.dayofyear.astype(float)
    rn = out[rn_col].astype(float)
    is_day = (out[dn_col] == 1)

    # Rows where the model is defined at all (daytime, drivers present)
    predictable = is_day & rn.notna() & nh.notna() & doy.notna()
    trainable = predictable & out[sw_col].notna()
    if fit_mask is not None:
        trainable = trainable & fit_mask.reindex(out.index).fillna(False).astype(bool)

    n_train = int(trainable.sum())
    if n_train < 100:
        raise ValueError(f'Only {n_train} rows available to fit {sw_col}; need at least 100')

    # Bases are built on the FULL predictable range, not just the training subset, so that
    # prediction never has to extrapolate beyond the knot span.
    k_rn = _knots(rn[predictable].to_numpy(), n_basis_rn, degree)
    k_hour = _knots(nh[predictable].to_numpy(), n_basis_hour, degree)
    k_doy = _knots(doy[predictable].to_numpy(), n_basis_doy, degree)

    def build(idx):
        b_rn = _design(rn[idx].to_numpy(), k_rn, degree)
        b_hour = _design(nh[idx].to_numpy(), k_hour, degree)
        b_doy = _design(doy[idx].to_numpy(), k_doy, degree)
        b_te = _row_kron(b_hour, b_doy)
        intercept = np.ones((b_rn.shape[0], 1))
        return np.hstack([intercept, b_rn, b_te]), b_rn.shape[1], b_te.shape[1]

    X_tr, p_rn, p_te = build(trainable)
    y_tr = out.loc[trainable, sw_col].to_numpy(dtype=float)

    # Block-diagonal penalties: roughness of the Rn smooth, and of the tensor product in
    # each of its two directions.
    p_total = X_tr.shape[1]
    S_rn = np.zeros((p_total, p_total))
    S_rn[1:1 + p_rn, 1:1 + p_rn] = _diff_penalty(p_rn)

    n_h, n_d = n_basis_hour, n_basis_doy
    S_te_block = (np.kron(_diff_penalty(n_h), np.eye(n_d))
                  + np.kron(np.eye(n_h), _diff_penalty(n_d)))
    S_te = np.zeros((p_total, p_total))
    S_te[1 + p_rn:, 1 + p_rn:] = S_te_block

    beta, info = _fit_penalised(X_tr, y_tr, [S_rn, S_te], lambda_grid)

    # Predict everywhere the model is defined; night and negatives handled as in R
    X_pred, _, _ = build(predictable)
    pred = pd.Series(np.nan, index=out.index, dtype=float)
    pred.loc[predictable] = X_pred @ beta
    pred = pred.clip(lower=0.0)
    pred.loc[out[dn_col] == 0] = 0.0

    out[out_col] = pred

    info.update({
        'n_predicted': int(pred.notna().sum()),
        'coverage_perc': 100.0 * pred.notna().mean(),
        'sw_col_untouched': bool(out[sw_col].equals(temp[sw_col])),
    })
    return(out, info)


def _reconstruct_clearness(out: pd.DataFrame, sw_col: str, rn_col: str, pot_col: str,
                           zenith_col: str, dn_col: str, out_col: str, degree: int,
                           lambda_grid: Sequence[float], fit_mask: Optional[pd.Series],
                           min_pot: float, n_basis: int = 10,
                           extra_cols: Sequence[str] = ()) -> Tuple[pd.DataFrame, dict]:
    """
    Clearness-index reconstruction (see reconstruct_swin(method='clearness')).

    Models kt = SW_IN / SW_pot as an additive function of Rn / SW_pot, solar elevation and
    any `extra_cols`. SW_pot is ANALYTIC, so the seasonal and diurnal cycle is computed
    rather than fitted; only the relationship between net radiation and atmospheric
    transmission is learned.

    WHY ADDITIVE, AND WHY 10 BASIS FUNCTIONS. All of the obvious ways to add flexibility
    were measured against a cross-sensor holdout at Jonkershoek and every one made
    prediction WORSE while improving the in-sample fit: a tensor interaction between the two
    main predictors gave R2 0.739 against the additive 0.748, adding clear-sky weighting
    0.738, and enlarging the basis to 14 gave 0.744. The model is not underfitting. Re-test
    before adding flexibility here.

    Nothing is clipped from above: kt may exceed 1 under cloud-edge enhancement.
    """
    if pot_col not in out.columns:
        raise KeyError(f'{pot_col!r} not found - compute clear-sky radiation first')

    elev = 90.0 - out[zenith_col]
    pot = out[pot_col].astype(float)
    rn = out[rn_col].astype(float)

    usable = (out[dn_col] == 1) & (pot > min_pot) & rn.notna() & elev.notna()
    for c in extra_cols:
        if c not in out.columns:
            raise KeyError(f'extra predictor {c!r} not found')
        usable = usable & out[c].notna()

    trainable = usable & out[sw_col].notna()
    if fit_mask is not None:
        trainable = trainable & fit_mask.reindex(out.index).fillna(False).astype(bool)
    if int(trainable.sum()) < 100:
        raise ValueError(f'Only {int(trainable.sum())} rows available to fit; need 100+')

    # Predictor 1 is trimmed because Rn/SW_pot has a heavy tail near sunrise and sunset,
    # where a small denominator turns ordinary noise into extreme leverage.
    x1 = (rn / pot).where(usable)
    x1 = x1.clip(*np.nanpercentile(x1[trainable], [0.5, 99.5]))
    preds = [x1, elev.where(usable)] + [out[c].astype(float).where(usable) for c in extra_cols]

    knots = [_knots(s[usable].to_numpy(), n_basis, degree) for s in preds]

    def build(idx):
        blocks = [np.ones((int(idx.sum()), 1))]
        blocks += [_design(s[idx].to_numpy(), k, degree) for s, k in zip(preds, knots)]
        return np.hstack(blocks)

    X_tr = build(trainable)
    y_tr = (out.loc[trainable, sw_col] / pot[trainable]).to_numpy(dtype=float)

    p = X_tr.shape[1]
    penalties = []
    for i in range(len(preds)):
        S = np.zeros((p, p))
        lo = 1 + n_basis * i
        S[lo:lo + n_basis, lo:lo + n_basis] = _diff_penalty(n_basis)
        penalties.append(S)
    beta, info = _fit_penalised(X_tr, y_tr, penalties, lambda_grid)

    kt = pd.Series(np.nan, index=out.index, dtype=float)
    kt.loc[usable] = build(usable) @ beta
    kt = kt.clip(lower=0.0)

    pred = (kt * pot).clip(lower=0.0)
    pred.loc[out[dn_col] == 0] = 0.0
    out[out_col] = pred
    out[out_col + '_kt'] = kt

    info.update({'method': 'clearness',
                 'predictors': ['Rn/SW_pot', 'solar_elevation'] + list(extra_cols),
                 'n_predicted': int(pred.notna().sum()),
                 'coverage_perc': 100.0 * pred.notna().mean(),
                 'sw_col_untouched': True})
    return(out, info)


def evaluate_swin(temp: pd.DataFrame, sw_col: str = 'SW_IN_1_1_1',
                  out_col: str = 'swin_reconstructed', dn_col: str = 'day_night',
                  by: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Compare a reconstruction against measured shortwave wherever both exist.

    A reconstruction can be good on average and still fail systematically at low sun
    angles, which is exactly where the tails of a light-response curve come from - so
    break the comparison down rather than reporting one number.

    Parameters
    ----------
    temp : pandas.DataFrame
        Data containing both columns.
    sw_col, out_col, dn_col : str, optional
        Measured, reconstructed and day/night column names.
    by : sequence of str or None, optional
        Extra grouping columns (e.g. ['season'] or ['hour']). Default None (overall only).

    Returns
    -------
    pandas.DataFrame
        n, bias, RMSE, R^2 and the mean of each series, per group.
    """
    d = temp.loc[(temp[dn_col] == 1) & temp[sw_col].notna() & temp[out_col].notna()].copy()
    if d.empty:
        return(pd.DataFrame())

    def stats(g):
        o, p = g[sw_col].to_numpy(float), g[out_col].to_numpy(float)
        resid = p - o
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((o - o.mean()) ** 2))
        return pd.Series({
            'n': len(g),
            'obs_mean': o.mean(),
            'pred_mean': p.mean(),
            'bias': resid.mean(),
            'rmse': float(np.sqrt(ss_res / len(g))),
            'r2': 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        })

    if by:
        return(d.groupby(list(by), observed=True).apply(stats, include_groups=False).reset_index())
    return(stats(d).to_frame().T)
