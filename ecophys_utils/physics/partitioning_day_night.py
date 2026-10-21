# Partitioning: Day/night determination functions
#---------------------------------
import datetime as _dt
import numpy as np
from typing import Optional, Union
import pandas as pd
import warnings

def _standard_tzinfo(ts: pd.Series, tz: str) -> _dt.timezone:
    """Fixed-offset timezone of the local standard time of `tz`, for the year of `ts`."""
    # Daylight saving moves clocks forward, so the standard offset is the smaller of the
    # January and July offsets. This also holds for zones that encode it the other way
    # round (Europe/Dublin), where taking the offset minus dst() would not.
    year = int(ts.dropna().dt.year.iloc[0]) if ts.notna().any() else 2000
    offset = min(pd.Timestamp(year, m, 1, 12).tz_localize(tz).utcoffset() for m in (1, 7))
    return _dt.timezone(offset)

def localize_timestamps(timestamps: pd.Series, tz: Optional[str] = None, time_ref: str = 'standard') -> pd.Series:
    """
    Attach a timezone to naive timestamps, according to how they were recorded.

    Parameters
    ----------
    timestamps : pandas.Series or list
        Naive timestamps. Timezone-aware timestamps are returned unchanged.
    tz : str or None, optional
        IANA timezone of the site (e.g. 'Europe/Zurich'). Not needed for time_ref='utc'.
    time_ref : {'standard', 'local', 'utc'}, optional
        'standard' (default): local standard time all year round, i.e. without daylight
        saving time, as most loggers record. 'local': local civil time including daylight
        saving time; timestamps that do not exist or are ambiguous at the clock change
        become NaT. 'utc': timestamps are in UTC and `tz` is ignored.

    Returns
    -------
    pandas.Series
        Timezone-aware timestamps.
    """
    ts = pd.to_datetime(pd.Series(timestamps))
    if ts.dt.tz is not None:
        return ts
    if time_ref == 'utc':
        return ts.dt.tz_localize('UTC')
    if tz is None:
        raise ValueError(f"tz is required for time_ref={time_ref!r}")
    if time_ref == 'local':
        return ts.dt.tz_localize(tz, nonexistent='NaT', ambiguous='NaT')
    if time_ref == 'standard':
        return ts.dt.tz_localize(_standard_tzinfo(ts, tz))
    raise ValueError(f"time_ref must be 'standard', 'local' or 'utc', got {time_ref!r}")

def to_standard_time(timestamps: pd.Series, tz: str, from_ref: str = 'local', ambiguous: Union[str, np.ndarray] = 'infer') -> pd.Series:
    """
    Convert timestamps to local standard time, i.e. local time without daylight saving.

    Parameters
    ----------
    timestamps : pandas.Series or list
        Timestamps to convert. Timezone-aware timestamps are converted from their own
        timezone, and `from_ref` is ignored.
    tz : str
        IANA timezone of the site (e.g. 'Europe/Zurich'), which defines the standard time.
    from_ref : {'local', 'utc', 'standard'}, optional
        How the naive timestamps were recorded: local time including daylight saving
        (default), UTC, or already in standard time (returned unchanged).
    ambiguous : 'infer', 'NaT' or bool array, optional
        Only for from_ref='local': how to resolve the hour that occurs twice when clocks go
        back. 'infer' (default) uses the order of the records, which works for a sorted,
        regularly logged series in which that hour is repeated. If it cannot be inferred,
        the ambiguous records become NaT with a warning. See pandas.Series.dt.tz_localize.
        Local times that do not exist (when clocks go forward) always become NaT.

    Returns
    -------
    pandas.Series
        Naive timestamps in local standard time, aligned to the input.
    """
    ts = pd.to_datetime(pd.Series(timestamps))
    std = _standard_tzinfo(ts, tz)
    if ts.dt.tz is not None:
        aware = ts
    elif from_ref == 'standard':
        return ts
    elif from_ref == 'utc':
        aware = ts.dt.tz_localize('UTC')
    elif from_ref == 'local':
        try:
            aware = ts.dt.tz_localize(tz, ambiguous=ambiguous, nonexistent='NaT')
        except Exception as exc:
            warnings.warn(f'Could not resolve ambiguous local times ({exc}); they are set to NaT.')
            aware = ts.dt.tz_localize(tz, ambiguous='NaT', nonexistent='NaT')
    else:
        raise ValueError(f"from_ref must be 'local', 'utc' or 'standard', got {from_ref!r}")
    return aware.dt.tz_convert(std).dt.tz_localize(None)

def is_day(timestamp_series: pd.Series, lat: float, lon: float, tz: Optional[str] = None, numeric: bool = True, time_ref: str = 'standard') -> pd.Series:
    """
    Determine if timestamps correspond to day or night based on sunrise/sunset.

    Parameters
    ----------
    timestamp_series : pandas.Series or list
        Series of timestamps.
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    tz : str or None, optional
        IANA timezone of the site (e.g. 'Europe/Zurich'). Not needed for time_ref='utc'.
    numeric : bool, optional
        If True, return 1 for day and 0 for night. If False, return 'Day' or 'Night'. Default is True.
    time_ref : {'standard', 'local', 'utc'}, optional
        How the timestamps were recorded. 'standard' (default): local standard time without
        daylight saving time. 'local': local time including daylight saving time. 'utc': UTC.
        See localize_timestamps().

    Returns
    -------
    pandas.Series
        Series indicating day (1 or 'Day') or night (0 or 'Night').
    """
    # Required to calculate Day/Night
    from astral import LocationInfo
    from astral.sun import sunrise, sunset, noon, elevation
    # Create the location object once
    location = LocationInfo(latitude=lat, longitude=lon)

    # Convert timestamps to timezone-aware datetimes
    ts = localize_timestamps(timestamp_series, tz, time_ref)
    tzinfo = ts.dt.tz
    local_date = ts.dt.date

    # Sunrise and sunset once per date of the timestamps rather than per record. The
    # timestamps' tzinfo is passed to astral so that the events fall on that date.
    # sunrise() and sunset() are used instead of sun(), because sun() also computes dawn and
    # dusk and fails in summer at high latitudes, where civil twilight never ends.
    rises, sets, never_day = {}, {}, set()
    for d in local_date.dropna().unique():
        try:
            rises[d] = pd.Timestamp(sunrise(location.observer, date=d, tzinfo=tzinfo)).tz_convert('UTC')
        except ValueError:
            rises[d] = pd.Timestamp.min.tz_localize('UTC')  # No sunrise: day from midnight
        try:
            sets[d] = pd.Timestamp(sunset(location.observer, date=d, tzinfo=tzinfo)).tz_convert('UTC')
        except ValueError:
            sets[d] = pd.Timestamp.max.tz_localize('UTC')  # No sunset: day until midnight
        # Neither event: polar day or polar night, decided by the sun at noon
        if rises[d].year == pd.Timestamp.min.year and sets[d].year == pd.Timestamp.max.year:
            if elevation(location.observer, noon(location.observer, date=d, tzinfo=tzinfo)) <= 0:
                never_day.add(d)

    ts_utc = ts.dt.tz_convert('UTC')
    rise = pd.to_datetime(local_date.map(rises), utc=True)
    set_ = pd.to_datetime(local_date.map(sets), utc=True)
    # If the timestamps' date is not the site's solar date (UTC at a site far from
    # Greenwich), sunset can come before sunrise on the same date; daylight then wraps
    # around midnight.
    wraps = set_ < rise
    day = (((ts_utc >= rise) & (ts_utc <= set_)) | (wraps & ((ts_utc >= rise) | (ts_utc <= set_)))) & ~local_date.isin(never_day)

    # Missing timestamps give None, as before
    if(numeric):
        out = day.astype(int)
        if ts.isna().any():
            out = out.astype(float)
            out[ts.isna()] = np.nan
    else:
        out = pd.Series(np.where(day, 'Day', 'Night'), index=ts.index, dtype=object)
        out[ts.isna()] = None
    return out
    
def create_doy_block_id(timestamps: pd.Series) -> np.ndarray:
    """
    Create day-of-year block IDs for timestamps, adjusted for midday.

    Parameters
    ----------
    timestamps : pandas.Series
        Series of timestamps.

    Returns
    -------
    numpy.ndarray
        Array of block IDs.
    """
    temp = timestamps.to_frame()
    temp['year'] = temp['timestamp'].dt.strftime('%Y').astype(int)
    # Create DOY ('day of year') variable
    temp['blockID'] = temp['timestamp'].dt.strftime('%j').astype(int)
    
    # calculate number of days to add to each doy
    temp['days_in_year'] = np.where(temp['timestamp'].dt.is_leap_year, 366, 365)
    year_lengths = temp[['year','days_in_year']].drop_duplicates()
    year_lengths['annual_day_sums'] = year_lengths['days_in_year'].cumsum().shift(1, fill_value=0)
    # Combine and add the numbers
    temp = temp.merge(year_lengths[['year','annual_day_sums']], on='year', how='left')
    temp['blockID'] = temp['blockID'] + temp['annual_day_sums']

    # Make sure that the blocks are not divided by midnight, but rather midday
    temp.loc[temp['timestamp'].dt.hour < 12, 'blockID'] = temp.loc[temp['timestamp'].dt.hour < 12, 'blockID'] - 1
    
    return(temp['blockID'].values)