# Radiation cleanup: night-time offset of shortwave sensors
#----------------------------------------------------------
#
# Pyranometers rarely read exactly 0 at night. Thermopile sensors in particular show a
# thermal offset (typically a few W m-2, usually negative) caused by net longwave exchange
# between the sensor dome and the sky. That offset does not switch off at sunrise: it is
# present in the daytime readings too, so clipping only the night values to 0 leaves the
# daytime bias untouched.
#
# correct_nighttime_swr_offset() therefore estimates the offset from the night records of
# each calendar day (midnight to midnight, i.e. the early-morning and the evening part of
# two different nights), subtracts it from EVERY record of that day, and only then forces
# the night values to 0.

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

def correct_nighttime_swr_offset(temp: pd.DataFrame, col: str = 'SW_IN_1_1_1',
                                 timestamp_col: str = 'timestamp',
                                 dn_col: Optional[str] = 'day_night',
                                 lat: Optional[float] = None, lon: Optional[float] = None,
                                 tz: Optional[str] = None,
                                 time_ref: str = 'standard',
                                 min_night_records: int = 6,
                                 return_offset: bool = False,
                                 silent: bool = False
                                 ) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Remove the night-time offset of a shortwave radiation sensor.

    Two steps:

    1. For each calendar day (00:00-24:00), the offset is the median of the night-time
       values of `col` on that day. It is subtracted from all records of that day, day and
       night alike. Days with fewer than `min_night_records` valid night values get their
       offset interpolated in time from neighbouring days.
    2. All night-time values are then set to 0, whatever their sign after step 1.

    Missing values stay missing. The input column is never modified.

    Parameters
    ----------
    temp : pandas.DataFrame
        Data containing the timestamp and radiation columns.
    col : str, optional
        Shortwave radiation column (e.g. incoming SW or PPFD). Default 'SW_IN_1_1_1'.
    timestamp_col : str, optional
        Timestamp column, in local time. Default 'timestamp'.
    dn_col : str or None, optional
        Day/night column (1 = day, 0 = night), e.g. from is_day(). Used if present in
        `temp`; otherwise night is detected with astral from `lat`, `lon` and `tz`.
        Default 'day_night'.
    lat, lon : float, optional
        Site latitude and longitude in decimal degrees, needed only if `dn_col` is absent.
    tz : str, optional
        IANA timezone of the site, needed only if `dn_col` is absent (and not for UTC).
    time_ref : {'standard', 'local', 'utc'}, optional
        How the timestamps were recorded, used only if `dn_col` is absent: local standard
        time without daylight saving (default), local time with daylight saving, or UTC.
        Calendar days are always those of the timestamps as recorded.
    min_night_records : int, optional
        Minimum number of valid night records needed to estimate a day's offset. Default 6
        (3 hours of half-hourly data).
    return_offset : bool, optional
        If True, also return the offset per calendar day. Default False.
    silent : bool, optional
        If False, print a summary of the correction.

    Returns
    -------
    pandas.Series
        Corrected `col`, aligned to `temp.index`.
    pandas.Series
        Only if `return_offset` is True: offset per calendar day (indexed by date), in the
        units of `col`. This is the value that was subtracted.
    """
    ts = pd.to_datetime(temp[timestamp_col])

    # Night mask
    if dn_col is not None and dn_col in temp.columns:
        dn = pd.to_numeric(temp[dn_col], errors='coerce')
    elif lat is not None and lon is not None and (tz is not None or time_ref == 'utc'):
        from ..physics.partitioning_day_night import is_day
        dn = pd.to_numeric(is_day(ts, lat, lon, tz, time_ref=time_ref), errors='coerce')
    else:
        raise ValueError(f'Day/night column {dn_col!r} not found; provide lat, lon and tz (or time_ref=\'utc\') to detect night with astral')
    night = (dn == 0) & ts.notna()

    raw = pd.to_numeric(temp[col], errors='coerce')
    day = ts.dt.normalize()

    # Step 1: median night-time value per calendar day
    night_vals = raw[night & raw.notna()]
    stats = night_vals.groupby(day[night_vals.index]).agg(['median', 'count'])
    offset = stats['median'].where(stats['count'] >= min_night_records)

    # Interpolate the offset for days without enough night data
    all_days = pd.DatetimeIndex(day.dropna().unique()).sort_values()
    offset = offset.reindex(all_days)
    n_estimated = int(offset.notna().sum())
    if n_estimated == 0:
        warnings.warn(f'No day has {min_night_records} or more night-time values of {col}; no offset subtracted.')
        offset = offset.fillna(0.0)
    else:
        offset = offset.interpolate(method='time', limit_direction='both')
    offset.index.name = 'date'
    offset.name = 'offset'

    corrected = raw - day.map(offset).fillna(0.0)

    # Step 2: night is 0, keeping gaps as gaps
    corrected[night & raw.notna()] = 0.0
    corrected.name = col

    if(not silent):
        print(f'    - {col}: night-time offset estimated for {n_estimated} of {len(offset)} days '
              f'({len(offset) - n_estimated} interpolated); median offset {offset.median():.2f}, '
              f'range [{offset.min():.2f}, {offset.max():.2f}]')

    if return_offset:
        return corrected, offset
    return corrected
