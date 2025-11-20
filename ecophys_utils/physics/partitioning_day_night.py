# Partitioning: Day/night determination functions
#---------------------------------
import numpy as np
from typing import Union
import pandas as pd

def is_day(timestamp_series: pd.Series, lat: float, lon: float, tz: str, numeric: bool = True) -> pd.Series:
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
    tz : str
        Timezone string.
    numeric : bool, optional
        If True, return 1 for day and 0 for night. If False, return 'Day' or 'Night'. Default is True.

    Returns
    -------
    pandas.Series
        Series indicating day (1 or 'Day') or night (0 or 'Night').
    """
    # Required to calculate Day/Night
    from astral import LocationInfo
    from astral.sun import sun
    # Create the location object once
    location = LocationInfo(latitude=lat, longitude=lon)
    
    # Convert timestamps to timezone-aware datetimes
    timestamp_series = pd.to_datetime(timestamp_series).dt.tz_localize(tz, nonexistent='NaT', ambiguous='NaT')

    # Apply the function element-wise
    def check_day_night(ts):
        if pd.isna(ts):  # Handle NaT values gracefully
            return None
        s = sun(location.observer, date=ts)
        if(numeric):
            return 1 if s['sunrise'] <= ts <= s['sunset'] else 0
        else:
            return 'Day' if s['sunrise'] <= ts <= s['sunset'] else 'Night'

    return timestamp_series.apply(check_day_night)
    
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
    temp['days_in_year'] = temp['timestamp'].dt.is_leap_year.replace({True: 366, False: 365})
    year_lengths = temp[['year','days_in_year']].drop_duplicates()
    year_lengths['annual_day_sums'] = year_lengths['days_in_year'].cumsum().shift(1, fill_value=0)
    # Combine and add the numbers
    temp = temp.merge(year_lengths[['year','annual_day_sums']], on='year', how='left')
    temp['blockID'] = temp['blockID'] + temp['annual_day_sums']

    # Make sure that the blocks are not divided by midnight, but rather midday
    temp.loc[temp['timestamp'].dt.hour < 12, 'blockID'] = temp.loc[temp['timestamp'].dt.hour < 12, 'blockID'] - 1
    
    return(temp['blockID'].values)