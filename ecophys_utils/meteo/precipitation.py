# Precipitation functions
#-----------------------
import pandas as pd
from typing import Optional

def calculate_last_precipitation(df: pd.DataFrame, timestamp_col: str = 'timestamp', precipitation_col: str = 'P_1_1_1', max_gap_in_event_h: int = 12) -> pd.DataFrame:
    """
    Calculate time since last precipitation event.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with timestamp and precipitation data.
    timestamp_col : str, optional
        Timestamp column name. Default is 'timestamp'.
    precipitation_col : str, optional
        Precipitation column name. Default is 'P_1_1_1'.
    max_gap_in_event_h : int, optional
        Max gap in hours for event continuity. Default is 12.

    Returns
    -------
    pandas.DataFrame
        DataFrame with cumulative precipitation and time since last event.
    """
    # Extract only relevant columns
    temp = df[[timestamp_col, precipitation_col]].copy()

    # Identify events: rain events are continuous, interruptions greater than x hours are treated as a new event
    max_gap = pd.Timedelta(hours = max_gap_in_event_h)

    # Remove NAs and 0 rainfall
    temp = temp.loc[(~temp[precipitation_col].isna()) & (temp[precipitation_col] > 0)].copy()

    # Calculate time differences between rows
    temp['time_diff'] = temp[timestamp_col].diff().fillna(pd.Timedelta(0))

    # Event breaks where the time difference exceeds max_gap (start a new event)
    temp['new_event'] = temp['time_diff'] > max_gap

    # Assign event ids: we will accumulate events, starting at 0 and incrementing when a new event starts
    temp['event_id'] = temp['new_event'].cumsum()

    # Merge precipitation event IDs back and fill them up
    temp = df[[timestamp_col, precipitation_col]].merge(temp[[timestamp_col,'event_id']], on=timestamp_col, how='outer')
    temp['prev_event_id'] = temp['event_id'].bfill()
    temp['event_id'] = temp['event_id'].ffill()
    
    # Calculate cumulative precipitation per event
    temp['P_cum'] = temp.groupby('event_id')[precipitation_col].cumsum()

    # Calculate time since the last event in s
    temp['time_since_last_event'] = temp.groupby('prev_event_id')[timestamp_col].transform(lambda x: x - x.min())
    temp.loc[temp['prev_event_id'] == temp['event_id'], 'time_since_last_event'] = pd.Timedelta('0s')
    temp['time_since_last_event_s'] = temp['time_since_last_event'].dt.total_seconds()

    return(temp[[timestamp_col,'P_cum','time_since_last_event_s']])