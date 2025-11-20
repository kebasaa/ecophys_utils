# Diurnal plotting functions
#---------------------------
import pandas as pd
from typing import List, Union

def create_diurnal_df(temp: pd.DataFrame, group_cols: List[str], data_cols: List[str], facet: bool = False) -> pd.DataFrame:
    """
    Create diurnal DataFrame for plotting.

    Parameters
    ----------
    temp : pandas.DataFrame
        DataFrame with timestamp and data columns.
    group_cols : list of str
        Columns to group by.
    data_cols : list of str
        Data columns to aggregate.
    facet : bool, optional
        Whether to facet. Default is False.

    Returns
    -------
    pandas.DataFrame
        Diurnal aggregated DataFrame.
    """
    temp['hour'] = temp['timestamp'].dt.strftime('%H%M')
    temp['year'] = temp['timestamp'].dt.strftime('%Y')

    group_cols = group_cols + ['hour','year']

    diurnal_df = temp[group_cols + data_cols].groupby(group_cols).agg(['median', 'std'], numeric_only=True)
    diurnal_df.reset_index(inplace=True)
    diurnal_df.columns = ['_'.join(col).strip('_') for col in diurnal_df.columns.values]

    diurnal_df['timestamp'] = pd.to_datetime(diurnal_df['hour'], format='%H%M')
    
    return(diurnal_df)
