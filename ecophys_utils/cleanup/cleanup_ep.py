# EddyPro cleanup functions
#---------------------------
import pandas as pd
import numpy as np
from typing import Union, Optional

def flagged_data_removal_ep(temp: pd.DataFrame, col: str, flag: Union[pd.Series, np.ndarray], silent: bool = False) -> pd.Series:
    """
    Remove flagged data by setting to NaN.

    Parameters
    ----------
    temp : pandas.DataFrame
        DataFrame with data.
    col : str
        Column to clean.
    flag : pandas.Series or numpy.ndarray
        Boolean flag for bad data.
    silent : bool, optional
        Whether to suppress output. Default is False.

    Returns
    -------
    pandas.Series
        Cleaned column.
    """
    temp = temp.copy()

    # Stats
    n_bad = len(temp.loc[(~temp[col].isna()) & flag].index)
    n     = len(temp.loc[(~temp[col].isna())].index)

    if(not silent and not n):
        print('    - WARNING: Removing', col, 'data failed: All NA')
        return(temp[col])

    if(not silent):
        print('    - Removing', str(np.round(n_bad/n*100, 2)) + '% flagged', col, 'data')

    # Remove bad data
    temp.loc[flag, col] = np.nan
    
    return(temp[col])
    
def remove_highly_variable_days(temp: pd.DataFrame, col: str = 'co2_flux', year: Optional[int] = None, threshold: int = 75, silent: bool = False) -> pd.Series:
    """
    Remove data from highly variable days.

    Parameters
    ----------
    temp : pandas.DataFrame
        DataFrame with data.
    col : str, optional
        Column to clean. Default is 'co2_flux'.
    year : int, optional
        Specific year to filter. Default is None.
    threshold : int, optional
        Variability threshold. Default is 75.
    silent : bool, optional
        Whether to suppress output. Default is False.

    Returns
    -------
    pandas.Series
        Cleaned column.
    """
    temp = temp.copy()

    temp['day'] = temp['timestamp'].dt.strftime('%Y-%m-%d')

    # Calculate daily variability
    daily_variability = temp.groupby(['day'])[col].agg(['max','min']) # Determin maximum and minimum values of this column, daily
    daily_variability['range'] = daily_variability['max'] - daily_variability['min'] # Calculate the range
    
    # Identify days with variability larger than the threshold
    high_variability_days = daily_variability[daily_variability['range'] > threshold].index

    # Stats
    n = len(temp.loc[~temp[col].isna()].index)
    if(year is not None):
        n_bad      = len(temp.loc[(temp['day'].isin(high_variability_days)) & (temp['timestamp'].dt.year == year) & (~temp[col].isna()), col].index)
        n_bad_days = sum(str(year) in s for s in high_variability_days)
        n_days     = len(temp.loc[temp['timestamp'].dt.year == year,'day'].unique())
    else:
        n_bad      = len(temp.loc[(temp['day'].isin(high_variability_days)) & (~temp[col].isna()), col].index)
        n_bad_days = sum('-' in s for s in high_variability_days)
        n_days     = len(temp['day'].unique())
        
    if ((not silent) and (year is not None)):
        print('    - Removing', str(np.round(n_bad/n*100, 2)) + '% bad', col, 'data in', year, 'due to highly variable days, i.e.', str(n_bad_days) + '/' + str(n_days), 'days')
    elif ((not silent) and (year is None)):
        print('    - Removing', str(np.round(n_bad/n*100, 2)) + '% bad', col, 'data due to highly variable days, i.e.', str(n_bad_days) + '/' + str(n_days), 'days')
    else:
        pass
    
    # Replace data of those days with NaN
    if(year is not None):
        temp.loc[(temp['day'].isin(high_variability_days)) & (temp['timestamp'].dt.year == year), col] = np.nan
    else:
        temp.loc[(temp['day'].isin(high_variability_days)), col] = np.nan

    return(temp[col])

def remove_outliers(temp: pd.DataFrame, col: str = 'co2_flux', stdevs: int = 2, silent: bool = False) -> pd.Series:
    """
    Remove outliers beyond std devs from daily median.

    Parameters
    ----------
    temp : pandas.DataFrame
        DataFrame with data.
    col : str, optional
        Column to clean. Default is 'co2_flux'.
    stdevs : int, optional
        Number of std devs. Default is 2.
    silent : bool, optional
        Whether to suppress output. Default is False.

    Returns
    -------
    pandas.Series
        Cleaned column.
    """
    temp = temp.copy()

    temp['day'] = temp['timestamp'].dt.strftime('%Y-%m-%d')

    # Calculate daily median & stddev
    daily_variability = temp.groupby(['day'])[col].agg(['median','std']).reset_index()

    # Merge it back
    temp = temp.merge(daily_variability, on='day', how='left')

    # Stats
    n     = len(temp.loc[~temp[col].isna()].index)
    n_bad = len(temp.loc[(temp[col] > temp['median'] + stdevs*temp['std']) | (temp[col] < temp['median'] - stdevs*temp['std']), col].index)
    if not silent:
        print('    - Removing', str(np.round(n_bad/n*100, 2)) + '% outlier data in', col)
    
    # Filter out the bad values
    temp.loc[(temp[col] > temp['median'] + stdevs*temp['std']) | (temp[col] < temp['median'] - stdevs*temp['std']), col] = np.nan
    
    return(temp[col])