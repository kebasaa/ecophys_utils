# Removes flagged data by making it NAN, based on a condition flag
def flagged_data_removal_ep(temp, col, flag, silent=False):
    import pandas as pd
    import numpy as np
    temp = temp.copy()

    # Stats
    n_bad = len(temp.loc[(~temp[col].isna()) & flag].index)
    n     = len(temp.loc[(~temp[col].isna())].index)

    if(not silent and not n):
        print('  - WARNING: Removing', col, 'data failed: All NA')
        return(temp[col])
    
    if not silent:
        print('  - Removing', str(np.round(n_bad/n*100, 2)) + '% flagged', col, 'data')

    # Remove bad data
    temp.loc[flag, col] = np.nan
    
    return(temp[col])
    
# Remove highly variable days above a certain threshold
def remove_highly_variable_days(temp, col='co2_flux', year=2019, threshold=75, silent=False):
    temp = temp.copy()

    temp['day'] = temp['timestamp'].dt.strftime('%Y-%m-%d')

    # Calculate daily variability
    daily_variability = temp.groupby(['day'])[col].agg(['max','min']) # Determin maximum and minimum values of this column, daily
    daily_variability['range'] = daily_variability['max'] - daily_variability['min'] # Calculate the range
    
    # Identify days with variability larger than the threshold
    high_variability_days = daily_variability[daily_variability['range'] > threshold].index

    # Stats
    n_bad      = len(temp.loc[(temp['day'].isin(high_variability_days)) & (temp['timestamp'].dt.year == year), col].index)
    n          = len(temp.loc[~temp[col].isna()].index)
    n_bad_days = sum(str(year) in s for s in high_variability_days)
    n_days     = len(temp.loc[temp['timestamp'].dt.year == year,'day'].unique())
    if not silent:
        print('  - Removing', str(np.round(n_bad/n*100, 2)) + '% bad', col, 'data in', year, 'due to highly variable days, i.e.', str(n_bad_days) + '/' + str(n_days), 'days')
    
    # Replace data of those days with NaN
    temp.loc[(temp['day'].isin(high_variability_days)) & (temp['timestamp'].dt.year == year), col] = np.nan

    return(temp[col])

# Remove datapoints >2 stddevs from daily median
def remove_outliers(temp, col='co2_flux', stdevs=2, silent=False):
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
        print('  - Removing', str(np.round(n_bad/n*100, 2)) + '% outlier data in', col)
    
    # Filter out the bad values
    temp.loc[(temp[col] > temp['median'] + stdevs*temp['std']) | (temp[col] < temp['median'] - stdevs*temp['std']), col] = np.nan
    
    return(temp[col])