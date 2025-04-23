def create_diurnal_df(temp, group_cols, data_cols, facet=False):
    import pandas as pd
    
    temp['hour'] = temp['timestamp'].dt.strftime('%H%M')
    temp['year'] = temp['timestamp'].dt.strftime('%Y')

    group_cols = group_cols + ['hour','year']

    diurnal_df = temp[group_cols + data_cols].groupby(group_cols).agg(['median', 'std'], numeric_only=True)
    diurnal_df.reset_index(inplace=True)
    diurnal_df.columns = ['_'.join(col).strip('_') for col in diurnal_df.columns.values]

    diurnal_df['timestamp'] = pd.to_datetime(diurnal_df['hour'], format='%H%M')
    
    return(diurnal_df)
