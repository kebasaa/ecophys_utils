# Interpolate to 1min
def upsample_interpolate_df(temp, freq='1min', interpolation_limit=30):
    import pandas as pd
    temp = temp.copy()
    # Create empty dataframe with a 1min frequency
    idx = pd.date_range(start=temp['timestamp'].tolist()[0],
                        end=temp['timestamp'].tolist()[-1],
                        freq=freq)
    time_df = pd.DataFrame(idx, index=None, columns=['timestamp'])
    # Merge back
    temp = time_df.merge(temp, on='timestamp', how='left')

    # Interpolate dataset
    temp = temp.interpolate(method='linear', limit=interpolation_limit, limit_direction='both')
    return(temp)