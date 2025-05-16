# Sanitize column names by:
# - Replacing spaces and special characters with underscores
# - Removing leading/trailing underscores
# - Ensuring unique column names
def sanitize_column_names(columns):
    import re
    
    sanitized = []
    seen = {}

    for col in columns:
        # Replace non-alphanumeric characters with underscores
        new_col = re.sub(r'\W+', '_', col).strip('_')

        # Ensure uniqueness
        if new_col in seen:
            seen[new_col] += 1
            new_col = f"{new_col}_{seen[new_col]}"
        else:
            seen[new_col] = 0

        sanitized.append(new_col)

    return(sanitized)
    
# Creates season labels based on 3-month seasons, not synoptic
def create_season_southern_hemisphere(timestamps):
    import warnings
    warnings.warn("Creates season labels based on 3-month seasons, not a synoptic definition", UserWarning)
    month_to_season = {
        12: "Summer", 1: "Summer", 2: "Summer",
        3: "Autumn", 4: "Autumn", 5: "Autumn",
        6: "Winter", 7: "Winter", 8: "Winter",
        9: "Spring", 10: "Spring", 11: "Spring"
    }
    return timestamps.dt.month.map(month_to_season)

def create_season_northern_hemisphere(timestamps):
    import warnings
    warnings.warn("Creates season labels based on 3-month seasons, not a synoptic definition", UserWarning)
    month_to_season = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"
    }
    return timestamps.dt.month.map(month_to_season)
    
def save_df(temp, output_path, output_fn, silent=True):
    import os
    if(not silent):
        print('Saving data to', out_fn)
    # Ensure the directory exists
    os.makedirs(output_path, exist_ok=True)
    # Create full output file path
    out_fn = os.path.join(output_path, output_fn)
    # Save data
    temp.to_csv(out_fn, sep=',', index=False)
    if(not silent):
        print('Done...')
    pass

# Sum function that ensures that the sum is nan if all elements were nan. Normally it would otherwise sum to 0
def nansum(x):
    import numpy as np
    if (x == np.nan).all():
        return np.nan
    else:
        return x.sum()

def create_categorical_order(col, cat_order):
    import pandas as pd
    col = pd.Categorical(col, categories=cat_order, ordered=True)
    return(col)

def complete_timestamps(temp, timestamp_col='timestamp', freq='30min'):
    # Create empty dataframe with a 1min frequency
    idx = pd.date_range(start=temp[timestamp_col].tolist()[0],
                        end=temp[timestamp_col].tolist()[-1],
                        freq=freq)
    time_df = pd.DataFrame(idx, index=None, columns=[timestamp_col])
    # Merge back
    out_df = time_df.merge(temp, on=timestamp_col, how='left')
    return(out_df)
