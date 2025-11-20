# Data processing utility functions
#-----------------------------------
import pandas as pd
import numpy as np
from typing import List, Union, Optional

def sanitize_column_names(columns: List[str]) -> List[str]:
    """
    Sanitize column names by replacing special characters and ensuring uniqueness.

    Parameters
    ----------
    columns : list of str
        List of column names.

    Returns
    -------
    list of str
        Sanitized column names.
    """
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
def create_season_southern_hemisphere(timestamps: pd.Series) -> pd.Series:
    """
    Create season labels for southern hemisphere.

    Creates season labels based on 3-month seasons, not synoptic definition.

    Parameters
    ----------
    timestamps : pandas.Series
        Series of timestamps.

    Returns
    -------
    pandas.Series
        Season labels.
    """
    import warnings
    warnings.warn("Creates season labels based on 3-month seasons, not a synoptic definition", UserWarning)
    month_to_season = {
        12: "Summer", 1: "Summer", 2: "Summer",
        3: "Autumn", 4: "Autumn", 5: "Autumn",
        6: "Winter", 7: "Winter", 8: "Winter",
        9: "Spring", 10: "Spring", 11: "Spring"
    }
    return timestamps.dt.month.map(month_to_season)

def create_season_northern_hemisphere(timestamps: pd.Series) -> pd.Series:
    """
    Create season labels for northern hemisphere.

    Creates season labels based on 3-month seasons, not synoptic definition.

    Parameters
    ----------
    timestamps : pandas.Series
        Series of timestamps.

    Returns
    -------
    pandas.Series
        Season labels.
    """
    import warnings
    warnings.warn("Creates season labels based on 3-month seasons, not a synoptic definition", UserWarning)
    month_to_season = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"
    }
    return timestamps.dt.month.map(month_to_season)
    
def save_df(temp: pd.DataFrame, output_path: str, output_fn: str, silent: bool = True) -> None:
    """
    Save DataFrame to CSV.

    Parameters
    ----------
    temp : pandas.DataFrame
        DataFrame to save.
    output_path : str
        Output directory path.
    output_fn : str
        Output filename.
    silent : bool, optional
        Whether to suppress output. Default is True.
    """
    import os
    # Ensure the directory exists
    os.makedirs(output_path, exist_ok=True)
    # Create full output file path
    out_fn = os.path.join(output_path, output_fn)
    if(not silent):
        print(f'Saving data to {out_fn}')
    # Save data
    temp.to_csv(out_fn, sep=',', index=False)
    if(not silent):
        print('Done...')

# Sum function that ensures that the sum is nan if all elements were nan. Normally it would otherwise sum to 0
def nansum(x: Union[np.ndarray, pd.Series]) -> Union[float, np.ndarray]:
    """
    Sum that returns NaN if all elements are NaN.

    Parameters
    ----------
    x : numpy.ndarray or pandas.Series
        Array to sum.

    Returns
    -------
    float or numpy.ndarray
        Sum or NaN.
    """
    if (x == np.nan).all():
        return np.nan
    else:
        return x.sum()

def create_categorical_order(col: pd.Series, cat_order: List[str]) -> pd.Categorical:
    """
    Create ordered categorical from series.

    Parameters
    ----------
    col : pandas.Series
        Series to convert.
    cat_order : list of str
        Category order.

    Returns
    -------
    pandas.Categorical
        Ordered categorical.
    """
    col = pd.Categorical(col, categories=cat_order, ordered=True)
    return(col)

def complete_timestamps(temp: pd.DataFrame, timestamp_col: str = 'timestamp', freq: str = '30min') -> pd.DataFrame:
    """
    Complete timestamps by filling missing time points.

    Parameters
    ----------
    temp : pandas.DataFrame
        DataFrame with timestamps.
    timestamp_col : str, optional
        Timestamp column name. Default is 'timestamp'.
    freq : str, optional
        Frequency for completion. Default is '30min'.

    Returns
    -------
    pandas.DataFrame
        DataFrame with complete timestamps.

    See Also
    --------
    upsample_interpolate_df : Upsample and interpolate DataFrame.
    """
    # Create empty dataframe with a 1min frequency
    idx = pd.date_range(start=temp[timestamp_col].tolist()[0],
                        end=temp[timestamp_col].tolist()[-1],
                        freq=freq)
    time_df = pd.DataFrame(idx, index=None, columns=[timestamp_col])
    # Merge back
    out_df = time_df.merge(temp, on=timestamp_col, how='left')
    return(out_df)

def upsample_interpolate_df(temp: pd.DataFrame, freq: str = '1min', interpolation_limit: Optional[int] = 30) -> pd.DataFrame:
    """
    Upsample and interpolate DataFrame to higher frequency.

    Parameters
    ----------
    temp : pandas.DataFrame
        Input DataFrame with 'timestamp' column.
    freq : str, optional
        Frequency for upsampling. Default is '1min'.
    interpolation_limit : int, optional
        Maximum number of consecutive NaNs to fill. Default is 30.

    Returns
    -------
    pandas.DataFrame
        Upsampled and interpolated DataFrame.

    See Also
    --------
    complete_timestamps : Fill missing timestamps.
    """
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