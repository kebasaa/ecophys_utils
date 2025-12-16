# Partitioning: Respiration calculation functions
#---------------------------------
import numpy as np
from typing import Union, Tuple
import pandas as pd
from scipy.optimize import curve_fit
import warnings

def respiration_from_nighttime_simple_interpolated(temp: pd.DataFrame, dn_col: str = 'dn', nee_col: str = 'nee') -> pd.Series:
    """
    Calculate ecosystem respiration using simple interpolation of nighttime data.

    Note: Only valid in tropical ecosystems, does not take temperature or PAR into account.

    Parameters
    ----------
    temp : pandas.DataFrame
        DataFrame containing timestamp, day/night column, and NEE data.
    dn_col : str, optional
        Column name for day/night indicator (1 for day, 0 for night). Default is 'dn'.
    nee_col : str, optional
        Column name for net ecosystem exchange. Default is 'nee'.

    Returns
    -------
    pandas.Series
        Series of interpolated respiration values.
    """
    warnings.warn("This simple method is only valid in tropical ecosystems, it does not take temperature or PAR into account!")

    temp = temp.copy()
    # Copy the GPP, then remove daytime data, for ecosystem respiration (Reco)
    temp['Reco'] = temp[nee_col]
    temp.loc[temp[dn_col] == 1, ['Reco']] = np.nan

    # Create day/night block IDs, i.e. from sunset to sunrise is a block
    from .partitioning_day_night import create_doy_block_id
    temp['blockID'] = create_doy_block_id(temp['timestamp'])

    # Night-time averaging (if there are more than 10 data points)
    night_mean_df = temp[['blockID','Reco']].groupby('blockID').agg(['median','count']).reset_index()
    night_mean_df.columns = ['_'.join(filter(None, col)).strip() for col in night_mean_df.columns]
    night_mean_df.loc[night_mean_df['Reco_count'] < 10, 'Reco_median'] = np.nan
    night_mean_df.drop(columns=['Reco_count'], inplace=True)
    night_mean_df.rename(columns={'Reco_median': 'Reco'}, inplace=True)

    # Remove now obsolete Reco column which has missing data, so it can be re-created from nighttime medians
    temp.drop(columns=['Reco'], inplace=True)

    # Make sure median Reco gets added at midnight only
    temp = temp.merge(night_mean_df, on='blockID', how='left')
    temp.loc[temp['timestamp'].dt.strftime('%H%M') != '0000', 'Reco'] = np.nan

    # Now interpolate Reco for all other times (limited to 1 day, or 48 half-hours)
    temp['Reco'].interpolate(method='polynomial', order=2, limit=48, limit_direction='forward', axis=0, inplace=True)
    return(temp['Reco'])

def respiration_from_nighttime_simple_blocks(temp: pd.DataFrame, dn_col: str = 'dn', nee_col: str = 'nee') -> pd.Series:
    """
    Calculate ecosystem respiration using simple blocks of nighttime data.

    Note: Only valid in tropical ecosystems, does not take temperature or PAR into account.

    Parameters
    ----------
    temp : pandas.DataFrame
        DataFrame containing timestamp, day/night column, and NEE data.
    dn_col : str, optional
        Column name for day/night indicator (1 for day, 0 for night). Default is 'dn'.
    nee_col : str, optional
        Column name for net ecosystem exchange. Default is 'nee'.

    Returns
    -------
    pandas.Series
        Series of respiration values.
    """
    warnings.warn("This simple method is only valid in tropical ecosystems, it does not take temperature or PAR into account!")
    
    temp = temp.copy()
    # Copy the GPP, then remove daytime data, for ecosystem respiration (Reco)
    temp['Reco'] = temp[nee_col]
    temp.loc[temp[dn_col] == 1, ['Reco']] = np.nan
    
    # Create DOY
    temp['doy'] = temp['timestamp'].dt.strftime('%d%m%Y').astype(int)

    # Night-time averageing (if there are more than 10 data points)
    night_mean_df = temp[['doy','Reco']].groupby('doy').agg(['median','count']).reset_index()
    night_mean_df.columns = ['_'.join(filter(None, col)).strip() for col in night_mean_df.columns]
    night_mean_df.loc[night_mean_df['Reco_count'] < 10, 'Reco_median'] = np.nan
    night_mean_df.drop(columns=['Reco_count'], inplace=True)
    night_mean_df.rename(columns={'Reco_median': 'Reco'}, inplace=True)

    # Remove now obsolete Reco column, so it can be imported from nighttime medians
    temp.drop(columns=['Reco'], inplace=True)

    # Make sure Reco gets added at midnight only
    temp = temp.merge(night_mean_df, on='doy', how='left')
    
    return(temp['Reco'])
    
def lloyd_taylor(T: Union[float, np.ndarray], R_ref: Union[float, np.ndarray], E0: Union[float, np.ndarray], T_ref: float = 15.0, T0: float = -46.02) -> Union[float, np.ndarray]:
    """
    Lloyd-Taylor (1994) respiration model.

    Parameters
    ----------
    T : float or numpy.ndarray
        Air temperature in degrees Celsius.
    R_ref : float or numpy.ndarray
        Respiration at reference temperature.
    E0 : float or numpy.ndarray
        Temperature sensitivity parameter.
    T_ref : float, optional
        Reference temperature in degrees Celsius. Default is 15.0.
    T0 : float, optional
        Temperature constant in degrees Celsius. Default is -46.02.

    Returns
    -------
    float or numpy.ndarray
        Modeled respiration R.
    """
    return(R_ref * np.exp(E0 * ((1.0 / (T_ref - T0)) - (1.0 / (T - T0)))))

def remove_nas(temp: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Remove rows with NA values in specified columns.

    Parameters
    ----------
    temp : pandas.DataFrame
        Input DataFrame.
    cols : list
        List of column names to check for NA values.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with NA rows removed.
    """
    filtered_df = temp[temp[cols].notna().all(axis=1)]
    return(filtered_df)

def fit_E0(temp: pd.DataFrame, dn_col: str = 'day_night', Tair_col: str = 'TA_1_1_1', nee_col: str = 'nee_f', initial_guess: Tuple[float, float] = (1.0, 300.0)) -> Tuple[float, float]:
    """
    Fit E0 using the Lloyd-Taylor model over nighttime dataset.

    This function fits both R_ref and E0 parameters.

    Parameters
    ----------
    temp : pandas.DataFrame
        DataFrame containing temperature and NEE data.
    dn_col : str, optional
        Column name for day/night indicator. Default is 'day_night'.
    Tair_col : str, optional
        Column name for air temperature. Default is 'TA_1_1_1'.
    nee_col : str, optional
        Column name for net ecosystem exchange. Default is 'nee_f'.
    initial_guess : tuple of float, optional
        Initial guess for R_ref and E0. Default is (1.0, 300.0).

    Returns
    -------
    tuple of float
        Fitted R_ref and E0 values.

    See Also
    --------
    lloyd_taylor : Lloyd-Taylor temperature response function.
    """
    # Extract night-time and remove NAs
    temp = temp[temp[dn_col] == 0].copy()
    filtered_df = remove_nas(temp, cols = [Tair_col, nee_col])

    Tair = filtered_df[Tair_col].values
    Reco = filtered_df[nee_col].values
    
    # Using curve_fit to fit the lloyd_taylor model.
    popt, pcov = curve_fit(lloyd_taylor, Tair, Reco, p0=initial_guess)
    R_ref_fit, E0_fit = popt
    return R_ref_fit, E0_fit

def estimate_R_ref_moving_window_overlapping(temp: pd.DataFrame, dn_col: str = 'day_night', Tair_col: str = 'TA_1_1_1', nee_col: str = 'nee_f', timestamp_col: str = 'timestamp', E0_col: str = 'E0_fit', window_days: int = 15, shift_days: int = 5, min_points: int = 10) -> pd.DataFrame:
    """
    Estimate R_ref in overlapping moving windows of nighttime data.

    Every window spans `window_days` days and windows are shifted by `shift_days` days.
    Returns a DataFrame with window midpoints and fitted R_ref values.

    Parameters
    ----------
    temp : pandas.DataFrame
        DataFrame containing temperature, NEE, and E0 data.
    dn_col : str, optional
        Column name for day/night indicator. Default is 'day_night'.
    Tair_col : str, optional
        Column name for air temperature. Default is 'TA_1_1_1'.
    nee_col : str, optional
        Column name for net ecosystem exchange. Default is 'nee_f'.
    timestamp_col : str, optional
        Column name for timestamps. Default is 'timestamp'.
    E0_col : str, optional
        Column name for E0 values. Default is 'E0_fit'.
    window_days : int, optional
        Number of days for each window. Default is 15.
    shift_days : int, optional
        Number of days to shift windows. Default is 5.
    min_points : int, optional
        Min. number of valid data points. Default is 10

    Returns
    -------
    pandas.DataFrame
        DataFrame with window midpoints and fitted R_ref values.

    See Also
    --------
    fit_E0 : Fit E0 parameter for Lloyd-Taylor model.
    interpolate_R_ref : Interpolate R_ref values.
    """
    
    # Determine the overall time range.
    start_time = temp[timestamp_col].min()
    end_time = temp[timestamp_col].max()
    
    # Filter for night-time data and remove NA values.
    temp = temp[temp[dn_col] == 0].copy()
    temp = remove_nas(temp, cols=[Tair_col, nee_col])

    # Ensure timestamp column is datetime (defensive).
    temp[timestamp_col] = pd.to_datetime(temp[timestamp_col], errors='coerce')
    
    # Determine the overall time range (after filtering to night rows).
    if temp.empty:
        # Return explicit empty DataFrame with expected columns if no night data
        return pd.DataFrame(columns=[timestamp_col, 'R_ref', 'E0'])
    # Define window and shift sizes as timedeltas.
    window_size = pd.Timedelta(days=window_days)
    shift_size = pd.Timedelta(days=shift_days)
    
    # Generate window start times such that the entire window fits within the data range.
    window_starts = []
    current_start = start_time
    # guard against NaT
    if pd.isna(current_start) or pd.isna(end_time):
        return {timestamp_col: pd.Timestamp(window_midpoint),
                    'R_ref': np.nan,
                    'E0': np.nan}
    while current_start + window_size <= end_time:
        window_starts.append(current_start)
        current_start += shift_size

    # Define a helper function for curve_fit that fixes E0 to E0_fixed.
    def lloyd_taylor_fixed(Tair, R_ref, E0_fixed):
        return lloyd_taylor(Tair, R_ref, E0_fixed)
    
    # Define a function to process a single window.
    def process_window(window_start):
        window_end = window_start + window_size
        window_data = temp[(temp[timestamp_col] >= window_start) & (temp[timestamp_col] < window_end)]
        # Calculate the midpoint of the window.
        window_midpoint = window_start + window_size / 2
        
        # Require a minimum temperature range and number of data points.
        if (len(window_data) < min_points) or ((window_data[Tair_col].max() - window_data[Tair_col].min()) < 5):
            return {timestamp_col: pd.Timestamp(window_midpoint),
                    'R_ref': np.nan,
                    'E0': np.nan}
        
        # Fit R_ref with E0 fixed.
        T_window = window_data[Tair_col].values
        Reco_window = window_data[nee_col].values
        # Compute mean of E0_col only if present
        if E0_col in window_data.columns:
            E0_fixed = window_data[E0_col].mean()
        else:
            E0_fixed = np.nan
         # If E0_fixed is NaN, cannot fit with fixed E0 -> return NaNs for this window
        if pd.isna(E0_fixed):
            return {timestamp_col: pd.Timestamp(window_midpoint),
                    'R_ref': np.nan,
                    'E0': np.nan}

        # initialize R_ref_window so it always exists even if fit fails
        R_ref_window = np.nan
        
        try:
            popt, _ = curve_fit(lambda T, R: lloyd_taylor_fixed(T, R, E0_fixed), T_window, Reco_window, p0=[1.0])
            R_ref_window = popt[0]
        except RuntimeError:
            # leave R_ref_window as NaN on fit failure
            R_ref_window = np.nan
        return {timestamp_col: pd.Timestamp(window_midpoint),
                'R_ref': R_ref_window,
                'E0': E0_fixed}

    # Apply the processing function on each window.
    results = [process_window(ws) for ws in window_starts]
    
    # Remove None results and create a DataFrame.
    results = [r for r in results if r is not None]
    if len(results) == 0:
        return pd.DataFrame(columns=[timestamp_col, 'R_ref', 'E0'])
        
    R_ref_df = pd.DataFrame(results)
    #R_ref_df['R_ref'] = R_ref_df['R_ref'].astype(float)
    #R_ref_df['E0'] = R_ref_df['E0'].astype(float)
    #R_ref_df[timestamp_col] = pd.to_datetime(R_ref_df[timestamp_col], unit='ns')
    # Avoid KeyError if column is missing
    if 'R_ref' in R_ref_df.columns:
        R_ref_df['R_ref'] = R_ref_df['R_ref'].astype(float)
    else:
        R_ref_df['R_ref'] = np.nan

    if 'E0' in R_ref_df.columns:
        R_ref_df['E0'] = R_ref_df['E0'].astype(float)
    else:
        R_ref_df['E0'] = np.nan
    R_ref_df[timestamp_col] = pd.to_datetime(R_ref_df[timestamp_col], unit='ns', errors='coerce')
    
    return(R_ref_df)

def interpolate_R_ref(full_df: pd.DataFrame, R_ref_df: pd.DataFrame, timestamp_col: str = 'timestamp') -> np.ndarray:
    """
    Interpolate R_ref estimates to obtain a continuous series.

    Parameters
    ----------
    full_df : pandas.DataFrame
        Full dataset DataFrame.
    R_ref_df : pandas.DataFrame
        DataFrame with R_ref estimates.
    timestamp_col : str, optional
        Column name for timestamps. Default is 'timestamp'.

    Returns
    -------
    numpy.ndarray
        Array of interpolated R_ref values.

    See Also
    --------
    estimate_R_ref_moving_window_overlapping : Estimate R_ref using moving windows.
    """
    full_df = full_df.copy()
    R_ref_df = R_ref_df.copy()
    # Convert to datetime and sort the full dataframe
    full_df[timestamp_col] = pd.to_datetime(full_df[timestamp_col])
    full_df = full_df.sort_values(timestamp_col)
    
    # Convert the R_ref_df timestamp column to datetime and sort
    R_ref_df[timestamp_col] = pd.to_datetime(R_ref_df[timestamp_col])
    R_ref_df = R_ref_df.sort_values(timestamp_col)
    
    # Set 'timestamp' as the index for both dataframes
    full_df.set_index(timestamp_col, inplace=True)
    R_ref_df.set_index(timestamp_col, inplace=True)
    
    # Reindex the R_ref dataframe with the full range of timestamps
    R_ref_full = R_ref_df.reindex(full_df.index)
    
    # Interpolate the missing R_ref values using time-based interpolation.
    R_ref_full['R_ref'] = R_ref_full['R_ref'].interpolate(
        method = 'time',
        limit_direction='both'
    )
     
    return(R_ref_full['R_ref'].values)