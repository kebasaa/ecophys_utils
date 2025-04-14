import numpy as np

# Function to determine day or night
def is_day(timestamp_series, lat, lon, tz, numeric=True):
    # Required to calculate Day/Night
    from astral import LocationInfo
    from astral.sun import sun
    import pandas as pd
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
    
def create_doy_block_id(timestamps):
    import pandas as pd
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
    
# Calculate ecosystem respiration (Reco)
# Note: Only valid in tropical ecosystems, does not take temperature or PAR into account
def respiration_from_nighttime_simple_interpolated(temp, dn_col='dn', nee_col='nee'):
    import pandas as pd
    import numpy as np
    import warnings
    warnings.warn("This simple method is only valid in tropical ecosystems, it does not take temperature or PAR into account!")
    
    temp = temp.copy()
    # Copy the GPP, then remove daytime data, for ecosystem respiration (Reco)
    temp['Reco'] = temp[nee_col]
    temp.loc[temp[dn_col] == 1, ['Reco']] = np.nan

    # Create day/night block IDs, i.e. from sunset to sunrise is a block
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

# Calculate ecosystem respiration (Reco)
# Simple interpolation of nighttime Reco data during daytime
# Note: Only valid in tropical ecosystems, does not take temperature or PAR into account
def respiration_from_nighttime_simple_blocks(temp, dn_col='dn', nee_col='nee'):
    import pandas as pd
    import numpy as np
    import warnings
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
    
def calculate_nee(co2_flux, storage_flux):
    nee = co2_flux + storage_flux
    return(nee)
    
# Calculates NPP from GPP and ecosystem respiration
def calculate_gpp(nee, reco):
    gpp = reco - nee
    gpp = np.where(gpp < 0, 0, gpp)
    return(gpp)

def calculate_wue(gpp_umol_m2_s1, ET_mm_h):
    # Constants
    from ..units.constants import M_C
    
    # Convert ET from mm h-1 to kgH2O m-2 s-1
    # 1 mm of water over 1 m² equals 1 kg, so per s, divide by 3600
    ET_kgH2O_m2_s1 = ET_mm_h/3600
    ET_kgH2O_m2_s1 = np.where(ET_kgH2O_m2_s1 < 0.00001, 0, ET_kgH2O_m2_s1)

    gpp_gC_m2_s1 = gpp_umol_m2_s1 * 10**(-6) * M_C
    
    wue_gC_kgH2O = gpp_gC_m2_s1 / ET_kgH2O_m2_s1
    wue_gC_kgH2O = np.where(np.isnan(wue_gC_kgH2O) | np.isinf(wue_gC_kgH2O), 0, wue_gC_kgH2O)
    
    return(wue_gC_kgH2O)
    
def calculate_wue_umol_mmol(gpp_umol_m2_s1, h2o_mmol_m2_s1):
    # Correct h2o to ET, i.e. no negative flux
    h2o_mmol_m2_s1 = np.where(h2o_mmol_m2_s1 < 0.00001, 0, h2o_mmol_m2_s1)
    
    wue_umolC_mmolH2O = gpp_umol_m2_s1 / h2o_mmol_m2_s1
    wue_umolC_mmolH2O = np.where(np.isnan(wue_umolC_mmolH2O) | np.isinf(wue_umolC_mmolH2O), 0, wue_umolC_mmolH2O)
    
    return(wue_umolC_mmolH2O)

# uStar filtering (similar to Reichstein et al. 2005 & Papale et al. 2006) function
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# filter_threshold = 0.99 determines at what % of NEE of the combined higher u* classes we accept a threshold
# use_night_only = True (Night-time only)
# min_uStar_threshold = 0.01 in grasslands, 0.1 in forests minimum
# na_uStar_threshold = 0.4 Threshold in case no threshold was found
# threshold_if_none_found = False, should a threshold be inserted if none was found by the algorithm?
def calculate_uStar_threshold_reichstein(df, Tair_col  = 'TA_1_1_1', dn_col    = 'day_night', uStar_col = 'u*', nee_col   = 'nee',
                               filter_threshold = 0.99, use_night_only = True, min_uStar_threshold = 0.01, na_uStar_threshold = 0.4, threshold_if_none_found = False):
    import pandas as pd
    import numpy as np
    from scipy import stats

    # Exclude rows where Tair is NA.
    mask = ~df[Tair_col].isna()
    temp = df[mask].copy()

    # Optionally use only night-time data (make sure to reassign!)
    if use_night_only:
        temp = temp.loc[temp[dn_col] == 0].copy()

    # Create 6 temperature classes based on quantiles.
    temp['Tair_class'] = pd.qcut(temp[Tair_col], 6, labels=False)

    # Within each Tair class, create 20 uStar classes.
    # We use groupby on Tair_class together with a lambda to avoid an explicit loop.
    temp['uStar_class'] = temp.groupby('Tair_class')[uStar_col].transform(
        lambda x: pd.qcut(x, 20, labels=False, duplicates='drop')
    )

    # Helper function for group processing:
    # For each temperature class (group), perform the linear regression between Tair and uStar.
    # If the absolute r-value is below 0.4, calculate the threshold by comparing mean NEE in 
    # each uStar bin versus the combination of all higher uStar bins.
    def compute_uStar_threshold(group):
        # Run linear regression on the entire group.
        res = stats.linregress(group[Tair_col], group[uStar_col])
        # If the correlation is too strong, do not compute a threshold.
        if abs(res.rvalue) >= 0.4:
            return np.nan

        # Initialise
        current_threshold = np.nan
        # Get sorted unique uStar_class values.
        # We drop NaN in case 'duplicates' were dropped during qcut.
        sorted_uStar_class = sorted(group['uStar_class'].dropna().unique())

        # Loop over each uStar_class except the highest (because it has no "higher" classes).
        for current_uStar_class in sorted_uStar_class[:-1]:
            # Create a mask for the current and higher classes.
            mask_current = group['uStar_class'] == current_uStar_class
            mask_higher  = group['uStar_class'] > current_uStar_class

            # Calculate mean NEE for current and combined higher uStar classes.
            F_current = group.loc[mask_current, nee_col].mean()
            F_higher  = group.loc[mask_higher, nee_col].mean()

            # Check the filter condition; if true, take the mean uStar of the higher classes.
            if F_current >= filter_threshold * F_higher:
                current_threshold = group.loc[mask_current, uStar_col].mean()
                # This stops the for loop so that the lowest threshold that meets
                # the criterion is used.
                break
    
        # In strict mode, you may optionally set a threshold if no value met the condition.
        if pd.isna(current_threshold) and threshold_if_none_found:
            current_threshold = na_uStar_threshold
        if(current_threshold < min_uStar_threshold):
            current_threshold = min_uStar_threshold

        # Enforce the minimum allowable threshold.
        if not pd.isna(current_threshold):
            current_threshold = max(current_threshold, min_uStar_threshold)
        return current_threshold

    # Apply across Tair classes:
    # Use groupby to process each Tair_class.
    thresholds_by_Tair = temp.groupby('Tair_class').apply(compute_uStar_threshold)

    # Compute the final uStar threshold as the median of thresholds across classes.
    uStar_threshold = thresholds_by_Tair.median()

    return(uStar_threshold)
    
def create_seasonal_uStar_threshold_list(df, groupby=['year', 'season'], 
                                           Tair_col='TA_1_1_1', dn_col='day_night', uStar_col='u*', nee_col='nee'):
    groups = df.groupby(groupby)
    grouped_thresholds = groups.apply(
        lambda group: calculate_uStar_threshold_reichstein(group,
                                             Tair_col=Tair_col, 
                                             dn_col=dn_col, 
                                             uStar_col=uStar_col, 
                                             nee_col=nee_col,
                                             filter_threshold=0.99, 
                                             use_night_only=True, 
                                             min_uStar_threshold=0.01, 
                                             na_uStar_threshold=0.4, 
                                             threshold_if_none_found=False)
    )
    thresholds_df = grouped_thresholds.reset_index()

    # Count rows of data for each group, for relevant columns only
    # Take the maximum number of rows and then some % of those as a limit
    relevant_cols = [Tair_col, dn_col, uStar_col, nee_col]
    grouped_rowcounts = groups.apply(lambda grp: grp[relevant_cols].dropna().shape[0])\
                              .reset_index(name='non_na_row_count')
    
    # Create pandas dataframe
    thresholds_df.columns = groupby + ['uStar_threshold']
    thresholds_df = thresholds_df.merge(grouped_rowcounts, on=groupby)

    return(thresholds_df)

# Remove seasons with too few data points, then calculate the threshold
def calculate_overall_uStar_threshold(thresholds_df, missing_fraction = 1, use_mean=True):
    max_rowcount = thresholds_df['non_na_row_count'].max()
    if(use_mean):
        threshold = thresholds_df.loc[thresholds_df['non_na_row_count'] > max_rowcount*missing_fraction,'uStar_threshold'].mean()
    else:
        threshold = thresholds_df.loc[thresholds_df['non_na_row_count'] > max_rowcount*missing_fraction,'uStar_threshold'].max()
    return(threshold)
    
def uStar_filtering_wrapper(temp, timestamp_col='timestamp', hemisphere='north', apply_to_cols=['nee','co2_flux','co2_strg','h2o_flux','H','LE'], silent=False):
    temp = temp.copy()
    if(not silent):
        print('Calculating u* filter threshold, following Reichstein et al. (2005):')

    # Add year variable
    temp['year']  = temp[timestamp_col].dt.year
    temp['month'] = temp[timestamp_col].dt.month
    # Create season
    if(hemisphere == 'north'):
        temp['season'] = create_season_northern_hemisphere(temp[timestamp_col])
    if(hemisphere == 'north'):
        temp['season'] = create_season_southern_hemisphere(temp[timestamp_col])
    else:
        print('Error: Choose north or south as a hemisphere!')
        return

    # Calculate seasonal thresholds
    thresholds_df  = create_seasonal_uStar_threshold_list(temp, groupby=['year', 'season'])
    if(not silent):
        print('  - u* thresholds, by season:')
    display(thresholds_df)

    # Calculate overall threshold
    threshold_overall  = calculate_overall_uStar_threshold(thresholds_df, missing_fraction = 0.75, use_mean=True)
    if(not silent):
        print('  - Final u* threshold:  ', threshold_overall)

    # Apply thresholds, remove NEE, LE, H
    if(not silent):
        print('  - Applying to:', apply_to_cols)
    for col in apply_to_cols:
        temp[col + '_f'] = temp[col]
        temp.loc[(temp['u*'] <= threshold_overall), col + '_f'] = np.nan

    if(not silent):
        print('Done applying u* filter...')
    return(temp)
    
# Respiration partitioning (Reichstein et al. 2005) using the Lloyd & Taylor model (1994)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Lloyd-Taylor (1994) respiration model
# Parameters:
# - T: air temperature (°C)
# - R_ref: respiration at reference temperature
# - E0: temperature sensitivity parameter
# - T_ref: reference temperature (default 15 °C)
# - T0: temperature constant (default -46.02 °C)
# Returns:
# - Modeled respiration R
def lloyd_taylor(T, R_ref, E0, T_ref=15.0, T0=-46.02):
    return(R_ref * np.exp(E0 * ((1.0 / (T_ref - T0)) - (1.0 / (T - T0)))))

def remove_nas(temp, cols):
    filtered_df = temp[temp[cols].notna().all(axis=1)]
    return(filtered_df)

# Step 1: Fit E0 using the Lloyd-Taylor model (1994) over the entire night-time dataset
# This function fits both parameters (R_ref & E0) to allow for a later fit of R_ref
def fit_E0(temp, dn_col='day_night', Tair_col='TA_1_1_1', nee_col='nee_f', initial_guess=(1.0, 300.0)):
    import pandas as pd
    # Extract night-time and remove NAs
    temp = temp[temp[dn_col] == 0].copy()
    filtered_df = remove_nas(temp, cols = [Tair_col, nee_col])

    Tair = filtered_df[Tair_col].values
    Reco = filtered_df[nee_col].values
    
    # Using curve_fit to fit the lloyd_taylor model.
    popt, pcov = curve_fit(lloyd_taylor, Tair, Reco, p0=initial_guess)
    R_ref_fit, E0_fit = popt
    return R_ref_fit, E0_fit

# Step 2: Estimate R_ref in overlapping moving windows of night-time data, keeping E0 fixed to previous fit.
# Every window spans `window_days` days and windows are shifted by `shift_days` days.
# Returns a DataFrame with window midpoints and fitted R_ref values.
def estimate_R_ref_moving_window_overlapping(temp, dn_col='day_night', Tair_col='TA_1_1_1', nee_col='nee_f',
                                             timestamp_col='timestamp', E0_col='E0_fit', window_days=15, shift_days=5): #E0_fixed=E0_fit
    from scipy.optimize import curve_fit
    import pandas as pd
    import numpy as np

    # Filter for night-time data and remove NA values.
    temp = temp[temp[dn_col] == 0].copy()
    temp = remove_nas(temp, cols=[Tair_col, nee_col])
    
    # Define window and shift sizes as timedeltas.
    window_size = pd.Timedelta(days=window_days)
    shift_size = pd.Timedelta(days=shift_days)
    
    # Determine the overall time range.
    start_time = temp[timestamp_col].min()
    end_time = temp[timestamp_col].max()
    
    # Generate window start times such that the entire window fits within the data range.
    window_starts = []
    current_start = start_time
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
        
        # Require a minimum temperature range and number of data points.
        if (len(window_data) < 10) or ((window_data[Tair_col].max() - window_data[Tair_col].min()) < 5):
            return None
        
        # Fit R_ref with E0 fixed.
        try:
            T_window = window_data[Tair_col].values
            Reco_window = window_data[nee_col].values
            E0_fixed = window_data[E0_col].mean() # NEW
            #popt, _ = curve_fit(lloyd_taylor_fixed, T_window, Reco_window, p0=[1.0])
            popt, _ = curve_fit(lambda T, R: lloyd_taylor_fixed(T, R, E0_fixed), T_window, Reco_window, p0=[1.0])
            R_ref_window = popt[0]
            #if(R_ref_window < 0.5):
            #    R_ref_window = np.nan
            # Calculate the midpoint of the window.
            window_midpoint = window_start + window_size / 2
            return pd.Series({timestamp_col: window_midpoint, 'R_ref': R_ref_window, 'E0': E0_fixed})
        except RuntimeError:
            # If the fitting fails, simply return None.
            return None

    # Apply the processing function on each window.
    results = [process_window(ws) for ws in window_starts]
    
    # Remove None results and create a DataFrame.
    R_ref_df = pd.DataFrame([res for res in results if res is not None])
    R_ref_df['R_ref'] = R_ref_df['R_ref'].astype(float)
    R_ref_df['E0'] = R_ref_df['E0'].astype(float)
    R_ref_df[timestamp_col] = pd.to_datetime(R_ref_df[timestamp_col], unit='ns')
    #R_ref_df['E0'] = E0_fixed
    return(R_ref_df)

# Step 3: Interpolate the R_ref estimates to obtain a continuous series for the full dataset.
def interpolate_R_ref(full_df, R_ref_df, timestamp_col='timestamp'):
    import pandas as pd
    
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
    
def partitioning_reichstein_wrapper(temp, timestamp_col='timestamp', dn_col='day_night', Tair_col='Tair', nee_col='nee_f', grouping_col='year'):
    # Step 1: Estimate E0 for each large (e.g. year) group
    # 1a) Define a wrapper function to use with .apply()
    def apply_fit_E0(group):
        R_ref_initial, E0_fit = fit_E0(group, Tair_col=Tair_col)
        n = len(group)
        return pd.Series({
            'R_ref_initial': R_ref_initial,
            'E0': E0_fit,
            'n_rows': n
        })
    
    # 1b) Apply the function to each year group
    result_E0 = temp.groupby(grouping_col).apply(apply_fit_E0).reset_index()

    # 1c) Merge back into temp
    temp = temp.merge(result_E0[[grouping_col,'E0_fit']], on=grouping_col, how='outer')

    # Step 2: Estimate R_ref for each smaller group in moving windows
    # 2a) Define the wrapper function
    def apply_estimate_R_ref(group):
        # Call the function with the appropriate columns and parameters
        R_ref_df = estimate_R_ref_moving_window_overlapping(
            group,
            dn_col=dn_col,
            Tair_col=Tair_col,
            nee_col=nee_col,
            E0_col='E0',
            window_days=15,
            shift_days=5
        )
        R_ref_df[grouping_col] = group.name
        return(R_ref_df)

    # 2b) Group by 'year' and apply the wrapper function
    result_R_ref = temp.groupby(grouping_col).apply(apply_estimate_R_ref).reset_index(drop=True)

    # Step 3: Interpolate R_ref across the year (or other grouping variable)
    # 3a) Define the wrapper function
    def interpolate_group(group):
        year = group.name
        R_ref_group = result_R_ref.loc[result_R_ref[grouping_col] == year]
        interpolated_values = interpolate_R_ref(group, R_ref_group)
        group = group.copy()
        group['R_ref'] = interpolated_values
        return(group)
    # 3b) Group and apply the wrapper function
    lloyd_taylor_params = temp.groupby(grouping_col).apply(interpolate_group).reset_index(drop=True)

    return(lloyd_taylor_params[[timestamp_col,'E0','R_ref']])
