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
# Note: Only valid in tropical ecosystems, does not take temperature or PAR into consideration
def respiration_from_nighttime_simple_interpolated(temp, dn_col='dn', nee_col='nee'):
    import pandas as pd
    import numpy as np
    import warnings
    warnings.warn("This simple method is only valid in tropical ecosystems, it does not take temperature or PAR into consideration!")
    
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
# Note: Only valid in tropical ecosystems, does not take temperature or PAR into consideration
def respiration_from_nighttime_simple_blocks(temp, dn_col='dn', nee_col='nee'):
    import pandas as pd
    import numpy as np
    import warnings
    warnings.warn("This simple method is only valid in tropical ecosystems, it does not take temperature or PAR into consideration!")
    
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
def uStar_filtering_reichstein(df, Tair_col  = 'TA_1_1_1', dn_col    = 'day_night', uStar_col = 'u*', nee_col   = 'nee',
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