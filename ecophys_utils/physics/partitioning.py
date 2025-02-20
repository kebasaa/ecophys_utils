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
def respiration_from_nighttime(temp, dn_col='dn', nee_col='co2_flux'):
    import pandas as pd
    import numpy as np
    temp = temp.copy()
    # Copy the GPP, then remove daytime data, for ecosystem respiration (Reco)
    temp['Reco'] = temp[nee_col]
    temp.loc[temp[dn_col] == 1, ['Reco']] = np.nan

    # Create day/night block IDs
    temp['blockID'] = create_doy_block_id(temp['timestamp'])

    # Night-time averageing (if there are more than 10 data points)
    night_mean_df = temp[['blockID','Reco']].groupby('blockID').agg(['median','count']).reset_index()
    night_mean_df.columns = ['_'.join(filter(None, col)).strip() for col in night_mean_df.columns]
    night_mean_df.loc[night_mean_df['Reco_count'] < 10, 'Reco_median'] = np.nan
    night_mean_df.drop(columns=['Reco_count'], inplace=True)
    night_mean_df.rename(columns={'Reco_median': 'Reco'}, inplace=True)

    # Remove now obsolete Reco column, so it can be imported from nighttime medians
    temp.drop(columns=['Reco'], inplace=True)

    # Make sure Reco gets added at midnight only
    temp = temp.merge(night_mean_df, on='blockID', how='left')
    temp.loc[temp['timestamp'].dt.strftime('%H%M') != '0000', 'Reco'] = np.nan

    # Now interpolate Reco for all other times (limited to 1 day, or 48 half-hours)
    temp['Reco'].interpolate(method='polynomial', order=2, limit=48, limit_direction='forward', axis=0, inplace=True)
    return(temp['Reco'])
    
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
    # Constants
    from ..units.constants import M_C
    
    # Correct h2o to ET, i.e. no negative flux
    h2o_mmol_m2_s1 = np.where(h2o_mmol_m2_s1 < 0.00001, 0, h2o_mmol_m2_s1)
    
    wue_umolC_mmolH2O = gpp_umol_m2_s1 / h2o_mmol_m2_s1
    wue_umolC_mmolH2O = np.where(np.isnan(wue_umolC_mmolH2O) | np.isinf(wue_umolC_mmolH2O), 0, wue_umolC_mmolH2O)
    
    return(wue_umolC_mmolH2O)
