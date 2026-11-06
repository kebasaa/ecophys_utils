# ecophys_utils: Python utilities library for ecophysiology tasks

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)

![Version](https://img.shields.io/badge/version-0.5-blue)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[![DOI](https://zenodo.org/badge/932117720.svg)](https://doi.org/10.5281/zenodo.20540557)

A library of support functions for ecophysiology research, including Eddy Covariance data processing, flux calculations, partitioning, and more. Functions are organised into modules for easy access.

## Features

- **Data Loading**: Support for various instruments (EddyPro, LI-COR devices, Campbell TOA5, CSV files)
- **Data Cleaning**: Outlier removal, flag-based cleaning, high variability detection
- **Physics Calculations**: Thermodynamics, flux calculations, conductance, energy budgets
- **Partitioning**: NEE partitioning into GPP and respiration using various methods
- **Utilities**: Data manipulation, interpolation, unit conversions
- **Visualization**: Diurnal plotting helpers

## Installation

### Option 1: Temporary Path Addition

Add to `sys.path` in your script:

```python
import sys
sys.path.append("/path/to/ecophys_utils")
import ecophys_utils as eco
```

### Option 2: Permanent PYTHONPATH

1. Find your environment's site-packages:
   ```bash
   python -c "import site; print(site.getsitepackages()[0])"
   ```
2. Create `ecophys_utils.pth` in that directory with:
   ```
   /path/to/ecophys_utils
   ```

### Option 3: Editable Install

```bash
pip install -e /path/to/ecophys_utils
```

## Dependencies

- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- astral >= 2.2

## Quick Start

```python
import ecophys_utils as eco
import pandas as pd

# Load EddyPro data
data = eco.load_all_eddypro('/path/to/eddypro_data')

# Clean data: remove flagged points
data = eco.flagged_data_removal_ep(data, 'co2_flux', data['qc_co2_flux'] > 1)

# Calculate vapor pressure deficit
data['VPD'] = eco.calculate_VPD(data['TA_1_1_1'], data['H2O_1_1_1'], data['PA_1_1_1'])

# Partition NEE into GPP and respiration
partitioned = eco.partitioning_reichstein_wrapper(data, timestamp_col='timestamp', nee_col='nee')

# Save results
eco.save_df(partitioned, '/output/path', 'partitioned_data.csv')
```

## User Guide

### Data Loading

Load data from various sources:

```python
# EddyPro files
eddy_data = eco.load_all_eddypro('/data/eddypro')

# LI-6800 files
li6800_data = eco.load_all_li6800('/data/li6800')

# CSV files
csv_data = eco.load_all_csv('/data/csv')
```

### Data Cleaning

Apply quality control:

```python
# Remove flagged data
clean_data = eco.flagged_data_removal_ep(data, 'nee', flags > 1)

# Remove outliers
clean_data = eco.remove_outliers(clean_data, 'nee', stdevs=3)

# Remove the night-time offset of a shortwave sensor (night detected with astral).
# Timestamps in local standard time (no daylight saving) by default; use time_ref='local' or 'utc' otherwise
clean_data['day_night'] = eco.is_day(clean_data['timestamp'], lat, lon, 'Europe/Zurich')
clean_data['SW_IN_corr'] = eco.correct_nighttime_swr_offset(clean_data, col='SW_IN_1_1_1')
```

### Flux Calculations

Calculate various fluxes:

```python
# Water vapor flux from chamber
h2o_flux = eco.calculate_h2o_flux(T_C, P_Pa, h2o_amb, h2o_cham, airflow, area)

# Gas flux
co2_flux = eco.calculate_gas_flux(T_C, P_Pa, h2o_amb, h2o_cham, co2_amb, co2_cham, airflow, area)
```

### Partitioning

Partition net ecosystem exchange:

```python
# Reichstein method
partitioned = eco.partitioning_reichstein_wrapper(data, timestamp_col='timestamp', nee_col='nee')
```

### Thermodynamics

Calculate atmospheric properties:

```python
# Vapor pressure deficit
vpd = eco.calculate_VPD(T_C, h2o_mmol, P_Pa)

# Saturation vapor pressure
es = eco.calculate_es(T_C, P_Pa)
```

## API Reference

### cleanup
Data cleaning and quality control functions.

- `flagged_data_removal_ep(temp, col, flag, silent=False)`: Remove flagged data points by setting to NaN, optimised for EddyPro quality flags.
- `remove_highly_variable_days(temp, col='co2_flux', year=None, threshold=75, silent=False)`: Remove days with excessive variability in specified column.
- `remove_outliers(temp, col='co2_flux', stdevs=2, silent=False)`: Remove outliers based on standard deviation threshold.
- `remove_outliers_by_regression(df, x_col, y_col, n_std=2, ddof=0)`: Remove outliers based on regression residual analysis.
- `correct_nighttime_swr_offset(temp, col='SW_IN_1_1_1', timestamp_col='timestamp', dn_col='day_night', lat=None, lon=None, tz=None, time_ref='standard', min_night_records=6, return_offset=False, silent=False)`: Subtract the median night-time value of each calendar day from that whole day, then set night-time values to 0. Night comes from `dn_col`, or from astral via `lat`, `lon`, `tz` and `time_ref`.

### dataloading
Functions for loading data from various instruments and formats.

#### Campbell Scientific
- `load_campbell_toa5(input_fn)`: Load Campbell Scientific TOA5 datalogger file.
- `load_campbell_tob1(input_fn, silent=False)`: Load single Campbell TOB1 binary file.

#### EddyPro
- `load_eddypro(fn, silent=False)`: Load single EddyPro output file (supports full_output and biomet formats).
- `load_all_eddypro(path, dataset='full_output', silent=False)`: Load all EddyPro files from directory.

#### LI-COR Devices
- `load_li600(input_fn, silent=True)`: Load single LI-600 photosynthesis system file.
- `load_all_li600(path, pattern='.csv', silent=False)`: Load all LI-600 files from directory.
- `remove_obsolete_cols_li600(temp, silent=False)`: Remove unnecessary columns from LI-600 data.
- `load_li6400(input_fn, silent=True)`: Load single LI-6400XT portable photosynthesis system file.
- `load_all_li6400(path, silent=False)`: Load all LI-6400 files from directory.
- `load_li6800(input_fn, silent=True)`: Load single LI-6800 portable photosynthesis system file.
- `load_all_li6800(path, silent=False)`: Load all LI-6800 files from directory.

#### Other Formats
- `load_csv(fn, timestamp_format='%Y-%m-%d %H:%M:%S', silent=False)`: Load single CSV file with timestamp parsing.
- `load_all_csv(path, dataset='.csv', timestamp_format='%Y-%m-%d %H:%M:%S', silent=False)`: Load all CSV files from directory.
- `load_all_zip(path, silent=False)`: Load all CSV files from within ZIP archives.
- `save_df(temp, output_path, output_fn, silent=True)`: Save DataFrame to CSV file.

### gases
Stomatal conductance adjustment factors

- Gas correction factors (R_ag_*): Stomatal conductance ratios for various gases relative to water vapor, allows to calculate conductance from known water vapor conductance

### graph
Visualization helper functions.

- `create_diurnal_df(temp, group_cols, data_cols, facet=False)`: Aggregate data for diurnal plotting with grouping and faceting support.

### meteo
Meteorological calculations.

- `calculate_last_precipitation(df, timestamp_col='timestamp', precipitation_col='P_1_1_1', max_gap_in_event_h=12)`: Calculate time since last precipitation and accumulated amount.

### physics
Physical calculations for atmospheric properties, fluxes, and conductance.

#### Thermodynamics
- `calculate_dewpointC(T_C, RH)`: Calculate dewpoint temperature from air temperature and relative humidity.
- `calculate_es(T_C, P_Pa)`: Calculate saturation vapor pressure using Campbell & Norman formulation.
- `calculate_VPD(T_C, h2o_mmol_mol, P_Pa)`: Calculate vapor pressure deficit.
- `convert_mmol_RH(T_C, h2o_mmol_mol, P_Pa)`: Convert water vapor concentration to relative humidity.
- `calculate_rho_dry_air(T_C, h2o_mmol_mol, P_Pa)`: Calculate density of dry air.
- `calculate_rho_moist_air(T_C, h2o_mmol_mol, P_Pa)`: Calculate density of moist air.
- `calculate_cp_dry_air(T_C)`: Calculate specific heat capacity of dry air.
- `calculate_cp_moist_air(T_C, h2o_mmol_mol, P_Pa)`: Calculate specific heat capacity of moist air.

#### Flux Calculations
- `calculate_h2o_flux(T_C, P_Pa, h2o_mmol_mol_ambient, h2o_mmol_mol_chamber, airflow_lpm, area_m2)`: Calculate water vapor flux from chamber measurements.
- `calculate_gas_flux(T_C, P_Pa, h2o_mmol_mol_ambient, h2o_mmol_mol_chamber, gas_mol_mol_ambient, gas_mol_mol_chamber, airflow_lpm, area_m2)`: Calculate gas flux accounting for water vapor dilution.

#### Conductance
- `calculate_cos_stomatal_conductance_ball_berry(T_C, h2o_mmol_mol, P_Pa, f_h2o_mmol_m2_s1, f_co2_umol_m2_s1, co2_umol_mol_ambient, PAR, diffusivity_ratio=R_ag_COS, bb_slope=17.207, bb_intercept=0.0487, bb_rh_min=70, bb_par_min=50)`: COS stomatal conductance (mol m⁻² s⁻¹) following Kooijmans et al. (2019): from transpiration and VPD, or from the Ball-Berry model at high RH in daylight, divided by the H2O/COS diffusivity ratio.
- `calculate_cos_total_conductance(f_cos_pmol_m2_s1, cos_pmol_mol_ambient)`: Calculate total COS conductance.
- `calculate_cos_internal_conductance(g_total, g_stomatal)`: Calculate internal COS conductance.
- `relative_uptake(f_cos_pmol_m2_s1, f_co2_umol_m2_s1, cos_pmol_mol_ambient, co2_umol_mol_ambient)`: Calculate leaf/soil relative COS uptake.

### units
Unit conversion utilities.

- `temperature_C_to_K(T_C)`: Convert temperature from Celsius to Kelvin.
- `temperature_K_to_C(T_K)`: Convert temperature from Kelvin to Celsius.
- `convert_RH_to_mmol(RH, T_C, P_Pa)`: Convert relative humidity to water vapor concentration.
- `convert_ppm_to_umol_m3(c_ppm, rho_dry_air)`: Convert gas concentration from ppm to µmol/m³.

### utils
Utility functions for data manipulation and processing (consolidated in data_processing.py).

- `sanitize_column_names(columns)`: Sanitise column names by removing special characters and ensuring uniqueness.
- `create_season_southern_hemisphere(timestamps)`: Create season labels for southern hemisphere (3-month seasons).
- `create_season_northern_hemisphere(timestamps)`: Create season labels for northern hemisphere (3-month seasons).
- `save_df(temp, output_path, output_fn, silent=True)`: Save DataFrame to CSV with optional logging.
- `nansum(x)`: Sum array treating NaN as zero (unlike numpy.nansum which returns NaN).
- `create_categorical_order(col, cat_order)`: Create ordered categorical series from column.
- `complete_timestamps(temp, timestamp_col='timestamp', freq='30min')`: Fill missing timestamps in DataFrame.
- `upsample_interpolate_df(temp, freq='1min', interpolation_limit=30)`: Upsample and interpolate DataFrame to higher frequency.

### partitioning
Functions for partitioning net ecosystem exchange into gross primary production and respiration.

#### Day/Night Detection
- `is_day(timestamp_series, lat, lon, tz=None, numeric=True, time_ref='standard')`: Determine day/night periods using astronomical calculations. `time_ref` says how the timestamps were recorded: `'standard'` (local standard time without daylight saving, default), `'local'` (with daylight saving) or `'utc'`.
- `localize_timestamps(timestamps, tz=None, time_ref='standard')`: Attach a timezone to naive timestamps according to `time_ref`.
- `to_standard_time(timestamps, tz, from_ref='local', ambiguous='infer')`: Convert timestamps recorded in local time with daylight saving (`from_ref='local'`) or in UTC (`from_ref='utc'`) to local standard time. The repeated hour when clocks go back is resolved from the record order.

#### Respiration Estimation
- `respiration_from_nighttime_simple_interpolated(temp, dn_col='dn', nee_col='nee')`: Estimate respiration using nighttime interpolation.
- `respiration_from_nighttime_simple_blocks(temp, dn_col='dn', nee_col='nee')`: Estimate respiration using nighttime blocks.
- `lloyd_taylor(T, R_ref, E0, T_ref=15.0, T0=-46.02)`: Lloyd-Taylor temperature response function for respiration.
- `fit_E0(temp, dn_col='day_night', Tair_col='TA_1_1_1', nee_col='nee_f', initial_guess=(1.0, 300.0))`: Fit E0 and R_ref of the Lloyd-Taylor model over all night-time data. Inflates E0 where R_ref follows the seasonal cycle of temperature; prefer `fit_E0_short_term`.
- `fit_E0_short_term(temp, dn_col='day_night', Tair_col='TA_1_1_1', nee_col='nee_f', timestamp_col='timestamp', window_days=15, shift_days=5, min_points=7, min_T_range=5.0, trim_percentile=5.0, n_best=3, E0_bounds=(30.0, 350.0))`: E0 from short-term windows of night-time data (Reichstein et al. 2005, as in REddyProc): Lloyd-Taylor fits in 15-day windows every 5 days with a temperature range of at least 5 K, residual tails trimmed, and the mean of the 3 plausible estimates with the smallest standard error. Returns E0 and the per-window table.
- `estimate_R_ref_moving_window_overlapping(temp, dn_col='day_night', Tair_col='TA_1_1_1', nee_col='nee_f', timestamp_col='timestamp', E0_col='E0_fit', window_days=15, shift_days=5, min_points=10, min_T_range=None)`: Estimate reference respiration with E0 fixed, in overlapping moving windows of night-time data. As in REddyProc, no temperature range is required by default and negative R_ref is discarded.
- `interpolate_R_ref(full_df, R_ref_df, timestamp_col='timestamp')`: Interpolate reference respiration values.

#### Core Partitioning
- `calculate_nee(co2_flux, storage_flux)`: Calculate net ecosystem exchange from CO2 flux and storage.
- `calculate_gpp(nee, reco, day_night=None, clip_negative=True)`: Calculate gross primary production from NEE and respiration. Pass `day_night` to force GPP to zero at night; without it, clipping night-time noise biases GPP upwards.
- `calculate_uStar_threshold_reichstein(df, Tair_col='TA_1_1_1', dn_col='day_night', uStar_col='u*', nee_col='nee', rules='reichstein', filter_threshold=None, use_night_only=True, min_uStar_threshold=0.01, na_uStar_threshold=0.4, threshold_if_none_found=False, n_Tair_classes=None, n_uStar_classes=20, n_plateau_bins=None, n_consecutive_bins=None, min_plateau_bins=None, corr_threshold=None, first_uStar_bin_max=None, min_valid_Tair_class_fraction=None, min_records=None, min_records_per_Tair_class=None)`: u* threshold (Reichstein et al. 2005). Settings left at None come from the rule set `rules` in `USTAR_RULES`: `'reichstein'` (default, as published: 6 temperature classes; a u* class is the threshold when its NEE reaches 99 % of the mean NEE of all higher classes; correlation limit 0.4) or `'reddyproc'` (7 temperature classes; a u* class and the next must reach 95 % of the mean NEE of the following 10 classes; correlation limit 0.5; temperature classes whose lowest u* class averages more than 0.2 are skipped). Median across temperature classes. Minimum record counts are optional (REddyProc: 160 per season, 100 per temperature class).
- `create_seasonal_uStar_threshold_list(df, groupby=['year', 'season'], Tair_col='TA_1_1_1', dn_col='day_night', uStar_col='u*', nee_col='nee', rules='reichstein', min_records=None, min_records_per_Tair_class=None)`: Create seasonal u* thresholds.
- `calculate_overall_uStar_threshold(thresholds_df, missing_fraction=1, use_mean=True)`: Calculate overall u* threshold from seasonal values.
- `uStar_filtering_wrapper(temp, timestamp_col='timestamp', hemisphere='north', apply_to_cols=['nee','co2_flux','co2_strg','h2o_flux','H','LE'], silent=False, uStar_aggregation='season', uStar_rules='reichstein', min_records=None, min_records_per_Tair_class=None)`: Apply u* filtering to data. By default every season is filtered with its own threshold, and seasons without one use the mean across seasons; `uStar_aggregation='overall'` applies that mean to the whole record. `uStar_rules='reddyproc'` switches the detection to the REddyProc rules. Adds the applied `uStar_threshold` column.
- `remove_nas(temp, cols)`: Remove rows with NaN values in specified columns.
- `partitioning_reichstein_wrapper(temp, timestamp_col='timestamp', dn_col='day_night', Tair_col='Tair', nee_col='nee_f', grouping_col='year', E0_method='short_term', window_days=15, shift_days=5)`: Complete partitioning workflow using Reichstein method: one E0 per group (short-term windows by default, `E0_method='annual'` for a single fit), then R_ref in moving windows, interpolated. Returns the timestamp, `E0` and `R_ref`.

#### Daytime Partitioning
- `partition_lasslop(temp, nee_col='nee_f', par_col=None, sw_col='SW_IN_1_1_1', sw_to_par=2.1, tair_col='Tair', vpd_col='VPD', E0_col='E0', dn_col='day_night', group_col=None, window_days=4, step_days=2, min_points=30, vpd0=None, vpd_unit='hPa', suffix='_lasslop')`: Daytime partitioning of Lasslop et al. (2010), fitted in moving windows. Adds `reco`, `gpp` and `nee` columns with the suffix, and returns the per-window fit table. `group_col` takes one column or a list; windows never span a group. Intended as a cross-check on the night-time method.
- `lasslop_nee(drivers, alpha, beta0, k, gamma, E0, vpd0=10.0)`: Lasslop light-response + Lloyd-Taylor NEE model.
- `fit_lasslop_window(par, T, vpd, nee, E0, vpd0=10.0, min_points=30)`: Fit that model to one window of daytime data.

#### Daytime Respiration Inhibition
Correction of the night-time partitioning for the light inhibition of daytime respiration (Kok effect), following Keenan et al. (2019) and their modified night-time method in REddyProc. Daytime reference respiration (RrefD) is the respiration intercept of the Lasslop light-response fit, with the night-time E0 held fixed; daytime Reco is then scaled by RrefD / RrefN, night-time Reco is unchanged, and GPP = Reco - NEE.
- `estimate_R_ref_day(temp, nee_col='nee_f', par_col=None, sw_col='SW_IN_1_1_1', sw_to_par=2.1, tair_col='Tair', vpd_col='VPD', E0_col='E0', R_ref_night_col='R_ref', dn_col='day_night', timestamp_col='timestamp', group_col=None, window_days=4, step_days=2, min_points=10, vpd0=None, vpd_unit='hPa', out_col='R_ref_day')`: Daytime R_ref in 4-day windows every 2 days, interpolated between window centres. Returns the frame with `out_col` added and the per-window table, including the inhibition and the filters of Keenan et al. `E0_col` must be the E0 that produced the night-time R_ref.
- `apply_respiration_inhibition(temp, reco_col, nee_col, R_ref_night_col='R_ref', R_ref_day_col='R_ref_day', dn_col='day_night', clip_negative=True, suffix='_inhib')`: Adds `reco_inhib`, `gpp_inhib` and `inhibition_inhib`. Works on a gap-filled Reco column.
- `calculate_respiration_inhibition(R_ref_night, R_ref_day)`: Inhibition I = (RrefN - RrefD) / RrefN, as a fraction.
- `summarise_inhibition(win_df, by_group=True)`: Monthly median inhibition of the valid windows, and its mean per calendar month over years.

```python
# After the night-time partitioning, with its 'E0' and 'R_ref' columns and Reco in 'reco'
data, windows = eco.estimate_R_ref_day(data, nee_col='nee_f', E0_col='E0', R_ref_night_col='R_ref',
                                       group_col='year')
data = eco.apply_respiration_inhibition(data, reco_col='reco', nee_col='nee_filled')
monthly, climatology = eco.summarise_inhibition(windows)
budget = eco.annual_carbon_budget(data, method='mds', reco_col='reco_inhib', ...)
```

### carbon_budget
Annual carbon sums that never treat gaps as zero.

- `mdc_annual_sum(temp, flux_col='nee_f', group_cols=(), time_bin='month', min_n=5, period_mins=30, reference_year=2021, dn_col='day_night', fallback_reco_col=None)`: Mean diurnal course binning (Falge et al. 2001): one bin per time of year x time of day, each converted to g C m-2.
- `mdc_total(bins, group_cols=())`: Annual total per group. NaN if any bin is empty; the sum of the non-empty bins is in `carbon_available.g_m2`.
- `mdc_cumulative(bins, group_cols=(), time_bin='month', reference_year=2021, period_mins=30)`: Cumulative within-year carbon series for plotting.
- `annual_carbon_budget(temp, method='mdc', nee_col='nee_f', reco_col='reco_umol_m2_s1', year_col='year', dn_col='day_night', ...)`: Per-year NEE, Reco, GPP and NEP from MDC bins (`method='mdc'`) or from a gap-filled series (`method='mds'`), with coverage statistics.

### gapfilling
- `gapfill_mds(temp, flux_col='nee_f', sw_col='SW_IN_1_1_1', tair_col='Tair', vpd_col='VPD', group_col=None, vpd_unit='hPa', ...)`: Marginal Distribution Sampling gap-filling (Reichstein et al. 2005). Adds the filled series plus quality flag, uncertainty and window columns. `group_col` takes one column or a list (e.g. `['site', 'period']`); look-up windows never cross a group.
- `gapfill_report(temp, flux_col='nee_f', suffix='_fill', group_col=None, dn_col='day_night', year_col='year')`: Measured and filled fractions by quality class, split by day and night.

### radiation
Solar geometry and shortwave reconstruction.

- `potential_radiation(timestamps, lat, lon, tz=None, transmissivity=0.75, offset_mins=0.0, time_ref='standard')`: Extraterrestrial and clear-sky shortwave, solar zenith and azimuth.
- `normalized_daylight_hour(temp, timestamp_col='timestamp', dn_col='day_night')`: Position within the daylight period, 0 at sunrise to 1 at sunset.
- `fit_horizon_profile(temp, rad_col='SW_IN_1_1_1', ...)`: Estimate the terrain horizon elevation by solar azimuth.
- `terrain_clearsky(temp, horizon, ...)`: Clear-sky shortwave with the direct beam removed where terrain blocks the sun.
- `reconstruct_swin(temp, sw_col='SW_IN_1_1_1', rn_col='NETRAD_1_1_1', method='clearness', ...)`: Reconstruct incoming shortwave from net radiation.
- `evaluate_swin(temp, sw_col='SW_IN_1_1_1', out_col='swin_reconstructed', dn_col='day_night', by=None)`: Compare reconstructed against measured shortwave.

### energy_budget
Energy budget analysis and gap-filling.

- `is_leap_year(year)`: Check if year is a leap year.
- `total_annual_periods(year, averaging_period_mins=30)`: Calculate total periods in a year.
- `annual_energy_budget(temp, H_col='H_tot_filled', LE_col='LE_tot_filled', Rn_col='Rn_filled', G_col='G_filled', period_mins=30, dn_col='day_night')`: Calculate annual energy budget statistics, including the daylight-only `EBR_day` and the 24 h `EBR`.
- `turbulent_energy_fluxes_gapfilling(temp, H_col='H', H_strg_col='H_strg', LE_col='LE', LE_strg_col='LE_strg', Rn_col='Rn', G_col='G', interp=True, interp_hours=2, dn_col=None, ebr_min_rn=50.0)`: Gap-fill turbulent energy fluxes. The half-hourly `EBR` is only computed where Rn - G exceeds `ebr_min_rn` (and in daylight when `dn_col` is given).
- `calculate_bowen_ratio(H, LE)`: Calculate Bowen ratio (H/LE).
- `calculate_albedo(SW_in, SW_out)`: Calculate surface albedo.
- `calculate_longwave_radiation(T_surf, emissivity=0.97, LW_in=None)`: Calculate longwave radiation components for a grey body; with `LW_in`, `LW_out` includes the reflected part.
- `calculate_surface_temperature_from_longwave(LW_out, emissivity=0.97, reflectance=None, LW_in=None)`: Calculate surface temperature from longwave radiation. Pass `LW_in` to remove the reflected part of the incoming longwave.

### stomatal_conductance
Stomatal conductance calculations.

- `total_water_conductance(T_leaf, P_Pa, E_mmol_m2_s, water_conc_air_mmol_mol)`: Calculate total water conductance.
- `leaf_conductance(T_leaf, P_Pa, E_mmol_m2_s, water_conc_air_mmol_mol)`: Calculate leaf conductance.
- `calculate_internal_concentration(conc_ambient, stomatal_cond, flux)`: Calculate internal concentration.
- `calculate_biochemical_conductance(T_leaf, LAI, E0=40000.0, T_ref=20.0)`: Biochemical conductance with an Arrhenius temperature response (E0 in J/mol).
- `calculate_mesophyll_conductance(T_leaf, LAI, T_opt=28.8, omega=0.61)`: Mesophyll conductance with a log-normal temperature response peaking at `T_opt` (°C).

### wue
Water use efficiency calculations.

- `calculate_wue(carbon_umol_m2_s1, ET_mm_h)`: Calculate WUE in gC/kgH2O.
- `calculate_wue_umol_mmol(carbon_umol_m2_s1, h2o_mmol_m2_s1)`: Calculate WUE in µmolC/mmolH2O.

## How to Cite

Muller (2025). *ecophys_utils: Python utilities library for ecophysiology tasks*. DOI: https://doi.org/10.5281/zenodo.20540557

## License

This software is distributed under the GNU GPL version 3. Any modification of the code in this repository may only be released under the same license, and with attribution of authorship of the original code (i.e., citation above).