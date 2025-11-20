# ecophys_utils: Python utilities library for ecophysiology tasks

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)

![Version](https://img.shields.io/badge/version-0.1-blue)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

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
- `calculate_cos_stomatal_conductance_ball_berry(T_C, h2o_mmol_mol, P_Pa, f_h2o_mmol_m2_s1, f_co2_umol_m2_s1, co2_umol_mol_ambient, PAR)`: Calculate COS stomatal conductance using Ball-Berry model.
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
- `is_day(timestamp_series, lat, lon, tz, numeric=True)`: Determine day/night periods using astronomical calculations.

#### Respiration Estimation
- `respiration_from_nighttime_simple_interpolated(temp, dn_col='dn', nee_col='nee')`: Estimate respiration using nighttime interpolation.
- `respiration_from_nighttime_simple_blocks(temp, dn_col='dn', nee_col='nee')`: Estimate respiration using nighttime blocks.
- `lloyd_taylor(T, R_ref, E0, T_ref=15.0, T0=-46.02)`: Lloyd-Taylor temperature response function for respiration.
- `fit_E0(temp, dn_col='day_night', Tair_col='TA_1_1_1', nee_col='nee_f', initial_guess=(1.0, 300.0))`: Fit E0 parameter for Lloyd-Taylor model.
- `estimate_R_ref_moving_window_overlapping(temp, dn_col='day_night', Tair_col='TA_1_1_1', nee_col='nee_f', window_days=7, step_days=1, min_points=10)`: Estimate reference respiration using moving window.
- `interpolate_R_ref(full_df, R_ref_df, timestamp_col='timestamp')`: Interpolate reference respiration values.

#### Core Partitioning
- `calculate_nee(co2_flux, storage_flux)`: Calculate net ecosystem exchange from CO2 flux and storage.
- `calculate_gpp(nee, reco)`: Calculate gross primary production from NEE and respiration.
- `calculate_uStar_threshold_reichstein(df, Tair_col='TA_1_1_1', dn_col='day_night', uStar_col='u*', nee_col='nee', filter_threshold=0.99, use_night_only=True, min_uStar_threshold=0.01, na_uStar_threshold=0.4, threshold_if_none_found=False)`: Calculate u* threshold using Reichstein method.
- `create_seasonal_uStar_threshold_list(df, groupby=['year', 'season'])`: Create seasonal u* thresholds.
- `calculate_overall_uStar_threshold(thresholds_df, missing_fraction=1, use_mean=True)`: Calculate overall u* threshold from seasonal values.
- `uStar_filtering_wrapper(temp, timestamp_col='timestamp', hemisphere='north', apply_to_cols=['nee','co2_flux','co2_strg','h2o_flux','H','LE'], silent=False)`: Apply u* filtering to data.
- `remove_nas(temp, cols)`: Remove rows with NaN values in specified columns.
- `partitioning_reichstein_wrapper(temp, timestamp_col='timestamp', dn_col='day_night', Tair_col='Tair', nee_col='nee_f', grouping_col='year')`: Complete partitioning workflow using Reichstein method.

### energy_budget
Energy budget analysis and gap-filling.

- `is_leap_year(year)`: Check if year is a leap year.
- `total_annual_periods(year, averaging_period_mins=30)`: Calculate total periods in a year.
- `annual_energy_budget(temp, H_col='H_tot_filled', LE_col='LE_tot_filled', Rn_col='Rn_filled', G_col='G_filled', period_mins=30)`: Calculate annual energy budget statistics.
- `turbulent_energy_fluxes_gapfilling(temp, H_col='H', H_strg_col='H_strg', LE_col='LE', LE_strg_col='LE_strg', Rn_col='Rn', G_col='G', interp=True, interp_hours=2)`: Gap-fill turbulent energy fluxes.

### stomatal_conductance
Stomatal conductance calculations.

- `total_water_conductance(T_leaf, P_Pa, E_mmol_m2_s, water_conc_air_mmol_mol)`: Calculate total water conductance.
- `leaf_conductance(T_leaf, P_Pa, E_mmol_m2_s, water_conc_air_mmol_mol)`: Calculate leaf conductance.
- `calculate_internal_concentration(conc_ambient, stomatal_cond, flux)`: Calculate internal concentration.

### wue
Water use efficiency calculations.

- `calculate_wue(carbon_umol_m2_s1, ET_mm_h)`: Calculate WUE in gC/kgH2O.
- `calculate_wue_umol_mmol(carbon_umol_m2_s1, h2o_mmol_m2_s1)`: Calculate WUE in µmolC/mmolH2O.

## How to Cite

Muller (2025). *ecophys_utils: Python utilities library for ecophysiology tasks*

## License

This software is distributed under the GNU GPL version 3. Any modification of the code in this repository may only be released under the same license, and with attribution of authorship of the original code (i.e., citation above).