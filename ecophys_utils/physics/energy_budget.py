# Energy budget functions
#------------------------
import numpy as np
from typing import Union
import pandas as pd
from scipy.stats import linregress

def is_leap_year(year: int) -> bool:
    """
    Check if a year is a leap year.

    Parameters
    ----------
    year : int
        Year to check.

    Returns
    -------
    bool
        True if leap year, False otherwise.
    """
    return((year % 4 == 0 and year % 100 != 0) or (year % 400 == 0))

def total_annual_periods(year: int, averaging_period_mins: int = 30) -> float:
    """
    Calculate total annual periods for a given averaging interval.

    Parameters
    ----------
    year : int
        Year.
    averaging_period_mins : int, optional
        Averaging period in minutes. Default is 30.

    Returns
    -------
    float
        Total number of periods in the year.
    """
    daily_periods = 24*60/averaging_period_mins
    return(365 * daily_periods + (daily_periods if is_leap_year(year) else 0))

def annual_energy_budget(temp: pd.DataFrame, H_col: str = 'H_tot_filled', LE_col: str = 'LE_tot_filled', Rn_col: str = 'Rn_filled', G_col: str = 'G_filled', period_mins: int = 30) -> pd.DataFrame:
    """
    Calculate annual energy budget statistics.

    Parameters
    ----------
    temp : pandas.DataFrame
        DataFrame with energy flux data.
    H_col : str, optional
        Column name for sensible heat flux. Default is 'H_tot_filled'.
    LE_col : str, optional
        Column name for latent heat flux. Default is 'LE_tot_filled'.
    Rn_col : str, optional
        Column name for net radiation. Default is 'Rn_filled'.
    G_col : str, optional
        Column name for ground heat flux. Default is 'G_filled'.
    period_mins : int, optional
        Period in minutes. Default is 30.

    Returns
    -------
    pandas.DataFrame
        DataFrame with annual energy budget statistics.

    See Also
    --------
    turbulent_energy_fluxes_gapfilling : Gap-fill turbulent fluxes.
    """
    results = []
    res_col='eb_residual.J_m2'

    for year, group in temp.groupby('year'):
        # Calculate residuals
        group[res_col] = (group[Rn_col] - group[H_col] - group[LE_col] - group[G_col])*period_mins*60

        # Statistics
        nb_halfhours = total_annual_periods(year, averaging_period_mins=period_mins)
        n_obs = group.loc[group[res_col].notna()].shape[0]
        # Calculate % of available data
        data_coverage = 100 * n_obs / nb_halfhours
        
        # Skip if too little data
        if n_obs == 0:
            continue

        # Sum residuals per year
        res_sum = group[res_col].sum()
        # Scale residuals to fraction of available data
        res_scaled = res_sum * (nb_halfhours / n_obs)
        
        # Calculate Energy Budget Ration (EBR)
        ebr = (group[H_col].sum() + group[LE_col].sum())/(group[Rn_col].sum() - group[G_col].sum())

        # Calculate regression
        radiative_fluxes = group[Rn_col].to_numpy() - group[G_col].to_numpy() # independent variable (x)
        turbulent_fluxes = group[H_col].to_numpy() + group[LE_col].to_numpy() # dependent variable (y)
        mask = ~np.isnan(radiative_fluxes) & ~np.isnan(turbulent_fluxes)
        radiative_fluxes, turbulent_fluxes = radiative_fluxes[mask], turbulent_fluxes[mask]
    
        if n_obs >= 2:
            lr = linregress(radiative_fluxes, turbulent_fluxes)
            slope     = lr.slope
            intercept = lr.intercept
            r2        = lr.rvalue**2
        else:
            slope = intercept = r2 = np.nan
    

        results.append({
            'year': year,
            'n_obs': n_obs,
            'data_coverage.perc': data_coverage,
            #'total_annual_halfhours': nb_halfhours,
            'residuals_sum': res_sum, # [J/m2]
            'residuals_scaled': res_scaled, # [J/m2]
            'slope': slope,
            'intercept': intercept,
            'r2': r2,
            'EBR': ebr
        })

    return pd.DataFrame(results)

def turbulent_energy_fluxes_gapfilling(temp: pd.DataFrame, H_col: str = 'H', H_strg_col: str = 'H_strg', LE_col: str = 'LE', LE_strg_col: str = 'LE_strg', Rn_col: str = 'Rn', G_col: str = 'G', interp: bool = True, interp_hours: int = 2) -> pd.DataFrame:
    """
    Gapfill turbulent energy fluxes.

    Parameters
    ----------
    temp : pandas.DataFrame
        DataFrame with flux data.
    H_col : str, optional
        Column name for sensible heat flux. Default is 'H'.
    H_strg_col : str, optional
        Column name for sensible heat storage. Default is 'H_strg'.
    LE_col : str, optional
        Column name for latent heat flux. Default is 'LE'.
    LE_strg_col : str, optional
        Column name for latent heat storage. Default is 'LE_strg'.
    Rn_col : str, optional
        Column name for net radiation. Default is 'Rn'.
    G_col : str, optional
        Column name for ground heat flux. Default is 'G'.
    interp : bool, optional
        Whether to interpolate gaps. Default is True.
    interp_hours : int, optional
        Hours to interpolate. Default is 2.

    Returns
    -------
    pandas.DataFrame
        DataFrame with gapfilled fluxes.
    """
    temp = temp.copy()

    # Calculate total H & LE (incl. storage)
    temp['H_tot']  = temp[H_col]  + temp[H_strg_col]
    temp['LE_tot'] = temp[LE_col] + temp[LE_strg_col]

    if(interp):
        # Interpolate gaps <=2h (i.e. 4 values)
        temp['H_tot_filled']  = temp['H_tot'].interpolate(method='linear', limit=interp_hours*2, limit_direction='both')
        temp['LE_tot_filled'] = temp['LE_tot'].interpolate(method='linear', limit=interp_hours*2, limit_direction='both')
        temp['Rn_filled'] = temp[Rn_col].interpolate(method='linear', limit=interp_hours*2, limit_direction='both')
        temp['H_filled']  = temp[H_col].interpolate(method='linear', limit=interp_hours*2, limit_direction='both')
        temp['LE_filled'] = temp[LE_col].interpolate(method='linear', limit=interp_hours*2, limit_direction='both')
        temp['G_filled']  = temp[G_col].interpolate(method='linear', limit=interp_hours*2, limit_direction='both')
        temp['turbulent_energy_fluxes'] = temp['H_tot_filled'] + temp['LE_tot_filled']
        temp['radiative_energy_fluxes'] = temp['Rn_filled'] - temp['G_filled']
        # Calculate half-hourly EBR
        temp['EBR'] = temp['turbulent_energy_fluxes']/temp['radiative_energy_fluxes']
    else:
        temp['turbulent_energy_fluxes'] = temp['H_tot'] + temp['LE_tot']
        temp['radiative_energy_fluxes'] = temp[Rn_col] - temp[G_col]
        # Calculate half-hourly EBR
        temp['EBR'] = temp['turbulent_energy_fluxes']/temp['radiative_energy_fluxes']

    return(temp)