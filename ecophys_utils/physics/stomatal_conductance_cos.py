# Physics: Conductance functions
#---------------------------------
import numpy as np
import pandas as pd
from typing import Union
from .physics_thermodynamics import calculate_VPD, convert_mmol_RH
from ..units.constants import R

# Conductance calculations
def calculate_cos_stomatal_conductance_ball_berry(T_C: Union[float, np.ndarray], h2o_mmol_mol: Union[float, np.ndarray], P_Pa: Union[float, np.ndarray], f_h2o_mmol_m2_s1: Union[float, np.ndarray], f_co2_umol_m2_s1: Union[float, np.ndarray], co2_umol_mol_ambient: Union[float, np.ndarray], PAR: Union[float, np.ndarray]) -> pd.Series:
    """
    Calculate COS stomatal conductance using the Ball-Berry model.

    Parameters
    ----------
    T_C : float or numpy.ndarray
        Air temperature in degrees Celsius.
    h2o_mmol_mol : float or numpy.ndarray
        Water vapor mole fraction in mmol/mol.
    P_Pa : float or numpy.ndarray
        Atmospheric pressure in Pascals.
    f_h2o_mmol_m2_s1 : float or numpy.ndarray
        Water vapor flux in mmol/m²/s.
    f_co2_umol_m2_s1 : float or numpy.ndarray
        CO2 flux in µmol/m²/s.
    co2_umol_mol_ambient : float or numpy.ndarray
        Ambient CO2 concentration in µmol/mol.
    PAR : float or numpy.ndarray
        Photosynthetically active radiation.

    Returns
    -------
    pandas.Series
        Stomatal conductance for COS.
    """
    # Input validation
    if np.any(P_Pa <= 0) or np.any(T_C < -273.15) or np.any(h2o_mmol_mol < 0) or np.any(co2_umol_mol_ambient <= 0) or np.any(PAR < 0):
        raise ValueError("Invalid input: P_Pa > 0, T_C > -273.15, h2o_mmol_mol >= 0, co2_umol_mol_ambient > 0, PAR >= 0")

    # Note: T_C is the air temperature in C
    # Calculate some initial values
    VPD_Pa = calculate_VPD(T_C, h2o_mmol_mol, P_Pa)
    RH = convert_mmol_RH(T_C, h2o_mmol_mol, P_Pa)
    
    # Pressure in hPa
    P_kPa   = P_Pa / 1000
    VPD_kPa = VPD_Pa / 1000
    
    # Stomatal conductance
    g_s_cos = (f_h2o_mmol_m2_s1 * P_kPa * 10 / VPD_kPa) * 2 / 10000
    
    # Temporarily convert to df
    temp = pd.DataFrame({'rh': RH, 'par': PAR, 'f_co2': f_co2_umol_m2_s1, 'co2_a': co2_umol_mol_ambient, 'g_s': g_s_cos})
    
    # Ball-Berry correction
    day_par_min = 50  # Define constant for minimum PAR for day
    temp['bbm'] = -temp['f_co2'] * temp['rh'] * 0.01 / temp['co2_a']
    temp.loc[(temp['rh'] > 70) & (temp['par'] > day_par_min), 'g_s'] = 17.207 * temp.loc[(temp['rh'] > 70) & (temp['par'] > day_par_min), 'bbm'] / 0.0487
    
    return(temp['g_s'])

def calculate_cos_total_conductance(f_cos_pmol_m2_s1: Union[float, np.ndarray], cos_pmol_mol_ambient: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate total conductance for COS.

    Parameters
    ----------
    f_cos_pmol_m2_s1 : float or numpy.ndarray
        COS flux in pmol/m²/s.
    cos_pmol_mol_ambient : float or numpy.ndarray
        Ambient COS concentration in pmol/mol.

    Returns
    -------
    float or numpy.ndarray
        Total conductance for COS.
    """
    g_t_cos = - (f_cos_pmol_m2_s1 / cos_pmol_mol_ambient)
    return(g_t_cos)

def calculate_cos_internal_conductance(g_total: Union[float, np.ndarray], g_stomatal: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate internal conductance for COS.

    Parameters
    ----------
    g_total : float or numpy.ndarray
        Total conductance.
    g_stomatal : float or numpy.ndarray
        Stomatal conductance.

    Returns
    -------
    float or numpy.ndarray
        Internal conductance for COS.

    See Also
    --------
    calculate_cos_total_conductance : Calculate total COS conductance.
    calculate_cos_stomatal_conductance_ball_berry : Calculate stomatal COS conductance.
    """
    g_i_cos = (g_total**(-1) - g_stomatal**(-1))**(-1)
    return(g_i_cos)

def relative_uptake(f_cos_pmol_m2_s1: Union[float, np.ndarray], f_co2_umol_m2_s1: Union[float, np.ndarray], cos_pmol_mol_ambient: Union[float, np.ndarray], co2_umol_mol_ambient: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate leaf/soil relative uptake (LRU/SRU).

    Parameters
    ----------
    f_cos_pmol_m2_s1 : float or numpy.ndarray
        COS flux in pmol/m²/s.
    f_co2_umol_m2_s1 : float or numpy.ndarray
        CO2 flux in µmol/m²/s.
    cos_pmol_mol_ambient : float or numpy.ndarray
        Ambient COS concentration in pmol/mol.
    co2_umol_mol_ambient : float or numpy.ndarray
        Ambient CO2 concentration in µmol/mol.

    Returns
    -------
    float or numpy.ndarray
        Relative uptake ratio.
    """
    ru = f_cos_pmol_m2_s1 / f_co2_umol_m2_s1 * co2_umol_mol_ambient / cos_pmol_mol_ambient
    return(ru)

