# Units conversion functions
#---------------------------
import numpy as np
from typing import Union
from ..physics.physics_thermodynamics import calculate_es

def temperature_C_to_K(T_C: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert temperature from Celsius to Kelvin.

    Parameters
    ----------
    T_C : float or numpy.ndarray
        Temperature in degrees Celsius.

    Returns
    -------
    float or numpy.ndarray
        Temperature in Kelvin.
    """
    T_K = T_C + 273.15
    return(T_K)
    
def temperature_K_to_C(T_K: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert temperature from Kelvin to Celsius.

    Parameters
    ----------
    T_K : float or numpy.ndarray
        Temperature in Kelvin.

    Returns
    -------
    float or numpy.ndarray
        Temperature in degrees Celsius.
    """
    T_C = T_K - 273.15
    return(T_C)
    
def convert_RH_to_mmol(RH: Union[float, np.ndarray], T_C: Union[float, np.ndarray], P_Pa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert relative humidity to water concentration in mmol/mol.

    Parameters
    ----------
    RH : float or numpy.ndarray
        Relative humidity in percent.
    T_C : float or numpy.ndarray
        Temperature in degrees Celsius.
    P_Pa : float or numpy.ndarray
        Atmospheric pressure in Pascals.

    Returns
    -------
    float or numpy.ndarray
        Water concentration in mmol/mol.
    """
    # From Eddypro manual: https://www.licor.com/env/support/EddyPro/topics/calculate-micromet-variables.html
    es = calculate_es(T_C, P_Pa)   # Water vapor partial pressure at saturation (Pa)
    e = RH/100 * es                # Water vapor partial pressure (Pa)
    h2o_mol_mol = e / P_Pa         # Water concentration (mol mol-1)
    
    # Unit conversions
    h2o_mmol_mol = h2o_mol_mol * 10**3 # water in [mmol mol-1]
    
    return(h2o_mmol_mol)
    
def convert_ppm_to_umol_m3(c_ppm: Union[float, np.ndarray], rho_dry_air: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert concentration from ppm to µmol/m³ of dry air.

    Parameters
    ----------
    c_ppm : float or numpy.ndarray
        Concentration in ppm.
    rho_dry_air : float or numpy.ndarray
        Density of dry air in kg/m³.

    Returns
    -------
    float or numpy.ndarray
        Concentration in µmol/m³.
    """
    # Constants
    M_d   = 0.02897              # molecular weights of dry air (kg mol-1)
    
    c_umol_m3 = c_ppm * rho_dry_air / M_d
    return(c_umol_m3)