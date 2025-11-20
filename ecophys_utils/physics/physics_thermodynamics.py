# Physics: Thermodynamics functions
#---------------------------------
import numpy as np
import pandas as pd
from typing import Union, Tuple

# Calculate the dewpoint temperature in C
def calculate_dewpointC(T_C: Union[float, np.ndarray], RH: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate the dewpoint temperature from temperature and relative humidity.

    Uses the Magnus formula approximation for dewpoint calculation.

    Parameters
    ----------
    T_C : float or numpy.ndarray
        Air temperature in degrees Celsius.
    RH : float or numpy.ndarray
        Relative humidity as a percentage (0-100).

    Returns
    -------
    float or numpy.ndarray
        Dewpoint temperature in degrees Celsius.

    Notes
    -----
    The formula used is an approximation and may have limitations at extreme temperatures.
    """
    # Input validation
    if np.any(RH < 0) or np.any(RH > 100):
        raise ValueError("Invalid input: RH must be between 0 and 100")

    dp = 243.04*(np.log(RH/100)+((17.625*T_C)/(243.04+T_C)))/(17.625-np.log(RH/100)-((17.625*T_C)/(243.04+T_C)))
    return(dp)

# Calculate saturation vapour pressure from pressure and temperature
# - 2 methods are available. Jones uses air pressure, Campbell & Norman do not
def calculate_es(T_C: Union[float, np.ndarray], P_Pa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate saturation vapor pressure from temperature and pressure.

    Uses the Campbell & Norman (1998) formulation.

    Parameters
    ----------
    T_C : float or numpy.ndarray
        Air temperature in degrees Celsius.
    P_Pa : float or numpy.ndarray
        Atmospheric pressure in Pascals.

    Returns
    -------
    float or numpy.ndarray
        Saturation vapor pressure in Pascals.

    References
    ----------
    Campbell, G. S., & Norman, J. M. (1998). An introduction to environmental biophysics.
    Springer Science & Business Media.
    """
    # Input validation
    if np.any(P_Pa <= 0) or np.any(T_C < -273.15):
        raise ValueError("Invalid input: P_Pa must be > 0, T_C must be > -273.15")

    # Jones p.348 (appendix 4)
    #es = (1.00072+(10**(-7)*P_Pa*(0.032+5.9*10**(-6)*T_C**2))) * (611.21*np.exp( (18.678-(T_C/234.5))*T_C/(257.14+T_C) ))

    # Eddypro manual: https://www.licor.com/env/support/EddyPro/topics/calculate-micromet-variables.html
    # Campbell & Norman (1998)
    T_K = T_C + 273.15
    es = T_K**(-8.2) * np.exp(77.345 + 0.0057*T_K - 7235 * T_K**(-1))
    return(es)

# Calculates VPD from different environmental variables
def calculate_VPD(T_C: Union[float, np.ndarray], h2o_mmol_mol: Union[float, np.ndarray], P_Pa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate vapor pressure deficit (VPD) from temperature, water vapor concentration, and pressure.

    Parameters
    ----------
    T_C : float or numpy.ndarray
        Air temperature in degrees Celsius.
    h2o_mmol_mol : float or numpy.ndarray
        Water vapor concentration in mmol/mol.
    P_Pa : float or numpy.ndarray
        Atmospheric pressure in Pascals.

    Returns
    -------
    float or numpy.ndarray
        Vapor pressure deficit in Pascals.

    See Also
    --------
    calculate_es : Calculate saturation vapor pressure.

    Notes
    -----
    VPD is calculated as the difference between saturation vapor pressure and actual vapor pressure.
    """
    # Input validation
    if np.any(P_Pa <= 0) or np.any(T_C < -273.15) or np.any(h2o_mmol_mol < 0):
        raise ValueError("Invalid input: P_Pa must be > 0, T_C must be > -273.15, h2o_mmol_mol must be >= 0")

    # Unit conversions 
    T_K = T_C + 273.15           # Temperature in K
    h2o_mol_mol = h2o_mmol_mol * 10**(-3) # water in [mol mol-1]

    # From Eddypro manual: https://www.licor.com/env/support/EddyPro/topics/calculate-micromet-variables.html
    e = h2o_mol_mol * P_Pa         # Water vapor partial pressure (Pa)
    es = calculate_es(T_C, P_Pa)   # Water vapor partial pressure at saturation (Pa)
    VPD = es - e                   # VPD (Pa)
    return(VPD)

# Converts water concentration [mmol mol] to RH [%]
def convert_mmol_RH(T_C: Union[float, np.ndarray], h2o_mmol_mol: Union[float, np.ndarray], P_Pa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert water vapor concentration to relative humidity.

    Parameters
    ----------
    T_C : float or numpy.ndarray
        Air temperature in degrees Celsius.
    h2o_mmol_mol : float or numpy.ndarray
        Water vapor concentration in mmol/mol.
    P_Pa : float or numpy.ndarray
        Atmospheric pressure in Pascals.

    Returns
    -------
    float or numpy.ndarray
        Relative humidity as a percentage (0-100).

    See Also
    --------
    calculate_VPD : Calculate vapor pressure deficit.
    """
    # Input validation
    if np.any(P_Pa <= 0) or np.any(T_C < -273.15) or np.any(h2o_mmol_mol < 0):
        raise ValueError("Invalid input: P_Pa must be > 0, T_C must be > -273.15, h2o_mmol_mol must be >= 0")

    # Unit conversions 
    T_K = T_C + 273.15           # Temperature in K
    h2o_mol_mol = h2o_mmol_mol * 10**(-3) # water in [mol mol-1]
    
    # From Eddypro manual: https://www.licor.com/env/support/EddyPro/topics/calculate-micromet-variables.html
    e = h2o_mol_mol * P_Pa         # Water vapor partial pressure (Pa)
    es = calculate_es(T_C, P_Pa)   # Water vapor partial pressure at saturation (Pa)
    RH = e/es * 100                # RH (%)
    return(RH)

# Density of dry air
# - https://www.licor.com/env/support/EddyPro/topics/calculate-micromet-variables.html
def calculate_rho_dry_air(T_C: Union[float, np.ndarray], h2o_mmol_mol: Union[float, np.ndarray], P_Pa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate the density of dry air.

    Parameters
    ----------
    T_C : float or numpy.ndarray
        Air temperature in degrees Celsius.
    h2o_mmol_mol : float or numpy.ndarray
        Water vapor concentration in mmol/mol.
    P_Pa : float or numpy.ndarray
        Atmospheric pressure in Pascals.

    Returns
    -------
    float or numpy.ndarray
        Density of dry air in kg/m³.
    """
    # Input validation
    if np.any(P_Pa <= 0) or np.any(T_C < -273.15) or np.any(h2o_mmol_mol < 0):
        raise ValueError("Invalid input: P_Pa must be > 0, T_C must be > -273.15, h2o_mmol_mol must be >= 0")

    # Constants
    from ...units.constants import R_dry_air, R, M_d, M_h2o
    
    # Unit conversions 
    T_K = T_C + 273.15           # Temperature in K
    h2o_mol_mol = h2o_mmol_mol * 10**(-3) # water in [mol mol-1]
    
    # Preparations
    e = h2o_mol_mol * P_Pa       # Water vapor partial pressure (Pa)
    P_d = P_Pa - e               # Dry air partial pressure (P_d, P_a)

    rho_dry_air = P_d / (R_dry_air * T_K) # Density of dry air (use for approximation)
    return(rho_dry_air)

# Density of moist air
def calculate_rho_moist_air(T_C: Union[float, np.ndarray], h2o_mmol_mol: Union[float, np.ndarray], P_Pa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate the density of moist air.

    Parameters
    ----------
    T_C : float or numpy.ndarray
        Air temperature in degrees Celsius.
    h2o_mmol_mol : float or numpy.ndarray
        Water vapor concentration in mmol/mol.
    P_Pa : float or numpy.ndarray
        Atmospheric pressure in Pascals.

    Returns
    -------
    float or numpy.ndarray
        Density of moist air in kg/m³.
    """
    # Input validation
    if np.any(P_Pa <= 0) or np.any(T_C < -273.15) or np.any(h2o_mmol_mol < 0):
        raise ValueError("Invalid input: P_Pa must be > 0, T_C must be > -273.15, h2o_mmol_mol must be >= 0")

    # Constants
    from ...units.constants import R, M_d, M_h2o
    
    # Unit conversions 
    T_K = T_C + 273.15           # Temperature in K
    h2o_mol_mol = h2o_mmol_mol * 10**(-3) # water in [mol mol-1]
    
    # Preparations
    e = h2o_mol_mol * P_Pa       # Water vapor partial pressure (Pa)
    P_d = P_Pa - e               # Dry air partial pressure (P_d, P_a)
    rho_d = P_d / (R / M_d * T_K) # Dry air mass density (rho_d, kg m-3)
    v_d = M_d / rho_d            # Dry air molar volume (vd, m3 mol-1)
    v_a = v_d * P_d/P_Pa         # Air molar volume (vd, m3mol-1) 
    rho_h2o = h2o_mol_mol * M_h2o / v_a # Ambient water vapor mass density (kg m-3)

    # Moist air mass density (ρa, kg m-3) 
    rho_air = rho_d + rho_h2o

    return(rho_air)

# Dry air heat capacity at constant pressure
# cp_d in [J kg-1 K-1]
 # https://www.licor.com/env/support/EddyPro/topics/calculate-micromet-variables.html
def calculate_cp_dry_air(T_C: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate the specific heat capacity of dry air at constant pressure.

    Parameters
    ----------
    T_C : float or numpy.ndarray
        Air temperature in degrees Celsius.

    Returns
    -------
    float or numpy.ndarray
        Specific heat capacity of dry air in J/(kg·K).
    """
    cp = 1005 + ((T_C + 23.12)**2)/3364
    return(cp)

# Specific heat capacity of moist air at constant pressure
# cp_m in [J kg-1 K-1]
# https://www.licor.com/env/support/EddyPro/topics/calculate-micromet-variables.html
def calculate_cp_moist_air(T_C: Union[float, np.ndarray], h2o_mmol_mol: Union[float, np.ndarray], P_Pa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate the specific heat capacity of moist air at constant pressure.

    Parameters
    ----------
    T_C : float or numpy.ndarray
        Air temperature in degrees Celsius.
    h2o_mmol_mol : float or numpy.ndarray
        Water vapor concentration in mmol/mol.
    P_Pa : float or numpy.ndarray
        Atmospheric pressure in Pascals.

    Returns
    -------
    float or numpy.ndarray
        Specific heat capacity of moist air in J/(kg·K).
    """
    # Constants
    from ...units.constants import R, M_d, M_h2o
    
    # Unit conversions 
    T_K = T_C + 273.15           # Temperature in K
    h2o_mol_mol = h2o_mmol_mol * 10**(-3) # water in [mol mol-1]

    # RH
    RH = convert_mmol_RH(T_C, h2o_mmol_mol, P_Pa)

    # Water vapor heat capacity at constant pressure (cp_h2o, J kg-1 K-1)
    cp_h2o = 1859 + 0.13*RH + (0.193 + 5.6*10**(-3) * RH)*T_C + (10**(-3) + 5 * 10**(-5)*RH)*T_C**2
    
    # Preparations
    e = h2o_mol_mol * P_Pa       # Water vapor partial pressure (Pa)
    P_d = P_Pa - e               # Dry air partial pressure (P_d, P_a)
    rho_d = P_d / (R / M_d * T_K) # Dry air mass density (rho_d, kg m-3)
    v_d = M_d / rho_d            # Dry air molar volume (vd, m3 mol-1)
    v_a = v_d * P_d/P_Pa         # Air molar volume (vd, m3mol-1) 
    rho_h2o = h2o_mol_mol * M_h2o / v_a # Ambient water vapor mass density (kg m-3)

    # Moist air mass density (ρa, kg m-3) 
    rho_air = rho_d + rho_h2o

    # Specific humidity (Q, kg kg-1) 
    Q = rho_h2o / rho_air

    # cp_moist
    cp = calculate_cp_dry_air(T_C) * (1-Q) + cp_h2o * Q
    return(cp)