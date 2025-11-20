# Physics: Flux calculation functions
#---------------------------------
import numpy as np
from typing import Union
from ..units.constants import R, M_d, M_h2o

# Calculates the flux of water
def calculate_h2o_flux(T_C: Union[float, np.ndarray], P_Pa: Union[float, np.ndarray], h2o_mmol_mol_ambient: Union[float, np.ndarray], h2o_mmol_mol_chamber: Union[float, np.ndarray], airflow_lpm: Union[float, np.ndarray], area_m2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate water vapor flux from chamber measurements.

    Uses the LI-6400 manual equations for flux calculation from concentration differences.

    Parameters
    ----------
    T_C : float or numpy.ndarray
        Chamber temperature in degrees Celsius.
    P_Pa : float or numpy.ndarray
        Atmospheric pressure in Pascals.
    h2o_mmol_mol_ambient : float or numpy.ndarray
        Ambient water vapor concentration in mmol/mol.
    h2o_mmol_mol_chamber : float or numpy.ndarray
        Chamber water vapor concentration in mmol/mol.
    airflow_lpm : float or numpy.ndarray
        Airflow rate in liters per minute.
    area_m2 : float or numpy.ndarray
        Leaf area in square meters.

    Returns
    -------
    float or numpy.ndarray
        Water vapor flux in mol/(m²·s).

    Notes
    -----
    This function assumes measurement of outflow from the chamber.
    """
    # Input validation
    if np.any(P_Pa <= 0) or np.any(T_C < -273.15):
        raise ValueError("Invalid input: P_Pa must be > 0, T_C must be > -273.15")

    # Constants
    # Unit conversions
    T_K = T_C + 273.15 # Temperature in K
    h2o_mol_mol_ambient = h2o_mmol_mol_ambient * 10**(-3)
    h2o_mol_mol_chamber = h2o_mmol_mol_chamber * 10**(-3)
    
    # Preparations
    e = h2o_mol_mol_chamber * P_Pa # Water vapor partial pressure (Pa)
    P_d = P_Pa - e               # Dry air partial pressure (P_d, P_a)
    rho_d = P_d / (R / M_d * T_K) # Dry air mass density (rho_d, kg m-3)
    v_d = M_d / rho_d            # Dry air molar volume (vd, m3 mol-1)
    v_a = v_d * P_d/P_Pa         # Air molar volume (v_a, m3mol-1)
    
    # Convert airflow from LPM to mol.s-1
    airflow_m3_s = airflow_lpm / 1000 / 60 # airflow conversion to total flow [m3.s-1]
    airflow_mol_s = airflow_m3_s / v_a     # airflow conversion to moist air [mol.s-1]
    
    # Note: We are measuring the outflow from the chamber, not the inflow (as opposed to the LI-6400)
    #       Therefore, we need to change eq. 1-2 (p. 1-7) to be ue = uo - sE
    #       This leads to eq. 1-3 changing to sE = uo*wo - (uo - sE)*we
    #       Finally, eq. 1-4 becomes E = uo*(wo - we) / (s*(1 - we))
    #       Where uo = airflow_mol_s
    #             wo = h2o_mol_mol_chamber (outgoing mol fraction)
    #             we = h2o_mol_mol_ambient (entering mol fraction)
    #             s  = area_m2 (leaf area)
       
    # LI-6400 manual, eq. 1-7
    h2o_flux = (airflow_mol_s * (h2o_mol_mol_chamber - h2o_mol_mol_ambient)) / (area_m2 * (1 - h2o_mol_mol_ambient))
    
    return(h2o_flux) # mol.m-2.s-1

# Inspired from LI-6400 manual: Uses the water flux because water changes the air density. If stomata are open and H2o is added, the gas is more 'diluted'
def calculate_gas_flux(T_C: Union[float, np.ndarray], P_Pa: Union[float, np.ndarray], h2o_mmol_mol_ambient: Union[float, np.ndarray], h2o_mmol_mol_chamber: Union[float, np.ndarray], gas_mol_mol_ambient: Union[float, np.ndarray], gas_mol_mol_chamber: Union[float, np.ndarray], airflow_lpm: Union[float, np.ndarray], area_m2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate gas flux from chamber measurements, accounting for water vapor dilution.

    Parameters
    ----------
    T_C : float or numpy.ndarray
        Chamber temperature in degrees Celsius.
    P_Pa : float or numpy.ndarray
        Atmospheric pressure in Pascals.
    h2o_mmol_mol_ambient : float or numpy.ndarray
        Ambient water vapor concentration in mmol/mol.
    h2o_mmol_mol_chamber : float or numpy.ndarray
        Chamber water vapor concentration in mmol/mol.
    gas_mol_mol_ambient : float or numpy.ndarray
        Ambient gas concentration in mol/mol.
    gas_mol_mol_chamber : float or numpy.ndarray
        Chamber gas concentration in mol/mol.
    airflow_lpm : float or numpy.ndarray
        Airflow rate in liters per minute.
    area_m2 : float or numpy.ndarray
        Leaf area in square meters.

    Returns
    -------
    float or numpy.ndarray
        Gas flux in mol/(m²·s).

    See Also
    --------
    calculate_h2o_flux : Calculate water vapor flux.

    Notes
    -----
    Accounts for the dilution effect of water vapor on gas concentrations.
    Assumes measurement of outflow from the chamber.
    """
    # Input validation
    if np.any(P_Pa <= 0) or np.any(T_C < -273.15) or np.any(area_m2 <= 0):
        raise ValueError("Invalid input: P_Pa and area_m2 must be > 0, T_C must be > -273.15")

    # Constants
    # Unit conversions
    T_K = T_C + 273.15 # Temperature in K
    h2o_mol_mol_ambient = h2o_mmol_mol_ambient * 10**(-3)
    h2o_mol_mol_chamber = h2o_mmol_mol_chamber * 10**(-3)
    
    # Preparations
    e = h2o_mol_mol_chamber * P_Pa # Water vapor partial pressure (Pa)
    P_d = P_Pa - e               # Dry air partial pressure (P_d, P_a)
    rho_d = P_d / (R / M_d * T_K) # Dry air mass density (rho_d, kg m-3)
    v_d = M_d / rho_d            # Dry air molar volume (vd, m3 mol-1)
    v_a = v_d * P_d/P_Pa         # Air molar volume (vd, m3mol-1)
    
    # Convert airflow from LPM to mol.s-1
    airflow_m3_s = airflow_lpm / 1000 / 60 # airflow conversion to total flow [m3.s-1]
    airflow_mol_s = airflow_m3_s / v_a     # airflow conversion to moist air [mol.s-1]
    
    # H2O transpiration flux [mol.m-2.s-1]
    h2o_flux_mol_m2_s = calculate_h2o_flux(T_C, P_Pa, h2o_mmol_mol_ambient, h2o_mmol_mol_chamber, airflow_lpm, area_m2)
    
    # Note: We are measuring the outflow from the chamber, not the inflow (as opposed to the LI-6400)
    #       Therefore, we need to change eq. 1-2 (p. 1-7) to be ue = uo - sE
    #       The LI-6400 uses assimilation, which is a positive flux. Previously, we used negative numbers due to CO2 uptake
    #       The sign should reflect the direction of the flux, i.e. emission is positive, and uptake negative, so assimilation should be negative
    #       The following calculations will therefore change eq. 1-11 (p. 1-9) to sa = uo*co - ue*ce
    #       This leads to eq. 1-12 (p. 1-9) changing to sa = uo*co - (uo - sE)*ce
    #       Finally, eq. 1-13 becomes a = uo*(co-ce) / s + E*ce
    #       Where uo = airflow_mol_s
    #             co = co2_mol_mol_chamber (outgoing mol fraction)
    #             ce = co2_mol_mol_ambient (entering mol fraction)
    #             s  = area_m2 (leaf area)
    #             E  = water flux (calculated above)
    
    # LI-6400 manual, eq. 1-13 (p. 1-10)
    gas_flux = (airflow_mol_s * (gas_mol_mol_chamber - gas_mol_mol_ambient)) / area_m2 + h2o_flux_mol_m2_s * gas_mol_mol_ambient
    
    return(gas_flux) # mol.m-2.s-1