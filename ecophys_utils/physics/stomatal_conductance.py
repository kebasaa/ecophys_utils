# Stomatal conductance functions
#-------------------------------
import numpy as np
from typing import Union
from ..units.constants import R

def total_water_conductance(T_leaf: Union[float, np.ndarray], P_Pa: Union[float, np.ndarray], E_mmol_m2_s: Union[float, np.ndarray], water_conc_air_mmol_mol: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate total water conductance.

    Parameters
    ----------
    T_leaf : float or numpy.ndarray
        Leaf temperature in degrees Celsius.
    P_Pa : float or numpy.ndarray
        Atmospheric pressure in Pascals.
    E_mmol_m2_s : float or numpy.ndarray
        Transpiration rate in mmol/m²/s.
    water_conc_air_mmol_mol : float or numpy.ndarray
        Water concentration in air in mmol/mol.

    Returns
    -------
    float or numpy.ndarray
        Total water conductance.
    """
    from . import calculate_es
    # Unit conversion
    E_mol_m2_s = E_mmol_m2_s * 10**(-3)
    water_conc_air = water_conc_air_mmol_mol * 10**(-3)
    
    # Water concentration inside the leaf [mol H2O mol-1 air]
    # assuming saturation in the sub-stomatal cavity (Monteith 1995)
    water_conc_sat = calculate_es(T_leaf, P_Pa) / P_Pa
    
    # water_conc_air is the water concentration in the ambient air plus the H2O molecules added by transpiration,
    # resulting in leaf-to-air VPD (Preisler et al 2023)
    
    # water flux (transpiration)
    g_tw = (E_mol_m2_s*(1 - (water_conc_sat + water_conc_air)/2))/(water_conc_sat - water_conc_air) # Was + at the end (LI-6400, eq. 1-7)
    
    # WARNING: Removed np.abs() to expose non-physical values (e.g. negative conductance).
    # Review inputs if negative values occur.
    return(g_tw)

def leaf_conductance(T_leaf: Union[float, np.ndarray], P_Pa: Union[float, np.ndarray], E_mmol_m2_s: Union[float, np.ndarray], water_conc_air_mmol_mol: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate leaf conductance using Buckley (2005) model.

    Parameters
    ----------
    T_leaf : float or numpy.ndarray
        Leaf temperature in degrees Celsius.
    P_Pa : float or numpy.ndarray
        Atmospheric pressure in Pascals.
    E_mmol_m2_s : float or numpy.ndarray
        Transpiration rate in mmol/m²/s.
    water_conc_air_mmol_mol : float or numpy.ndarray
        Water concentration in air in mmol/mol.

    Returns
    -------
    float or numpy.ndarray
        Leaf conductance.

    References
    ----------
    Buckley (2005) https://doi.org/10.1111/j.1469-8137.2005.01543.x
    """
    from . import calculate_es
    # Unit conversion
    E_mol_m2_s = E_mmol_m2_s * 10**(-3)
    water_conc_air = water_conc_air_mmol_mol * 10**(-3)
    
    # Water concentration inside the leaf [mol H2O mol-1 air]
    # assuming saturation in the sub-stomatal cavity (Monteith 1995)
    water_conc_sat = calculate_es(T_leaf, P_Pa) / P_Pa
    
    g_w = E_mol_m2_s / (water_conc_sat - water_conc_air)
    #g_tw = E / VPD * 100
    
    # WARNING: Removed np.abs() to expose non-physical values (e.g. negative conductance).
    # Review inputs if negative values occur.
    return(g_w)

def total_water_conductance_wrapper(df: pd.DataFrame, T_leaf_col: str = 'T_leaf', P_Pa_col: str = 'P_Pa', E_col: str = 'E_mmol_m2_s', h2o_air_col: str = 'water_conc_air_mmol_mol') -> pd.DataFrame:
    """
    Wrapper for total_water_conductance that adds the result and a negative value flag to a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    T_leaf_col : str
        Column name for leaf temperature.
    P_Pa_col : str
        Column name for pressure.
    E_col : str
        Column name for transpiration rate.
    h2o_air_col : str
        Column name for ambient water concentration.

    Returns
    -------
    pandas.DataFrame
        DataFrame with 'g_tw' and 'g_tw_neg_flag' columns added.
    """
    df = df.copy()
    g_tw = total_water_conductance(df[T_leaf_col], df[P_Pa_col], df[E_col], df[h2o_air_col])
    df['g_tw'] = g_tw
    df['g_tw_neg_flag'] = g_tw < 0
    return df

def calculate_internal_concentration(conc_ambient: Union[float, np.ndarray], stomatal_cond: Union[float, np.ndarray], flux: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate internal concentration.

    Based on F = g * (Ci - Ca), where Ci is internal concentration.

    Parameters
    ----------
    conc_ambient : float or numpy.ndarray
        Ambient concentration.
    stomatal_cond : float or numpy.ndarray
        Stomatal conductance.
    flux : float or numpy.ndarray
        Flux.

    Returns
    -------
    float or numpy.ndarray
        Internal concentration.
    """
    Ci = flux / stomatal_cond + conc_ambient
    return(Ci)

def calculate_biochemical_conductance(T_leaf: Union[float, np.ndarray], LAI: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate biochemical conductance.

    Parameters
    ----------
    T_leaf : float or numpy.ndarray
        Leaf temperature in degrees Celsius.
    LAI : float or numpy.ndarray
        Leaf area index.

    Returns
    -------
    float or numpy.ndarray
        Biochemical conductance.
    """
    # Constants
    E0 = 40
    ref_T_C = 20

    # Preparations
    ref_T_K = ref_T_C + 273.15
    T_leaf_K = T_leaf + 273.15

    # Biochemical conductance
    g_ca = 0.8*0.055*LAI * np.exp((E0/R) * (1/ref_T_K - 1/T_leaf_K))

    return(g_ca)

def calculate_mesophyll_conductance(T_leaf: Union[float, np.ndarray], LAI: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate mesophyll conductance.

    Parameters
    ----------
    T_leaf : float or numpy.ndarray
        Leaf temperature in degrees Celsius.
    LAI : float or numpy.ndarray
        Leaf area index.

    Returns
    -------
    float or numpy.ndarray
        Mesophyll conductance.
    """
    g_m = 0.188*LAI * np.exp(-0.5*(np.log((T_leaf/28.8)/0.61))**2)

    return(g_m)