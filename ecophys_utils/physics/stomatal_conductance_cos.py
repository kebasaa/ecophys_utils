# Physics: Conductance functions
#---------------------------------
import numpy as np
import pandas as pd
from typing import Union
from .physics_thermodynamics import calculate_VPD
from ..units.constants import R
from ..units.units import convert_mmol_to_RH
from ..gases.gas_parameters import R_ag_COS

# Conductance calculations
def calculate_cos_stomatal_conductance_ball_berry(T_C: Union[float, np.ndarray], h2o_mmol_mol: Union[float, np.ndarray], P_Pa: Union[float, np.ndarray], f_h2o_mmol_m2_s1: Union[float, np.ndarray], f_co2_umol_m2_s1: Union[float, np.ndarray], co2_umol_mol_ambient: Union[float, np.ndarray], PAR: Union[float, np.ndarray],
                                                  diffusivity_ratio: float = R_ag_COS, bb_slope: float = 17.207, bb_intercept: float = 0.0487,
                                                  bb_rh_min: float = 70.0, bb_par_min: float = 50.0) -> pd.Series:
    """
    Calculate COS stomatal conductance, from transpiration or from the Ball-Berry model.

    Follows Kooijmans et al. (2019), PNAS 116:2470-2475. Stomatal conductance to water
    vapour is taken from transpiration normalised by VPD, g_s,H2O = E * P / VPD. Where RH
    is high and the sun is up, measured transpiration is unreliable (in chambers, water
    adsorbs on the walls) and VPD is small, so g_s,H2O comes instead from the Ball-Berry
    model, g_s,H2O = m * A * h / Ca + g0 (Ball et al. 1987; Kooijmans et al. 2019, eq. 3),
    with m and g0 fitted to the site's own low-RH data. Net CO2 uptake (-f_co2) stands in
    for assimilation A, as in Kooijmans et al. Both are converted to COS by dividing by the
    ratio of the H2O and COS diffusivities.

    Air VPD is used, not leaf-to-air VPD, and there is no boundary-layer or aerodynamic
    resistance: a big-leaf approximation.

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
        CO2 flux in µmol/m²/s, negative for uptake.
    co2_umol_mol_ambient : float or numpy.ndarray
        Ambient CO2 concentration in µmol/mol.
    PAR : float or numpy.ndarray
        Photosynthetically active radiation (µmol/m²/s).
    diffusivity_ratio : float, optional
        Ratio of the H2O and COS diffusivities. Default R_ag_COS (1.955). Kooijmans et
        al. (2019) use 2.00 (Seibt et al. 2010); Stimler et al. (2010) and SiB4 (Cho et
        al. 2023) use 1.94.
    bb_slope, bb_intercept : float, optional
        Ball-Berry slope m (dimensionless) and intercept g0 (mol/m²/s) for H2O.
        Defaults 17.207 and 0.0487 are site-fitted values; refit them for other sites.
    bb_rh_min, bb_par_min : float, optional
        The Ball-Berry model is used where RH > bb_rh_min (%) and PAR > bb_par_min.
        Defaults 70 and 50.

    Returns
    -------
    pandas.Series
        Stomatal conductance for COS (mol/m²/s).

    References
    ----------
    Kooijmans, L. M. J., et al. (2019). Influences of light and humidity on carbonyl
    sulfide-based estimates of photosynthesis. PNAS, 116(7), 2470-2475.
    """
    # Input validation
    if np.any(P_Pa <= 0) or np.any(T_C < -273.15) or np.any(h2o_mmol_mol < 0) or np.any(co2_umol_mol_ambient <= 0) or np.any(PAR < 0):
        raise ValueError("Invalid input: P_Pa > 0, T_C > -273.15, h2o_mmol_mol >= 0, co2_umol_mol_ambient > 0, PAR >= 0")

    # Note: T_C is the air temperature in C
    # Calculate some initial values
    VPD_Pa = calculate_VPD(T_C, h2o_mmol_mol, P_Pa)
    RH = convert_mmol_to_RH(h2o_mmol_mol, T_C, P_Pa)

    # Stomatal conductance to H2O from transpiration [mol m-2 s-1], then to COS
    g_s_h2o = f_h2o_mmol_m2_s1 * 1e-3 * P_Pa / VPD_Pa
    g_s_cos = g_s_h2o / diffusivity_ratio

    # Temporarily convert to df
    temp = pd.DataFrame({'rh': RH, 'par': PAR, 'f_co2': f_co2_umol_m2_s1, 'co2_a': co2_umol_mol_ambient, 'g_s': g_s_cos})

    # Ball-Berry at high RH in daylight: A * h / Ca [mol m-2 s-1], with h as a fraction
    temp['bbm'] = -temp['f_co2'] * temp['rh'] * 0.01 / temp['co2_a']
    use_bb = (temp['rh'] > bb_rh_min) & (temp['par'] > bb_par_min)
    temp.loc[use_bb, 'g_s'] = (bb_slope * temp.loc[use_bb, 'bbm'] + bb_intercept) / diffusivity_ratio

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

