# Water Use Efficiency functions
#-------------------------------
#
# There are 2 Water Use Efficiencies:
#
# - Physiological Water Use Efficiency (WUE_GPP), to study stomatal physiology (i.e., transpiration <-> photosynthesis link),
#   because respiration (Reco) is not directly related to stomatal control. This is mainly used in plant physiological studies
#   It is calculated from GPP/ET
#   
# - Ecosystem Water Use Efficiency (WUE_NEP), to study net ecosystem carbon balance (including respiration losses) per unit water used.
#   calculated from NEP/ET, where NEP (Net Ecosystem Production) = GPP − Reco (ecosystem C gain), or NEP = -NEE

import numpy as np
from typing import Union

# Water Use Efficiency (gC/kgH2O)
def calculate_wue(carbon_umol_m2_s1: Union[float, np.ndarray], ET_mm_h: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate water use efficiency in gC/kgH2O.

    Parameters
    ----------
    carbon_umol_m2_s1 : float or numpy.ndarray
        Carbon flux in µmol/m²/s.
    ET_mm_h : float or numpy.ndarray
        Evapotranspiration in mm/h.

    Returns
    -------
    float or numpy.ndarray
        Water use efficiency in gC/kgH2O.
    """
    # Constants
    from ..units.constants import M_C
    
    # Convert ET from mm h-1 to kgH2O m-2 s-1
    # 1 mm of water over 1 m² equals 1 kg, so per s, divide by 3600
    ET_kgH2O_m2_s1 = ET_mm_h/3600
    ET_kgH2O_m2_s1 = np.where(ET_kgH2O_m2_s1 < 0.00001, 0, ET_kgH2O_m2_s1)

    carbon_gC_m2_s1 = carbon_umol_m2_s1 * 10**(-6) * M_C
    
    wue_gC_kgH2O = carbon_gC_m2_s1 / ET_kgH2O_m2_s1
    wue_gC_kgH2O = np.where(np.isnan(wue_gC_kgH2O) | np.isinf(wue_gC_kgH2O), np.nan, wue_gC_kgH2O)
    
    return(wue_gC_kgH2O)
    
def calculate_wue_umol_mmol(carbon_umol_m2_s1: Union[float, np.ndarray], h2o_mmol_m2_s1: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate water use efficiency in µmolC/mmolH2O.

    Parameters
    ----------
    carbon_umol_m2_s1 : float or numpy.ndarray
        Carbon flux in µmol/m²/s.
    h2o_mmol_m2_s1 : float or numpy.ndarray
        Water flux in mmol/m²/s.

    Returns
    -------
    float or numpy.ndarray
        Water use efficiency in µmolC/mmolH2O.

    See Also
    --------
    calculate_wue : Calculate WUE in gC/kgH2O.
    """
    # Correct h2o to ET, i.e. no negative flux
    h2o_mmol_m2_s1 = np.where(h2o_mmol_m2_s1 < 0.00001, 0, h2o_mmol_m2_s1)
    
    wue_umolC_mmolH2O = carbon_umol_m2_s1 / h2o_mmol_m2_s1
    wue_umolC_mmolH2O = np.where(np.isnan(wue_umolC_mmolH2O) | np.isinf(wue_umolC_mmolH2O), np.nan, wue_umolC_mmolH2O)
    
    return(wue_umolC_mmolH2O)
