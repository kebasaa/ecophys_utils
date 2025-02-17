def temperature_C_to_K(T_C):
    T_K = T_C + 273.15
    return(T_K)
    
def temperature_K_to_C(T_K):
    T_Cs = T_K - 273.15
    return(T_C)
    
# Converts relative humidity RH [%] to water concentration [mmol mol]
def convert_RH_to_mmol(RH, T_C, P_Pa):
    # From Eddypro manual: https://www.licor.com/env/support/EddyPro/topics/calculate-micromet-variables.html
    es = calculate_es(T_C, P_Pa)   # Water vapor partial pressure at saturation (Pa)
    e = RH/100 * es                # Water vapor partial pressure (Pa)
    h2o_mol_mol = e / P_Pa         # Water concentration (mol mol-1)
    
    # Unit conversions
    h2o_mmol_mol = h2o_mol_mol * 10**3 # water in [mmol mol-1]
    
    return(h2o_mmol_mol)
    
# Converts ppm to umol m-3 of dry air
def convert_ppm_to_umol_m3(c_ppm, rho_dry_air):
    # Constants
    M_d   = 0.02897              # molecular weights of dry air (kg mol-1)
    
    c_umol_m3 = c_ppm * rho_dry_air / M_d
    return(c_umol_m3)