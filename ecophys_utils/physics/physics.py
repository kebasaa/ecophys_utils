# Physics: Unit converstions, fluxes, etc.
#-----------------------------------------
import numpy as np
import pandas as pd

# Calculate the dewpoint temperature in C
def calculate_dewpointC(T_C, RH):
    dp = 243.04*(np.log(RH/100)+((17.625*T_C)/(243.04+T_C)))/(17.625-log(RH/100)-((17.625*T_C)/(243.04+T_C)))
    return(dp)

# Calculate saturation vapour pressure from pressure and temperature
# - 2 methods are available. Jones uses air pressure, Campbell & Norman do not
def calculate_es(T_C, P_Pa):
    # Jones p.348 (appendix 4)
    #es = (1.00072+(10**(-7)*P_Pa*(0.032+5.9*10**(-6)*T_C**2))) * (611.21*np.exp( (18.678-(T_C/234.5))*T_C/(257.14+T_C) ))

    # Eddypro manual: https://www.licor.com/env/support/EddyPro/topics/calculate-micromet-variables.html
    # Campbell & Norman (1998)
    T_K = T_C + 273.15
    es = T_K**(-8.2) * np.exp(77.345 + 0.0057*T_K - 7235 * T_K**(-1))
    return(es)

# Calculates VPD from different environmental variables
def calculate_VPD(T_C, h2o_mmol_mol, P_Pa):
    # Unit conversions 
    T_K = T_C + 273.15           # Temperature in K
    h2o_mol_mol = h2o_mmol_mol * 10**(-3) # water in [mol mol-1]

    # From Eddypro manual: https://www.licor.com/env/support/EddyPro/topics/calculate-micromet-variables.html
    e = h2o_mol_mol * P_Pa         # Water vapor partial pressure (Pa)
    es = calculate_es(T_C, P_Pa)   # Water vapor partial pressure at saturation (Pa)
    VPD = es - e                   # VPD (Pa)
    return(VPD)

# Converts water concentration [mmol mol] to RH [%]
def convert_mmol_RH(T_C, h2o_mmol_mol, P_Pa):
    # Unit conversions 
    T_K = T_C + 273.15           # Temperature in K
    h2o_mol_mol = h2o_mmol_mol * 10**(-3) # water in [mol mol-1]
    
    #es = calculate_es(T_C, P_Pa)
    #RH <- 0.263*P_Pa*((h2o_mmol_mol*18.02/28.97)/1000)*np.exp(17.67*(T_C)/(T_K-29.65))**(-1)
    #RH = 100 if (RH > 100) else RH
    #RH = np.nan if (RH < 5) else RH

    # From Eddypro manual: https://www.licor.com/env/support/EddyPro/topics/calculate-micromet-variables.html
    e = h2o_mol_mol * P_Pa         # Water vapor partial pressure (Pa)
    es = calculate_es(T_C, P_Pa)   # Water vapor partial pressure at saturation (Pa)
    RH = e/es * 100                # RH (%)
    return(RH)

# Density of dry air
# - https://www.licor.com/env/support/EddyPro/topics/calculate-micromet-variables.html
def calculate_rho_dry_air(T_C, h2o_mmol_mol, P_Pa):
    # Constants
    from from ..units.constants import R_dry_air, R, M_d, M_h2o
    
    # Unit conversions 
    T_K = T_C + 273.15           # Temperature in K
    h2o_mol_mol = h2o_mmol_mol * 10**(-3) # water in [mol mol-1]
    
    # Preparations
    e = h2o_mol_mol * P_Pa       # Water vapor partial pressure (Pa)
    P_d = P_Pa - e               # Dry air partial pressure (P_d, P_a)
    
    rho_dry_air = P_d / (R_dry_air * T_K) # Density of dry air (use for approximation)
    return(rho_dry_air)

# Density of moist air
def calculate_rho_moist_air(T_C, h2o_mmol_mol, P_Pa):
    # Constants
    from ..units.constants import R, M_d, M_h2o
    
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
def calculate_cp_dry_air(T_C):
    cp = 1005 + ((T_C + 23.12)**2)/3364
    return(cp)

# Specific heat capacity of moist air at constant pressure
# cp_m in [J kg-1 K-1]
# https://www.licor.com/env/support/EddyPro/topics/calculate-micromet-variables.html
def calculate_cp_moist_air(T_C, h2o_mmol_mol, P_Pa):
    # Constants
    from ..units.constants import R, M_d, M_h2o
    
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

# Calculates the flux of water
def calculate_h2o_flux(T_C, P_Pa, h2o_mmol_mol_ambient, h2o_mmol_mol_chamber, airflow_lpm, area_m2):
    # Constants
    from ..units.constants import R, M_d, M_h2o
    
    # Unit conversions
    T_K = T_C + 273.15 # Temperature in K
    h2o_mol_mol_ambient = h2o_mmol_mol_ambient * 10**(-3)
    h2o_mol_mol_chamber = h2o_mmol_mol_chamber * 10**(-3)
    
    # Preparations
    e = h2o_mol_mol_chamber * P_Pa # Water vapor partial pressure (Pa)
    P_d = P_Pa - e               # Dry air partial pressure (P_d, P_a)
    rho_d = P_d / (R / M_d * T_K) # Dry air mass density (rho_d, kg m-3)
    v_d = M_d / rho_d            # Dry air molar volume (v_d, m3 mol-1)
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
def calculate_gas_flux(T_C, P_Pa, h2o_mmol_mol_ambient, h2o_mmol_mol_chamber, gas_mol_mol_ambient, gas_mol_mol_chamber, airflow_lpm, area_m2):
    # Constants
    from ..units.constants import R, M_d, M_h2o
    
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

# Conductance calculations
def calculate_cos_stomatal_conductance_ball_berry(T_C, h2o_mmol_mol, P_Pa, f_h2o_mmol_m2_s1, f_co2_umol_m2_s1, co2_umol_mol_ambient, PAR):
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
    temp['bbm'] = -temp['f_co2'] * temp['rh'] * 0.01 / temp['co2_a']
    temp.loc[(temp['rh'] > 70) & (temp['par'] > day_par_min), 'g_s'] = 17.207 * temp.loc[(temp['rh'] > 70) & (temp['par'] > day_par_min), 'bbm'] / 0.0487
    
    return(temp['g_s'])

def calculate_cos_total_conductance(f_cos_pmol_m2_s1, cos_pmol_mol_ambient):
    # Total conductance
    g_t_cos = - (f_cos_pmol_m2_s1 / cos_pmol_mol_ambient)
    return(g_t_cos)

def calculate_cos_internal_conductance(g_total, g_stomatal):
    # Total conductance
    g_i_cos = (g_total**(-1) - g_stomatal**(-1))**(-1)
    return(g_i_cos)

# Calculates leaf/soil relative uptake, i.e. LRU and SRU
def relative_uptake(f_cos_pmol_m2_s1, f_co2_umol_m2_s1, cos_pmol_mol_ambient, co2_umol_mol_ambient):
    ru = f_cos_pmol_m2_s1 / f_co2_umol_m2_s1 * co2_umol_mol_ambient / cos_pmol_mol_ambient
    return(ru)

def calculate_biochemical_conductance(T_leaf, LAI):
    # Constants
    from ..units.constants import R
    
    # Constants
    E0 = 40
    ref_T_C = 20
    
    # Preparations
    ref_T_K = ref_T_C + 273.15
    T_leaf_K = T_leaf + 273.15
    
    # Biochemical conductance
    g_ca = 0.8*0.055*LAI * np.exp((E0/R) * (1/ref_T_K - 1/T_leaf_K))

    return(g_ca)

def calculate_mesophyll_conductance(T_leaf, LAI):
    g_m = 0.188*LAI * np.exp(-0.5*(np.log((T_leaf/28.8)/0.61))**2)

    return(g_m)