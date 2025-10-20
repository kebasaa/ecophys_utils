def total_water_conductance(T_leaf, P_Pa, E_mmol_m2_s, water_conc_air_mmol_mol):
    # Unit conversion
    E_mol_m2_s = E_mmol_m2_s * 10**(-3)
    water_conc_air = water_conc_air_mmol_mol * 10**(-3)
    
    # Water concentration inside the leaf [mol H2O mol-1 air]
    # assuming saturation in the sub-stomatal cavity (Monteith 1995)
    water_conc_sat = calculate_es(T_leaf) / P_Pa
    
    # water_conc_air is the water concentration in the ambient air plus the H2O molecules added by transpiration,
    # resulting in leaf-to-air VPD (Preisler et al 2023)
    
    # water flux (transpiration)
    g_tw = (E_mol_m2_s*(1 - (water_conc_sat + water_conc_air)/2))/(water_conc_sat - water_conc_air) # Was + at the end (LI-6400, eq. 1-7)
    
    return(np.abs(g_tw))

def total_water_conductance2(T_leaf, P_Pa, E_mmol_m2_s, water_conc_air_mmol_mol):
    # Unit conversion
    E_mol_m2_s = E_mmol_m2_s * 10**(-3)
    water_conc_air = water_conc_air_mmol_mol
    
    # Water concentration inside the leaf [mmol H2O mol-1 air]
    # assuming saturation in the sub-stomatal cavity (Monteith 1995)
    water_conc_sat = 0.61365*np.exp((17.592*T_leaf)/(240.97+T_leaf))
    
    # water_conc_air is the water concentration in the ambient air plus the H2O molecules added by transpiration,
    # resulting in leaf-to-air VPD (Preisler et al 2023)
    
    # water flux (transpiration)
    g_tw = (E_mmol_m2_s*(1000 - (water_conc_sat + water_conc_air)/2))/(water_conc_sat - water_conc_air)
    
    return(np.abs(g_tw))

# Buckley (2005)  https://doi.org/10.1111/j.1469-8137.2005.01543.x
def leaf_conductance(T_leaf, P_Pa, E_mmol_m2_s, water_conc_air_mmol_mol):
    # Unit conversion
    E_mol_m2_s = E_mmol_m2_s * 10**(-3)
    water_conc_air = water_conc_air_mmol_mol * 10**(-3)
    
    # Water concentration inside the leaf [mol H2O mol-1 air]
    # assuming saturation in the sub-stomatal cavity (Monteith 1995)
    water_conc_sat = calculate_es(T_leaf) / P_Pa
    
    g_w = E_mol_m2_s / (water_conc_sat - water_conc_air)
    #g_tw = E / VPD * 100
    return(np.abs(g_w))

def conc_internal(conc_ambient, stomatal_cond, flux):
    # Based on F=g [Ci-Ca]
    Ci = flux / stomatal_cond + conc_ambient
    return(Ci)