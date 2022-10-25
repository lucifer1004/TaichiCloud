# Specific heat capacities
c_pd = 1005.0  # Dry air
c_pv = 1850.0  # Water vapor
c_pw = 4218.0  # Liquid water

# Molar masses (kg / moles)
M_H = 1e-3
M_C = 12e-3
M_N = 14e-3
M_O = 16e-3
M_S = 32e-3

# O2, O3
M_O2 = 2 * M_O
M_O3 = 3 * M_O

# H2O
M_OH = M_O + M_H
M_H2O = 2 * M_H + M_O
M_H2O2 = 2 * M_H + M_O2

# SO2 * H2O
M_SO2 = M_S + 2 * M_O
M_SO2_H2O = M_SO2 + M_H2O
M_HSO3 = M_OH + M_SO2
M_SO3 = M_O + M_SO2

# NH3 * H2O
M_NH3 = M_N + 3 * M_H
M_NH3_H2O = M_NH3 + M_H2O
M_NH4 = M_NH3 + M_H

# NO3
M_NO3 = M_N + 3 * M_O
M_HNO3 = M_H + M_NO3

# CO2 * H2O
M_CO2 = M_C + 2 * M_O
M_CO2_H2O = M_CO2 + M_H2O
M_HCO3 = M_OH + M_CO2
M_CO3 = M_O + M_CO2

# H2SO4
M_H2SO4 = 2 * M_H + M_S + 4 * M_O
M_HSO4 = M_H + M_S + 4 * M_O
M_SO4 = M_S + 4 * M_O

M_d = 0.02897  # Dry air (Curry &Webster / Seinfeld & Pandis)
M_v = M_H2O
eps = M_v / M_d

# Universal gas constant (J / K / mol)
# i.e. k_B * N_A (the Boltzmann times the Avogadro constants)
kaBoNA = 8.3144621

# Gas constants
R_d = kaBoNA / M_d
R_v = kaBoNA / M_v

# Exner function exponent for dry air
R_d_over_c_pd = R_d / c_pd

# Water density
rho_w = 1e3

# Vapor diffusivity in air (see Properties of air, Tracy, Welch & Porter 1980)
D_0 = 2.26e-5  # m^2 / s

# Standard pressure and temperature
p_0 = 1e5  # Pa
T_0 = 273.15  # K

# Thermal conductivity of air
K_0 = 2.4e-2  # J / m / s / K

# Water triple point parameters
p_tri = 611.73  # Pa
T_tri = 273.16  # K
l_tri = 2.5e6  # J / kg
