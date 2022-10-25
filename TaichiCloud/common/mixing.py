import taichi as ti
from common.constants import *


@ti.func
def mix(dry: float, vap: float, r: float) -> float:
    return (dry + r * vap) / (1 + r)


@ti.func
def R(r: float) -> float:
    return mix(R_d, R_v, r)


@ti.func
def c_p(r: float) -> float:
    return mix(c_pd, c_pv, r)


@ti.func
def p_v(p: float, r: float) -> float:
    return p * r / (r + eps)


@ti.func
def D(T: float, p: float) -> float:
    return D_0 * (T / T_0) ** 1.81 * (p_0 / p)


@ti.func
def p_vs(T: float) -> float:
    """Saturated vapor pressure for water assuming constant c_p_v and c_p_w"""
    return p_tri * ti.exp((l_tri + (c_pw - c_pv) * T_tri) / R_v * (1 / T_tri - 1 / T) - (c_pw - c_pv) / R_v * ti.log(T / T_tri))


@ ti.func
def r_vs(T: float, p: float) -> float:
    """Saturated vapor mixing ratio for water as a function of pressure and temperature"""
    return eps / (p / p_vs(T) - 1)


@ ti.func
def l_v(T: float) -> float:
    """Latent heat for constant c_p"""
    return l_tri + (c_pw - c_pv) * (T - T_tri)
