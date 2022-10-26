import taichi as ti
from common.constants import *


@ti.func
def mix(dry: ti.template(), vap: ti.template(), r: ti.template()):
    return (dry + r * vap) / (1 + r)


@ti.func
def R(r: ti.template()):
    return mix(R_d, R_v, r)


@ti.func
def c_p(r: ti.template()):
    return mix(c_pd, c_pv, r)


@ti.func
def p_v(p: ti.template(), r: ti.template()):
    return p * r / (r + eps)


@ti.func
def D(T: ti.template(), p: ti.template()):
    return D_0 * (T / T_0) ** 1.81 * (p_0 / p)


@ti.func
def p_vs(T: ti.template()):
    """Saturated vapor pressure for water assuming constant c_p_v and c_p_w"""
    return p_tri * ti.exp((l_tri + (c_pw - c_pv) * T_tri) / R_v * (1 / T_tri - 1 / T) - (c_pw - c_pv) / R_v * ti.log(T / T_tri))


@ ti.func
def r_vs(T: ti.template(), p: ti.template()):
    """Saturated vapor mixing ratio for water as a function of pressure and temperature"""
    return eps / (p / p_vs(T) - 1)


@ ti.func
def l_v(T: ti.template()):
    """Latent heat for constant c_p"""
    return l_tri + (c_pw - c_pv) * (T - T_tri)
