import taichi as ti
from common.constants import *
from common.mixing import *


@ti.func
def rhod(p: float, th_std: float, rv: float) -> float:
    """Dry air density as a function of p, theta and rv"""
    return (p - p_v(p, rv)) / ((p / p_0) ** (R_d / c_pd) * R_d * th_std)


@ti.func
def exner(p: float) -> float:
    """Exner pressure"""
    return (p / p_0) ** (R_d / c_pd)


@ti.func
def T(th: float, rhod: float) -> float:
    return (th * (rhod * R_d / p_0) ** (R_d / c_pd)) ** (c_pd / (c_pd - R_d))


@ti.func
def p(rhod: float, r: float, T: float) -> float:
    return rhod * (R_d + r * R_v) * T


@ti.func
def d_th_d_rv(T: float, th: float) -> float:
    return -th / T * l_v(T) / c_pd


@ti.func
def std2dry(th_std: float, r: float) -> float:
    return th_std * (1 + r * R_v / R_d) ** (R_d / c_pd)


@ti.func
def dry2std(th_dry: float, r: float) -> float:
    return th_dry / (1 + r * R_v / R_d) ** (R_d / c_pd)
