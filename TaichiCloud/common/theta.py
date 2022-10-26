import taichi as ti
from common.constants import *
from common.mixing import *


@ti.func
def rhod(p: ti.template(), th_std: ti.template(), rv: ti.template()):
    """Dry air density as a function of p, theta and rv"""
    return (p - p_v(p, rv)) / ((p / p_0) ** (R_d / c_pd) * R_d * th_std)


@ti.func
def exner(p: ti.template()):
    """Exner pressure"""
    return (p / p_0) ** (R_d / c_pd)


@ti.func
def T(th: ti.template(), rhod: ti.template()):
    return (th * (rhod * R_d / p_0) ** (R_d / c_pd)) ** (c_pd / (c_pd - R_d))


@ti.func
def p(rhod: ti.template(), r: ti.template(), T: ti.template()):
    return rhod * (R_d + r * R_v) * T


@ti.func
def d_th_d_rv(T: ti.template(), th: ti.template()):
    return -th / T * l_v(T) / c_pd


@ti.func
def std2dry(th_std: ti.template(), r: ti.template()):
    return th_std * (1 + r * R_v / R_d) ** (R_d / c_pd)


@ti.func
def dry2std(th_dry: ti.template(), r: ti.template()):
    return th_dry / (1 + r * R_v / R_d) ** (R_d / c_pd)
