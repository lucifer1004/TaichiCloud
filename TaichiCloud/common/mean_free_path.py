import taichi as ti
from common.constants import D_0, K_0, R_v, R_d


@ti.func
def lambda_D(T: ti.template()):
    return 2 * D_0 / ti.sqrt(2 * R_v * T)


@ti.func
def lambda_K(T: ti.template(), p: ti.template()):
    return 0.8 * K_0 * T / p / ti.sqrt(2 * R_d * T)
