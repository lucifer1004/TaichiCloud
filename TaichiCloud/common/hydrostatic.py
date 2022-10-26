import taichi as ti
import common.constants as constants
import common.theta as theta


@ti.func
def p(z: ti.template(), th_0: ti.template(), r_0: ti.template(), z_0: ti.template(), p_0: ti.template()):
    return constants.p_0 * ((p_0 / constants.p_0) ** constants.R_d_over_c_pd - constants.R_d_over_c_pd * constants.g / th_0 / theta.R(r_0) * (z - z_0)) ** (1 / constants.R_d_over_c_pd)
