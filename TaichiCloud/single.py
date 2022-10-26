import taichi as ti
from typing import Tuple
from common.formulae import *
from common.constants import *
import common.mixing as mixing
import common.theta as theta

ti.init(arch=ti.gpu, debug=True)

# Constants
# ========================
r_c0 = 5e-4  # Auto-conversion threshold
r_eps = 2e-5  # Absolute tolerance
# Kessler auto-conversion (eq. 5a in Grabowski & Smolarkiewicz 1996)
k_acnv = 1e-3
# Number of iterations in Newton-Raphson saturation adjustment
nwtrph_iters = 3

# Options
# ========================
enable_condensation = True
enable_cloud_evaporation = True
enable_rain_evaporation = True
enable_auto_conversion = True
enable_accretion = True
enable_sedimentation = True

# Resolution
# ========================
nx = 1  # Number of grid points in x direction
nz = 1  # Number of grid points in z direction
spinup = 2000  # Number of time steps to spin up


@ti.func
def copysign(x: ti.template(), y: ti.template()):
    sign = 1
    if y < 0:
        sign = -1
    return sign * ti.abs(x)


@ti.data_oriented
class SingleMoment:
    """
    th, thp: potential temperature and its derivative
    rv, rvp: water vapor mixing ratio and its derivative
    rc, rcp: cloud water mixing ratio and its derivative
    rr, rrp: rain water mixing ratio and its derivative
    """

    def __init__(self, size):
        self.rhod = ti.field(dtype=ti.f32)
        self.p = ti.field(dtype=ti.f32)
        self.th = ti.field(dtype=ti.f32)
        self.rv = ti.field(dtype=ti.f32)
        self.rc = ti.field(dtype=ti.f32)
        self.rr = ti.field(dtype=ti.f32)
        self.thp = ti.field(dtype=ti.f32)
        self.rvp = ti.field(dtype=ti.f32)
        self.rcp = ti.field(dtype=ti.f32)
        self.rrp = ti.field(dtype=ti.f32)
        self.f_rhod = ti.field(dtype=ti.f32)
        self.f_p = ti.field(dtype=ti.f32)
        self.f_r = ti.field(dtype=ti.f32)
        self.f_rs = ti.field(dtype=ti.f32)
        self.f_T = ti.field(dtype=ti.f32)
        ti.root.dense(ti.j, size[1]).dense(ti.i, size[0]).place(
            self.rhod, self.p, self.th, self.rv, self.rc, self.rr, self.thp, self.rvp,
            self.rcp, self.rrp, self.f_rhod, self.f_p, self.f_r, self.f_rs, self.f_T,
        )

        self.init()

    @ti.kernel
    def init(self):
        for i in ti.grouped(self.th):
            self.rhod[i] = 1.0
            self.th[i] = 300.0
            self.p[i] = p_0
            self.rc[i] = 0.01

    @ti.kernel
    def adj_cellwise_nwtrph(self, dt: ti.template()):
        for i in ti.grouped(self.p):
            assert(self.rv[i] >= 0.0)
            assert(self.rc[i] >= 0.0)
            assert(self.th[i] >= 273.15)

            p = self.p[i]
            drc = 0.0
            rv_tmp = self.rv[i]
            th_tmp = self.th[i]
            exner_p = theta.exner(self.p[i])
            T = th_tmp * exner_p
            L0 = mixing.l_v(T)
            for iter in range(nwtrph_iters):
                p_vs = mixing.p_vs(T)
                L = mixing.l_v(T)
                coeff = L * L0 / (c_pd * R_v) / T ** 2 / (1 - p_vs / self.p[i])
                r_vs = mixing.r_vs(T, p)
                drc += (rv_tmp - r_vs) / (1 + coeff * r_vs)
                rv_tmp = self.rv[i] - drc
                th_tmp = self.th[i] + L0 / (c_pd * exner_p) * drc
                T = th_tmp * exner_p

            drc = min(self.rv[i], max(-self.rc[i], drc))
            self.rv[i] -= drc
            self.rc[i] += drc
            self.th[i] += L0 / (c_pd * exner_p) * drc

            assert(self.rv[i] >= 0.0)
            assert(self.rc[i] >= 0.0)
            assert(self.th[i] >= 273.15)

    @ti.func
    def init_F(self, i, const_p):
        self.f_rhod[i] = self.rhod[i]
        self.f_p[i] = self.p[i]
        self.update_F(i, self.th[i], self.rv[i], const_p)

    @ti.func
    def update_F(self, i, th, rv, const_p):
        self.f_r[i] = rv
        if const_p == 0:
            self.f_T[i] = theta.T(th, self.f_rhod[i])
            self.f_p[i] = theta.p(self.f_rhod[i], rv, self.f_T[i])
        else:
            self.f_T[i] = th * theta.exner(self.f_p[i])
        self.f_rs[i] = mixing.r_vs(self.f_T[i], self.f_p[i])

    @ti.func
    def eval_F(self, i, th, rv, const_p):
        self.update_F(i, th, rv, const_p)
        return theta.d_th_d_rv(self.f_T[i], th)

    @ti.func
    def runge_kutta_4(self, i, th, rv, drv, const_p):
        """Runge-Kutta 4th order integration for a single cell"""
        k_1 = self.eval_F(i, th, rv, const_p)
        y_1 = th + k_1 * 0.5 * drv
        k_2 = self.eval_F(i, y_1, rv + 0.5 * drv, const_p)
        y_2 = th + k_2 * 0.5 * drv
        k_3 = self.eval_F(i, y_2, rv + 0.5 * drv, const_p)
        y_3 = th + k_3 * drv
        k_4 = self.eval_F(i, y_3, rv + drv, const_p)
        return th + (drv / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)

    @ti.kernel
    def adj_cellwise_hlpr(self, dt: ti.template(), const_p: ti.template()):
        if enable_condensation:
            for i in ti.grouped(self.p):
                rhod = self.rhod[i]
                p = self.p[i]

                assert(self.rv[i] >= 0.0)
                assert(self.rc[i] >= 0.0)
                assert(self.rr[i] >= 0.0)
                assert(self.th[i] >= 273.15)

                self.init_F(i, const_p)
                vapor_excess = 0.0
                drr_max = 0.0
                incloud = False

                if self.f_rs[i] > self.f_r[i] and self.rr[i] > 0 and enable_rain_evaporation:
                    drr_max = dt * \
                        evaporation_rate(
                            self.f_r[i], self.f_rs[i], self.rr[i], rhod, p)

                while True:
                    vapor_excess = self.rv[i] - self.f_rs[i]

                    # FIXME: need to call the following statement
                    # otherwise f_r will be reset to 0.0
                    # I do not know why!
                    # Only Taichi v1.2.0 has this issue. v1.1.3 is OK.
                    assert(self.f_r[i] == self.rv[i])

                    if vapor_excess <= r_eps:
                        if not enable_cloud_evaporation or vapor_excess >= -r_eps:
                            break

                        incloud = self.rc[i] > 0
                        if not incloud and (not enable_rain_evaporation or self.rr[i] <= 0 or drr_max <= 0):
                            break

                    drv = -copysign(min(0.5 * r_eps, 0.5 *
                                    vapor_excess), vapor_excess)
                    if vapor_excess < 0:
                        if incloud:
                            drv = min(self.rc[i], drv)
                        else:
                            drv = min(drr_max, self.rr[i], drv)

                    assert(drv != 0)

                    # Do step
                    self.th[i] = self.runge_kutta_4(
                        i, self.th[i], self.rv[i], drv, const_p)
                    self.rv[i] += drv

                    assert(self.rv[i] >= 0)

                    if vapor_excess > 0 or incloud:
                        self.rc[i] -= drv
                        assert(self.rc[i] >= 0)
                    else:
                        assert(enable_rain_evaporation)
                        self.rr[i] -= drv
                        assert(self.rr[i] >= 0)
                        drr_max -= drv
                        if drr_max == 0:
                            break

                assert(self.f_r[i] == self.rv[i])
                assert(self.rc[i] >= 0)
                assert(self.rv[i] >= 0)
                assert(self.rr[i] >= 0)
                assert(self.th[i] >= 273.15)

    def adj_cellwise(self, dt):
        self.adj_cellwise_hlpr(dt, 0)

    def adj_cellwise_const_p(self, dt):
        self.adj_cellwise_hlpr(dt, 1)

    @ti.kernel
    def rhs_cellwise(self):
        """Handle coalescence"""
        for i in ti.grouped(self.p):
            tmp = 0.0

            if enable_auto_conversion:
                tmp += auto_conversion_rate(self.rc[i], r_c0, k_acnv)

            if enable_accretion:
                tmp += collection_rate(self.rc[i], self.rr[i])

            self.rrp[i] += tmp
            self.rcp[i] -= tmp

    @ti.kernel
    def rhs_cellwise_nwtrph(self, dt: ti.template()):
        for i in ti.grouped(self.p):
            tmp = 0.0

            if enable_auto_conversion:
                tmp += auto_conversion_rate(self.rc[i], r_c0, k_acnv)

            if enable_accretion:
                tmp += collection_rate(self.rc[i], self.rr[i])

            self.rrp[i] += tmp
            self.rcp[i] -= tmp

            T = self.th[i] * theta.exner(self.p[i])
            r_vs = mixing.r_vs(T, self.p[i])
            tmp2 = evaporation_rate(
                self.rv[i], r_vs, self.rr[i], self.rhod[i], self.p[i]) * dt

            tmp2 = min(self.rr[i], tmp2) / dt
            self.rvp[i] += tmp2
            self.rrp[i] -= tmp2
            self.thp[i] -= mixing.l_v(T) / \
                (c_pd * theta.exner(self.p[i])) * tmp2

    @ti.kernel
    def rhs_columnwise(self, dz: ti.template()) -> float:
        """Handle sedimentation"""
        sedimentation = 0.0
        if enable_sedimentation:
            for col in range(nx):
                flux_in = 0.0  # kg / m^3 / s
                rhod_0 = self.rhod[0, col]
                rhod = 0.0
                rr = 0.0
                for irow in range(nz):
                    row = nz - 1 - irow
                    rhod_below = self.rhod[row, col]
                    rr_below = self.rr[row, col]
                    if row != nz - 1:  # all but top grid cell
                        flux_out = 0.5 * (rhod_below * v_term(rr_below, rhod_below,
                                          rhod_0) + rhod * v_term(rr, rhod, rhod_0)) * rr / dz
                        self.rrp[row + 1, col] -= (flux_in - flux_out) / rhod
                        flux_in = flux_out
                    rhod = rhod_below
                    rr = rr_below
                flux_out = -rhod * v_term(rr, rhod, rhod_0) * rr / dz
                self.rrp[0, col] -= (flux_in - flux_out) / rhod
                sedimentation += flux_out
        return sedimentation


def main():
    cld = SingleMoment((nz, nx))
    for t in range(5):
        print('t = ', t)
        cld.adj_cellwise(1)
        cld.rhs_cellwise()
        print('flux = ', cld.rhs_columnwise(1))


if __name__ == '__main__':
    main()
