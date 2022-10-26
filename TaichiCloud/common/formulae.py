import taichi as ti

# Constants
k_2 = 2.2
vterm_A = 36.34
vterm_B = 1e-3


@ti.func
def auto_conversion_rate(rc: ti.template(), rc_thresh: ti.template(), k_autoconv: ti.template()):
    return k_autoconv * max(0, rc - rc_thresh)


@ti.func
def collection_rate(rc: ti.template(), rr: ti.template()):
    return k_2 * rc * rr ** 0.875


@ti.func
def evaporation_rate(rv: ti.template(), rvs: ti.template(), rr: ti.template(), rhod: ti.template(), p: ti.template()):
    return (1 - rv / rvs) / rhod \
        * (1.6 + 124.9 * (1e-3 * rhod * rr) ** 0.2046) \
        * (1e-3 * rhod * rr) ** 0.525 \
        / (5.4e2 + 2.55e5 * (1 / p / rvs))


@ti.func
def v_term(rr: ti.template(), rhod: ti.template(), rhod_0: ti.template()):
    return vterm_A * (rhod * rr * vterm_B) ** 0.1346 \
        * ti.sqrt(rhod_0 / rhod)
