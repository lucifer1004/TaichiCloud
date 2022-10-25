import taichi as ti

# Constants
k_2 = 2.2
vterm_A = 36.34
vterm_B = 1e-3


@ti.func
def auto_conversion_rate(rc: float, rc_thresh: float, k_autoconv: float) -> float:
    return k_autoconv * max(0, rc - rc_thresh)


@ti.func
def collection_rate(rc: float, rr: float) -> float:
    return k_2 * rc * rr ** 0.875


@ti.func
def evaporation_rate(rv: float, rvs: float, rr: float, rhod: float, p: float) -> float:
    return (1 - rv / rvs) / rhod \
        * (1.6 + 124.9 * (1e-3 * rhod * rr) ** 0.2046) \
        * (1e-3 * rhod * rr) ** 0.525 \
        / (5.4e2 + 2.55e5 * (1 / p / rvs))


@ti.func
def v_term(rr: float, rhod: float, rhod_0: float) -> float:
    return vterm_A * (rhod * rr * vterm_B) ** 0.1346 \
        * ti.sqrt(rhod_0 / rhod)
