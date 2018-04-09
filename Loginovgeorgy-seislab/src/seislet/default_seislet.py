DEFAULT_DT = .00025
DEFAULT_NS = 64
DEFAULT_I = 100
DEFAULT_L = 64
DEFAULT_LET = 'berlage'


def default_a0(*args):
    return 1


def default_f(dt, *args):
    f_n = 1 / 2 / dt
    f = f_n / 8
    return f


def default_tau(dt, ns):
    tau = dt * ns / 4
    return tau


def default_alpha(dt, ns):
    return 4


def default_theta(dt, ns):
    return 4.5


def default_gamma(dt, ns):
    return 0.01


def default_k(dt, ns):
    return 100


def default_betta(dt, ns):
    return 6
