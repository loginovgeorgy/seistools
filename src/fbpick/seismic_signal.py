import numpy as np

from .helpers import cast_input_to_array
DEFAULT_DT = 1e-3
DEFAULT_NS = 256


def _default_f(dt, ns):
    """
    Central frequency
    :param dt:
    :param ns:
    :return:
    """

    return 1 / 2 / dt / 10


def _default_tau(dt, ns):
    """
    timeshift
    :param dt:
    :param ns:
    :return:
    """

    return dt * ns / 2


def _default_alpha(dt, ns):
    """
    bandwidth factor (decay) (0, 1)
    :param dt:
    :param ns:
    :return: alpha > 0
    """
    return 1. / 3


def _default_theta(dt, ns):
    """
    theta in (0, 2pi)
    :param dt:
    :param ns:
    :return:
    """

    return 0


def _default_gamma(dt, ns):
    """
    Chirp rate
    :param dt:
    :param ns:
    :return:
    """
    return 0.01


def _default_k(dt, ns):
    return 100


def _default_betta(dt, ns):
    """
    asymmetry factor
    :param dt:
    :param ns:
    :return:
    """
    return 1


def _ricker(
        t,
        dt=None,
        tau=None,
        f=None,
        **kwargs
):
    t = t - tau
    alpha = np.pi * np.pi * f * f
    spectrum = (1 - 2 * alpha * (t ** 2))
    envelope = np.exp(alpha * (- t ** 2))

    return spectrum * envelope


def _berlage(
        t,
        dt,
        tau=None,
        f=None,
        alpha=None,
        theta=None,
        **kwargs
):
    t = t - tau
    alpha *= ((np.pi * f) ** 2)
    spectrum = np.sin(2 * np.pi * f * t + theta)
    envelope = t * np.exp(alpha * (- t ** 2))

    return envelope * spectrum * (t > 0)


def _chirplet(
        t,
        dt,
        tau=None,
        f=None,
        alpha=None,
        theta=None,
        gamma=None,
        betta=None,
        k=None,
        **kwargs
):
    t = t - tau

    alpha *= ((np.pi * f) ** 2)
    gamma *= (np.pi * f) ** 2
    k = 1 / dt / k
    betta = np.abs(betta)

    spectrum = np.cos(2 * np.pi * f * t + gamma * (t ** 2) + theta)
    envelope = np.exp(alpha * (1 - betta * np.tanh(k * t)) * (- t ** 2))

    return spectrum * envelope


SIGNAL_TYPE = {
    'ricker': _ricker,
    'berlage': _berlage,
    'chirplet': _chirplet,
}

DEFAULT_PARAMETERS = {
    'tau': _default_tau,
    'f': _default_f,
    'alpha': _default_alpha,
    'theta': _default_theta,
    'gamma': _default_gamma,
    'betta': _default_betta,
    'k': _default_k,
}


def _cast_n_check_signal_parameters(dt, ns, kwargs):
    dt = np.float32(dt)
    ns = np.int32(ns)
    for key in kwargs:
        tmp = np.array(kwargs[key], ndmin=2, dtype=np.float32)
        is_nan = np.isnan(tmp)
        tmp[is_nan] = DEFAULT_PARAMETERS[key](dt, ns)
        kwargs[key] = tmp
    return dt, ns, kwargs


def seismic_signal(
        signal='ricker',
        t=None,
        dt=DEFAULT_DT,
        ns=DEFAULT_NS,
        tau=None,
        f=None,
        alpha=None,
        theta=None,
        gamma=None,
        betta=None,
        k=None,
        verbose=False,
):
    """

    :param signal:
    :param t:
    :param dt:
    :param ns:
    :param tau:
    :param f:
    :param alpha:
    :param theta:
    :param gamma:
    :param betta:
    :param k:
    :param verbose:
    :return:
    """

    kwargs = dict(
        tau=tau,
        f=f,
        alpha=alpha,
        theta=theta,
        gamma=gamma,
        betta=betta,
        k=k,
    )

    dt, ns, kwargs = _cast_n_check_signal_parameters(dt, ns, kwargs)

    if not isinstance(t, np.ndarray):
        t = cast_input_to_array(np.arange(0, ns * dt, dt))

    if verbose:
        for k in kwargs:
            print('>>> ', k, kwargs[k])
    return SIGNAL_TYPE[signal](t, dt, **kwargs)
