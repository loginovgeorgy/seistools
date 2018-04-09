import numpy as np
from .default_seislet import *

SEISLET_DEFAULT_PARAMETERS = {
    'f': default_f,
    'tau': default_tau,
    'alpha': default_alpha,
    'a0': default_a0,
    'theta': default_theta,
    'gamma': default_gamma,
    'K': default_k,
    'betta': default_betta
}


def check_parameters(dt, ns, s_let_default_parameters=SEISLET_DEFAULT_PARAMETERS, **parameters):
    parameters_value = []
    for parameter_name in parameters:
        parameter = parameters[parameter_name]
        if parameter is None:
            set_default_parameter_func = s_let_default_parameters[parameter_name]
            parameter = set_default_parameter_func(dt, ns)

        parameters_value.append(parameter)

    return parameters_value


def berlage_function(dt=DEFAULT_DT, ns=DEFAULT_NS, a0=None, tau=None, f=None, alpha=None, **kwargs):
    parameters = check_parameters(dt, ns, a0=a0, tau=tau, f=f, alpha=alpha)

    a0, tau, f, alpha = parameters

    if alpha <= 0:
        alpha = 1

    alpha = f / dt / (alpha * 2)

    t = np.arange(0, ns * dt, dt)

    envelope = np.sin(2 * f * np.pi * (t - tau))
    spectrum = (t - tau) * np.exp(-alpha * (t - tau) ** 2)

    signal = spectrum * envelope
    signal[t < tau] = 0

    signal = a0 * signal / np.abs(signal).max()

    return signal


def puzirev_function(dt=DEFAULT_DT, ns=DEFAULT_NS, a0=None, tau=None, f=None, alpha=None, theta=None, **kwargs):
    parameters = check_parameters(dt, ns, a0=a0, tau=tau, f=f, alpha=alpha, theta=theta)

    a0, tau, f, alpha, theta = parameters

    if alpha <= 0:
        alpha = 1

    alpha = f / dt / (alpha * 2)

    t = np.arange(0, ns * dt, dt)

    envelope = np.sin(2 * np.pi * f * (t - tau) + theta)
    spectrum = np.exp(-(alpha ** 2) * (t - tau) ** 2)

    signal = a0 * spectrum * envelope

    return signal


def chirplet_function(dt=DEFAULT_DT,
                      ns=DEFAULT_NS,
                      a0=None,
                      tau=None,
                      f=None,
                      alpha=None, theta=None, gamma=None, K=None, betta=None, **kwargs):
    parameters = check_parameters(dt, ns,
                                  a0=a0,
                                  tau=tau,
                                  f=f,
                                  alpha=alpha,
                                  theta=theta,
                                  betta=betta,
                                  K=K,
                                  gamma=gamma)

    a0, tau, f, alpha, theta, gamma, K, betta = parameters

    if alpha <= 0:
        alpha = 1

    alpha = f / dt / (alpha * 2)

    t = np.arange(0, ns * dt, dt)

    envelope = np.exp(-alpha * (1 - betta * np.tanh(K * (t - tau))) * ((t - tau) ** 2))
    spectrum = np.cos(f * (t - tau) + gamma * (t - tau) ** 2 + theta)

    signal = a0 * spectrum * envelope

    return signal


def ricker_function(dt=DEFAULT_DT, ns=DEFAULT_NS, a0=None, tau=None, f=None, **kwargs):
    parameters = check_parameters(dt, ns, a0=a0, tau=tau, f=f)

    a0, tau, f = parameters

    t = np.arange(0, ns * dt, dt)
    # signal = a0*(1-2*pi*pi*f*f*(t-tau).^2).*exp(-pi*pi*f*f*(t-tau).^2)

    envelope = (1 - 2 * np.pi * np.pi * f * f * (t - tau) ** 2)
    spectrum = np.exp(-np.pi * np.pi * f * f * (t - tau) ** 2)

    signal = a0 * spectrum * envelope
    signal[t < tau] = 0

    return signal
