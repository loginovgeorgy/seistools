from .seislet_functions import *

LET_FUNCTIONS = {'berlage': berlage_function,
                 'puzirev': puzirev_function,
                 'chirplet': chirplet_function,
                 'ricker': ricker_function}


def gen_seislet(seislet,
                dt=DEFAULT_DT,
                ns=DEFAULT_NS,
                s_let_funcs=LET_FUNCTIONS,
                **kwargs):
    signal_function = s_let_funcs[seislet]
    signal = signal_function(dt, ns, **kwargs)
    return signal 