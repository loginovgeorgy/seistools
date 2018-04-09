import numpy as np
import sys
from pylab import *


def plot_seislet(s):

    if not isinstance(s, list):
        s = [s]
    for p in s:
        plot(p)
    show()

