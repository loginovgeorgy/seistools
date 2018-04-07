import numpy as np
DEFAULT_DT = .25
DEFAULT_RECEIVER_NAME = 'name'
DEFAULT_SOURCE_NAME = 'event'
DEFAULT_RECEIVER_ORIENTATION = np.ones(3, 3)


class Units(object):
    def __init__(self, distance='m', time='s', amplitude='m/s2', **kwargs):
        """
        class for determining the units of measurements


        :param distance: 'm', 'km', 'ft'
        :param time: 's', 'ms',
        :param amplitude: speed, acceleration, count
        :param kwargs:
        """
        self.distance = distance
        self.time = time
        self.amplitude = amplitude

    def _get_description(self):

        return 'Units: \n distance "{}" \n time "{}" \n amplitude "{}"'.format(
            self.distance,
            self.time,
            self.amplitude
        )

    def __str__(self):
        return self._get_description()

    def __repr__(self):
        return self._get_description()


class Receiver(object):
    def __init__(
            self,
            location,
            orientation=DEFAULT_RECEIVER_ORIENTATION,
            name=DEFAULT_RECEIVER_NAME,
            dt=DEFAULT_DT,
            **kwargs
    ):
        """
        class for seismic receivers description

        :param location:
        :param orientation:
        :param name:
        :param dt:
        :param kwargs:
        """
        self.location = np.array(location).ravel()
        self.orientation = orientation
        self.name = name
        self.dt = dt
        self.units = Units(**kwargs)

    def _get_description(self):
        return 'Receiver "{}", loc ({}) \n{}'.format(
            self.name,
            self.location,
            self.units
        )

    def __str__(self):
        return self._get_description()

    def __repr__(self):
        return self._get_description()


class Source(object):
    def __init__(self, location, m0=1, name=DEFAULT_SOURCE_NAME, moment=None, **kwargs):
        """
        class for seismic source description

        :param location:
        :param m0:
        :param name:
        :param moment:
        :param kwargs:
        """
        self.location = np.array(location).ravel()
        self.m0 = m0
        self.name = name
        self.moment = moment
        self.units = Units(**kwargs)

    def _get_description(self):
        return 'Source "{}", loc ({})'.format(
            self.name,
            self.location,
        )

    def __str__(self):
        return self._get_description()

    def __repr__(self):
        return self._get_description()


class Velmod(object):
    def __init__(self, **kwargs):
        """
        class do define the velocity model as the list of horizons and layers

        :param kwargs:
        """
        self.horizons = []
        self.layers = []

    def __add__(self, other):

        return


def _plane():

def _flat_horizon(n=(0,0,1), depth=0, anchor=(0,0), dip=0, azimuth=None, **kwargs):
    def plane(x):
        return (depth - np.sum(n*x)) / (n[2] + 1e-16)
    # yield ax + by + cz + d
    # yield (d - ax - by) / (n[-1] + 1e-16)
    return plane

def _grid_horizon():
    return 1
HORIZON_TYPES = {
    'flat': _flat_horizon,
    'f': _flat_horizon,
    'fh': _flat_horizon,
    'horizontal': _flat_horizon,
    'grid': _grid_horizon,

}
class Horizon(object):
    def __init__(self, type, *args, **kwargs):
        self._model = HORIZON_TYPES[type](**kwargs)
        self._properties = kwargs

        self.type = type
        self.units = Units(**kwargs)

    def fit(self, x, y):k
        self._model = 1

    def predict(self, x):
        return self._model(x)


class Layer(object):
    def __init__(self, **kwargs):
        self.vp = None
        self.vs = None

        self.units = Units(**kwargs)

