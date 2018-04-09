import numpy as np
from .units import Units

DEFAULT_DT = .25
DEFAULT_RECEIVER_NAME = 'name'
DEFAULT_SOURCE_NAME = 'event'
DEFAULT_RECEIVER_ORIENTATION = np.ones(3, 3)


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


class Layer(object):
    def __init__(self, **kwargs):
        self.vp = None
        self.vs = None

        self.units = Units(**kwargs)

