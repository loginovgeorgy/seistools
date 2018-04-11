import numpy as np
from .units import Units
from .utils import plot_points_3d

DEFAULT_SOURCE_NAME = 'event'
DEFAULT_MOMENT_TENSOR = np.eye(3, 3)
DEFAULT_MAGNITUDE = 1
DEFAULT_LOCATION = np.array([0, 0, 0])


class Source(object):
    def __init__(
            self,
            location=DEFAULT_LOCATION,
            name=DEFAULT_SOURCE_NAME,
            mechanism=None,
            **kwargs
    ):
        """
        class for seismic source description

        :param location:
        :param m0:
        :param name:
        :param moment:
        :param kwargs:
        """
        self.location = np.array(location).ravel()
        self.name = name
        self.units = Units(**kwargs)
        self.layer = None

        # TODO class for seismic source mechanism
        self.mechanism = mechanism

    def _get_description(self):
        return 'Source "{}", loc ({})'.format(
            self.name,
            self.location,
        )

    def __str__(self):
        return self._get_description()

    def __repr__(self):
        return self._get_description()

    def plot(self, **kwargs):
        plot_points_3d(self.location, **kwargs)




