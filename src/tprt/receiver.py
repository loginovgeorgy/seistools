import numpy as np
from .units import Units
from .utils import plot_points_3d

DEFAULT_DT = .25
DEFAULT_RECEIVER_NAME = 'geophone'
DEFAULT_RECEIVER_ORIENTATION = np.eye(3, 3)
DEFAULT_LOCATION = np.array([0, 0, 0])


class Receiver(object):
    def __init__(
            self,
            location=DEFAULT_LOCATION,
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
        self.location = np.array(location, dtype=float).ravel()
        self.orientation = orientation
        self.name = name
        self.dt = dt
        self.units = Units(**kwargs)
        self.layer = None

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

    def plot(self, **kwargs):
        plot_points_3d(self.location, **kwargs)
        # TODO prettify using plt.show()
