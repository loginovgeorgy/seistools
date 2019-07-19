import numpy as np
from .units import Units
from .utils import plot_points_3d

DEFAULT_SOURCE_NAME = 'event'
DEFAULT_MAGNITUDE = 1
DEFAULT_LOCATION = np.array([0, 0, 0])
DEFAULT_FREQUENCY = 40


class Source(object):
    def __init__(
            self,
            vel_model,
            name=DEFAULT_SOURCE_NAME,
            location=DEFAULT_LOCATION,
            dom_ferq=DEFAULT_FREQUENCY,  # dominant frequency in the Ricker wavelet (Hz)
            magnitude=DEFAULT_MAGNITUDE
    ):
        self.name = name
        self.location = np.array(location, dtype=float).ravel()
        self.dom_ferq = dom_ferq
        self.magnitude = magnitude
        self.layer = vel_model.get_location_layer(self.location)

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

    def scalar_radiation(self, r, polarization):
        """

        :param r: radius-vector of a point in 3D
        :param polarization: desired polarization
        :return: scalar amplitude in given point for desired polarization (in assumption that the source's layer is
        infinite)
        """
        pass


class DilatCenter(Source):
    # A class for sources-centers of dilation.

    def scalar_radiation(self, r, polarization):

        # This type of source irradiates only P-wave.

        direction = r - self.location

        return np.dot(direction, polarization) / (np.linalg.norm(direction) * np.linalg.norm(polarization))


class RotatCenter(Source):
    # A class for sources-centers of rotation.

    def __init__(
            self,
            axis,
            vel_model,
            name=DEFAULT_SOURCE_NAME,
            location=DEFAULT_LOCATION,
            dom_ferq=DEFAULT_FREQUENCY,  # dominant frequency in the Ricker wavelet (Hz)
            magnitude=DEFAULT_MAGNITUDE
    ):

        self.axis = axis / np.linalg.norm(axis)  # axis of rotation

        Source.__init__(self,
                        vel_model,
                        name,
                        location,
                        dom_ferq,
                        magnitude)

    def scalar_radiation(self, r, polarization):

        # This type of source irradiates only S-wave.

        direction = r - self.location
        cross_prod = np.cross(direction, self.axis)

        return np.dot(cross_prod, polarization) / (np.linalg.norm(cross_prod) * np.linalg.norm(polarization))

