import numpy as np
from .units import Units
from .utils import plot_points_3d

DEFAULT_SOURCE_NAME = 'event'
DEFAULT_MAGNITUDE = 1
DEFAULT_LOCATION = np.array([0, 0, 0])


class Source(object):
    def __init__(
            self,
            frequency,
            vel_model,
            location=DEFAULT_LOCATION,
            name=DEFAULT_SOURCE_NAME,
    ):
        self.name = name
        self.location = np.array(location).ravel()
        self.layer = vel_model.get_location_layer(self.location)
        self.frequency = frequency

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

    def psi0(self, r0, vec):
        # returns a coefficient which defines the amplitude at point with radius-vector r0 in vicinity of the source.
        # Argument vec specifies the desired direction of polarization.
        pass


class DilatCenter(Source):
    # A class for sources-centers of dilatation.

    def psi0(self, r0, vec):

        # This type of source irradiates only P-wave.

        r = r0 - self.location

        return np.dot(r, vec) / np.linalg.norm(r) / np.linalg.norm(vec) * \
               self.frequency / ( 2 * self.layer.get_velocity(1)["vp"] ** 3 * self.layer.get_density() )


class RotatCenter(Source):
    # A class for sources-centers of rotation.

    def __init__(
            self,
            frequency,
            vel_model,
            axis,
            location=DEFAULT_LOCATION,
            name=DEFAULT_SOURCE_NAME,
    ):
        self.name = name
        self.location = np.array(location).ravel()
        self.layer = vel_model.get_location_layer(self.location)
        self.frequency = frequency
        self.axis = axis / np.linalg.norm(axis) # axis of rotation


    def psi0(self, r0, vec):

        # This type of source irradiates only S-wave.

        r = r0 - self.location

        cross_prod = np.cross(r / np.linalg.norm(r), self.axis)

        return np.dot(cross_prod, vec) / np.linalg.norm(vec) * \
               self.frequency / ( 2 * self.layer.get_velocity(1)["vs"] ** 3 * self.layer.get_density())

