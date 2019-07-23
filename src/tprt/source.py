import numpy as np
from .units import Units
from .utils import plot_points_3d

from ..seislet.seislet_functions import DEFAULT_PARAMETERS  # Not sure if this is a good import

DEFAULT_SOURCE_NAME = 'event'
DEFAULT_WAVELET_NAME = "ricker"
DEFAULT_LOCATION = np.array([0, 0, 0])
DEFAULT_MOMENT = np.eye(3)
DEFAULT_MAGNITUDE = 1


class Source(object):
    def __init__(
            self,
            vel_model,
            name=DEFAULT_SOURCE_NAME,
            location=DEFAULT_LOCATION,
            moment=DEFAULT_MOMENT,
            wavelet_name=DEFAULT_WAVELET_NAME,
            wavelet_parameters=DEFAULT_PARAMETERS,
            magnitude=DEFAULT_MAGNITUDE
    ):
        self.name = name
        self.location = np.array(location, dtype=float).ravel()
        self.moment = moment
        self.wavelet_name = wavelet_name
        self.wavelet_parameters = wavelet_parameters
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

    def source_radiation(self, r, vtype):
        """Computes far-field displacement vector for desired wave type in the point with radius-vector r.

        :param r: radius-vector of a point in 3D space
        :param vtype: name of the desired wave's velocity: "vp" or "vs"
        :return: polarization vector of the far-field displacement (in assumption that the source's layer is
        infinite)
        """

        # This function uses formula:
        # A_i = M_kl * G_ik,l
        # where A_i is i-th component of the sought displacement vector, M_kl is the moment tensor of the source
        # and G_ik,l is derivative of the Green's tensor (found for homogeneous medium) with respect to the l-th
        # space component (x, y or z). Einstein's summation rule is applied.

        distance = np.linalg.norm(r - self.location)  # distance between source's location and the point r
        n = (r - self.location) / distance  # unit vector pointed from the source to the point r

        # G_ik,l consist of "tensor part" and some scalar factors:
        if vtype == "vp":

            g_ikl = np.einsum("i, k, l", n, n, n)

        else:

            g_ikl = - (np.einsum("i, k, l", n, n, n) - np.einsum("ik, l", np.eye(3), n))

        # Scalar factors:
        g_ikl = g_ikl / (4 * np.pi * self.layer.get_density() * self.layer.get_velocity(0)[vtype] ** 3)
        g_ikl = g_ikl / distance  # geometrical spreading

        # Finally, our displacement:
        displacement = np.einsum("kl, ikl", self.moment, g_ikl)

        return displacement


DEFAULT_FREQUENCY = 40


class DilatCenter(object):
    # A class for sources-centers of dilation.

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

    def plot(self, **kwargs):
        plot_points_3d(self.location, **kwargs)

    def source_radiation(self, r, vtype):
        """Computes far-field displacement vector for desired wave type in the point with radius-vector r.

        :param r: radius-vector of a point in 3D space
        :param vtype: name of the desired wave's velocity: "vp" or "vs"
        :return: polarization vector of the far-field displacement (in assumption that the source's layer is
        infinite)
        """

        # This function uses formula:
        # A_i = n_i / (4 * Pi * rho * vp^3 * R) if vtype == "vp"; 0 if vtype == "vs
        # where A_i is i-th component of the sought displacement vector, n_i is i-th component of the unit vector
        # connecting source's location with the point r and R is distance between source's location with the point r.
        # rho and vp are density and P-wave velocity in the source's layer respectively.

        distance = np.linalg.norm(r - self.location)  # distance between source's location and the point r
        n = (r - self.location) / distance  # unit vector pointed from the source to the point r

        if vtype == "vp":

            displacement = n / (4 * np.pi * self.layer.get_density() * self.layer.get_velocity(0)["vp"] ** 3)
            displacement = displacement / distance  # geometrical spreading

        else:

            displacement = np.zeros(3)

        return displacement


class RotatCenter(object):
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
        self.name = name
        self.location = np.array(location, dtype=float).ravel()
        self.dom_ferq = dom_ferq
        self.magnitude = magnitude
        self.layer = vel_model.get_location_layer(self.location)
        self.axis = axis / np.linalg.norm(axis)  # axis of rotation

    def plot(self, **kwargs):
        plot_points_3d(self.location, **kwargs)

    def source_radiation(self, r, vtype):
        """Computes far-field displacement vector for desired wave type in the point with radius-vector r.

        :param r: radius-vector of a point in 3D space
        :param vtype: name of the desired wave's velocity: "vp" or "vs"
        :return: polarization vector of the far-field displacement (in assumption that the source's layer is
        infinite)
        """

        # This function uses formula:
        # A = 0 if vtype == "vp"; cross(n, l) / (4 * Pi * rho * vs^3 * R) if vtype == "vs"
        # where A is sought displacement vector, n is unit vector connecting source's location with the point r and R
        # is distance between source's location with the point r. l is unit vector defining axis of source's rotation.
        # rho and vs are density and S-wave velocity in the source's layer respectively.

        distance = np.linalg.norm(r - self.location)  # distance between source's location and the point r
        n = (r - self.location) / distance  # unit vector pointed from the source to the point r

        if vtype == "vp":

            displacement = np.zeros(3)

        else:

            displacement = np.cross(n, self.axis)
            displacement = displacement / (4 * np.pi * self.layer.get_density() * self.layer.get_velocity(0)["vs"] ** 3)
            displacement = displacement / distance  # geometrical spreading

        return displacement


