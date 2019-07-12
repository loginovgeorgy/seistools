import numpy as np
from .ray import Ray

class Survey:
    def __init__(self, sources, receivers, vel_mod):
        """
        :param sources:     list of objects of type of Source
        :param receivers:   list of objects of type of Receiver
        :param vel_mod:     object of type of Velocity_model
        """
        self.sources = sources
        self.receivers = receivers
        self.velmod = vel_mod

        self.rays = []
        self.traveltimes = None

    def initialize_rays(self, reflect_horizon=2, vtype='vp', forward=False):
        """

        :param reflect_horizon: number of reflect horizon with respect to source layer
        :param forward: Is it forward wave? (boolean)
        :param vtype: type of wave ('vs', 'vp')
        :return: update property "rays"
        """

        for sou in self.sources:
            for rec in self.receivers:
                raycode_ij = None
                if not forward:
                    raycode_ij = Ray.get_raycode(sou, rec, reflect_horizon, self.velmod, vtype)
                self.rays.append(Ray(sou, rec, self.velmod, raycode=raycode_ij, vtype=vtype))

    def calculate(self, method='BFGS', survey2D=False):
        for ray_i in self.rays:
            ray_i.optimize(method=method, survey2D=survey2D)

    def plot(self, ax):
        for rec in self.receivers:
            rec.plot(ax=ax)
        for ray in self.rays:
            ray.plot(ax=ax)
        for sou in self.sources:
            sou.plot(ax=ax, marker='^', color='k')

    def get_traveltimes(self):
        n, m = len(self.sources), len(self.receivers)
        shape = (n, m)
        T = np.empty(shape=shape)
        for i, ray in enumerate(self.rays):
            T[i // m, i % m] = ray.traveltime

        self.traveltimes = T
        return T
