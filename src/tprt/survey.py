import numpy as np
from tqdm import tqdm_notebook
from .ray import Ray


class Survey:
    def __init__(self, sources, receivers, vel_mod):
        """
        :param sources:     array of objects of type of Source
        :param receivers:   array of objects of type of Receiver
        :param vel_mod:     object of type of VelocityModel
        """
        self.sources = np.array(sources, ndmin=1)
        self.receivers = np.array(receivers, ndmin=1)
        self.velmod = vel_mod

        self.rays = None
        self.traveltimes = None
        self.amplitudes = None

    def initialize_rays(self, reflect_horizon=2, vtype='vp', forward=False):
        """
        :param reflect_horizon: number of reflect horizon with respect to source layer
        :param forward: Is it forward wave? (boolean)
        :param vtype: type of wave ('vs', 'vp')
        :return: update property "rays"
        """
        self.rays = np.empty(shape=self.sources.shape+self.receivers.shape, dtype=Ray)
        for i, sou in np.ndenumerate(self.sources):
            for j, rec in np.ndenumerate(self.receivers):
                raycode_ij = Ray.get_raycode(sou, rec, reflect_horizon, self.velmod, vtype, forward=forward)
                self.rays[i+j] = Ray(sou, rec, self.velmod, raycode=raycode_ij, vtype=vtype)
                
    def calculate(self, method='BFGS', survey2D=False):
        self.traveltimes = np.empty(shape=self.rays.shape, dtype=float)
        self.amplitudes = np.empty(shape=self.rays.shape+(3,), dtype=complex)
        with tqdm_notebook(total=np.size(self.rays), desc='Rays calculating:') as p_bar:
            for i, ray in np.ndenumerate(self.rays):
                ray.optimize(method=method, survey2D=survey2D)
                ray.ray_amplitude = ray.compute_ray_amplitude(survey2D=survey2D)
                self.amplitudes[i] = ray.ray_amplitude
                self.traveltimes[i] = ray.traveltime
                p_bar.update()

    def plot(self, **kwargs):
        if not np.any(kwargs.get('ax')):
            fig = plt.figure()
            ax = Axes3D(fig)
        else:
            ax = kwargs['ax']
            kwargs.pop('ax')
        
        for i, rec in np.ndenumerate(self.receivers):
            rec.plot(ax=ax)
        for i, ray in np.ndenumerate(self.rays):
            ray.plot(ax=ax)
        for i, sou in np.ndenumerate(self.sources):
            sou.plot(ax=ax, marker='^', color='k')

    def get_traveltimes(self):
        pass
