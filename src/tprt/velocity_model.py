import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from .units import Units
from src.tprt import Layer, FlatHorizon

class Velocity_model(object):
    def __init__(self, velocity, density, name, horizons):
        self.horizons = horizons
        self.layers = self.make_layers(velocity, density, name)

    @staticmethod
    def make_flat_horizons(depths, anchors, dips, azimuths):        # Подразумеватся что все отсортировано по глубине от 0 до ...
        FlatHorizons = []
        for depth, anchor, dip, azimuth in zip(depths, anchors, dips, azimuths):
            FlatHorizons.append(FlatHorizon(depth, anchor, dip, azimuth))
        return FlatHorizons

    def make_layers(self, velocities, densities, names):            # Подразумеавется, что все подано в нужной последовательности
        Layers = []
        for i, (velocity, density, name) in enumerate(zip(velocities, densities, names)):

            if i==0: top = FlatHorizon(depth = - np.inf)
            else: top = self.horizons[i-1]

            if i<len(velocities)-1: bottom = self.horizons[i]
            else: bottom = FlatHorizon(depth = np.inf)
            Layers.append(Layer(velocity, density, top, bottom, name=name))
        return Layers

    def add_horizon(self, Horizon):
        self.horizons.append(Horizon)

    def add_layer(self, Layer):
        self.mid_layers.append(Layer)

    def __repr__(self):
        return self.layers
