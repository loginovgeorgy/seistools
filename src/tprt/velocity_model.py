import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from .units import Units
from src.tprt import Layer, FlatHorizon

class Velocity_model(object):
    def __init__(self, velocity, density, name, horizons):
        self.horizons = horizons
        self.top_layer = None
        self.mid_layers = []
        self.bottom_layer = None
        self.split_layers(self.make_layers(velocity, density, name))

    @staticmethod
    def make_flat_horizons(depths, anchors, dips, azimuths):        # Подразумеватся что все отсортировано по глубине от 0 до ...
        FlatHorizons = []
        for depth, anchor, dip, azimuth in zip(depths, anchors, dips, azimuths):
            FlatHorizons.append(FlatHorizon(depth, anchor, dip, azimuth))
        return FlatHorizons

    def make_layers(self, velocities, densities, names):            # Подразумеавется, что все подано в нужной последовательности
        Layers = []
        for i, (velocity, density, name) in enumerate(zip(velocities, densities, names)):
            if i!=0: top = self.horizons[i-1]
            else: top = None
            if i<len(self.horizons): bottom = self.horizons[i]
            else: bottom = None
            Layers.append(Layer(velocity, density, top, bottom, name=name))
        return Layers

    def split_layers(self, all_layers):
        self.top_layer = all_layers[0]
        self.mid_layers = all_layers[1:-1]
        if len(all_layers) == 1:
            self.bottom_layer = None
        else:
            self.bottom_layer = all_layers[-1]


    def add_horizon(self, Horizon):
        self.horizons.append(Horizon)

    def add_layer(self, Layer):
        self.mid_layers.append(Layer)

    def __repr__(self):
        return self.layers
