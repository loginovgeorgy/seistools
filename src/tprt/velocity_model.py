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

    def get_location_layer(self, location):

        # returns a Layer object where the location = np.array([x, y, z]) is located

        layer_num = 0 # initially we think that the location layer is the very first one

        for i in range(len(self.layers)):

            if self.layers[i].bottom.get_depth(location[0:2]) < location[2]:

                layer_num = layer_num + 1 # if the bottom of the current layer is higher (closer to zero depth) than
                # z-coordinate of location then add 1 to the layer_num.

        return self.layers[layer_num]
