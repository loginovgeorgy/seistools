import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from .units import Units
from src.tprt import Layer, FlatHorizon

class Velocity_model(object):
    def __init__(self, velocity, density, name, depth, anchor, dip, azimuth):
        self.velocities = velocity
        self.densities  = density
        self.names      = name
        self.depths     = depth
        self.anchors    = anchor
        self.dips       = dip
        self.azimuths   = azimuth

        self.horizons = self.make_horizons()
        self.layers = self.make_layers()

    def make_horizons(self):        # Подразумеватся что все отсортировано по глубине от 0 до ...
        FlatHorizons = []
        if (self.depths).any()!=0: FlatHorizons.append(FlatHorizon())       # Все-таки я считаю, что нужно хранить дневную поверхность
        for depth, anchor, dip, azimuth in zip(self.depths, self.anchors, self.dips, self.azimuths):
            FlatHorizons.append(FlatHorizon(depth, anchor, dip, azimuth))
        return FlatHorizons

    def make_layers(self):
        Layers = []
        for i, (velocity, density, name) in enumerate(zip(self.velocities, self.densities, self.names)):
            top = self.horizons[i]
            if (i+1)<len(self.horizons): bottom = self.horizons[i+1]
            else: bottom = None
            Layers.append(Layer(velocity, density, top, bottom, name=name))
        return Layers

    def add_horizon(self, Horizon):
        self.horizons.append(Horizon)

    def add_layer(self, Layer):
        self.layers.append(Layer)

    def __repr__(self):
        return self.layers
