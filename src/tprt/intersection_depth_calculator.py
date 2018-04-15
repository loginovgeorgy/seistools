import numpy as np
'''
IDC_TYPES = {
    'flat': _flat_horizon,
    'f': _flat_horizon,
    'fh': _flat_horizon,
    'horizontal': _flat_horizon,
    'grid': _grid_horizon,
}
'''


class IntersectionDepthCalculator:
    def get_depth(self, x):
        pass


class FlatHorizonIDC(IntersectionDepthCalculator):
    def __init__(self, depth=0, anchor=(0, 0), dip=0, azimuth=0):
        self.depth = depth
        dip = np.deg2rad(dip)
        azimuth = np.deg2rad(azimuth)
        self.n = np.array([np.sin(dip)*np.cos(azimuth), np.sin(dip)*np.sin(azimuth)], ndmin=2)
        self.n3 = np.cos(dip)
        self.anchor = np.array(anchor, ndmin=2)

    def get_depth(self, x):
        x = np.array(x, ndmin=2)
        x -= self.anchor
        z = (self.depth - (x * self.n).sum(axis=1, keepdims=True)) / (self.n3 + 1e-16)
        return z


class GridHorizonIDC(IntersectionDepthCalculator):
    def get_depth(self, x):
        return 1


