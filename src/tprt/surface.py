import numpy as np
from scipy.spatial import Delaunay
'''
IDC_TYPES = {
    'flat': _flat_horizon,
    'f': _flat_horizon,
    'fh': _flat_horizon,
    'horizontal': _flat_horizon,
    'grid': _grid_horizon,
}
'''


class Surface:
    def get_depth(self, x):
        pass
    def intersect(self, sou, rec):
        pass
    def get_gradient(self, x):
        pass


class FlatSurface(Surface):
    def __init__(self, depth=0, anchor=(0, 0), dip=0, azimuth=0):
        self.depth = depth
        dip = np.deg2rad(dip)
        azimuth = np.deg2rad(azimuth)
        self.normal = np.array([np.sin(dip)*np.cos(azimuth), np.sin(dip)*np.sin(azimuth), np.cos(dip)])
        self.n = np.array(self.normal[:-1], ndmin=2)
        self.n3 = self.normal[-1]
        self.anchor = np.array(anchor, ndmin=2)
        self.gradient = self.get_gradient

    def get_depth(self, x):
        x = np.array(x, ndmin=2)
        x -= self.anchor
        z = (self.depth - (x * self.n).sum(axis=1, keepdims=True)) / (self.n3 + 1e-16)
        return z

    def get_gradient(self, x):
        return -self.normal[:-1]

    def intersect(self, sou, rec):
        points = np.zeros((3, len(sou)))
        points[0], points[1] = sou, rec
        points[2, :2] = np.array([sou[0], rec[1]])
        for i in range(len(points)):
            points[i, 2] = self.get_depth(points[i, :2])
        tri = Delaunay(points[:, :2])
        intersection = []
        for index in tri.simplices:
            p1, p2, p3 = points[index]

            N = np.cross(p2 - p1, p3 - p1)
            if np.dot(rec - sou, N) == 0:
                break
            p0 = sou + np.dot(p1 - sou, N) / np.dot(rec - sou, N) * (rec - sou)

            x_max, x_min = max(sou[0], rec[0]), min(sou[0], rec[0])
            y_max, y_min = max(sou[1], rec[1]), min(sou[1], rec[1])  # Т.к. у нас отрезок, а не бесконечная линия
            z_max, z_min = max(sou[2], rec[2]), min(sou[2], rec[2])  # то создадим ограничения

            if (x_min < p0[0] < x_max and
                y_min < p0[1] < y_max and
                z_min < p0[2] < z_max):
                intersection = p0
                break
        intersection = np.array(intersection, ndmin=1)
        return intersection


class GridSurface(Surface):
    def __init__(self, points, gradient=None):
        self.points = np.array(points, ndmin=2)
        self.gradient = gradient

    def get_depth(self, x):
        return self.points[0,-1]

    def get_gradient(self, x):
        return self.gradient

    def intersect(self, sou, rec):
        points = self.points
        tri = Delaunay(points[:, :2])
        intersection = []
        for index in tri.simplices:
            p1, p2, p3 = points[index]

            N = np.cross(p2 - p1, p3 - p1)
            if np.dot(rec - sou, N) == 0:
                break
            p0 = sou + np.dot(p1 - sou, N) / np.dot(rec - sou, N) * (rec - sou)

            x_max, x_min = max(sou[0], rec[0]), min(sou[0], rec[0])
            y_max, y_min = max(sou[1], rec[1]), min(sou[1], rec[1])
            z_max, z_min = max(sou[2], rec[2]), min(sou[2], rec[2])

            if (np.dot(np.cross(p2 - p1, p0 - p1), N) >= 0 and
                np.dot(np.cross(p3 - p2, p0 - p2), N) >= 0 and
                np.dot(np.cross(p1 - p3, p0 - p3), N) >= 0 and
                x_min < p0[0] < x_max and
                y_min < p0[1] < y_max and
                z_min < p0[2] < z_max):
                intersection = p0
                break
        intersection = np.array(intersection, ndmin=1)
        return intersection


