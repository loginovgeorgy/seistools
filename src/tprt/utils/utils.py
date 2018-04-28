from mpl_toolkits.mplot3d import Axes3D
import pylab as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import Delaunay

def plot_points_3d(location, **kwargs):
    if not np.any(kwargs.get('ax')):
        fig = plt.figure()
        ax = Axes3D(fig)
    else:
        ax = kwargs['ax']
        kwargs.pop('ax')

    ax.scatter3D(*location, **kwargs)


def plot_line_3d(location, **kwargs):
    if not np.any(kwargs.get('ax')):
        fig = plt.figure()
        ax = Axes3D(fig)
    else:
        ax = kwargs['ax']
        kwargs.pop('ax')

    ax.plot3D(*location, **kwargs)


def flat_intersect(sou, rec, horizon):
    points = np.zeros((3,len(sou)))
    points[0], points[1] = sou, rec
    points[2,:2] = np.array([sou[0],rec[1]])
    for i in range(len(points)):
        points[i,2] = horizon.get_depth(points[i,:2])
    tri = Delaunay(points[:, :2])
    intersection = []
    for index in tri.simplices:
        p1, p2, p3 = points[index]

        N = np.cross(p2 - p1, p3 - p1)
        if np.dot(rec - sou, N)==0: break
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

def grid_intersect(sou, rec, horizon):
    points = horizon.points
    tri = Delaunay(points[:, :2])
    intersection = []
    for index in tri.simplices:
        p1, p2, p3 = points[index]

        N = np.cross(p2 - p1, p3 - p1)
        p0 = sou + np.dot(p1 - sou, N) / np.dot(rec - sou, N) * (rec - sou)

        x_max, x_min = max(sou[0], rec[0]), min(sou[0], rec[0])
        y_max, y_min = max(sou[1], rec[1]), min(sou[1], rec[1])     # Т.к. у нас отрезок, а не бесконечная линия
        z_max, z_min = max(sou[2], rec[2]), min(sou[2], rec[2])     # то создадим ограничения

        if (np.dot(np.cross(p2 - p1, p0 - p1), N) >= 0 and          # Эти условия проверяют лежит ли точка в заданном треугольнике
            np.dot(np.cross(p3 - p2, p0 - p2), N) >= 0 and
            np.dot(np.cross(p1 - p3, p0 - p3), N) >= 0 and
            x_min < p0[0] < x_max and
            y_min < p0[1] < y_max and
            z_min < p0[2] < z_max):
            intersection = p0
            break
    intersection = np.array(intersection, ndmin=1)
    return intersection


INTERSECTION_TYPES = {
    'flat': flat_intersect,
    'f': flat_intersect,
    'fh': flat_intersect,
    'horizontal': flat_intersect,
    'grid': grid_intersect,
}

def is_ray_intersect_surf(sou, rec, horizon, eps=1e-8):
    return INTERSECTION_TYPES[horizon.kind](sou,rec,horizon)



