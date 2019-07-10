import numpy as np
from itertools import product
from pylab import griddata


def init_equidistant_sphere(n=256):
    """

    :param n:
    :return:
    """

    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1/n, 1/n - 1, n)
    radius = np.sqrt(1 - z**2)

    points = np.zeros((n, 3))
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = z
    return points


def plot_equidistant_sphere(points):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], zs=points[:, 2])


def sdr_generator(n=10):
    """
    Grid of strike, dip, rake
    :param n:
    :return:
    """
    p = np.linspace(0, 2*np.pi, 2*n)
    d = np.linspace(0, np.pi/2, n)
    l = np.linspace(-np.pi, np.pi, 2*n)
    grid = product(p, d, l)
    return grid


def grid_model(xz, media, n=30):
    """

    :param xz:
    :param media:
    :param n:
    :return:
    """

    xi = np.linspace(min(xz[:,0]),max(xz[:,0]),n)
    zi = np.linspace(min(xz[:,1]),max(xz[:,1]),n)

    media = griddata(xz[:, 0], xz[:, 1], media.ravel(), xi, zi, interp='linear')
    return media, xi, zi
