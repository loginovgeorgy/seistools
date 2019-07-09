import pylab as plt
import numpy as np
from .utils import projection, pol2cart


def plot_moments(c_clvd, c_iso, centers=None, labels=None, s=40):
    if isinstance(labels, type(None)):
        if not isinstance(labels, list):
            labels = [labels]

    if isinstance(labels, type(None)):
        if not isinstance(centers, list):
            centers = [centers]

    plt.figure(figsize=(15, 5))

    for i, (l, cn) in enumerate(zip(labels, centers)):
        axes = plt.subplot(1, len(labels), i + 1)
        plt.plot([1, 0, -1, 0, 1],
                 [0, 1, 0, -1, 0], 'k')

        plt.scatter(c_clvd, c_iso, c=l, s=s)
        plt.plot(cn[:, 0],
                 cn[:, 1], '*r', ms=10)

        axes.set_xlim([-1.1, 1.1])
        axes.set_ylim([-1.1, 1.1])
        axes.set_aspect('equal')


def plot_moments_vu(c_clvd, c_iso, centers=None, labels=None, s=40):
    if isinstance(labels, type(None)):
        if not isinstance(labels, list):
            labels = [labels]

    if isinstance(labels, type(None)):
        if not isinstance(centers, list):
            centers = [centers]

    plt.figure(figsize=(15, 5))

    for i, temp in enumerate(zip(labels, centers)):
        l, cn = temp
        axes = plt.subplot(1, len(labels), i + 1)
        plt.plot([-4 / 3, 0, 4 / 3, 0, -4 / 3],
                 [-1 / 3, 1, 1 / 3, -1, -1 / 3], 'k')

        plt.scatter(c_clvd, c_iso, c=l, s=s)
        plt.plot(cn[:, 0],
                 cn[:, 1], '*r', ms=10)

        axes.set_xlim([-1.1 * 4 / 3, 1.1 * 4 / 3])
        axes.set_ylim([-1.1, 1.1])
        axes.set_aspect('equal')


def plot_hemisphere(dx, dy, radians=True):
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='w')
    ax.set_axis_off()

    polar = np.arange(0, 91, 15)
    azimuth = np.arange(0, 360, 30)
    rho = projection(np.deg2rad(polar))
    for r, p in zip(rho, polar):
        x1, y1 = pol2cart(np.linspace(0, 2 * np.pi, 50), r)
        ax.plot(x1, y1, color=[.5, .5, .5, .5])
        ax.text(r, 0, p)

    vertical_alignment = [
        'center', 'baseline', 'baseline', 'bottom', 'baseline',
        'baseline', 'center', 'top', 'top', 'top', 'top', 'top'
    ]

    horizontal_alignment = [
        'left', 'left', 'left', 'center', 'right', 'right',
        'right', 'right', 'right', 'center', 'left', 'left'
    ]

    for i, a in enumerate(azimuth):
        x1, y1 = pol2cart(np.deg2rad(a), np.array([0, 1]))
        ax.plot(x1, y1, color=[.5, .5, .5, .5])
        if i > 0:
            ax.text(
                x1[1], y1[1], a,
                verticalalignment=vertical_alignment[i],
                horizontalalignment=horizontal_alignment[i]
            )

    dx = np.array(dx, ndmin=2)
    dy = np.array(dy, ndmin=2)
    if not radians:
        dx = np.deg2rad(dx)
        dy = np.deg2rad(dy)

    yy = projection(dy)
    x1, y1 = pol2cart(dx, yy)
    ax.scatter(x1, y1)

    return ax