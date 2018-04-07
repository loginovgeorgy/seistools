import numpy as np
from scipy.interpolate import Rbf
import pickle

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

# все "магические" константы должны быть определены в одном месте. Если они вне класса - пишем большими буквами
STEP = 5
RATIO = 10
OVERLAP = 1
RBF_RADIUS = 10


class Approx:
    def __init__(self, x_grd, y_grd, step=STEP):
        self.step = step
        self.x_grd = x_grd
        self.y_grd = y_grd

        # это переменные, которые не нужны пользователю - скроем их при помощи "_"
        self._dx = (x_grd[1, 0] - x_grd[0, 0])
        self._dy = (y_grd[0, 1] - y_grd[0, 0])
        self._nx, self._ny = x_grd.shape
        self._list_of_funcs = []
        self._nc = len(np.arange(0, self._ny, self.step))
        self._nl = len(np.arange(0, self._nx, self.step))

    @staticmethod
    def _cut_grid_area(x, xl, xr, yl, yr):
        if not isinstance(x, list):
            x = [x]

        y = [_x[xl: xr, yl: yr] for _x in x]
        return y

    def _apply_approx_func(self, x, y):
        i = np.floor(x / (self._dx * self.step))
        j = np.floor(y / (self._dy * self.step))
        arg = int(i * self._nc + j)
        return self._list_of_funcs[arg](x, y)

    @staticmethod
    def plot_horizon(x, y, z):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # Plot the surface.
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(z.min(), z.max())
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def fit(self, z_grd, ratio=RATIO, overlap=OVERLAP, rbd_radius=RBF_RADIUS):
        # это локальные переменные, она не должны быть self
        grid = [self.x_grd, self.y_grd, z_grd]
        iter_x = int(self._nx / self.step)
        iter_y = int(self._ny / self.step)

        ######################################

        lst = []  # список функций
        for i in np.arange(0, self._nx, self.step):
            for j in np.arange(0, self._ny, self.step):
                xl, yl = i, j
                xr, yr = i + self.step, j + + self.step

                if (overlap < i < self._nx - overlap) and (overlap < j < self._ny - overlap):
                    xl -= overlap
                    yl -= overlap
                    xr += overlap
                    yr += overlap

                    x, y, z = self._cut_grid_area(grid, xl, xr, yl, yr)
                else:
                    x, y, z = self._cut_grid_area(grid, xl, xr, yl, yr)
                    # тут можно выбрать другой метод аппроксимации на маленьких квадратиках
                rbf = Rbf(x, y, z, epsilon=rbd_radius)
                lst.append(rbf)
                del rbf

        self._list_of_funcs = lst
        return  # self.ListOfFunc, iterY

    def predict(self, x_grd, y_grd):
        nx, ny = x_grd.shape
        z_grd = np.zeros(x_grd.shape)
        for i in range(0, nx):  # !!!!!!!!!!!!!-1
            for j in range(0, ny):
                z_grd[i, j] = self._apply_approx_func(x_grd[i, j], y_grd[i, j])
        return z_grd
# функции вне класса


def load_horizons(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data


def _gap(z_grd, x, y):
    diff = np.zeros(8)
    diff[0] = z_grd[x - 1, y] - z_grd[x, y]
    diff[1] = z_grd[x + 1, y] - z_grd[x, y]
    diff[2] = z_grd[x, y - 1] - z_grd[x, y]
    diff[3] = z_grd[x, y + 1] - z_grd[x, y]
    diff[4] = z_grd[x - 1, y - 1] - z_grd[x, y]
    diff[5] = z_grd[x - 1, y + 1] - z_grd[x, y]
    diff[6] = z_grd[x + 1, y + 1] - z_grd[x, y]
    diff[7] = z_grd[x + 1, y - 1] - z_grd[x, y]

    return np.abs(diff).max()


def gap_array(z_grid):
    arr = np.zeros(z_grid.shape)
    nx, ny = z_grid.shape
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            arr[i, j] = _gap(z_grid, i, j)
    return arr
