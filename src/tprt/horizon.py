import numpy as np
import pylab as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from .units import Units
from scipy.spatial import Delaunay
from .bicubic_interpolation import *


class Horizon:
    def get_depth(self, x): # probably this method returns depth of the "horizon" in the given point x = [x, y]
        pass

    def intersect(self, sou, rec):
        pass

    def get_gradient(self, x):
        pass

    def get_normal(self, x, y): # returns unit normal to the point [x, y, z(x,y)]
        pass

    @staticmethod
    def _plot_horizon_3d(x, z, ax=None):
        ax.plot_trisurf(x[:, 0], x[:, 1], np.squeeze(z))


class FlatHorizon(Horizon):
    def __init__(self, depth=0, anchor=(0, 0), dip=0, azimuth=0):
        self.units = Units()
        self.depth = depth
        dip = np.deg2rad(dip)
        azimuth = np.deg2rad(azimuth)
        self.normal = np.array([np.sin(dip) * np.cos(azimuth), np.sin(dip) * np.sin(azimuth), np.cos(dip)])
        self.n = np.array(self.normal[:-1], ndmin=2)
        self.n3 = self.normal[-1]
        self.anchor = np.array(anchor, ndmin=2)
        self.gradient = self.get_gradient

    def get_depth(self, x):
        x = np.array(x, ndmin=2)
        x -= self.anchor
        z = (self.depth - (x * self.n).sum(axis=1, keepdims=True)) / (self.n3 + 1e-16)
        return z

    def get_normal(self, x, y):

        return self.normal

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

            n = np.cross(p2 - p1, p3 - p1)
            if np.dot(rec - sou, n) == 0:
                break
            p0 = sou + np.dot(p1 - sou, n) / np.dot(rec - sou, n) * (rec - sou)

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

    def plot(self, x=None, extent=(0, 100, 0, 100), ns=2, ax=None):
        if not np.any(x):
            _x, _y = np.meshgrid(
                np.linspace(extent[0], extent[1], ns),
                np.linspace(extent[2], extent[3], ns)
            )
            x = np.vstack((_x.ravel(), _y.ravel())).T

        z = self.get_depth(x)

        # TODO prettify using plt.show()
        if not np.any(ax):
            fig = plt.figure()
            ax = Axes3D(fig)

        self._plot_horizon_3d(x, z, ax=ax)


class GridHorizon(Horizon):

    # This class describes curvilinear smooth interfaces. They all have to be set as three arrays:
    # X = np.array([x_0, x_1, ... , x_m-1]) - array of X-coordinates of the points of the surface
    # Y = np.array([y_0, y_1, ... , y_n-1]) - array of Y-coordinates of the points of the surface
    # Z = np.array([[z_0_0, z_0_1, ... , z_0_n-1],
    #               [z_1_0, z_1_1, ... , z_1_n-1],
    #               ...
    #               [z_m-1_0, z_m-1-1, ... , z_m-1_n-1]]) - array of Z-coordinates of the points of the surface.
    #                                                       In our notation: z_i_j = z(x_i, y_j)

    # All given points are supposed to be points of an unknown smooth function z = z(x,y). The surface is supposed to be
    # smooth enough to have a normal vector at any point and curvature in this point
    # as well. So, corresponding conditions are imposed on the method of the interpolation/approximation.
    # Here we decided to use cubic spline interpolation. All formulas are explained at address:
    # http://statistica.ru/branches-maths/interpolyatsiya-splaynami-teor-osnovy/

    # These formulas allow us to construct a surface with second-order smoothness.

    def __init__(self, X, Y, Z, gradient=None):
        # Here arrays X and Y form up a rectangular coordinate grid. The grid is supposed to be regular.

        self.X = X
        self.Y = Y
        self.Z = Z

        self.gradient = gradient

    def get_depth(self, x):

        return two_dim_inter(self.X, self.Y, self.Z, x[0], x[1])

    def get_gradient(self, x):
        return self.gradient

    def get_normal(self, x, y):
        # We shall use the following fact: z = f(x, y) => g(x, y, z) = z - f(x, y) == 0 => grad(g(x,y,z)) is orthogonal
        # to our surface.
        # grad(g(x,y,z)) = [df/dx, df/dy, 1]

        # Since the interpolation provides second-order smoothness we can calculate df/dx and df/dy just by
        # interpolating vector of derivatives provided by the corresponding function. In order to find this vector we
        # have to find vectors [f(x_i, y)] and [f(x, y_i)]:

        x_vector = np.zeros(self.X.shape[0]) # [f(x_i, y)]
        y_vector = np.zeros(self.Y.shape[0]) # [f(x, y_i)]

        for i in range(self.X.shape[0]):

            x_vector[i] = one_dim_inter(self.Y, self.Z[i, :], y) # exactly in this way: we are
            # interpolating particular "cross-section" of the surface along axis Y

        for j in range(self.Y.shape[0]):

            y_vector[j] = one_dim_inter(self.X, self.Z[:, j], x) # exactly in this way: we are
            # interpolating particular "cross-section" of the surface along axis X

        # The "derivatives" function needs the discretization step as an argument. Remember that sets X and Y are
        # supposed to be regular!

        x_step = self.X[1] - self.X[0]

        y_step = self.Y[1] - self.Y[0]

        # Now we are ready to compute df/dx and df/dy at the point (x, y).

        x_der = one_dim_inter_ddx(self.X, x_vector, x)

        y_der = one_dim_inter_ddx(self.Y, y_vector, y)

        # The normal to the surface can be set as:

        n = np.array([- x_der, - y_der, 1])

        # But have to return unit normal:

        return n / np.linalg.norm(n)

    def intersect(self, sou, rec):
        # по заданной поверхности surf = [x_set, y_set, z] и координатам источника и приёмника sou, rec строит точку
        # пересечения интерполированной поверхности с прямой, соединяющей источник и приёмник

        if sou[0] == rec[0] and sou[1] == rec[1]: # если источник и приёмник находятся на одной вертикали, то всё
            # просто

            return sou[0], sou[1], two_dim_inter(self.X, self.Y, self.Z, sou[0], sou[1])

        # если же нет, то:

        # будем работать в вертикальной плоскости, включающей огогворенную выше прямую. Эта плоскость пересекает
        # горизонтальню плоскость вдоль прямой y = ax + b. Коэффициенты считаются из системы уравнений
        # sou[1] = a * sou[0] + b,
        # rec[1] = a * rec[0] + b

        a = (sou[1] - rec[1])/(sou[0] - rec[0])
        b = sou[1] - a * sou[0]

        # Также очевидно, что в вертикальной плоскости прямая "источник-приёмник" задаётся уравнением

        # z = +-c*sqrt((x - sou[0])**2 + (y - sou[1])**2) + d =
        # = sign(x - sou[0]) * c*sqrt((x - sou[0])**2 + (ax + b - sou[1])**2) + d =
        # = sign(x - sou[0]) * c*sqrt((a**2 + 1)*x**2 + (2a(b - sou[1]) - 2sou[0])*x + (b - sou[1])**2 + sou[0]**2) + d

        # Коэффициенты c, d ищутся по системе:
        # sou[2] = c*sqrt((sou[0] - sou[0])**2 + (sou[1] - sou[1])**2) + d = d
        # rec[2] = sign(rec[0] - sou[0]) * c*sqrt((rec[0] - sou[0])**2 + (rec[1] - sou[1])**2) + d

        d = sou[2]

        c = np.sign(rec[0] - sou[0]) * (rec[2] - d)/np.sqrt((rec[0] - sou[0])**2 + (rec[1] - sou[1])**2)

        # Хорошо. Теперь осталось найти точку пересечения поверхности с прямой. Заметим, что теперь X и Y меняются не
        # независимо.
        # Т.е. надо решить НЕ surf(x, y) = z(x, y), а surf(x, y(x)) = z(x, y(x))

        # => surf(x) = sign(x - sou[0]) * c*sqrt(x**2 + (ax + b)**2) + d =
        # = sign(x - sou[0]) * c*sqrt((a**2 + 1)*x**2 + 2ab*x + b**2) + d

        # Слева в уравнении стоит z-координата точки интерполированной поверхности.

        # Это уравнение мы будем решать численно. Для алгоритма минимизации нужно какое-то нулевое приближение.
        # В качестве такового возьмём решение уравнения:

        # surf_aver = sign(x - sou[0]) * c*sqrt((a**2 + 1)*x**2 +
        # + (2a(b - sou[1]) - 2sou[0])*x + (b - sou[1])**2 + sou[0]**2) + d

        # surf_aver - среднее значение z по дискретно заданной поверхности

        surf_aver = np.average(self.Z)

        #     x0 = np.roots([a**2 + 1, 2*a*b, b**2 - ((surf_aver - d)/c)**2])[0]. Надо помнить, что вовсе не
        # обязательно, чтобы это начальное приближение попадало в исследуемую оюласть. Так что возьмём один из корней
        # этого уравнения и на этом успокоимся.
        x0 = (- (a * (b - sou[1]) - sou[0])  - np.sqrt((a * (b - sou[1]) - sou[0])**2 -
                                                       (a**2 + 1) * ((b - sou[1])**2 + sou[0]**2 -
                                                                     ((surf_aver - d)/c)**2 )))/(a**2 + 1)

        # if x0 < self.X[0] or x0 > self.X[-1]: # если один из корней в область не попадает
        #     x0 = (- (a * (b - sou[1]) - sou[0])  + np.sqrt((a * (b - sou[1]) - sou[0])**2 -
        #                                                    (a**2 + 1) * ((b - sou[1])**2 + sou[0]**2 -
        #                                                                  ((surf_aver - d)/c)**2 )))/(a**2 + 1)
        #
        # if x0 < self.X[0] or x0 > self.X[-1]: # если и второй корень не попадает в область, то точки пересечения нет.
        #
        #     return []

        #     Теперь надо построить точку пересечения прямой с интерполированной поверхностью.
        # Нужно, чтобы минимизатор не увёл нас за пределы области:
        bound = np.array([self.X[0], self.X[-1]])

        minimization = minimize(difference, x0, args = (sou, self.X, self.Y, self.Z,
                                                        a, b, c, d), method ='SLSQP', bounds = np.array([bound]))

        # If the minimum exists but it is not real point of intersection, target function would have big absolute value.
        # So, let's check whether we have find the intersection point or not:

        if abs(minimization.fun) > 10 ** (-6): # we'll need to reconsider the condition: should it just 10 ** (-6)
            #  or less

            return [] # if the target function is still big then we did not really find the intersection point

        else: # if the target function is relatively small, return the found point:

            return minimization.x[0], a * minimization.x[0] + b,\
               two_dim_inter(self.X, self.Y, self.Z, minimization.x[0], a * minimization.x[0] + b)

    def plot(self, ax=None):

        # We'd like to plot a smooth interface, so let's construct new X and Y sets which will be 5 times more frequent
        # than original ones::

        X_new = np.linspace(self.X[0], self.X[-1], 5 * self.X.shape[0])
        Y_new = np.linspace(self.Y[0], self.Y[-1], 5 * self.Y.shape[0])

        # And let's interpolate our initial grid to the new one:

        Z_new = np.zeros((X_new.shape[0],  Y_new.shape[0]))

        for i in range(X_new.shape[0]):
            for j in range(Y_new.shape[0]):

                Z_new[i, j] = two_dim_inter(self.X, self.Y, self.Z, X_new[i], Y_new[j])

        # That's all. We can plot it.

        # But we have to mesgrid X_new and Y_new first:

        YY_new, XX_new = np.meshgrid(Y_new, X_new)

        # TODO prettify using plt.show()

        if not np.any(ax):
            fig = plt.figure()
            ax = Axes3D(fig)

        self._plot_horizon_3d(np.vstack((XX_new.ravel(), YY_new.ravel())).T, Z_new.ravel(), ax=ax)
