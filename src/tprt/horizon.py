import numpy as np
import pylab as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from .units import Units
from scipy.spatial import Delaunay
from .bicubic_interpolation import *


class Horizon:
    def get_depth(self, x): # returns depth of the "horizon" in the given point x = [x, y]
        pass

    def intersect(self, sou, rec): # returns the point of intersection of a straight line connecting source and receiver
        # with the "horizon".
        pass

    def get_gradient(self, x): # returns gradient of the "horizon" function: grad( z(x,y) ) = [dz / dx, dz / dy]
        pass

    def get_normal(self, x): # returns unit normal to the point [x, y, z(x,y)] (x in the arguments is x = [x, y])
        pass

    def get_sec_deriv(self, x, vec): # 1. constructs local system of coordinates e1, e2 and n, where n is unit normal
        # to the point [x, y, z[x,y]], d1 and d2 are tangent vectors to the surface at this point. It will be
        # explained later what tangent vectors are used. 2. returns second partial derivatives of the surface
        # calculated in local system at the point [x, y, z[x,y]] and the transformation matrix [d1, d2, n].

        # vec is vector of the incident ray.
        # x in the arguments is x = [x, y]

        # Further explanations are presented below.
        pass

    @staticmethod
    def _plot_horizon_3d(x, y, z, ax=None):
        # x and y are 1D-arrays which define the grid.
        yy, xx = np.meshgrid(y, x) # exactly in this order

        ax.plot_surface(xx, yy, z, alpha = 0.5)

        return


class FlatHorizon(Horizon):
    def __init__(self, depth = 0, anchor = np.array([0, 0]),  dip = 0, azimuth = 0):

        # Constructor accepts as arguments the following items:

        # depth = Z-coordinate of a point of the plane

        # anchor = [y, z] = X- and Y-coordinates of this point

        # azimuth = azimuth of incidence (азимут падения). It is counted clockwise from the axis X.
        # Lies in [0, 360). Measured in degrees.

        # dip = angle of incidence (угол падения). It is counted from the horizontal plane downward.
        # Lies in [0, 90). Measured in degrees.

        # x0 = np.array([x0, y0, z0]) = a point in 3D space which lies at the plane of the "horizon".


        self.units = Units()

        # Any plane in 3D space can be set by an equation:
        # A * x + B * y + C * z + D = 0

        # It would be convenient to use four constants A, B, C and D as fields of this class:

        self.A = np.sin(np.radians(dip)) * np.cos(np.radians(azimuth))
        self.B = np.sin(np.radians(dip)) * np.sin(np.radians(azimuth))
        self.C = - np.cos(np.radians(dip))
        self.D = - (self.A * anchor[0] + self.B * anchor[1] + self.C * depth)

    def get_depth(self, x):
        # "Horizons" are supposed to be non-vertical. Consequently C != 0. So, the depth is defined by formula:
        return - (self.A * x[0] + self.B * x[1] + self.D) / self.C

    def get_normal(self, x = np.array([0, 0])):
        # Gradient of any function f(x,y,z) is perpendicular to constant level surfaces. In case of a plane:
        # f(x,y,z) = A * x + B * y + C * z + D = 0 everywhere on the plane => grad(f) = [A, B, C] is the
        # sought normal vector. We just have to normalize it.
        return np.array([self.A, self.B, self.C]) / np.linalg.norm(np.array([self.A, self.B, self.C]))

    def get_gradient(self, x):
        return - np.array([self.A, self.B]) / self.C # it is obvious from the formula of the gradient of z(x,y)

    def intersect(self, sou, rec):

        # Для начала надо удостовериться, что пересечение существует. Критерий простой: если источник и приёмник
        # находятся выше или ниже границы, то пересечения нет. (случай сильно криволинейных границ мы не рассматриваем)

        # Стоит иметь ввиду, что произвольная точка [x,y,z] находится выше "горизонта". если разница
        # z - self.get_depth(x,y)
        # принимает положительные значения. И наоборот, эта точка находится ниже "горизонта", если эта разница
        # отрицательна.
        # Таким образом, критерий может быть записан довольно просто:

        if ( sou[2] - self.get_depth([sou[0], sou[1]]) ) * ( rec[2] - self.get_depth([rec[0], rec[1]]) ) > 0:

            return []

        # Зададим вектор в направлении от источника к приёмнику:

        vector = np.array([rec[0] - sou[0], rec[1] - sou[1], rec[2] - sou[2]])

        # Радиус-веткор любой точки на луче, соединяющем источник и приёмник, задаётся выраженим:

        # r = np.array([sou[0] + s * vector[0], sou[1] + s * vector[1], sou[2] + s * vector[2]])

        # Здесь s ~ длина пути вдоль луча (пропорциональна - т.к. вектор vector не единичен). В теории, она может быть и
        # отрицательной.

        # Таким образом, для пересечения нам нужно добиться равенства:

        # z_surface(x, y) = z_ray
        # => z_surface(sou[0] + s * vector[0], sou[1] + s * vector[1]) = sou[2] + s * vector[2]

        # Эта задача - нахождение подходящего s - в случае плоскости решается просто:

        # z_surface(sou[0] + s * vector[0], sou[1] + s * vector[1]) = sou[2] + s * vector[2]
        # => - (self.A * (sou[0] + s * vector[0]) + self.B * (sou[1] + s * vector[1]) + self.D) / self.C =
        # = sou[2] + s * vector[2]

        # => (self.A * vector[0] + self.B * vector[1] + self.C * vector[2]) * s =
        # = - (self.A * sou[0] + self.B * sou[1] + self.C * sou[2] + self.D)

        s = - (self.A * sou[0] + self.B * sou[1] + self.C * sou[2] + self.D) /\
            (self.A * vector[0] + self.B * vector[1] + self.C * vector[2])

        # НАДО ПРОВЕРИТЬ, НЕ ВЫШЛИ ЛИ МЫ ЗА ПРЕДЕЛЫ ОБЛАСТИ!!!
        # А вот здесь надо быть осторожным!!! Мы не знаем точно границ области!
        #
        # if sou[0] + s * vector[0] < 0 or \
        #         sou[0] + s * vector[0] > 100 or \
        #         sou[1] + s * vector[1] < 0 or \
        #         sou[1] + s * vector[1] > 100:
        #
        #         return []

        # Ну, а если не вышли, то возвращаем найденную точку:

        return np.array([sou[0] + s * vector[0], sou[1] + s * vector[1], sou[2] + s * vector[2]])

    def get_sec_deriv(self, x, vec):

        # n is unit normal to the surface at the current point. It is pointed to the medium where the outgoing ray
        # comes.

        n = self.get_normal(x)
        n = np.sign(np.dot(n, vec)) * n # it has to be pointed in direction of vec

        # d1 is normed projection of vec (from arguments) on the tangent plane to the surface at the current point.
        # d2 is cross product [n x d1].

        # The tangent plain is formed up by vectors dr/dx = [1, 0, z_x] and dr/dy = [0, 1, z_y]. But they are not
        # unit and even not orthogonal to each other. Let's orthogonalize them:

        tang_x = np.array([1, 0, - self.A / self.C]) # dr/dx
        tang_y = np.array([0, 1, - self.B / self.C]) # dr/dy

        tang_y = tang_y - np.dot(tang_y, tang_x) / np.dot(tang_x, tang_x) * tang_x # now tang_y is perpendicular to
        # tang_x.

        tang_x = tang_x / np.linalg.norm(tang_x)
        tang_y = tang_y / np.linalg.norm(tang_y) # and now they both are unit

        # So, we are ready to introduce d1 and d2. But we should note that if vec is collinear with n (i.e. we deal
        # with normal incidence) than d1 and d2 are not defined. In that case we shall set d1 as follows:

        if np.linalg.norm(np.cross(vec, n)) == 0:

            d1 = tang_x

        else:

            d1 = np.dot(vec, tang_x) * tang_x + np.dot(vec, tang_y) * tang_y


        # Let's normalize it:

        d1 = d1 / np.linalg.norm(d1)

        # d2 is still defined as a cross-product:

        d2 = np.cross(n, d1)

        # second derivatives on a plane are equal to zero in any coordinate system. So, let's return the result:

        return 0, 0, 0, np.array([d1, d2, n]).T

    def plot(self,extent=(0, 100, 0, 100), ax=None):

        x = np.array([extent[0], extent[1]])
        y = np.array([extent[2], extent[4]])

        z = np.zeros(x.shape)

        for i in range(x.shape[0]):
            for j in range(y.shape[0]):

                z[i, j] = self.get_depth([x[i], y[j]])

        # TODO prettify using plt.show()
        if not np.any(ax):
            fig = plt.figure()
            ax = Axes3D(fig)

        self._plot_horizon_3d(x, y, z, ax=ax)


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
    # Here we decided to use cubic spline interpolation. Corresponding formulas allow us to construct a surface with
    # second-order smoothness.

    def __init__(self, X, Y, Z, bool_parab = 1):
        # Here the input arrays X and Y form up a rectangular coordinate grid. The grid is supposed to be regular for
        # each direction (X and Y).
        # In order to decrease influence of breaking out points, we shall construct more frequent grid using
        # so-called averaged parabolic interpolation defined in the corresponding module. However, you can switch off
        # this feature and keep only points given by X, Y and Z by setting bool_parab equal to 0.

        if bool_parab == 1:

            self.X = np.linspace(X[0], X[-1], 2 * X.shape[0] - 1)
            self.Y = np.linspace(X[0], X[-1], 2 * X.shape[0] - 1)
            # minus one - since we would like to save all input points.

            self.Z = two_dim_parab_inter_surf(X, Y, Z, self.X, self.Y)

        else:

            self.X = X
            self.Y = Y
            self.Z = Z

        # In addition it will be needed to compute and keep array of all polynomial coefficients of our bicubic splines:

        self.polynomial = two_dim_polynomial(self.X, self.Y, self.Z)

    def get_depth(self, x):

        return two_dim_inter(self.polynomial, self.X, self.Y, x[0], x[1])

    def get_gradient(self, x):

        # grad(z(x,y)) = [dz/dx, dz/dy]

        grad = np.array([two_dim_inter_dx(self.polynomial, self.X, self.Y, x[0], x[1]),
                         two_dim_inter_dy(self.polynomial, self.X, self.Y, x[0], x[1])])

        return grad

    def get_normal(self, x):

        # We shall use the following fact: z = f(x, y) => g(x, y, z) = z - f(x, y) == 0 => grad(g(x,y,z)) is orthogonal
        # to our surface.
        # grad(g(x,y,z)) = [- df/dx, - df/dy, df/dz] = [- dz/dx, - dz/dy, 1]

        # dz/dx and dz/dy are present in self.get_gradient() vector. So:

        x_der, y_der = self.get_gradient(x)

        # Now the normal to the surface can be set as:

        n = np.array([- x_der, - y_der, 1])

        # But have to return unit normal:

        return n / np.linalg.norm(n)

    def intersect(self, sou, rec):
        # По заданным координатам источника и приёмника sou, rec строит точку пересечения прямой, соединяющей источник и
        # приёмник, с интерполированной поверхностью.

        # Для начала надо удостовериться, что пересечение существует. Критерий простой: если источник и приёмник
        # находятся выше или ниже границы, то пересечения нет. (случай сильно криволинейных границ мы не рассматриваем)

        # Стоит иметь ввиду, что произвольная точка [x,y,z] находится выше "горизонта". если разница
        # z - self.get_depth(x,y)
        # принимает положительные значения. И наоборот, эта точка находится ниже "горизонта", если эта разница
        # отрицательна.
        # Таким образом, критерий может быть записан довольно просто:

        if ( sou[2] - self.get_depth([sou[0], sou[1]]) ) * ( rec[2] - self.get_depth([rec[0], rec[1]]) ) > 0:

            return []

        # Зададим вектор в направлении от источника к приёмнику:

        vector = np.array([rec[0] - sou[0], rec[1] - sou[1], rec[2] - sou[2]])

        # Радиус-веткор любой точки на луче, соединяющем источник и приёмник, задаётся выраженим:

        # r = np.array([sou[0] + s * vector[0], sou[1] + s * vector[1], sou[2] + s * vector[2]])

        # Здесь s ~ длина пути вдоль луча (пропорциональна - т.к. вектор vector не единичен). В теории, она может быть и
        # отрицательной.

        # Таким образом, для пересечения нам нужно добиться равенства:

        # z_surface(x, y) = z_ray
        # => z_surface(sou[0] + s * vector[0], sou[1] + s * vector[1]) = sou[2] + s * vector[2]

        # Эту задачу - нахождение подходящего s - будем решать с помошью минимизации. Начальное приближение - s = 0.5,
        # что означает точку пересечения посредине между источникм и приёмником. Ограничения - величина s должна быть
        # не отрицательна (т.к. мы всегда идём по направлению к приёмнику) и не может превышать 1, т.к. s = 1
        # соответствует точке приёмника.

        s = minimize(difference, np.array([0.5]), args = (self.X, self.Y, self.polynomial, sou, vector),
                     method ='SLSQP', bounds = np.array([[0, 1]])).x[0]

        # Надо проверить, не вышли ли мы за пределы сетки задания поверхности:

        if sou[0] + s * vector[0] < self.X[0] or\
                sou[0] + s * vector[0] > self.X[-1] or \
                sou[1] + s * vector[1] < self.Y[0] or\
                sou[1] + s * vector[1] > self.Y[-1]:

            return []

        # Ну, а если не вышли, то возвращаем найденную точку:

        return np.array([sou[0] + s * vector[0], sou[1] + s * vector[1], sou[2] + s * vector[2]])

    def get_sec_deriv(self, x, vec):

        # First of all we need to calculate first and second derivatives of the surface in the global system of
        # coordinates.

        # In order to do that let's compute z(x, y_search) and z(x_search, y) arrays:

        z_x, z_y = self.get_gradient(x) # dz/dx and dz/dy

        z_xx = two_dim_inter_dx_dx(self.polynomial, self.X, self.Y, x[0], x[1])# d2z/dx2
        z_yy = two_dim_inter_dy_dy(self.polynomial, self.X, self.Y, x[0], x[1])# d2z/dy2

        z_xy = two_dim_inter_dx_dy(self.polynomial, self.X, self.Y, x[0], x[1])# d2z/dxdy

        # Very well. Now we have to construct a local coordinate system. It consists of three mutually orthogonal
        # unit vectors: d1, d2 and n.

        # n is unit normal to the surface at the current point. It is pointed to the medium where the outgoing ray
        # comes.

        n = self.get_normal(x)
        n = np.sign(np.dot(n, vec)) * n # it has to be pointed in direction of vec

        # d1 is normed projection of vec (from arguments) on the tangent plane to the surface at the current point.
        # d2 is cross product [n x d1].

        # The tangent plain is formed up by vectors dr/dx = [1, 0, z_x] and dr/dy = [0, 1, z_y]. But they are not
        # unit and even not orthogonal to each other. Let's orthogonalize them:

        tang_x = np.array([1, 0, z_x]) # dr/dx
        tang_y = np.array([0, 1, z_y]) # dr/dy

        tang_y = tang_y - np.dot(tang_y, tang_x) / np.dot(tang_x, tang_x) * tang_x # now tang_y is perpendicular to
        # tang_x.

        tang_x = tang_x / np.linalg.norm(tang_x)
        tang_y = tang_y / np.linalg.norm(tang_y) # and now they both are unit

        # So, we are ready to introduce d1 and d2. But we should note that if vec is collinear with n (i.e. we deal
        # with normal incidence) than d1 and d2 are not defined. In that case we shall set d1 as follows:

        if np.linalg.norm(np.cross(vec, n)) == 0:

            d1 = tang_x

        else:

            d1 = np.dot(vec, tang_x) * tang_x + np.dot(vec, tang_y) * tang_y

        # Let's normalize it:

        d1 = d1 / np.linalg.norm(d1)

        # d2 is still defined as a cross-product:

        d2 = np.cross(n, d1)

        # OK. The last thing to do is to recalculate known d2z/dx2, d2z/dxdy and d2z/dy2 into second partial derivatives
        # of z(x, y) written in local coordinates along e1 and e2 axes. Let's call the latter h11, h12 and h22.

        # It can be proven that h11, h12, h22 are solutions of a particular system of linear equations which depends on
        # e1, e2, dz/dx and dz/dy with n and d2z/dx2, d2z/dxdy, d2z/dy2 in the right part. Let's define the matrix
        # of this system and its right part:

        A = np.array([[(d1[0] + d1[2] * z_x)**2,
                       2 * (d1[0] + d1[2] * z_x) * (d2[0] + d2[2] * z_x),
                       (d2[0] + d2[2] * z_x)**2],

                      [(d1[0] + d1[2] * z_x) * (d1[1] + d1[2] * z_y),
                       (d1[0] + d1[2] * z_x) * (d2[1] + d2[2] * z_y) + (d1[1] + d1[2] * z_y) * (d2[0] + d2[2] * z_x),
                       (d2[0] + d2[2] * z_x) * (d2[1] + d2[2] * z_y)],

                      [(d1[1] + d1[2] * z_y)**2,
                       2 * (d1[1] + d1[2] * z_y) * (d2[1] + d2[2] * z_y),
                       (d2[1] + d2[2] * z_y)**2]])

        B = np.array([n[2] * z_xx, n[2] * z_xy, n[2] * z_yy])

        # Now we solve this system and return the result (including the transformation matrix):

        h11, h12, h22 = np.linalg.solve(A, B)

        return h11, h12, h22, np.array([d1, d2, n]).T

    def plot(self, ax=None):

        # We'd like to plot a smooth interface, so let's construct new X and Y sets which will be 2 times more frequent
        # than original ones::

        # X_new = np.linspace(self.X[0], self.X[-1], 2 * self.X.shape[0] - 1)
        # Y_new = np.linspace(self.Y[0], self.Y[-1], 2 * self.Y.shape[0] - 1)
        #
        # # And let's interpolate our initial grid to the new one:
        #
        # Z_new = np.zeros((X_new.shape[0], Y_new.shape[0]))
        #
        # for i in range(X_new.shape[0]):
        #     for j in range(Y_new.shape[0]):
        #
        #         Z_new[i, j] = two_dim_inter(self.polynomial, self.X, self.Y, X_new[i], Y_new[j])
        #
        # # That's all. We can plot it.
        #
        # # But we have to mesgrid X_new and Y_new first:
        #
        # YY_new, XX_new = np.meshgrid(Y_new, X_new)

        # TODO prettify using plt.show()

        if not np.any(ax):
            fig = plt.figure()
            ax = Axes3D(fig)

        self._plot_horizon_3d(self.X, self.Y, self.Z, ax=ax)
