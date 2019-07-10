import numpy as np
import pylab as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from .units import Units
from .bicubic_interpolation import *


class Horizon:
    def get_depth(self, xy_target):
        '''

        :param xy_target: target value of (x, y); xy_target = [x_target, y_target]
        :return: value of z(x_target, y_target)
        '''
        pass

    def intersect(self, sou, rec):
        '''
        Returns the point of intersection of a straight line connecting source and receiver with the "horizon".

        :param sou: coordinates of the starting point; sou = [sou1, sou2, sou3]
        :param rec: coordinates of the ending point; sou = [sou1, sou2, sou3]
        :return: coordinates of the intersection point
        '''
        pass

    def get_gradient(self, xy_target):
        '''

        :param xy_target: target value of (x, y); xy_target = [x_target, y_target]
        :return: gradient of the "horizon" function: grad( z(x,y) ) = [dz / dx, dz / dy] |(x -> x_target, y -> y_target)
        '''
        pass

    def get_normal(self, xy_target):
        '''

        :param xy_target: target value of (x, y); xy_target = [x_target, y_target]
        :return: unit normal to the point [x, y, z(x,y)] with x = x_target, y = y_target
        '''
        pass

    def get_local_properties(self, xy_target, vec):
        '''
        1. Constructs local system of coordinates e1, e2 and n, where n is unit normal to the point [x, y, z(x,y)],
        d1 and d2 are tangent vectors to the surface at this point. It will be explained later what tangent vectors are
        used.
        2. Returns second partial derivatives of the surface calculated in local system at the point [x, y, z(x,y)] and
        the transformation matrix [d1, d2, n].
        Further explanations are presented below.

        :param xy_target: target value of (x, y); xy_target = [x_target, y_target]
        :param vec: incident vector vec = [vec1, vec2, vec3]
        :return: matrix of second partial derivatives of the surface calculated in local system at the point of
        incidence [x_target, y_target, z(x_target, y_target)] and the transformation matrix [d1, d2, n].
        '''
        pass

    @staticmethod
    def _plot_horizon_3d(x, y, z, ax=None):
        '''

        :param x: strictly ascending 1D-numerical array
        :param y: strictly ascending 1D-numerical array
        :param z: 2D-numerical array of corresponding values z = z(x, y)
        :param ax: axes3d object
        :return: surface plot of function z(x, y)
        '''

        # x and y are 1D-arrays which define the grid.
        yy, xx = np.meshgrid(y, x) # exactly in this order

        ax.plot_surface(xx, yy, z, alpha = 0.5)
        # ax.set_zlim(bottom=200, top=0)

        return


class FlatHorizon(Horizon):
    def __init__(self,
                 depth=0,
                 anchor=np.array([0, 0]),
                 dip=0,
                 azimuth=0,
                 region=np.array([[- 1000, - 1000], [1000, 1000]])):

        # Constructor accepts as arguments the following items:

        # depth = Z-coordinate of a point of the plane

        # anchor = [y, z] = X- and Y-coordinates of this point

        # azimuth = azimuth of incidence (азимут падения). It is counted clockwise from the axis X.
        # Lies in [0, 360). Measured in degrees.

        # dip = angle of incidence (угол падения). It is counted from the horizontal plane downward.
        # Lies in [0, 90). Measured in degrees.

        # x0 = np.array([x0, y0, z0]) = a point in 3D space which lies at the plane of the "horizon".

        # region - explicitly defined rectangular region in form [[x_min, y_min], [x_max, y_max]]

        self.units = Units()

        # Any plane in 3D space can be set by an equation:
        # A * x + B * y + C * z + D = 0

        # It would be convenient to use four constants A, B, C and D as fields of this class:

        self.A = np.sin(np.radians(dip)) * np.cos(np.radians(azimuth))
        self.B = np.sin(np.radians(dip)) * np.sin(np.radians(azimuth))
        self.C = - np.cos(np.radians(dip))
        self.D = - (self.A * anchor[0] + self.B * anchor[1] + self.C * depth)

        self.region = region

    def get_depth(self, xy_target):
        # "Horizons" are supposed to be non-vertical. Consequently C != 0. Thus, the depth is defined by formula:
        return - (self.A * xy_target[0] + self.B * xy_target[1] + self.D) / self.C

    def get_normal(self, xy_target):
        # Gradient of any function f(x,y,z) is perpendicular to constant level surfaces. In our case:
        # z = - A / C * x - B / C * y - D/ C => f(x, y, z) = A / C * x + B / C * y + D / C + z = 0 everywhere on the
        # plane => grad(f) = [A / C, B / C, 1] is the sought normal vector. We just have to normalize it.
        return np.array([self.A / self.C, self.B / self.C, 1]) / np.linalg.norm(np.array([self.A / self.C, self.B / self.C, 1]))

    def get_gradient(self, xy_target):
        return - np.array([self.A, self.B]) / self.C # it is obvious from the formula of the gradient of z(x,y)

    def intersect(self, sou, rec):

        # Для начала надо удостовериться, что пересечение существует. Критерий простой: если источник и приёмник
        # находятся выше или ниже границы, то пересечения нет.

        # Стоит иметь ввиду, что произвольная точка [x,y,z] находится выше "горизонта". если разница
        # z - self.get_depth(x,y)
        # принимает отрицательные значения. И наоборот, эта точка находится ниже "горизонта", если эта разница
        # положительна.
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

        # НАДО ПРОВЕРИТЬ, НЕ ВЫШЛИ ЛИ МЫ ЗА ПРЕДЕЛЫ ОБЛАСТИ
        #
        if sou[0] + s * vector[0] < self.region[0, 0] or \
                sou[0] + s * vector[0] > self.region[1, 0] or \
                sou[1] + s * vector[1] < self.region[0, 1] or \
                sou[1] + s * vector[1] > self.region[1, 1]:

                return []

        # Ну, а если не вышли, то возвращаем найденную точку:

        return sou + s * vector

    def get_local_properties(self, xy_target, vec):

        # n is unit normal to the surface at the current point. It is pointed to the medium where the outgoing ray
        # comes.

        n = self.get_normal(xy_target)
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

        return np.array([[0, 0], [0, 0]]), np.array([d1, d2, n]).T

    def plot(self, ax=None):

        x = np.array([self.region[0, 0], self.region[1, 0]])
        y = np.array([self.region[0, 1], self.region[1, 1]])

        z = np.array([[self.get_depth([x[0], y[0]]), self.get_depth([x[0], y[1]])],
                      [self.get_depth([x[1], y[0]]), self.get_depth([x[1], y[1]])]])

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
    # smooth enough to have a normal vector at any point and curvature in this point as well. Thus, corresponding
    # conditions are imposed on the method of the interpolation/approximation.
    # Here we decided to use cubic spline interpolation. Corresponding formulas allow us to construct a surface with
    # second-order smoothness.

    def __init__(self, x_set, y_set, z_set, bool_parab=0, *args):
        # Here the input arrays x_set and y_set form up a rectangular coordinate grid. The grid is supposed to be
        # regular in each direction (x_set and y_set).
        # In order to decrease influence of breaking out points, we can construct more frequent grid using
        # so-called averaged parabolic interpolation defined in the corresponding module. One can switch on
        # this feature and keep only points given by X, Y and Z by setting bool_parab equal to 1.

        if len(args) != 0:

            self.x_set = x_set
            self.y_set = y_set

            self.polynomial = args[0]

        else:

            if bool_parab == 1:

                self.x_set = np.linspace(x_set[0], x_set[-1], 2 * x_set.shape[0] - 1)
                self.y_set = np.linspace(y_set[0], y_set[-1], 2 * y_set.shape[0] - 1)
                # minus one - since we would like to save all input points.

                new_z_set = two_dim_parab_inter_surf(x_set, y_set, z_set, self.x_set, self.y_set)

                self.polynomial = two_dim_polynomial(self.x_set, self.y_set, new_z_set)

            else:

                self.x_set = x_set
                self.y_set = y_set

                self.polynomial = two_dim_polynomial(self.x_set, self.y_set, z_set)

            self.region = np.array([[self.x_set[0], self.y_set[0]],
                                    [self.x_set[- 1], self.y_set[- 1]]])

    def get_depth(self, xy_target):

        return two_dim_inter(self.polynomial, self.x_set, self.y_set, xy_target)

    def get_gradient(self, xy_target):

        # grad(z(x,y)) = [dz/dx, dz/dy]

        grad = np.array([two_dim_inter_dx(self.polynomial, self.x_set, self.y_set, xy_target),
                         two_dim_inter_dy(self.polynomial, self.x_set, self.y_set, xy_target)])

        return grad

    def get_normal(self, xy_target):

        # We shall use the following fact: z = f(x, y) => g(x, y, z) = z - f(x, y) == 0 => grad(g(x,y,z)) is orthogonal
        # to our surface.
        # grad(g(x,y,z)) = [- df/dx, - df/dy, dz/dz] = [- dz/dx, - dz/dy, 1]

        # dz/dx and dz/dy are present in self.get_gradient() vector. So:

        x_der, y_der = self.get_gradient(xy_target)

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

        s = minimize(difference, np.array([0.5]), args = (self.x_set, self.y_set, self.polynomial, sou, vector),
                     method ='SLSQP', bounds = np.array([[0, 1]])).x[0]

        # Надо проверить, не вышли ли мы за пределы сетки задания поверхности:

        if sou[0] + s * vector[0] < self.region[0, 0] or \
                sou[0] + s * vector[0] > self.region[1, 0] or \
                sou[1] + s * vector[1] < self.region[0, 1] or \
                sou[1] + s * vector[1] > self.region[1, 1]:

            return []

        # Ну, а если не вышли, то возвращаем найденную точку:

        return np.array([sou[0] + s * vector[0], sou[1] + s * vector[1], sou[2] + s * vector[2]])

    def get_local_properties(self, xy_target, vec):

        # First of all we need to calculate first and second derivatives of the surface in the global system of
        # coordinates.

        # In order to do that let's compute z(x, y_search) and z(x_search, y) arrays:

        z_x, z_y = self.get_gradient(xy_target) # dz/dx and dz/dy

        z_xx = two_dim_inter_dx_dx(self.polynomial, self.x_set, self.y_set, xy_target)# d2z/dx2
        z_yy = two_dim_inter_dy_dy(self.polynomial, self.x_set, self.y_set, xy_target)# d2z/dy2

        z_xy = two_dim_inter_dx_dy(self.polynomial, self.x_set, self.y_set, xy_target)# d2z/dxdy

        # Very well. Now we have to construct a local coordinate system. It consists of three mutually orthogonal
        # unit vectors: d1, d2 and n.

        # n is unit normal to the surface at the current point. It is pointed to the medium where the outgoing ray
        # comes.

        n = self.get_normal(xy_target)
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

        return np.array([[h11, h12], [h12, h22]]), np.array([d1, d2, n]).T

    def plot(self, ax=None):

        yy_set, xx_set = np.meshgrid(self.y_set, self.x_set)

        z_set = np.zeros((self.x_set.shape[0], self.y_set.shape[0]))

        # print(z_set[- 1, :].shape)
        # print(self.polynomial[:, - 1, 0, 0].shape)
        # print(self.y_set[:].shape)

        for i in range(4):
            for j in range(4):

                z_set[0: - 1, 0: - 1] = z_set[0: - 1, 0: - 1] +\
                                        self.polynomial[:, :, i, j] * \
                                        xx_set[0: - 1, 0: - 1]**i * \
                                        yy_set[0: - 1, 0: - 1]**j

                z_set[- 1, 0: - 1] = z_set[- 1, 0: - 1] +\
                                     self.polynomial[- 1, :, i, j] *\
                                     self.x_set[- 1]**i *\
                                     self.y_set[0: - 1]**j

                z_set[0: - 1, - 1] = z_set[0: - 1, - 1] +\
                                     self.polynomial[:, - 1, i, j] *\
                                     self.x_set[0: - 1]**i *\
                                     self.y_set[- 1]**j

                z_set[- 1, - 1] = z_set[- 1, - 1] +\
                                  self.polynomial[- 1, - 1, i, j] * \
                                  self.x_set[- 1]**i * \
                                  self.y_set[- 1]**j

        # TODO prettify using plt.show()

        if not np.any(ax):
            fig = plt.figure()
            ax = Axes3D(fig)

        self._plot_horizon_3d(self.x_set, self.y_set, z_set, ax=ax)
