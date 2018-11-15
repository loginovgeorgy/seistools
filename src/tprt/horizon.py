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

    def get_curvature(self, x): # returns main curvatures and main directions for the point [x, y, z(x,y)]
        # (x in the arguments is x = [x, y])
        pass

    @staticmethod
    def _plot_horizon_3d(x, z, ax=None):
        ax.plot_trisurf(x[:, 0], x[:, 1], np.squeeze(z))


class FlatHorizon(Horizon):
    def __init__(self, depth = 0, anchor = np.array([0, 0]), azimuth = 0, dip = 0):

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
        self.C = np.cos(np.radians(dip))
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

    def get_curvature(self, x):

        return 0, 0, np.array([]), np.array([]) # normal curvature of a plane in any direction is zero and
        # the main directions are not defined.

    def plot(self, x=None, extent=(0, 100, 0, 100), ns=2, ax=None):
        if not np.any(x):
            _x, _y = np.meshgrid(
                np.linspace(extent[0], extent[1], ns),
                np.linspace(extent[2], extent[3], ns)
            )
            x = np.vstack((_x.ravel(), _y.ravel())).T


        z = []

        for i in range(x.shape[0]):

            z.append(self.get_depth([x[i,0], x[i,1]]))

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

    def __init__(self, X, Y, Z):
        # Here the input arrays X and Y form up a rectangular coordinate grid. The grid is supposed to be regular for
        # each direction (X and Y).
        # In order to decrease influence of breaking out points, we shall construct more frequent grid using
        # so-called averaged parabolic interpolation defined in the corresponding module.

        self.X = np.linspace(X[0], X[-1], 2 * X.shape[0] - 1)
        self.Y = np.linspace(X[0], X[-1], 2 * X.shape[0] - 1)
        # minus one - since we would like to save all input points.

        self.Z = two_dim_parab_inter_surf(X, Y, Z, self.X, self.Y)

        # In addition it would be convenient to keep an array of partial derivatives along X axis:

        self.der_X = np.zeros(self.Z.shape)

        step_X = X[1] - X[0] # step along X axis

        for j in range(Y.shape[0]):

            self.der_X[:, j] = derivatives(self.Z[:, j], step_X)

    def get_depth(self, x):

        return two_dim_inter(self.X, self.Y, self.Z, self.der_X, x[0], x[1])

    def get_gradient(self, x):

        # grad(z(x,y)) = [dz/dx, dz/dy]

        # We want to find derivatives in a certain point on the surface. So, in order to do so we
        # have to find vectors [z(x_i, y)] and [z(x, y_i)]:

        x_vector = np.zeros(self.X.shape[0]) # [z(x_i, y)]
        y_vector = np.zeros(self.Y.shape[0]) # [z(x, y_i)]

        # We'll need to know step along X and Y axes:

        step_X = self.X[1] - self.X[0]
        step_Y = self.Y[1] - self.Y[0]

        for i in range(self.X.shape[0]):

            x_vector[i] = one_dim_inter(self.Y, self.Z[i, :], derivatives(self.Z[i, :], step_Y), x[1]) # exactly in this
            # way: we are interpolating particular "cross-section" of the surface along axis Y

        for j in range(self.Y.shape[0]):

            y_vector[j] = one_dim_inter(self.X, self.Z[:, j], self.der_X[:, j], x[0]) # exactly in this way: we are
            # interpolating particular "cross-section" of the surface along axis X

        # Now we are ready to compute dz/dx and dz/dy at the point (x, y).

        x_der = one_dim_inter_ddx(self.X, x_vector, derivatives(x_vector, step_X), x[0])

        y_der = one_dim_inter_ddx(self.Y, y_vector, derivatives(y_vector, step_Y), x[1])

        # Finally:

        grad = np.array([x_der, y_der])

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

        s = minimize(difference, np.array([0.5]), args = (self.X, self.Y, self.Z, self.der_X, sou, vector),
                     method ='SLSQP', bounds = np.array([[0, 1]])).x[0]

        # Надо проверить, не вышли ли мы за пределы сетки задания поверхности:

        if sou[0] + s * vector[0] < self.X[0] or\
                sou[0] + s * vector[0] > self.X[-1] or \
                sou[1] + s * vector[1] < self.Y[0] or\
                sou[1] + s * vector[1] > self.Y[-1]:

            return []

        # Ну, а если не вышли, то возвращаем найденную точку:

        return np.array([sou[0] + s * vector[0], sou[1] + s * vector[1], sou[2] + s * vector[2]])

    def get_curvature(self, x):
        # Теория здесь: http://ium.mccme.ru/postscript/s17/DG2017.pdf
        # и здесь: http://info.sernam.ru/book_elv.php?id=140

        # Будем считать главные кривизны поверхности.

        # Для этой цели нам надо найти коэффициенты первой и второй квадратичной форм поверхности в данной точке:

        # E = dr/dx * dr/dx = 1 + dz/dx * dz/dx
        # G = dr/dy * dr/dy = 1 + dz/dy * dz/dy
        # F = dr/dx * dr/dy = dz/dx * dz/dy

        # D = n * d2r/dx2 = n[2] * d2z/dx2 (также встречается обозначение L)
        # D_2 = n * d2r/dy2 = n[2] * d2z/dy2 (также встречается обозначение N)
        # D_1 = n * d2r/dxdy = n[2] * d2z/dxdy (также встречается обозначение M)

        # n - единичная нормаль к поверхности в точке

        # Нужно взять производные. Для этого надо задать массивы z(x, y_search) и z(x_search, y):

        step_X = self.X[1] - self.X[0] # шаг по X
        step_Y = self.Y[1] - self.Y[0] # шаг по Y

        ddx = np.zeros(self.X.shape[0]) # сюда прямо сейчас запишем z(x, y_search)
        dx_dy = np.zeros(self.X.shape[0]) # сюда прямо сейчас запишем dz(x, y_search)/dx
        ddy = np.zeros(self.Y.shape[0]) # сюда прямо сейчас запишем z(x_search, y):

        for q in range(self.X.shape[0]):

            ddx[q] = one_dim_inter(self.Y, self.Z[q, :], derivatives(self.Z[q, :], step_Y), x[1])

            dx_dy[q] = one_dim_inter_ddx(self.Y, self.Z[q, :], derivatives(self.Z[q, :], step_Y), x[1])

        for q in range(self.Y.shape[0]):

            ddy[q] = one_dim_inter(self.X, self.Z[:, q], self.der_X[:, q], x[0])

        # Находим векторы кривизны (их ненулевые компоненты):

        z_x = one_dim_inter_ddx(self.X, ddx, derivatives(ddx, step_X), x[0]) # dz/dx
        z_xx = one_dim_inter_ddx2(self.X, ddx, derivatives(ddx, step_X), x[0]) # d2z/dx2

        z_y = one_dim_inter_ddx(self.Y, ddy, derivatives(ddy, step_Y), x[1]) # dz/dy
        z_yy = one_dim_inter_ddx2(self.Y, ddy, derivatives(ddy, step_Y), x[1]) # d2z/dy2

        z_xy = one_dim_inter_ddx(self.X, dx_dy, derivatives(dx_dy, step_X), x[0]) # d2z/dxdy

        # Коэффициенты квадратичных форм поверхности:

        E = 1 + z_x * z_x
        F = z_x * z_y
        G = 1 + z_y * z_y

        # D = self.get_normal(self, x)[2] * z_xx
        # D_1 = self.get_normal(self, x)[2] * z_xy
        # D_2 = self.get_normal(self, x)[2] * z_yy - можно и так написать, но чтобы не вычислять несколько раз нормаль,
        # напишем в одну строчку:

        D, D_1, D_2 = self.get_normal(self, x)[2] * np.array([z_xx, z_xy, z_yy])

        # Главные кривизны будт отношениями второй и первой квадратичных форм при определённых соотношениях dy / dx:

        # K = (D * dx**2 + 2 D_1 * dx * dy + D_2 * dy**2) / (E * dx**2 + 2 F * dx * dy + G * dy**2)

        # Искомые соотношения dy / dx будут задавать направления в плоскости XY. Эти направления называются главными.
        # Они перпендикулярны друг другу. Найдём их.

        # Обозначим dy / dx = h. Относительно h можно составить квадратное уравнение, корни которого и зададут главные
        # направления:

        # (GD_1 - FD_2) * h **2 + (GD - ED_2) * h + (FD - ED_1) = 0

        # В случае когда наше уравнение становится тождеством, главные направления не определены. Так может получиться,
        # например, на полюсе сферы или на плоскости. Поэтому сначала проверка:

        if abs(G* D_1 - F * D_2) < 1e-15 and abs(G * D - E * D_2) < 1e-15  and abs(F * D - E * D_1) < 1e-15:

            # В каком направлении брать кривизны, совершенно неважно. Возьмём в направлении координатных осей.

            main_curv = np.array([D  / E, D_2 / G]) # массив главных кривизн

            main_dir_1 = np.array([]) # главные направления не определены
            main_dir_2 = np.array([]) # главные направления не определены

            # Отметим, что на плоскости D = D_2 = 0, и поэтому кривизны будут равны нулю.

            return np.min(main_curv), np.max(main_curv), main_dir_1, main_dir_2

        # Если же мы всё-таки не на плоскости, то надо считать главные направления.

        # Однако, стоит отметить, что если эти направления совпадают с координатными осями, то наше уравнение
        # некорректно (т.к. на прямой x = 0 всюду dy/dx = infinity). В числах бесконечность вряд ли получится, но
        # зашкаливающе больше значение будет. Чтобы это не привело к вычислительным ошибкам, просто будем считать, что
        # если коэффициент при h**2 мал, то главные направления соответствуют координатным осям:

        if abs(G* D_1 - F * D_2) < 1e-15:

            main_curv = np.array([D  / E, D_2 / G]) # массив главных кривизн

            main_dir_1 = np.array([0, 1, z_x]) / np.sqrt(E) # касательный к поверхности единичный
            # вектор в направлении dr / dx

            main_dir_2 = np.array([1, 0, z_y])  / np.sqrt(G) # касательный к поверхности единичный
            # вектор в направлении dr / dy

            # Эти векторы уже единичны, в чём легко убедиться, вспомнив определения E и G.

            return np.min(main_curv), np.max(main_curv), \
                   np.array([main_dir_1, main_dir_2])[np.argmin(main_curv)],\
                   np.array([main_dir_1, main_dir_2])[np.argmax(main_curv)]

        # Если же и эту проверку не прошли, то считаем честно:

        h = np.roots([G* D_1 - F * D_2, G * D - E * D_2, F * D - E * D_1])
        # массив корней нашего уравнения

        # Зададим кривизны:

        main_curv = (D  + 2 * D_1 * h + D_2 * h**2) / (E + 2 * F * h + G * h**2)
        # массив главных кривизн

        # Зададим главные направления. Первое направление будет соответствовать меньшей кривизне, второе - большей.
        # По построению, это будут производные радиус-вектора в нушном направлении по
        # элементу длины, т.е. они заведомо будут единичными.

        main_dir_1 = ( np.array([1, 0, z_x]) + h[np.argmin(main_curv)] * np.array([0, 1, z_y]) ) / \
                     np.sqrt( E + 2 * F * h[np.argmin(main_curv)] + G * h[np.argmin(main_curv)]**2 )
        # касательный к поверхности единичный вектор в направлении dr / dx + np.max(h) * dr / dy

        main_dir_2 = ( np.array([1, 0, z_x]) + h[np.argmax(main_curv)] * np.array([0, 1, z_y]) ) / \
                     np.sqrt( E + 2 * F * h[np.argmax(main_curv)] + G * h[np.argmax(main_curv)]**2 )
        # касательный к поверхности единичный вектор в направлении dr / dx + np.min(h) * dr / dy

        # возвращаем величины в том же порядке:

        return np.min(main_curv), np.max(main_curv), main_dir_1, main_dir_2

    def plot(self, ax=None):

        # We'd like to plot a smooth interface, so let's construct new X and Y sets which will be 2 times more frequent
        # than original ones::

        X_new = np.linspace(self.X[0], self.X[-1], 2 * self.X.shape[0])
        Y_new = np.linspace(self.Y[0], self.Y[-1], 2 * self.Y.shape[0])

        # And let's interpolate our initial grid to the new one:

        Z_new = two_dim_inter_surf(self.X, self.Y, self.Z, self.der_X, X_new, Y_new)

        # That's all. We can plot it.

        # But we have to mesgrid X_new and Y_new first:

        YY_new, XX_new = np.meshgrid(Y_new, X_new)

        # TODO prettify using plt.show()

        if not np.any(ax):
            fig = plt.figure()
            ax = Axes3D(fig)

        self._plot_horizon_3d(np.vstack((XX_new.ravel(), YY_new.ravel())).T, Z_new.ravel(), ax=ax)
