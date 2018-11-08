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
    def __init__(self, azimuth = 0, angle = 0, x0 = np.array([0, 0, 0])):
        # Constructor accepts as arguments the following items:

        # azimuth = azimuth of incidence (азимут падения). It is counted clockwise from the axis X.
        # Lies in [0, 360). Measured in degrees.

        # angle = angle of incidence (угол падения). It is counted from the horizontal plane downward.
        # Lies in [0, 90). Measured in degrees.

        # x0 = np.array([x0, y0, z0]) = a point in 3D space which lies at the plane of the "horizon".


        self.units = Units()

        # Any plane in 3D space can be set by an equation:
        # A * x + B * y + C * z + D = 0

        # It would be convenient to use four constants A, B, C and D as fields of this class:

        self.A = np.sin(np.radians(angle)) * np.cos(np.radians(azimuth))
        self.B = np.sin(np.radians(angle)) * np.sin(np.radians(azimuth))
        self.C = np.cos(np.radians(angle))
        self.D = - (self.A * x0[0] + self.B * x0[1] + self.C * x0[2])

    def get_depth(self, x):
        # "Horizons" are supposed to be non-vertical. Consequently C != 0. So, the depth is defined by formula:
        return - (self.A * x[0] + self.B * x[1] + self.D) / self.C

    def get_normal(self, x):
        # Gradient of any function f(x,y,z) is perpendicular to constant level surfaces. In case of a plane:
        # f(x,y,z) = A * x + B * y + C * z + D = 0 everywhere on the plane => grad(f) = [A, B, C] is the
        # sought normal vector. We just have to normalize it.
        return np.array([self.A, self.B, self.C]) / np.linalg.norm(np.array([self.A, self.B, self.C]))

    def get_gradient(self, x):
        return - np.array([self.A, self.B]) / self.C # it is obvious from the formula of the gradient of z(x,y)

    def intersect(self, sou, rec):

        # First of all we have to be sure that there is an intersection. The criterion is simple:
        # if the source point and the receiver point lie both higher or lower the surface, there will be no
        # intersection point.
        # Remember that any point [x,y,z] is higher than the "horizon" if the difference
        # z - self.get_depth(x,y)
        # is positive. Vice versa, this point is lower than the "horizon" if this difference is negative.
        # So, the criterion takes simple form:

        if ( sou[2] - self.get_depth([sou[0], sou[1]]) ) * ( rec[2] - self.get_depth([rec[0], rec[1]]) ) > 0:

            return []

        # If the intersection exists, there is still a simple case when we can find it without sophisticated
        # calculations:

        if sou[0] == rec[0] and sou[1] == rec[1]: # If the source and the receiver lie on one vertical line
            # it is simple.

            return sou[0], sou[1], self.get_depth([sou[0], sou[1]])

        # But in general we have to carry out some calculations.
        # Let's build two planes which include line "sou - rec". In order to do it we have to calculate some
        # properties of this line.

        sou_rec = (np.array(rec) - np.array(sou)) / np.linalg.norm((np.array(rec) - np.array(sou))) # unit vector
        # directed from sou to rec. This vector is not vertical. Consequently, sou_rec[0] != 0 or sou_rec[1] != 0.

        # Both planes should contain whole line defined by sou_rec vector. Both planes will be defined by A, B, C and
        # D constants. As it was stated before A, B and C are components of normal vector to this plane. D can be found
        # from the fact that "sou" point belongs to the plane.
        # So, let's set the constants.

        # We'd like to have to non-vertical planes. In order to do it let's find nonzero component of the sou_rec:

        if sou_rec[0] != 0:

            # Components of the first normal:

            A1, B1, C1 = - sou_rec[2] / sou_rec[0], 0, 1

            # The second normal:

            A2, B2, C2 = - (sou_rec[1] + sou_rec[2]) / sou_rec[0], 1, 1

        else:

            # Components of the first normal:

            A1, B1, C1 = 0, - sou_rec[2] / sou_rec[1], 1

            # The second normal:

            A2, B2, C2 = 1, - (sou_rec[0] + sou_rec[2]) / sou_rec[0], 1

        # All these vectors are evidently orthogonal to sou_rec and since that they include the straight line between
        # sou and rec

        # Remaining D1 and D2 can be found from the following equation:

        D1 = - (A1 * sou[0] + B1 * sou[1] + C1 * sou[2])

        D2 = - (A2 * sou[0] + B2 * sou[1] + C2 * sou[2])

        # Now we just have to solve a system of two equations. Matrix of the system takes the following form:

        M = np.array([[A1 / C1 - self.A / self.C, B1 / C1 - self.B / self.C],
                      [A2 / C2 - self.A / self.C, B2 / C2 - self.B / self.C]])

        # Right part of this system:

        Y = np.array([- D1 / C1 + self.D / self.C, - D2 / C2 + self.D / self.C])

        # Let's solve this system and use the result in the returned vector:

        x, y = np.linalg.solve(M, Y)

        return np.array([x, y, self.get_depth([x, y])])

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

    def __init__(self, X, Y, Z, gradient=None):
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
        # По заданным координатам источника и приёмника sou, rec строит точку
        # пересечения интерполированной поверхности с прямой, соединяющей источник и приёмник

        # Для начала надо удостовериться, что пересечение существует. Критерий простой: если источник и приёмник
        # находятся выше или ниже границы, то пересечения нет. (случай сильно криволинейных границ мы не рассматриваем)

        # Стоит иметь ввиду, что произвольная точка [x,y,z] находится выше "горизонта". если разница
        # z - self.get_depth(x,y)
        # принимает положительные значения. И наоборот, эта точка находится ниже "горизонта", если эта разница
        # отрицательна.
        # Таким образом, критерий может быть записан довольно просто:

        if ( sou[2] - self.get_depth([sou[0], sou[1]]) ) * ( rec[2] - self.get_depth([rec[0], rec[1]]) ) > 0:

            return []

        # Если же пересечение сущетсвует, то всё ещё остаётся простой случай, в котором не надо будет производить
        # сложные вычисления и построения:

        if sou[0] == rec[0] and sou[1] == rec[1]: # если источник и приёмник находятся на одной вертикали, то всё
            # просто

            return sou[0], sou[1], two_dim_inter(self.X, self.Y, self.Z, self.der_X, sou[0], sou[1])

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

        minimization = minimize(difference, x0, args = (sou, self.X, self.Y, self.Z, self.der_X,
                                                        a, b, c, d), method ='SLSQP', bounds = np.array([bound]))

        # If the minimum exists but it is not real point of intersection, target function would have big absolute value.
        # So, let's check whether we have find the intersection point or not:

        if abs(minimization.fun) > 10 ** (-6): # we'll need to reconsider the condition: should it be just 10 ** (-6)
            #  or less

            print(abs(minimization.fun))

            return [] # if the target function is still big then we did not really find the intersection point

        else: # if the target function is relatively small, return the found point:

            return minimization.x[0], a * minimization.x[0] + b,\
               two_dim_inter(self.X, self.Y, self.Z, self.der_X, minimization.x[0], a * minimization.x[0] + b)


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

            main_dir_1 = np.array([]) # главные направления не определены
            main_dir_2 = np.array([]) # главные направления не определены

            # В каком направлении брать кривизны, совершенно неважно. Возьмём в направлении координатных осей.

            main_curv_1 = D  / E # кривизна при dy = 0

            main_curv_2 = D_2 / G # кривизна при dx = 0

            # Отметим, что на плоскости D = D_2 = 0, и поэтому кривизны будут равны нулю.

            return main_curv_1, main_curv_2, main_dir_1, main_dir_2

        # Если же мы всё-таки не на плоскости, то надо считать главные направления.

        # Однако, стоит отметить, что если эти направления совпадают с координатными осями, то наше уравнение
        # некорректно (т.к. на прямой x = 0 всюду dy/dx = infinity). В числах бесконечность вряд ли получится, но
        # зашкаливающе больше значение будет. Чтобы это не привело к вычислительным ошибкам, просто будем считать, что
        # если коэффициент при h**2 мал, то главные направления соответствуют координатным осям:

        if abs(G* D_1 - F * D_2) < 1e-15:

            main_dir_1 = np.array([1, 0, z_x]) / np.linalg.norm(np.array([1, 0, z_x])) # касательный к поверхности единичный
            # вектор в направлении dr / dx
            main_dir_2 = np.array([0, 1, z_y]) / np.linalg.norm(np.array([1, 0, z_y])) # касательный к поверхности единичный
            # вектор в направлении dr / dy

            main_curv_1 = D  / E # кривизна при dy = 0
            main_curv_2 = D_2 / G # кривизна при dx = 0

            return main_curv_1, main_curv_2, main_dir_1, main_dir_2

        # Если же и эту проверку не прошли, то считаем честно:

        h1, h2 = np.roots([G* D_1 - F * D_2, G * D - E * D_2, F * D - E * D_1]) # корни нашего уравнения

        main_dir_1 = (np.array([1, 0, z_x]) /np.linalg.norm(np.array([1, 0, z_x])) + \
                      h1 * np.array([0, 1, z_y]) /np.linalg.norm(np.array([0, 1, z_y])))
        main_dir_1 = main_dir_1 / np.linalg.norm(main_dir_1)
        # касательный к поверхности единичный вектор в направлении dr / dx + h1 * dr / dy

        main_dir_2 = (np.array([1, 0, z_x]) /np.linalg.norm(np.array([1, 0, z_x])) + \
                      h2 * np.array([0, 1, z_y]) /np.linalg.norm(np.array([0, 1, z_y])))
        main_dir_2 = main_dir_2 / np.linalg.norm(main_dir_2)
        # касательный к поверхности единичный вектор в направлении dr / dx + h2 * dr / dy

        main_curv_1 = (D  + 2 * D_1 * h1 + D_2 * h1**2) / (E + 2 * F * h1 + G * h1**2)
        # кривизна в первом главном направлении

        main_curv_2 = (D  + 2 * D_1 * h2 + D_2 * h2**2) / (E + 2 * F * h2 + G * h2**2)

        return main_curv_1, main_curv_2, main_dir_1, main_dir_2

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
