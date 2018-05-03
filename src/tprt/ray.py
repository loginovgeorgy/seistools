import numpy as np
from .utils import plot_line_3d
from scipy.optimize import minimize
from functools import partial
from .rt_coefficients import rt_coefficients


class Ray(object):
    def __init__(self, sou, rec, vel_mod):
        self.source = sou
        self.receiver = rec
        self._r0 = sou.location
        _v0 = rec.location - sou.location
        self.distance = np.sqrt((_v0**2).sum())
        self._v0 = _v0/self.distance
        self.segments = self._get_segments(vel_mod)
        self._trajectory = self._get_trajectory()
        self.reflection_coefficients, self.transmission_coefficients = self.ampl_coefficients()

    def _get_segments(self, vel_mod):
        # TODO: make more pythonic
        source = np.array(self.source.location, ndmin=1)
        receiver = np.array(self.receiver.location, ndmin=1)
        intersections = []
        distance = []

        for layer in vel_mod:
            rec = layer.top.surface.intersect(source, receiver)
            if len(rec) == 0:
                continue
            dist = np.sqrt(((source - rec) ** 2).sum())
            intersections.append((rec, layer))
            distance.append(dist)

        intersections = [x for _, x in sorted(zip(distance, intersections))]

        segments = []
        sou = np.array(self.source.location, ndmin=1)
        for x in intersections:
            rec = x[0]
            hor = x[1].top.get_depth
            hor_normal = x[1].top.surface.normal
            vec = (rec - sou)
            vec /= np.sqrt((vec**2).sum())
            layer = self._get_location_layer(sou + vec, vel_mod)
            segments.append(Segment(sou, rec, layer.get_velocity, layer, hor, hor_normal))
            sou = rec

        rec = np.array(self.receiver.location, ndmin=1)
        layer = self._get_location_layer(rec, vel_mod)
        segments.append(Segment(sou, rec, layer.get_velocity, layer, layer.top.get_depth, layer.top.surface.normal))

        return segments

    def _get_trajectory(self):
        # TODO: make more "pythonic"
        trj = self.segments[0].source
        for x in self.segments:
            trj = np.vstack((trj, x.segment[-1]))
        return trj

    @staticmethod
    def _get_location_layer(x, vel_mod):
        higher = [l for l in vel_mod if l.top.get_depth(x[:2]) > x[-1]]
        distance = [(l.top.get_depth(x[:2]) - x[-1]) for l in higher]
        layer = higher[np.array(distance).argmin()]

        return layer

    def travel_time(self, x=None, vtype='vp'):
        # TODO make more pythonic and faster
        if np.any(x):
            self._trajectory[1:-1, :2] = np.reshape(x, (-1, 2))

        for i, (loc, seg) in enumerate(zip(self._trajectory[1:-1], self.segments)):
            self._trajectory[i+1, -1] = seg.horizon(loc[:2])

        time = 0
        sou = self._trajectory[0]
        # TODO prettify the way of segments update
        new_segments = []
        for segment, rec in zip(self.segments, self._trajectory[1:]):

            vector = rec - sou
            distance = np.sqrt((vector ** 2).sum())
            vector = vector / distance

            new_segments.append(Segment(sou, rec, segment.velocity, segment, segment.horizon, segment.horizon_normal))
            # here I pass segment itself as an argument because in the constructor of the "Segment" class we call
            # .density field of the corresponding argument. segment (which is passed) possesses such field.
            time += (distance / segment.velocity(vector)[vtype])
            sou = rec
        self.segments = new_segments
        return time

    def optimize(self, vtype='vp', method="Nelder-Mead", tol=1e-32):
        # TODO: Add derivatives and Snels Law check
        x0 = self._trajectory[1:-1, :2]
        fun = partial(self.travel_time, vtype=vtype)
        if not np.any(x0):
            return fun()

        xs = minimize(fun, x0.ravel(), method=method, tol=tol)
        time = xs.fun

        return time

    def plot(self, style='trj', **kwargs):
        if style == 'trj':
            plot_line_3d(self._trajectory.T, **kwargs)
            return
        for s in self.segments:
            plot_line_3d(s.segment.T, **kwargs)

    def ampl_coefficients(self):

        r_coefficients = np.zeros((len(self.segments) - 1, 3), dtype=complex) # сюда мы будем записывать коэффициенты отражения на каждой границе
        t_coefficients = np.zeros((len(self.segments) - 1, 3), dtype=complex) # сюда мы будем записывать коэффициенты прохождения на каждой границе
        # "минус один", т.к. последний сегмент кончается в приёмнике, а не на границе раздела

        for i in range(len(self.segments)-1):  # "минус один" - по той же причине
            angle_of_incidence_deg = np.degrees(np.arccos(abs(self.segments[i].vector.dot(self.segments[i].horizon_normal))))  # очень длинная формула. Но она оправдана:
            # np.degrees self.segments[i].vector.dot(self.segments[i].horizon_normal)- т.к. формула для расчёта коэффициентов принимает на вход угол падения в градусах
            # self.segments[i].vector.dot(self.segments[i].horizon_normal) - скалярное произведение направляющего вектора сегмента и вектора нормали к границе
            # np.arccos - т.к. оба вышеупомянутых вектора единичны. Их скалярное произведение - косинус угла падения
            # abs - чтобы избежать проблем с выбором нормали к границе; угол падения - всегда острый

            # создадим массив коэффициентов на данной границе:
            new_coefficients = rt_coefficients(self.segments[i].density, self.segments[i + 1].density,
                                               self.segments[i].velocity(1)['vp'], self.segments[i].velocity(1)['vs'],
                                               self.segments[i + 1].velocity(1)['vp'], self.segments[i + 1].velocity(1)['vs'],
                                               0, angle_of_incidence_deg) # пока что я рассматриваю падающую волну как P-волну

            # и присоединим коэффициенты из полученного массива к "глобальным" массивам коэффициентов для всего луча:
            # (индексация связана с порядком следования коэффициентов на выходи функции rt_coefficients)
            for j in range(3):

                r_coefficients[i, j] = new_coefficients[j]
                t_coefficients[i, j] = new_coefficients[j+3]

        # возвращаем массивы коэффициентов отражения и прохождения, возникших на пути луча:
        return r_coefficients, t_coefficients

    def check_snellius(self, eps=1e-5):
        amount = len(self.segments) - 1
        points = []
        for i in range(amount+1):
            points.append(self.segments[i].source)
        points.append(self.segments[-1].receiver)
        points = np.array(points, ndmin=2)

        normal = np.array([self.segments[k].horizon_normal for k in range(amount)])
        v = np.array([self.segments[k].velocity(points)['vp'] for k in range(amount+1)])

        critic = []
        snell = []
        for i in range(amount):
            r = points[i + 1] - points[i]
            r_1 = points[i + 2] - points[i + 1]

            r = r / np.linalg.norm(r)
            r_1 = r_1 / np.linalg.norm(r_1)
            normal_r = normal[i] / np.linalg.norm(normal[i])

            sin_r_1 = np.sqrt(1 - r_1.dot(normal_r) ** 2)
            sin_r = np.sqrt(1 - r.dot(normal_r) ** 2)

            if v[i] < v[i + 1]:
                critic.append(sin_r >= v[i] / v[i + 1])
            else:
                critic.append(False)
            if np.array(critic).any():
                raise SnelliusError('На границе {} достигнут критический угол'.format(i + 1))
            snell.append(abs(sin_r / sin_r_1 - v[i] / v[i + 1]) <= eps)
            if not np.array(snell).any():
                raise SnelliusError('При точности {} на границе {} нарушен закон Снеллиуса'.format(eps, i + 1))


class Segment(object):
    def __init__(self, source, receiver, velocity, layer, horizon, horizon_normal):
        self.source = source
        self.receiver = receiver
        self.segment = np.vstack((source, receiver))
        vec = receiver - source
        self.distance = np.sqrt((vec**2).sum())
        self.vector = vec / self.distance
        self.velocity = velocity
        self.density = layer.density
        self.horizon = horizon
        self.horizon_normal = horizon_normal

    def __repr__(self):
        return self.segment

    def __str__(self):
        return self.segment


class SnelliusError(Exception):
    pass;
