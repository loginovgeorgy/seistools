import numpy as np
from .utils import plot_line_3d
from scipy.optimize import minimize
from functools import partial
from .rt_coefficients import rt_coefficients


class Ray(object):
    def __init__(self, sou, rec, vel_mod, wavecode=None):
        self.source = sou
        self.receiver = rec
        self.vel_mod = vel_mod.layers       # Нужно ли хранить всю модель?
        self.wavecode = wavecode

        self._r0 = sou.location                 # ????
        _v0 = rec.location - sou.location       # ????
        self.distance = np.sqrt((_v0**2).sum()) # ????
        self._v0 = _v0/self.distance            # ????
        self.segments = self._get_init_segments()
        self._trajectory = self._get_trajectory()

        #self.reflection_coefficients, self.transmission_coefficients = self.ampl_coefficients()

    def _get_init_segments(self):
        # TODO: make more pythonic
        source = np.array(self.source.location, ndmin=1)
        receiver = np.array(self.receiver.location, ndmin=1)
        intersections = []
        distance = []

        for layer in self.vel_mod:
            rec = layer.bottom.intersect(source, receiver)
            if len(rec) == 0:
                continue
            dist = np.sqrt(((rec - source) ** 2).sum())
            intersections.append((rec, layer))
            distance.append(dist)

        intersections = [x for _, x in sorted(zip(distance, intersections))]

        segments = []
        sou = np.array(self.source.location, ndmin=1)
        for x in intersections:
            rec = x[0]

            vec = (rec - sou)
            vec /= np.sqrt((vec**2).sum())
            layer = self._get_location_layer(sou + vec, self.vel_mod)
            segments.append(Segment(sou, rec, layer))
            sou = rec

        layer = self._get_location_layer(receiver, self.vel_mod)
        segments.append(Segment(sou, receiver, layer))

        return segments

    def _get_trajectory(self):
        # TODO: make more "pythonic"
        trj = self.segments[0].source
        for x in self.segments:
            trj = np.vstack((trj, x.receiver))
        return trj

    @staticmethod
    def _get_location_layer(x, vel_mod): # ВАЖНО! ПОГРАНИЧНЫЕ СЛУЧАИ ЗАГЛУБЛЯЮТСЯ
        for l in vel_mod:
            if (l.bottom.get_depth(x[:2]) > x[2]+1e-8 > l.top.get_depth(x[:2])): return l

    def travel_time(self, x=None, vtype='vp'):
        # TODO: make more pythonic and faster
        # Если даны новые координаты траектории, тогда обновляются сегменты и следовательно траектория
        if np.any(x):
            new_segments = []
            sou = self._get_trajectory()[0]
            for seg, rec in zip(self.segments, np.reshape(x, (-1, 2))):
                receiver = np.array([rec[0], rec[1], seg.end_horizon.get_depth(rec)])

                new_segments.append(Segment(sou, receiver, seg.layer))
                sou = receiver
            new_segments.append(Segment(sou, self.segments[-1].receiver, self.segments[-1].layer))
            self.segments = new_segments
            self._trajectory = self._get_trajectory()

        time = 0
        for segment in self.segments:
            time += (segment.distance / segment.layer.velocity.get_velocity(segment.vector)[vtype])

        return time

    def dtravel(self, r=None, vtype='vp'):

        amount_of_borders = len(self.segments) - 1
        dt = np.zeros((amount_of_borders, 2))        # there is only two derivatives of time, over dx and dy
        if not np.any(r):
            r = self._trajectory                    # if points are not given, they will be trajectory by default
        for ind_border in range(amount_of_borders):
            x = r[ind_border:ind_border+3]          # The points along the ray around given point

            vector = np.array([x[1]-x[0], x[2]-x[1]])
            distance = np.array([np.sqrt((vector[0]**2).sum()), np.sqrt((vector[1]**2).sum())])
            gradient = self.segments[ind_border].end_horizon.get_gradient(x[1])
            vector[0], vector[1] = vector[0]/distance[0], vector[1]/distance[1]

            v = np.zeros(2)
            v[0] = self.segments[ind_border].layer.velocity.get_velocity(vector[0])[vtype]
            v[1] = self.segments[ind_border + 1].layer.velocity.get_velocity(vector[1])[vtype]

            dv = np.zeros((2,2))
            dv[0] = self.segments[ind_border].layer.velocity.get_dv(vector[0])[vtype]
            dv[1] = self.segments[ind_border + 1].layer.velocity.get_dv(vector[1])[vtype]

            dt[ind_border] += (x[1,:-1] - x[0,:-1] + (x[1,-1]-x[0,-1])*gradient)/distance[0]/v[0]
            dt[ind_border] -= distance[0] * dv[0] / (v[0] ** 2)
            dt[ind_border] -= (x[2,:-1] - x[1,:-1] + (x[2,-1]-x[1,-1])*gradient)/distance[1]/v[1]
            dt[ind_border] += distance[1] * dv[1] / (v[1] ** 2)
            # I will attach a photo with the formula of calculating this derivative of time
            # I request so that anybody will check it
        return dt

    def optimize(self, vtype='vp', method="Nelder-Mead", tol=1e-32):
        # TODO: Add derivatives and Snels Law check
        x0 = self._get_trajectory()[1:-1, :2]

        fun = partial(self.travel_time, vtype=vtype)
        if not np.any(x0):
            return fun()

        xs = minimize(fun, x0.ravel(), method=method, tol=tol)
        time = xs.fun

        return time

    def plot(self, style='trj', **kwargs):
        if style == 'trj':
            plot_line_3d(self._get_trajectory().T, **kwargs)
            return
        for s in self.segments:
            plot_line_3d(s.segment.T, **kwargs)

    def ampl_coefficients(self):

        r_coefficients = np.zeros((len(self.segments) - 1, 3), dtype=complex) # сюда мы будем записывать коэффициенты отражения на каждой границе
        t_coefficients = np.zeros((len(self.segments) - 1, 3), dtype=complex) # сюда мы будем записывать коэффициенты прохождения на каждой границе
        # "минус один", т.к. последний сегмент кончается в приёмнике, а не на границе раздела

        for i in range(len(self.segments)-1):  # "минус один" - по той же причине
            angle_of_incidence_deg = np.degrees(np.arccos(abs(self.segments[i].vector.dot(self.segments[i].horizon.normal))))  # очень длинная формула. Но она оправдана:
            # np.degrees self.segments[i].vector.dot(self.segments[i].horizon_normal)- т.к. формула для расчёта коэффициентов принимает на вход угол падения в градусах
            # self.segments[i].vector.dot(self.segments[i].horizon_normal) - скалярное произведение направляющего вектора сегмента и вектора нормали к границе
            # np.arccos - т.к. оба вышеупомянутых вектора единичны. Их скалярное произведение - косинус угла падения
            # abs - чтобы избежать проблем с выбором нормали к границе; угол падения - всегда острый

            # создадим массив коэффициентов на данной границе:
            new_coefficients = rt_coefficients(self.segments[i].layer.get_density(), self.segments[i + 1].layer.get_density(),
                                               self.segments[i].layer.velocity.get_velocity(1)['vp'], self.segments[i].layer.velocity.get_velocity(1)['vs'],
                                               self.segments[i + 1].layer.velocity.get_velocity(1)['vp'], self.segments[i + 1].layer.velocity.get_velocity(1)['vs'],
                                               0, angle_of_incidence_deg) # пока что я рассматриваю падающую волну как P-волну

            # и присоединим коэффициенты из полученного массива к "глобальным" массивам коэффициентов для всего луча:
            # (индексация связана с порядком следования коэффициентов на выходи функции rt_coefficients)
            for j in range(3):

                r_coefficients[i, j] = new_coefficients[j]
                t_coefficients[i, j] = new_coefficients[j+3]

        # возвращаем массивы коэффициентов отражения и прохождения, возникших на пути луча:
        return r_coefficients, t_coefficients

    def check_snellius(self, eps=1e-5):
        amount = len(self.segments) - 1             # Amount of boundaries

        points = self._get_trajectory()             # Points of the trajectory

        normal = np.array([self.segments[k].end_horizon.normal for k in range(amount)])     # Normal vectors of each boundary
        v = np.array([self.segments[k].layer.velocity.get_velocity(self.segments[k].vector)['vp'] for k in range(amount+1)])

        critic = []
        snell = []
        for i in range(amount):
            r = points[i + 1] - points[i]           # vector before boundary
            r_1 = points[i + 2] - points[i + 1]     # vector after boundary

            r = r / np.linalg.norm(r)
            r_1 = r_1 / np.linalg.norm(r_1)
            normal_r = normal[i] / np.linalg.norm(normal[i])

            sin_r_1 = np.sqrt(1 - r_1.dot(normal_r) ** 2)   # sin of angle between normal and r_1
            sin_r = np.sqrt(1 - r.dot(normal_r) ** 2)       # -//- and r

            if v[i] < v[i + 1]:
                critic.append(sin_r >= v[i] / v[i + 1])     # checking of critic angle
            else:
                critic.append(False)

            if np.array(critic).any():
                raise SnelliusError('На границе {} достигнут критический угол'.format(i + 1))

            snell.append(abs(sin_r / sin_r_1 - v[i] / v[i + 1]) <= eps)

            if not np.array(snell).any():
                raise SnelliusError('При точности {} на границе {} нарушен закон Снеллиуса'.format(eps, i + 1))


class Segment(object):
    def __init__(self, source, receiver, layer):
        self.source = source                # Just np.array([x0,y0,z0]), it's not Object of type Source
        self.receiver = receiver            # Just np.array([x1,y1,z1]), it's not Object of type Receivers
        self.segment = np.vstack((source, receiver))
        vec = receiver - source
        self.distance = np.sqrt((vec**2).sum())
        self.vector = vec / self.distance
        self.layer = layer
        self.begin_horizon = self._what_horizon(source)                          # object of type of Horizon
        self.end_horizon = self._what_horizon(receiver)                          # object of type of Horizon

    def _what_horizon(self, point):
        top = self.layer.top
        bottom = self.layer.bottom
        if top == None: return top
        if bottom == None: return bottom
        dist_top = abs(top.get_depth(point[:2]) - point[2])
        dist_bottom = abs(bottom.get_depth(point[:2]) - point[2])
        if dist_top<dist_bottom: return top
        else: return bottom

    def __repr__(self):
        return self.segment

    def __str__(self):
        return self.segment


class SnelliusError(Exception):
    pass;
