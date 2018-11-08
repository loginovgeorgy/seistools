import numpy as np
from .utils import plot_line_3d
from scipy.optimize import minimize
from functools import partial
from .rt_coefficients import rt_coefficients

WAVECODE = {0: 'vp', 1: 'vs', 2: 'vs'}

class Ray(object):
    def __init__(self, sou, rec, vel_mod, raycode=None):
        """

        :param sou: object of type of Source
        :param rec: object of type of Receiver
        :param vel_mod: object of type of Velocity_model
        :param raycode: np.array([[+1 (down) or -1 (up), number of a layer, type of wave 0,1 or 2]])
        """

        self.source = sou
        self.receiver = rec
        self.raycode = raycode
        self.segments = self._get_init_segments(vel_mod, raycode)
        #self.reflection_coefficients, self.transmission_coefficients = self.ampl_coefficients()

    def _get_init_segments(self, vel_mod, raycode):
        if raycode==None: return self._get_forward(vel_mod)

        sou = np.array(self.source.location, ndmin=1)
        receiver = np.array(self.receiver.location, ndmin=1)
        dist = np.sqrt(((sou-receiver)**2).sum())
        segments = []
        # МНОГО, ОЧЕНЬ МНОГО if'ов
        for k, (sign, i, vtype) in enumerate(raycode):
            # raycode: [вниз +1 или вверх -1, номер слоя, тип волны, см. WAVECODE]
            first = k==0
            last = k==len(raycode)-1
            layer = vel_mod.layers[i]

            shifted_sou = (sou+dist/20)

            rec = np.array([shifted_sou[0], shifted_sou[1], layer.top.get_depth(shifted_sou[:2])])       # ЭТО ОСТАЕТСЯ ОТКРЫТЫМ ВОПРОСОМ
            end_horizon = layer.top
            start_horizon = layer.bottom
            if sign > 0 and not last:
                rec = np.array([shifted_sou[0], shifted_sou[1], layer.bottom.get_depth(shifted_sou[:2])])
                end_horizon = layer.bottom
                start_horizon = layer.top
            elif last:
                rec = receiver

            if first: start_horizon = None
            if last: end_horizon = None

            segments.append(Segment(sou, rec, layer, start_horizon, end_horizon, vtype=WAVECODE[vtype]))
            sou = rec

        return segments

    def _get_forward(self, vel_mod):
        # TODO: make more pythonic
        source = np.array(self.source.location, ndmin=1)
        receiver = np.array(self.receiver.location, ndmin=1)
        intersections = []
        distance = []
        horizons = []
        for hor in vel_mod.horizons:
            rec = hor.intersect(source, receiver)
            if len(rec) == 0:
                continue
            dist = np.sqrt(((rec - source) ** 2).sum())
            intersections.append(rec)
            distance.append(dist)
            horizons.append(hor)

        intersections = [x for _, x in sorted(zip(distance, intersections))]
        horizons = [x for _, x in sorted(zip(distance, horizons))]

        segments = []
        sou = np.array(self.source.location, ndmin=1)
        for i, rec in enumerate(intersections):
            first = (i == 0)
            layer = self._get_location_layer(sou/2 + rec/2, vel_mod)
            end_horizon = horizons[i]
            if first: start_horizon = None
            else: start_horizon = horizons[i-1]
            segments.append(Segment(sou, rec, layer, start_horizon, end_horizon))
            sou = rec

        layer = self._get_location_layer(receiver, vel_mod)
        if len(horizons)==0: start_horizon = None
        else: start_horizon = horizons[-1]
        segments.append(Segment(sou, receiver, layer, start_horizon, None))

        return segments

    def _get_trajectory(self):
        # TODO: make more "pythonic"
        trj = np.array(self.source.location, ndmin=1)
        for i, seg in enumerate(self.segments):
            trj = np.vstack((trj, seg.receiver))
        return trj

    @staticmethod
    def _get_location_layer(x, vel_mod): # ВАЖНО! ПОГРАНИЧНЫЕ СЛУЧАИ ЗАГЛУБЛЯЮТСЯ
        for l in vel_mod.layers:
            if (l.bottom.get_depth(x[:2]) > x[2] +1e-8 > l.top.get_depth(x[:2])): return l

    def travel_time(self, x=None):
        # TODO: make more pythonic and faster
        # Если даны новые координаты траектории, тогда обновляются сегменты и следовательно траектория
        if np.any(x):
            new_segments = []
            sou = self.segments[0].source
            for seg, rec in zip(self.segments, np.reshape(x, (-1, 2))):
                receiver = np.array([rec[0], rec[1], seg.end_horizon.get_depth(rec)])

                new_segments.append(Segment(sou, receiver, seg.layer, seg.start_horizon, seg.end_horizon))
                sou = receiver
            new_segments.append(Segment(sou, self.segments[-1].receiver, self.segments[-1].layer,
                                        self.segments[-1].start_horizon, self.segments[-1].end_horizon))
            self.segments = new_segments

        time = 0.0
        for segment in self.segments:
            time += segment.time
        return time

    # НУЖНО ПЕРЕПИСАТЬ
    def dtravel(self, r=None):
        amount_of_borders = len(self.segments) - 1
        dt = np.zeros((amount_of_borders, 2))             # Производные по dx & dy соответственно, на каждой пересекающей луч границе

        for ind_border in range(amount_of_borders):
            seg1 = self.segments[ind_border]              # Соседние 2 сегмента, около точки на границе
            seg2 = self.segments[ind_border + 1]

            dist1, dist2 = seg1.get_distance(), seg2.get_distance()
            vec1, vec2 = seg1.vector, seg2.vector
            gradient = seg1.end_horizon.get_gradient(seg1.receiver[:-1])

            v1 = seg1.layer.get_velocity(vec1)[seg1.vtype]
            v2 = seg2.layer.get_velocity(vec2)[seg2.vtype]

            dv1 = seg1.layer.get_dv(vec1)[seg1.vtype]
            dv2 = seg2.layer.get_dv(vec2)[seg2.vtype]

            dt[ind_border] += (seg1.receiver[:-1] - seg1.source[:-1] + (seg1.receiver[-1] - seg1.source[-1])*gradient)/dist1/v1
            dt[ind_border] -= dist1 * dv1 / (v1 ** 2)
            dt[ind_border] -= (seg2.receiver[:-1] - seg2.source[:-1] + (seg2.receiver[-1] - seg2.source[-1])*gradient)/dist2/v2
            dt[ind_border] += dist2 * dv2 / (v2 ** 2)
        return dt

    def optimize(self, method="Nelder-Mead", tol=1e-32,
                 penalty=False, projection=True, only_snells_law=False):
        # TODO: Add derivatives and Snels Law check
        x0 = self._get_trajectory()[1:-1, :2]

        if not np.any(x0):
            return self.travel_time()

        def _fun(x):                # Добавлена штрафная функция, которая представлена разницей отношений синусов к скоростям в соседних слоях
            f = ((self.snells_law(projection=projection))**2).mean()
            if not only_snells_law: f += self.travel_time(x)
            return f

        fun = self.travel_time
        if penalty: fun = _fun

        xs = minimize(fun, x0.ravel(), method=method, tol=tol)
        time = xs.fun

        return time

    def plot(self, style='trj', **kwargs):
        if style == 'trj':
            plot_line_3d(self._get_trajectory().T, **kwargs)
            return
        for s in self.segments:
            plot_line_3d(s.segment.T, **kwargs)

    def snells_law(self, projection=True):
        if not projection: return self.snells_law_by_sin()
        return self.snells_law_by_projection()

    def snells_law_by_sin(self):
        amount = len(self.segments) - 1  # Amount of boundaries

        critic = []
        snell = []

        for i in range(amount):
            r1 = self.segments[i].vector  # vector before boundary
            r2 = self.segments[i + 1].vector  # vector after boundary

            normal = self.segments[i].end_horizon.normal

            sin_r1 = np.sqrt(1 - r1.dot(normal) ** 2)  # -//- and r
            sin_r2 = np.sqrt(1 - r2.dot(normal) ** 2)  # sin of angle between normal and r_1

            v1 = self.segments[i].layer.get_velocity(r1)[self.segments[i].vtype]
            v2 = self.segments[i + 1].layer.get_velocity(r2)[self.segments[i + 1].vtype]

            if v1 < v2:
                critic.append(sin_r1 >= v1 / v2)  # checking of critic angle
            else:
                critic.append(False)

            # if np.array(critic).any():
            #     raise SnelliusError('На границе {} достигнут критический угол'.format(i + 1))

            snell.append(abs(sin_r1 / v1 - sin_r2 / v2))

        return np.array(snell)

    def snells_law_by_projection(self):
        amount = len(self.segments) - 1  # Amount of boundaries

        snell = []

        for i in range(amount):
            r1 = self.segments[i].vector  # vector before boundary
            r2 = self.segments[i + 1].vector

            normal = self.segments[i].end_horizon.normal

            v1 = self.segments[i].layer.get_velocity(r1)[self.segments[i].vtype]
            v2 = self.segments[i + 1].layer.get_velocity(r2)[self.segments[i + 1].vtype]

            r2 = r1 - np.dot(normal, r1) * normal * (1 - (v2 / v1))

            pr_r1 = np.dot(normal, r1 / v1)

            pr_r2 = np.dot(normal, r2 / v2)

            snell.append(abs(pr_r1 - pr_r2))

        return np.array(snell)

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


class Segment(object):
    def __init__(self, source, receiver, layer, start_horizon, end_horizon, vtype='vp'):
        self.source = source                # Just np.array([x0,y0,z0]), it's not Object of type Source
        self.receiver = receiver            # Just np.array([x1,y1,z1]), it's not Object of type Receivers
        self.vector = self.get_vector()
        self.vtype = vtype
        self.layer = layer
        self.time = self.get_time()
        self.start_horizon = start_horizon                          # object of type of Horizon
        self.end_horizon = end_horizon                              # object of type of Horizon

    def get_distance(self):
        return np.sqrt(((self.receiver - self.source)**2).sum())

    def get_vector(self):
        dist = self.get_distance()
        vec = (self.receiver - self.source) / (dist+1e-16)
        return vec

    def get_time(self):
        dist = self.get_distance()
        time = dist/self.layer.get_velocity(self.vector)[self.vtype]
        return time

    def _what_horizon(self, point):
        top = self.layer.top
        bottom = self.layer.bottom
        if top == None and bottom != None: return bottom
        if bottom == None and top != None: return top
        if bottom == None and top == None: return None
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
