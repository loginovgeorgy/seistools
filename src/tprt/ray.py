import numpy as np
from .utils import plot_line_3d
from scipy.optimize import least_squares
from scipy.optimize import minimize
from functools import partial
from pylab import *


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

    def _get_segments(self, vel_mod):
        # TODO: make more pythonic
        source = np.array(self.source.location, ndmin=1)
        receiver = np.array(self.receiver.location, ndmin=1)
        intersections = []
        distance = []

        for layer in vel_mod:
            rec = layer.top.surface.intersect(source, receiver)
            if len(rec)==0:
                continue
            dist = np.sqrt(((source - rec) ** 2).sum())
            intersections.append((rec, layer))
            distance.append(dist)

        intersections = [x for _, x in sorted(zip(distance, intersections))]

        segments = []
        sou = np.array(self.source.location, ndmin=1)
        for x in intersections:
            rec = x[0]
            hor = x[1].top

            vec = (rec - sou)
            vec /= np.sqrt((vec**2).sum())
            layer = self._get_location_layer(sou + vec, vel_mod)
            segments.append(Segment(sou, rec, layer.velocity, hor))
            sou = rec

        layer = self._get_location_layer(receiver, vel_mod)
        segments.append(Segment(sou, receiver, layer.velocity, layer.top))

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
            self._trajectory[i+1, -1] = seg.horizon.get_depth(loc[:2])

        time = 0
        sou = self._trajectory[0]
        # TODO prettify the way of segments update
        new_segments = []
        for segment, rec in zip(self.segments, self._trajectory[1:]):

            vector = rec - sou
            distance = np.sqrt((vector ** 2).sum())
            vector = vector / distance

            new_segments.append(Segment(sou, rec, segment.velocity, segment.horizon))
            time += (distance / segment.velocity.get_velocity(vector)[vtype])
            sou = rec
        self.segments = new_segments
        return time

    def dtravel(self, r=None, vtype='vp'):

        amount_of_borders = len(self.segments) - 1
        dt = np.zeros((amount_of_borders,2))        # there is only two derivatives of time, over dx and dy
        if not np.any(r):
            r = self._trajectory                    # if points are not given, they will be trajectory by default
        for ind_border in range(amount_of_borders):
            x = r[ind_border:ind_border+3]          # The points along the ray around given point

            vector = np.array([x[1]-x[0], x[2]-x[1]])
            distance = np.array([np.sqrt((vector[0]**2).sum()), np.sqrt((vector[1]**2).sum())])
            gradient = self.segments[ind_border].horizon.surface.get_gradient(x[1])
            vector[0], vector[1] = vector[0]/distance[0], vector[1]/distance[1]

            v = np.zeros(2)
            v[0] = self.segments[ind_border].velocity.get_velocity(vector[0])[vtype]
            v[1] = self.segments[ind_border + 1].velocity.get_velocity(vector[1])[vtype]

            dv = np.zeros((2,2))
            dv[0] = self.segments[ind_border].velocity.get_dv(vector[0])[vtype]
            dv[1] = self.segments[ind_border + 1].velocity.get_dv(vector[1])[vtype]

            dt[ind_border] += (x[1,:-1] - x[0,:-1] + (x[1,-1]-x[0,-1])*gradient)/distance[0]/v[0]
            dt[ind_border] -= distance[0] * dv[0] / (v[0] ** 2)
            dt[ind_border] -= (x[2,:-1] - x[1,:-1] + (x[2,-1]-x[1,-1])*gradient)/distance[1]/v[1]
            dt[ind_border] += distance[1] * dv[1] / (v[1] ** 2)
                                                        # I will attach a photo with the formula of calculating this derivative of time
                                                        # I request so that anybody will check it
        return dt

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

    def check_snellius(self, eps=1e-5, *args, **kwargs):

        amount = len(self.segments) - 1             # Amount of boundaries
        points = []
        for i in range(amount+1):
            points.append(self.segments[i].source)
        points.append(self.segments[-1].receiver)
        points = np.array(points, ndmin=2)          # Points of the trajectory

        normal = np.array([self.segments[k].horizon.surface.normal for k in range(amount)])     # Normal vectors of each boundary
        v = np.array([self.segments[k].velocity.get_velocity(points)['vp'] for k in range(amount+1)])

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

            if (v[i] < v[i + 1]):
                critic.append(sin_r >= v[i] / v[i + 1])     # checking of critic angle
            else:
                critic.append(False)
            if np.array(critic).any() == True:
                raise SnelliusError('На границе {} достигнут критический угол'.format(i + 1))

            snell.append(abs(sin_r / sin_r_1 - v[i] / v[i + 1]) <= eps)
            if np.array(snell).any() == False:
                raise SnelliusError('При точности {} на границе {} нарушен закон Снеллиуса'.format(eps, i + 1))

class Segment(object):
    def __init__(self, source, receiver, velocity, horizon):
        self.source = source
        self.receiver = receiver
        self.segment = np.vstack((source, receiver))
        vec = receiver - source
        self.distance = np.sqrt((vec**2).sum())
        self.vector = vec / self.distance
        self.velocity = velocity                        # Now, it will be the whole object of type of Velocity
        self.horizon = horizon                          # Now, it will be the whole object of type of Horizon
        

    def __repr__(self):
        return self.segment

    def __str__(self):
        return self.segment


class SnelliusError(Exception):
    pass;
