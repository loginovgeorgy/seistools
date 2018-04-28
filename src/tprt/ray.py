import numpy as np
from .utils import is_ray_intersect_surf, plot_line_3d
from scipy.optimize import least_squares
from scipy.optimize import minimize
from functools import partial

from .Reflection_And_Transmission_Coefficients import Reflection_And_Transmission_Coefficients_By_Honest_Solving

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
        intersections = []
        distance = []

        for layer in vel_mod:
            is_intersect, rec = is_ray_intersect_surf(source, self._v0, self.distance, layer.top)
            if not is_intersect:
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
            vec = (rec - sou)
            vec /= np.sqrt((vec**2).sum())
            layer = self._get_location_layer(sou + vec, vel_mod)
            segments.append(Segment(sou, rec, layer.get_velocity, hor))
            sou = rec

        rec = np.array(self.receiver.location, ndmin=1)
        layer = self._get_location_layer(rec, vel_mod)
        segments.append(Segment(sou, rec, layer.get_velocity, layer.top.get_depth))

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

            new_segments.append(Segment(sou, rec, segment.velocity, segment.horizon))
            time += (distance / segment.velocity(vector)[vtype])
            sou = rec
        self.segments = new_segments
        return time

    def optimize(self, vtype='vp'):
        # TODO: Add derivatives and Snels Law check
        x0 = self._trajectory[1:-1, :2]
        fun = partial(self.travel_time, vtype=vtype)
        if not np.any(x0):
            return fun()

        xs = minimize(fun, x0.ravel())
        time = xs.fun

        return time

    def plot(self, style='trj', **kwargs):
        if style == 'trj':
            plot_line_3d(self._trajectory.T, **kwargs)
            return
        for s in self.segments:
            plot_line_3d(s.segment.T, **kwargs)

    def Reflection_And_Transmission_Coefficients(self):

        Reflection_Coefficients = np.zeros((len(self.segments),3)) #We shall write here reflection coefficients at every boundary.
        Transmission_Coefficients = np.zeros((len(self.segments),3)) #We shall write here transmission coefficients at every boundary.

        for i in range(len(self.segments)-1): #"minus one" because the last segment ends in the receiver.

            Angle_Of_Incidence_Deg = np.degrees(np.arccos(-self.segments[i].vector.dot(np.array([0,0,1]))))  # Very long formula. But the formula for coefficients
            #accepts angle in degrees only. EXACT NORMAL VECTOR TO THE SURFACE NEEDED!!!

            #Let's create an array of coefficients at the current boundary.
            #For I don't see different velocities in the "segments" I shall consider the existing "velocity" as Vp
            #and derive the Vs by dividing Vp by 2.
            New_Coefficients = Reflection_And_Transmission_Coefficients_By_Honest_Solving(self.segments[i].density, self.segments[i + 1].density,
                                                                                          self.segments[i].velocity, self.segments[i].velocity / 2,
                                                                                          self.segments[i + 1].velocity, self.segments[i + 1].velocity / 2,
                                                                                          0, Angle_Of_Incidence_Deg) #Correct Vp and Vs are needed!!!

            #Let's add new coefficients at the current boundary to the array of coefficients in the whole medium.
            for j in range(3):

                Reflection_Coefficients[i,j] = New_Coefficients[j]
                Transmission_Coefficients[i,j] = New_Coefficients[j+2]

        #Let's return two arrays of coefficients occuring on the ray's path.
        return Reflection_Coefficients, Transmission_Coefficients

class Segment(object):
    def __init__(self, source, receiver, velocity, density, horizon):
        self.source = source
        self.receiver = receiver
        self.segment = np.vstack((source, receiver))
        vec = receiver - source
        self.distance = np.sqrt((vec**2).sum())
        self.vector = vec / self.distance
        self.velocity = velocity
        self.density = density
        self.horizon = horizon

    def __repr__(self):
        return self.segment

    def __str__(self):
        return self.segment