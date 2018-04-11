import numpy as np
from .utils import is_ray_intersect_surf, plot_line_3d


class Ray(object):
    def __init__(self, sou, rec, vel_mod):
        self.source = sou
        self.receiver = rec
        self._r0 = sou.location
        _v0 = rec.location - sou.location
        self.distance = np.sqrt((_v0**2).sum())
        self._v0 = _v0/self.distance
        self.source.layer = self._get_location_layer(self.receiver.location, vel_mod)
        self.receiver.layer = self._get_location_layer(self.receiver.location, vel_mod)
        self.segments = self._get_segments(vel_mod)

    def _get_segments(self, vel_mod):
        sou = np.array(self.source.location, ndmin=1)
        # segments = [Segment(sou,  self._v0, self.source.layer)]
        segments = []
        distance = []
        for layer in vel_mod:
            is_intersect, rec = is_ray_intersect_surf(sou, self._v0, self.distance, layer.top)

            if not is_intersect:
                continue
            # TODO: 'is_ray_intersect_surf' must be applied to segment.Temporary solution: p -= p0
            # TODO: solution does not work if source is upper the receiver

            dist = np.sqrt(((sou - rec) ** 2).sum())
            segments.append(Segment(rec,  self._v0, layer))
            distance.append(dist)

        segments = [x for _, x in sorted(zip(distance, segments))]

        return segments

    @staticmethod
    def _get_location_layer(x, vel_mod):
        higher = [l for l in vel_mod if l.top.predict(x[:2]) > x[-1]]
        distance = [(l.top.predict(x[:2]) - x[-1]) for l in higher]
        layer = higher[np.array(distance).argmin()]

        return layer

    def plot(self, **kwargs):
        sou = np.array(self.source.location, ndmin=2)
        for s in self.segments:
            rec = np.array(s.source, ndmin=2)
            x = np.vstack((sou, rec))
            sou = rec
            plot_line_3d(x.T, **kwargs)
        x = np.vstack((sou, np.array(self.receiver.location, ndmin=2)))
        plot_line_3d(x.T, **kwargs)


class Segment(object):
    def __init__(self, source, vec, layer):
        self.source = source
        self.distance = None
        self.vec = vec
        self.velocity = layer.predict(vec)



