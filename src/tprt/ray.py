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
        self.segments = self._get_segments(vel_mod)

    def _get_segments(self, vel_mod):
        sou = np.array(self.source.location, ndmin=2)
        segments = []
        p0 = 0
        for layer in vel_mod:
            # TODO: check case if receiver lies on horizon
            # TODO: is_ray_intersect_surf must be applied to segment
            is_intersect, p = is_ray_intersect_surf(self, layer.top)
            if not is_intersect:
                continue
            # TODO: 'is_ray_intersect_surf' must be applied to segment.Temporary solution: p -= p0
            # TODO: solution does not work if source is upper the receiver
            p -= p0
            p0 = p
            rec = self.predict(*p, r0=sou)
            segments.append(Segment(sou, rec, layer))
            sou = rec

        rec = self.receiver.location
        # vel_mod
        # rec_layer = vel_mod[0]
        rec_layer = self._get_location_layer(rec, vel_mod)
        segments.append(Segment(sou, rec, rec_layer))
        return segments

    @staticmethod
    def _get_location_layer(x, vel_mod):
        higher = [l for l in vel_mod if l.top.predict(x[:2]) > x[-1]]
        distance = [(l.top.predict(x[:2]) - x[-1]) for l in higher]
        layer = higher[np.array(distance).argmin()]

        return layer

    def predict(self, r, r0=None, v0=None):
        if not np.any(r0):
            r0 = self._r0
        if not np.any(v0):
            v0 = self._v0
        # azimuth = np.deg2rad(azimuth)
        # polar = np.deg2rad(polar)
        # n1 = r * np.sin(polar) * np.cos(azimuth)
        # n2 = r * np.sin(polar) * np.sin(azimuth)
        # n3 = r * np.cos(polar)
        # v = self._r0 + np.array([n1, n2, n3])
        v = r0 + v0 * r
        return v

    def plot(self, **kwargs):
        for s in self.segments:
            plot_line_3d(s.line.T, **kwargs)


class Segment(object):
    def __init__(self, sou, rec, layer):
        _v0 = rec - sou
        self.line = np.vstack([sou, rec])
        self.distance = np.sqrt((_v0 ** 2).sum())
        self._v0 = _v0 / self.distance
        self.velocity = layer.predict(_v0)



