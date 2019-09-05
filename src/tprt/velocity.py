import numpy as np


class Velocity:
    def get_velocity(self, x):
        pass

    def get_dv(self, x):
        pass


class ISOVelocity(Velocity):
    def __init__(self, vp, vs):
        self.vp = vp
        self.vs = vs
        self.dvp = np.array([0, 0])
        self.dvs = np.array([0, 0])

    def get_velocity(self, x=None):
        return {'vp': self.vp, 'vs': self.vs}

    def get_dv(self, x=None):
        return {'vp': self.dvp, 'vs': self.dvs}