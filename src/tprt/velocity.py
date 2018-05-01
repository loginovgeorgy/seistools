import numpy as np
'''
LAYER_KIND = {
    'iso': _iso_model,
    'ani': None,
}
'''


class Velocity:
    def get_velocity(self, x):
        pass
    def get_dv(self, x):
        pass


class ISOVelocity(Velocity):
    def __init__(self, vp, vs):
        self.vp = vp
        self.vs = vs
        self.dvp = np.array([0,0])
        self.dvs = np.array([0,0])


    def get_velocity(self, x):
        return {'vp': self.vp, 'vs': self.vs}

    def get_dv(self, x):
        return {'vp': self.dvp, 'vs': self.dvs}