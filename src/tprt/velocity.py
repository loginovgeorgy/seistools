class Velocity:
    def get_velocity(self, x):
        pass


class ISOVelocity(Velocity):
    def __init__(self, vp, vs):
        self.vp = vp
        self.vs = vs

    def get_velocity(self, x):
        return {'vp': self.vp, 'vs': self.vs}
