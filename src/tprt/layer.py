from .units import Units


class Layer(object):
    def __init__(self, velocity, density, top_horizon=None, bottom_horizon=None, kind='iso', name='flat'):
        self.kind = kind
        self.velocity = velocity
        self.density = density
        self.top = top_horizon
        self.bottom = bottom_horizon
        self.units = Units()
        self.name = name
        self.predict = None

    def get_velocity(self, x):
        return self.velocity.get_velocity(x)

    def get_dv(self, x):
        return self.velocity.get_dv(x)

    def get_density(self):
        return self.density
