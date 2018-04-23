from .horizon import Horizon
from .units import Units


class Layer(object):
    def __init__(self, velocity, horizon, kind='iso', name='flat'):
        self.kind = kind
        self.velocity = velocity
        self.top = horizon
        self.units = Units()
        self.name = name
        self.predict = None

    def get_velocity(self, x):
        return self.velocity.get_velocity(x)
