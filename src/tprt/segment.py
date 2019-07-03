import numpy as np


class Segment(object):
    def __init__(self, source, receiver, layer, start_horizon, end_horizon, vtype):
        # Changeable
        self.source         =   source                # np.array([x0,y0,z0]), it's not Object of type Source
        self.receiver       =   receiver              # np.array([x1,y1,z1]), it's not Object of type Receiver

        # Unchangeable
        self.vtype          =   vtype
        self.layer          =   layer
        self.start_horizon  =   start_horizon                            # the object of the Horizon type
        self.end_horizon    =   end_horizon                              # the object of the Horizon type

    def get_distance(self):
        return np.linalg.norm(self.receiver - self.source) + 1e-16

    def get_vector(self):
        return (self.receiver - self.source) / self.get_distance()

    def get_traveltime(self):
        return self.get_distance() / self.layer.get_velocity(self.get_vector())[self.vtype]

    def __repr__(self):
        return np.array([self.source, self.receiver])

    def __str__(self):
        return np.array([self.source, self.receiver])