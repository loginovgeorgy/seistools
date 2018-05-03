import numpy as np

class Density():
    def get_density(self, x):
        pass
    def get_gradient(self, x):
        pass

class ISODensity(Density):
    def __init__(self, rho, gradient=np.array([0,0,0])):
        self.rho = rho
        self.gradient = gradient

    def get_density(self, x):
        return self.rho

    def get_gradient(self, x):
        return self.gradient