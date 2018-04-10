import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import sys

from .tprt import Receiver, Layer, Source, Units, Horizon, Ray

source = Source([0, 0, 0])
print(Source)

receivers = []
for depth in [0, 30, 60, 90, 120, 150, 180, 210]:
    receivers.append(Receiver([100,100,depth]))

print(receivers[0])


vel_mod = []
for vp, vs, depth, name in zip([3500, 3300, 2800], [2700, 2550, 2150], [100, 150, 200], ['1', '2', '3']):
    vel_mod.append(Layer(vp=vp, vs=vs, depth=depth, dip=0, azimuth=30, name=name))


rec = receivers[4]
ray = Ray(source, rec, vel_mod)

fig = plt.figure()
ax = Axes3D(fig)
for l in vel_mod:
    l.top.plot(ax=ax)

fig = plt.figure()
ax = Axes3D(fig)
rec.plot(ax=ax, color='k', marker='^', s=50)
source.plot(ax=ax, color='r', marker='p', s=50)
ray.plot(ax=ax)
