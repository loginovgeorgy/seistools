import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import sys

from src.tprt import Receiver, Layer, Source, Units, Horizon, Ray

source = Source([0, 0, 0])
print(source)

receivers = []
for depth in [0, 30, 60, 90, 120, 150, 180, 210]:
    receivers.append(Receiver([100, 100, depth]))

print(receivers[0])

# TODO: make class for vel_mod, for now init at least one horizon upper the top receiver
# TODO: check procedure of "is_ray_intersect_boundary"
vel_mod = []
for vp, vs, depth, name, dip, az in zip(
        [1000, 3500, 1500, 2500], # vp
        [2700, 2500, 2100, 1000], # vs
        [20, 50, 120, 250], # depth
        ['1', '2', '3', '4'], #name
        [0, 15, 45, 0], #dip
        [0, 5, 15, 0]  #azimuth
):
    vel_mod.append(Layer(vp=vp, vs=vs, depth=depth, dip=dip, azimuth=az, name=name))

#vel_mod.append(Layer(vp=vp, vs=vs, depth=240, dip=0, azimuth=0, name=name))
rays = [Ray(source, rec, vel_mod) for rec in receivers]

fig = plt.figure()
ax = Axes3D(fig)
for l in vel_mod:
    l.top.plot(ax=ax)


source.plot(ax=ax, color='r', marker='p', s=50)
for ray, rec in zip(rays, receivers):
    ray.optimize()
    rec.plot(ax=ax, color='k', marker='^', s=50)
    # keep segments colored to check correctness of procedure
    ray.plot(ax=ax)
plt.show()

for i in range(len(rays[-1].segments)):
    print(rays[-1].segments[i].source)

