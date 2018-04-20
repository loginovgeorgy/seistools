import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import sys

from src.tprt import Receiver, Layer, Source, Units, Horizon, Ray, get_ray_xyz, get_ray_xy

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
        [1400, 2500, 2000, 2100], # vp
        [1000, 2000, 700, 1300], # vs
        [80, 250, 120, 250], # depth
        ['1', '2', '3', '4'], #name
        [0, 0, 0, 0], #dip
        [0, 0]  #azimuth
):
    vel_mod.append(Layer(vp=vp, vs=vs, depth=depth, dip=dip, azimuth=az, name=name))

rays = [Ray(source, rec, vel_mod) for rec in receivers]

fig = plt.figure()
ax = Axes3D(fig)
for l in vel_mod:
    l.top.plot(ax=ax)

# R = []
# for i in range(len(rays[-1].segments)):
#     R.append(rays[-1].segments[i].source)
# R.append(rays[-1].segments[-1].receiver)
# R = np.array(R)
# print(R)

source.plot(ax=ax, color='r', marker='p', s=50)
for i, (ray, rec) in enumerate(zip(rays, receivers)):
    #get_ray_xy(ray)
    ray.optimize()
    try:
        ray.check_snellius(eps=1e-5)
    except:
        print('Вдоль луча под номером {} до приемника {} не выполняется закон Cнеллиуса'.format(i+1, rec.location))
    rec.plot(ax=ax, color='k', marker='^', s=50)
    # keep segments colored to check correctness of procedure
    ray.plot(ax=ax,style='s')
plt.show()


R = []
for i in range(len(rays[-1].segments)):
    R.append(rays[-1].segments[i].source)
R.append(rays[-1].segments[-1].receiver)
R = np.array(R)
print(R)

