import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from src.tprt import Receiver, Layer, Source, Ray, FlatHorizon, ISOVelocity, Velocity_model
from src.tprt.ray import SnelliusError

source = Source([40, 60, 130])
print(source)

receivers = []
for depth in [5, 30, 60, 90, 120, 150, 180, 210]:
    receivers.append(Receiver([100, 100, depth]))

print(receivers[0])

# TODO: make class for vel_mod, for now init at least one horizon upper the top receiver
# TODO: check procedure of "is_ray_intersect_boundary"


vp =        np.array([1000, 3300, 2800, 2300, 1800])  # vp
vs =        np.array([700, 2550, 2150, 1900, 1000])  # vs
velocities = np.array([ISOVelocity(vp[i], vs[i]) for i in range(len(vp))])
depth =     np.array([20, 70, 120, 180])  # depth
name =      np.array(['1', '2', '3', '4', '5'])  # name
density =   np.array([2500, 2500, 2500, 2500, 2500])  # Density
anchor =    [(0,0), (0,0), (0,0), (0,0), (0,0)]
dip =       np.array([0, 0, 15, 0, 0])  # dip
azimuth =   np.array([0, 0, 90, 0, 0])  # azimuth


vel_mod = Velocity_model(velocities, density, name, depth, anchor, dip, azimuth)

rays = [Ray(source, rec, vel_mod) for rec in receivers]

fig = plt.figure()
ax = Axes3D(fig)
for l in vel_mod.mid_layers+[vel_mod.bottom_layer]:
    l.top.plot(ax=ax)

#rays[-1].optimize()

source.plot(ax=ax, color='r', marker='p', s=50)
for i, (ray, rec) in enumerate(zip(rays, receivers)):
    ray.optimize()
    # try:
    #     ray.check_snellius(eps=1e-7)
    # except SnelliusError as e:
    #     print('Вдоль луча под номером {} до приемника {} не выполняется закон Cнеллиуса'.format(i+1, rec.location))
    rec.plot(ax=ax, color='k', marker='^', s=50)
    # keep segments colored to check correctness of procedure
    ray.plot(ax=ax)
plt.show()

n=4
print(rays[n]._get_trajectory())
#
#
# print(rays[-1].dtravel())
