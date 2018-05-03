import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from src.tprt import Receiver, Layer, Source, Horizon, Ray, FlatSurface, ISOVelocity
from src.tprt.ray import SnelliusError

source = Source([0, 0, 0])
print(source)

receivers = []
for depth in [0, 30, 60, 90, 120, 150, 180, 210]:
    receivers.append(Receiver([100, 100, depth]))

print(receivers[0])

# TODO: make class for vel_mod, for now init at least one horizon upper the top receiver
# TODO: check procedure of "is_ray_intersect_boundary"
vel_mod = []

for vp, vs, Density, depth, name in zip(
        [1000, 3300, 2800, 2300, 1800],  # vp
        [500, 1650, 1400, 1150, 900],  # vs
        [2500, 2500, 2500, 2500, 2500],  # Density
        [20, 30, 70, 150, 215],  # depth
        ['1', '2', '3', '4', '5']  # name
):
    vel_mod.append(Layer(ISOVelocity(vp, vs), Density, Horizon(FlatSurface(depth=depth, dip=0, azimuth=0)), name=name))

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
    ray.optimize()
    # ray.check_snellius()
    try:
        ray.check_snellius(eps=1e-6)
    except SnelliusError as e:
        print('Вдоль луча под номером {} до приемника {} не выполняется закон Cнеллиуса'.format(i+1, rec.location))
    rec.plot(ax=ax, color='k', marker='^', s=50)
    # keep segments colored to check correctness of procedure
    ray.plot(ax=ax, style='s')

plt.show()

n = 1
R = []
for i in range(len(rays[n].segments)):
    R.append(rays[n].segments[i].source)
R.append(rays[n].segments[n].receiver)
R = np.array(R)
print(R)
