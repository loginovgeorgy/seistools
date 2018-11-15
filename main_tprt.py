import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from src.tprt import Receiver, Layer, Source, Ray, FlatHorizon, GridHorizon, ISOVelocity, Velocity_model
from src.tprt.ray import SnelliusError

source = Source([40, 60, 90])

print(source)

receivers = []
for depth in [5, 220]:
    receivers.append(Receiver([100, 100, depth]))

print(receivers[0])

# TODO: make class for vel_mod, for now init at least one horizon upper the top receiver
# TODO: check procedure of "is_ray_intersect_boundary"


vp =        np.array([1000, 1300, 1600, 3800])  # vp
vs =        np.array([500, 650, 800, 900])  # vs
velocities = np.array([ISOVelocity(vp[i], vs[i]) for i in range(len(vp))])
depth =     np.array([20, 70])  # depth
name =      np.array(['0', '1', '2', '3'])  # name
density =   np.array([2500, 2500, 2500, 2500])  # Density
anchor =    [(0,0), (0,0)]
dip =       np.array([0, 0])  # dip
azimuth =   np.array([0, 0])  # azimuth

X = np.linspace(0, 100, 11)
Y = np.linspace(0, 100, 11)

Z = np.zeros((X.shape[0], Y.shape[0]))

for i in range(X.shape[0]):
    for j in range(Y.shape[0]):

        Z[i, j] = X[i] * X[i] / 1000 + Y[j] * Y[j] / 1000 + 130

# Это list из горизонтов, в данном случае только flat. Если требуется grid_horizon то можно воткнуть его куда нужно
# Главное соблюдать сортированность по глубине
horizons = Velocity_model.make_flat_horizons(depth, anchor, dip, azimuth)

# horizons = Velocity_model.make_flat_horizons(np.array([depth[0]]), np.array([anchor[0]]),
#                                              np.array([dip[0]]), np.array([azimuth[0]]))

horizons.append(GridHorizon(X, Y, Z))
#
horizons.append(FlatHorizon(190))


vel_mod = Velocity_model(velocities, density, name, horizons)

rays = [Ray(source, rec, vel_mod) for rec in receivers]
#
# raycode = [[-1, 0, 0],
#            [-1, 1, 0]]
#
# rays.append(Ray(source, Receiver([100, 100, 90]), vel_mod, raycode))

fig = plt.figure()
ax = Axes3D(fig)
for l in vel_mod.layers[:-1]:
    l.bottom.plot(ax=ax)

rays[-1].optimize()

source.plot(ax=ax, color='r', marker='p', s=50)
for i, (ray, rec) in enumerate(zip(rays, receivers)):
    ray.optimize(penalty=True)
    rec.plot(ax=ax, color='k', marker='^', s=50)
    # keep segments colored to check correctness of procedure
    ray.plot(ax=ax)
rays[-1].optimize()
rays[-1].plot(ax=ax)
plt.show()

n=-1
# print(rays[n]._get_trajectory())
# print(rays[n].snells_law(projection=True))
# print(rays[n].dtravel())
