import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from src.tprt import Receiver, Layer, Source, Ray, FlatHorizon, GridHorizon, ISOVelocity
from src.tprt.ray import SnelliusError

source = Source([0, 0, 5])
print(source)

receivers = []
for depth in [240]:
    receivers.append(Receiver([90, 90, depth]))

# receivers.append(Receiver([100,100,500]))

print(receivers[0])

# TODO: make class for vel_mod, for now init at least one horizon upper the top receiver
# TODO: check procedure of "is_ray_intersect_boundary"
vel_mod = []

X = np.linspace(0, 100, 10)
Y = np.linspace(0, 100, 10)

Z = np.zeros((X.shape[0], Y.shape[0]))

for i in range(X.shape[0]):
    for j in range(Y.shape[0]):

        Z[i, j] = (X[i] - 50) * (X[i] - 50) / 5000 + (Y[j] - 50) * (Y[j] - 50) / 5000  + 0 * X[i] * Y[j] + 100

vel_mod.append(Layer(ISOVelocity(1800,1000), 2500, GridHorizon(X, Y, Z)))

for vp, vs, depth, density, angle, azimuth in zip(
        [1800], # vp
        [1000], # vs
        [300],    # depth
        [2500],  # Density
        [5],         # angle
        [0]          # azimuth
):
    vel_mod.append(Layer(ISOVelocity(vp, vs), density, FlatHorizon(azimuth=azimuth, angle=angle, x0=np.array([0, 0, depth]))))

# vel_mod.append(Layer(ISOVelocity(3000, 1500), 2500, FlatHorizon(azimuth=0, angle=0, x0=[0,0,100]), name=1))

rays = [Ray(source, rec, vel_mod) for rec in receivers]

fig = plt.figure()
ax = Axes3D(fig)
for l in vel_mod:
    l.top.plot(ax=ax)


source.plot(ax=ax, color='r', marker='p', s=50)
for i, (ray, rec) in enumerate(zip(rays, receivers)):
    ray.optimize()
    try:
        ray.check_snellius(eps=1e-6)
    except SnelliusError as e:
        print('Вдоль луча под номером {} до приемника {} не выполняется закон Cнеллиуса'.format(i+1, rec.location))
    rec.plot(ax=ax, color='k', marker='^', s=50)
    # keep segments colored to check correctness of procedure
    ray.plot(ax=ax)
plt.show()

n=-1
R = []
for i in range(len(rays[n].segments)):
    R.append(rays[n].segments[i].source)
R.append(rays[n].segments[n].receiver)
R = np.array(R)
print(R)


# print(rays[-1].dtravel())
