import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from src.tprt import Receiver, Layer, Source, DilatCenter, RotatCenter, Ray, FlatHorizon, GridHorizon, ISOVelocity, Velocity_model
from src.tprt.ray import SnelliusError


# TODO: make class for vel_mod, for now init at least one horizon upper the top receiver
# TODO: check procedure of "is_ray_intersect_boundary"

# Let's define the interface.

X = np.linspace(- 2000, 2000, 41)
Y = np.linspace(- 2000, 2000, 41)

Z = np.zeros((X.shape[0], Y.shape[0]))

for i in range(X.shape[0]):
    for j in range(Y.shape[0]):

        Z[i, j] = 700 - 200 * np.exp(- X[i] * X[i] / 1000000 - Y[j] * Y[j] / 1000000)
        # Z[i, j] = 700

horizons = [GridHorizon(X, Y, Z)]
# horizons = [FlatHorizon(100,np.array([0,0]),30,0)]

vel_mod = Velocity_model(np.array([ISOVelocity(2000, 1100), ISOVelocity(2800, 1600)]),
                         np.array([1800, 2100]), np.array([1, 2]), horizons)





sou_line = np.arange(- 1300, 0, 100) # source line starting from - 1000 and ending at 0 with step 100
rec_line = np.linspace(100, 1300, sou_line.shape[0]) # source line starting from 0 and ending at 1000 with step 100

# Let's set sources and receivers along profile:

sources = np.empty(sou_line.shape[0], dtype = Source)
receivers_s = np.empty(rec_line.shape[0], dtype = Receiver)
receivers_h = np.empty(rec_line.shape[0], dtype = Receiver)

for i in range(sou_line.shape[0]):

    sources[i] = DilatCenter(50, vel_mod,np.array([sou_line[i], 0, 0]))
    receivers_s[i] = Receiver([rec_line[-1 - i], 0, 0])
    receivers_h[i] = Receiver([rec_line[-1 - i], 0, 400])


# Let's construct raycode for reflected waves:
raycode = [[1, 0, 0],
           [-1, 0, 0]]

# And initiate rays themselves:
rays_s = np.empty(sou_line.shape[0], dtype = Ray)
rays_h = np.empty(sou_line.shape[0], dtype = Ray)

# Geometrical spreading along all rays:
geom_spread_s_curv = np.zeros(rays_s.shape)
geom_spread_s_plane = np.zeros(rays_s.shape)
geom_spread_h_curv = np.zeros(rays_h.shape)
geom_spread_h_plane = np.zeros(rays_h.shape)

# ampl = Ray(sources[-1], receivers_s[-1], vel_mod, [[1, 0, 0],
#                                                    [-1, 0, 0]])
#
# ampl1 = Ray(sources[-1], receivers_s[-1], vel_mod, [[1, 0, 0],
#                                                     [-1, 0, 1]])

# ampl.optimize(penalty=True)
# ampl1.optimize(penalty=True)

fig = plt.figure()
ax = Axes3D(fig)
ax.invert_zaxis()
for l in vel_mod.layers[:-1]:
    l.bottom.plot(ax=ax)

    # ampl.plot(ax=ax)
    # ampl1.plot(ax=ax)

for i in range(sources.shape[0]):

    sources[i].plot(ax=ax, color='r', marker='p', s=50)
    receivers_s[i].plot(ax=ax, color='k', marker='^', s=50)
    # receivers_h[i].plot(ax=ax, color='k', marker='^', s=50)

    rays_s[i] = Ray(sources[i], receivers_s[i], vel_mod, raycode)
    rays_s[i].optimize(penalty=True)

    geom_spread_s_curv[-1 - i] = rays_s[i].spreading(1)[1] / 2000000
    geom_spread_s_plane[-1 - i] = rays_s[i].spreading(0)[1] / 2000000

    rays_s[i].plot(ax=ax)

    # rays_h[i] = Ray(sources[i], receivers_h[i], vel_mod, raycode)
    # rays_h[i].optimize(penalty=True)
    #
    # geom_spread_h_curv[-1 - i] = rays_h[i].spreading(1)
    # geom_spread_h_plane[-1 - i] = rays_h[i].spreading(0)
    #
    # rays_h[i].plot(ax=ax)

# # keep segments colored to check correctness of procedure
# #
plt.xlabel("x")
plt.ylabel("y")
plt.show()

fig2 = plt.figure()

plt.plot(rec_line, geom_spread_s_curv, 'r-', label = "curv_s")
plt.plot(rec_line, geom_spread_s_plane, 'r--', label = "plane_s")

# plt.plot(rec_line, geom_spread_h_curv / 1000000, 'b-', label = "curv_h")
# plt.plot(rec_line, geom_spread_h_plane / 1000000, 'b--', label = "plane_h")

# plt.plot(rec_line, 4 * (500 ** 2 + rec_line**2) / 1000000, 'ko', label="distance^2")

plt.legend()

# plt.yticks(np.arange(0, int(geom_spread_s_curv[-1]) + 3, 1))
plt.grid()

plt.show()

# print(2.74 - 0.17)

# print(ampl1.segments[1].vtype)