import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from src.tprt import Receiver, Layer, Source, DilatCenter, RotatCenter, Ray, FlatHorizon, GridHorizon, ISOVelocity, Velocity_model
from src.tprt.ray import SnelliusError


# TODO: make class for vel_mod, for now init at least one horizon upper the top receiver
# TODO: check procedure of "is_ray_intersect_boundary"

# Let's define the interface.

X = np.linspace(- 2000, 2000, 11)
Y = np.linspace(- 2000, 2000, 11)

Z = np.zeros((X.shape[0], Y.shape[0]))

for i in range(X.shape[0]):
    for j in range(Y.shape[0]):

        Z[i, j] = 700 - 200 * np.exp(- X[i] * X[i] / 1000000 - Y[j] * Y[j] / 1000000)
        # Z[i, j] = 700

horizons = [GridHorizon(X, Y, Z)]
# horizons = [FlatHorizon(700,np.array([0,0]),0,0)]

vel_mod = Velocity_model(np.array([ISOVelocity(2000, 1100), ISOVelocity(2800, 1600)]),
                         np.array([1800, 2100]), np.array([1, 2]), horizons)

sou_line = np.arange(- 1300, 50, 50) # source line starting from - 1300 and ending at 0 with step 100
rec_line = np.linspace(0, 1300, sou_line.shape[0]) # source line starting from 0 and ending at 1300 with step 100

# Let's set sources and receivers along profile:

sources = np.empty(sou_line.shape[0], dtype = Source)
receivers_s = np.empty(rec_line.shape[0], dtype = Receiver)
receivers_h = np.empty(rec_line.shape[0], dtype = Receiver)

for i in range(sou_line.shape[0]):

    sources[i] = DilatCenter(0.01, vel_mod, np.array([sou_line[i], 0, 0]))
    receivers_s[i] = Receiver([rec_line[-1 - i], 0, 0])
    receivers_h[i] = Receiver([rec_line[-1 - i] / 5, 0, 400])


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

# Travel time of waves from sources to the surface receivers
travel_time_s = np.zeros(rays_s.shape)

# Record time at the station

max_time = 2 * np.sqrt(900 ** 2 + rec_line[-1]**2) / vel_mod.layers[0].get_velocity(0)['vp']

# Time array for seismic gathers. Grid spacing is 1 millisecond.

record_time = np.arange(0, max_time, 0.001)

# A two-dimensional arrays for gathers:

gathers_sx = np.zeros((rec_line.shape[0], record_time.shape[0]))
gathers_sy = np.zeros((rec_line.shape[0], record_time.shape[0]))
gathers_sz = np.zeros((rec_line.shape[0], record_time.shape[0]))

fig = plt.figure()
ax = Axes3D(fig)
ax.invert_zaxis()
for l in vel_mod.layers[:-1]:
    l.bottom.plot(ax=ax)

for i in range(sources.shape[0]):
# #
    sources[i].plot(ax=ax, color='r', marker='p', s=50)
    receivers_s[i].plot(ax=ax, color='k', marker='^', s=50)
    # receivers_h[i].plot(ax=ax, color='k', marker='^', s=50)

    rays_s[i] = Ray(sources[i], receivers_s[i], vel_mod, raycode)
    rays_s[i].optimize(penalty=True)

    travel_time_s[-1 - i] = rays_s[i].travel_time()

    geom_spread_s_curv[-1 - i] = rays_s[i].spreading(1)[1] / 1000000
    geom_spread_s_plane[-1 - i] = rays_s[i].spreading(0)[1] / 1000000

    for j in range(record_time.shape[0]):

        gathers_sx[i, j], gathers_sy[i, j], gathers_sz[i, j] = rays_s[i].amplitude_t_dom(record_time[j])

    rays_s[i].plot(ax=ax)

    # rays_h[i] = Ray(sources[i], receivers_h[i], vel_mod, raycode)
    # rays_h[i].optimize(penalty=True)
    #
    # geom_spread_h_curv[-1 - i] = rays_h[i].spreading(1)[1] / 1000000
    # geom_spread_h_plane[-1 - i] = rays_h[i].spreading(0)[1] / 1000000
    #
    # rays_h[i].plot(ax=ax)

# # keep segments colored to check correctness of procedure
# #
ax.set_xlabel("Расстояние по оси x, м")
ax.set_ylabel("Расстояние по оси y, м",)
ax.set_zlabel("Глубина, м")

plt.show()

fig2 = plt.figure()

plt.title("Геометрическое расхождение на приёмной линии на поверхности")

plt.plot(rec_line, geom_spread_s_curv, 'r-', label = "С учётом кривизны границы")
plt.plot(rec_line, geom_spread_s_plane, 'r--', label = "Без учёта кривизны границы")

plt.plot(rec_line, 2000 * np.sqrt(500 ** 2 + rec_line**2) * 2 / 1000000, 'ko', label="~ distance^2")

plt.legend()
plt.grid()

plt.xlabel("Координаты вдоль профиля, м")
plt.ylabel("Геометрическое расхождение, км^2 / с")

plt.show()

# fig3 = plt.figure()
#
# plt.title("Геометрическое расхождение на заглублённой приёмной линии")
#
# plt.plot(rec_line / 5, geom_spread_h_curv, 'b-', label = "С учётом кривизны границы")
# plt.plot(rec_line / 5, geom_spread_h_plane, 'b--', label = "Без учёта кривизны границы")
#
# plt.legend()
# plt.grid()

# plt.xlabel("Координаты вдоль профиля, м")
# plt.ylabel("Геометрическое расхождение, км^2 / с")

# plt.show()

fig4 = plt.figure()

plt.title("Годограф на приёмной линии на поверхности")

plt.plot(rec_line, travel_time_s, 'r-', label = "Практический годограф отражённой PP-волны")
plt.plot(rec_line, 2 * np.sqrt(900 ** 2 + rec_line**2) / vel_mod.layers[0].get_velocity(0)['vp'],
         'ko', label = "Теоретический годограф отражённой PP-волны")

plt.legend()
plt.grid()

plt.xlabel("Координаты вдоль профиля, м")
plt.ylabel("Время первых вступлений, с")

plt.show()

fig5 = plt.figure()

plt.title("Сейсмограммы на приёмной линии на поверхности")

for i in range(rays_s.shape[0]):

    plt.plot(record_time, gathers_sx[i, :] * 1000000 + i, 'r-', linewidth = 0.3)
    plt.plot(record_time, gathers_sy[i, :] * 1000000 + i + 0.1, 'g-', linewidth = 0.3)
    plt.plot(record_time, gathers_sz[i, :] * 1000000 + i + 0.2, 'b-', linewidth = 0.3)

# plt.legend()
plt.grid()

plt.xlabel("Время, с")

plt.show()

print("max x-displacement = ", np.max(gathers_sx))
print("max y-displacement = ", np.max(gathers_sy))
print("max z-displacement = ", np.max(gathers_sz))
# print(2.74 - 0.17)

# print(ampl1.segments[1].vtype)