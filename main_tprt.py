import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import timeit as tt
from src.tprt import Receiver, Layer, Source, Segment, FlatHorizon, \
                        GridHorizon, ISOVelocity, VelocityModel, Ray, Survey


#############################################################
################## MODEL INITIALIAZING ######################
vp =        np.array([1000, 1500, 2300, 2600, 2700], dtype=float)  # vp
vs =        np.array([700, 1100, 2000, 2200, 2350], dtype=float)  # vs
velocities = [ISOVelocity(vpi, vsi) for vpi, vsi in zip(vp,vs)]

depth =     np.array([20, 70, 120, 180], dtype=float)  # depth
name =      np.array([0, 1, 2, 3, 4])  # name
density =   np.array([2500, 2500, 2500, 2500, 2500], dtype=float)  # Density
anchor =    np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=float)
dip =       np.array([0, 0, 15, 0, 0], dtype=float)  # dip
azimuth =   np.array([0, 0, 90, 0, 0], dtype=float)  # azimuth

# Это list из горизонтов, в данном случае только flat.
# Если требуется grid_horizon то можно воткнуть его куда нужно
# Главное соблюдать сортированность по глубине
horizons = VelocityModel.make_flat_horizons(depth, anchor, dip, azimuth)
vel_mod = VelocityModel(velocities, density, name, horizons)
print('1. Model has been initialized')
#############################################################
#################### SURVEY NETWORK #########################

sources = [Source(location=[0.0, 0.0, 0.0], vel_model=vel_mod)]
receivers = []
# for x,y in zip(np.arange(0.0, 100/np.sqrt(2), 10/np.sqrt(2)), np.arange(0, 100/np.sqrt(2), 10/np.sqrt(2))):
#     receivers.append(Receiver([x, y, 0.0]))

tic = tt.default_timer()
for x in np.arange(-100, 100, 10.0):
        receivers.append(Receiver([x, 0.0, 0.0]))

survey = Survey(sources, receivers, vel_mod)
toc = tt.default_timer()
print('2.1 Survey network has been initialized', toc-tic)

survey.initialize_rays(reflect_horizon=2, vtype='vp', forward=False)

# tic = tt.default_timer()
# survey.calculate(method='bfgs', survey2D=True)
# toc = tt.default_timer()
# print('Time for 2D minimization = ', toc-tic)
# T_2D = survey.get_traveltimes()

tic = tt.default_timer()
survey.calculate(method='bfgs', survey2D=False)
toc = tt.default_timer()
print('Time for 3D minimization = ', toc-tic)
T_3D = survey.get_traveltimes()

print('2.2 Survey has been calculated')

# T_analytic = np.empty(shape=(len(receivers),))
#############################################################
###################### PLOT FIGURES #########################

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlim(left=-100, right=100)
ax.set_ylim(bottom=-100, top=100)
ax.set_zlim(bottom=200, top=0)
vel_mod.plot(ax)
survey.plot(ax)
plt.show()

plt.figure()
# plt.plot(T_analytic[:], '*r', label='analytic')
# plt.plot(T_2D[0,:], label='2D')
plt.plot(T_3D[0,:], label='3D')
plt.legend(loc='best')
plt.show()

# head_wave_raycode = np.array([
#                     [+1, 3, 0],
#                     [-1, 4, 0],
#                     [-1, 3, 0],
#                     ], dtype=int)

# init_trj = np.array([[50.0, 50.0],[80.0, 80.0]])
# rays.append(Ray(source, Receiver([100, 100, 90]), vel_mod, raycode=raycode))
# rays.append(Ray(source, receivers[1], vel_mod, raycode=raycode_test))


#rays[-1].optimize()

# source.plot(ax=ax, color='r', marker='p', s=50)
# for i, (ray, rec) in enumerate(zip(rays, receivers)):
#     ray.optimize(snells_law=False, Ferma=True, dtravel=True, survey2D=True)
#     rec.plot(ax=ax, color='k', marker='^', s=50)
#     # keep segments colored to check correctness of procedure
#     ray.plot(ax=ax)
# rays[-1].optimize(snells_law=False, Ferma=True, dtravel=True, survey2D=True)
# rays[-1].plot(ax=ax)
# plt.show()

# n=-1
# print('Trajectory of {0} ray:\n'.format(n), rays[n]._get_trajectory())
# print('Checking the snellius law:\n', rays[n].snells_law(projection=False))
# print('Checking the derivative along ray\n', rays[n].dtravel())

# seg1 = rays[-1].segments[0]
# seg2 = rays[-1].segments[-1]
# seg3 = rays[-1].segments[1]
# print('\n', 180/np.pi*np.arccos(abs(np.dot(seg1.get_vector(), seg1.end_horizon.get_normal(seg1.receiver[:-1])))))
# print('\n', 180/np.pi*np.arccos(np.dot(seg2.get_vector(), seg1.end_horizon.get_normal(seg2.source[:-1]))))
# print('\n Critic angle = ', 180/np.pi*np.arcsin(seg1.layer.get_velocity(0)['vp']/seg3.layer.get_velocity(0)['vp']))
