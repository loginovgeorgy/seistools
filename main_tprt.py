import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from src.tprt import Receiver, Layer, Source, DilatCenter, RotatCenter, Ray, FlatHorizon, GridHorizon, ISOVelocity, Velocity_model
from src.tprt.ray import SnelliusError
from src.tprt.rt_coefficients import rt_coefficients
from src.tprt.bicubic_interpolation import *

import time # for runtime
import os # for folder's creation

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

import datetime # for folder's naming

start_time = time.time()

# TODO: make class for vel_mod, for now init at least one horizon upper the top receiver
# TODO: check procedure of "is_ray_intersect_boundary"

# Let's define the interfaces.

# Grid:
X = np.linspace(- 2000, 2000, 51)
Y = np.linspace(- 2000, 2000, 51)

YY, XX = np.meshgrid(Y, X)

# Interfaces:

Convex_Gauss_500_Center = np.array(list(map(lambda x, y: 700 - 200 * np.exp(- x * x / 1000000 -
                                                                            y * y / 1000000), XX, YY)))
Convex_Gauss_1000_Center = np.array(list(map(lambda x, y: 1200 - 200 * np.exp(- x * x / 1000000 -
                                                                              y * y / 1000000), XX, YY)))

Convex_Gauss_500_Incident = np.array(list(map(lambda x, y: 700 - 200 * np.exp(- (x + 816.045) *
                                                                              (x + 816.045) / 1000000 -
                                                                              y * y / 1000000), XX, YY)))
Convex_Gauss_500_Upcoming = np.array(list(map(lambda x, y: 700 - 200 * np.exp(- (x - 816.045) *
                                                                              (x - 816.045) / 1000000 -
                                                                              y * y / 1000000), XX, YY)))
Concave_Gauss_500_Center = np.array(list(map(lambda x, y: 700 + 200 * np.exp(- x * x / 1000000 -
                                                                             y * y / 1000000), XX, YY)))

# Let's construct raycodes for reflected waves:
raycode_model_1_3 = [[1, 0, 0],
                     [-1, 0, 0]]

raycode_model_2 = [[1, 0, 0],
                   [1, 1, 0],
                   [- 1, 1, 0],
                   [- 1, 0, 0]]

# Let's set a particular model and a particular raycode as "current". The current model will be specified by a number
# (starting with 1) and the current raycode will be set by a rule: model 1 or 3 -> raycode_model_1_3, any kind of
# model 2 -> raycode_model_2.
# model numbers:

# 1 - vel_mod_1
# 2 - vel_mod_2a
# 3 - vel_mod_2b
# 4 - vel_mod_2c
# 5 - vel_mod_3

model_number = 2

if model_number == 1:

    horizons = [GridHorizon(X, Y, Convex_Gauss_500_Center, bool_parab = 0)]
    current_mod = Velocity_model(np.array([ISOVelocity(2000, 1100), ISOVelocity(2800, 1600)]),
                                 np.array([1800, 2100]), np.array([1, 2]), horizons)

elif model_number == 2:

    horizons = [GridHorizon(X, Y, Convex_Gauss_500_Incident, bool_parab = 0),
                GridHorizon(X, Y, Convex_Gauss_1000_Center, bool_parab = 0)]
    current_mod = Velocity_model(np.array([ISOVelocity(2000, 1100), ISOVelocity(2800, 1600), ISOVelocity(3600, 2100)]),
                                 np.array([1800, 2100, 2400]), np.array([1, 2, 3]), horizons)

elif model_number == 3:

    horizons = [GridHorizon(X, Y, Convex_Gauss_500_Center, bool_parab = 0),
                GridHorizon(X, Y, Convex_Gauss_1000_Center, bool_parab = 0)]
    current_mod = Velocity_model(np.array([ISOVelocity(2000, 1100), ISOVelocity(2800, 1600), ISOVelocity(3600, 2100)]),
                                 np.array([1800, 2100, 2400]), np.array([1, 2, 3]), horizons)

elif model_number == 4:

    horizons = [GridHorizon(X, Y, Convex_Gauss_500_Upcoming, bool_parab = 0),
                GridHorizon(X, Y, Convex_Gauss_1000_Center, bool_parab = 0)]
    current_mod = Velocity_model(np.array([ISOVelocity(2000, 1100), ISOVelocity(2800, 1600), ISOVelocity(3600, 2100)]),
                                 np.array([1800, 2100, 2400]), np.array([1, 2, 3]), horizons)

elif model_number == 5:

    horizons = [GridHorizon(X, Y, Concave_Gauss_500_Center, bool_parab = 0)]
    current_mod = Velocity_model(np.array([ISOVelocity(2000, 1100), ISOVelocity(2800, 1600)]),
                                 np.array([1800, 2100]), np.array([1, 2]), horizons)

else: # in any oser case set the default model:

    horizons = [GridHorizon(X, Y, Convex_Gauss_500_Center, bool_parab = 0)]
    current_mod = Velocity_model(np.array([ISOVelocity(2000, 1100), ISOVelocity(2800, 1600)]),
                                 np.array([1800, 2100]), np.array([1, 2]), horizons)

# And current raycode:
current_raycode = raycode_model_1_3 # by default

if 1 < model_number <= 4:

    current_raycode = raycode_model_2


# Let's construct the observation system:
sou_line = np.arange(- 1300, 25, 25) # source line starting from - 1300 and ending at 0 with step 100
rec_line = np.linspace(0, 1300, sou_line.shape[0]) # source line starting from 0 and ending at 1300 with step 100

# Let's set sources and receivers along profile:
sources = np.empty(sou_line.shape[0], dtype = Source)
receivers = np.empty(rec_line.shape[0], dtype = Receiver)

# So, let's set Sources and Receivers.

frequency_dom = 39 # dominant frequency of the source

for i in range(sou_line.shape[0]):

    sources[i] = DilatCenter(frequency_dom, current_mod, np.array([sou_line[i], 0, 0]))
    receivers[i] = Receiver([rec_line[-1 - i], 0, 0])

# And initiate rays themselves:
rays = np.empty(sou_line.shape[0], dtype = Ray)

# Geometrical spreading along all rays:
geom_spread_curv_1 = np.zeros(rays.shape)
geom_spread_plane_1 = np.zeros(rays.shape)

geom_spread_curv_2 = np.zeros(rays.shape)
geom_spread_plane_2 = np.zeros(rays.shape)

# Travel time of waves from sources to the surface receivers
travel_time = np.zeros(rays.shape)

# Record time at the station in seconds:
max_time = 2

# Time array for seismic gathers. Grid spacing is 1 millisecond:
record_time = np.arange(0, max_time, 0.001)

# A two-dimensional arrays for gathers:
gathers_x = np.zeros((rec_line.shape[0], record_time.shape[0]))
gathers_y = np.zeros((rec_line.shape[0], record_time.shape[0]))
gathers_z = np.zeros((rec_line.shape[0], record_time.shape[0]))

# Later it will be needed to insert the current model's name into titles of plots. In order to do this, let's
# define a dictionary:

number_string = {1:'1', 2:'2a', 3:'2b', 4:'2c', 5:'3'}

# Create a description file. We shall create a folder for this file and others. So, the name of the directory is:

dir_name = "C:/Users/USER/Documents/Лучевой метод/AVO, коэффициенты отражения-преломления/Результаты вычислений/" \
           "Модель №{}/Вычисления {} {}-{}".format(number_string[model_number],
                                                   datetime.datetime.now().date(),
                                                   datetime.datetime.now().hour,
                                                   datetime.datetime.now().minute)

createFolder(dir_name)

description_file = open("{}/Описание.txt".format(dir_name), "w+")

description_file.write("Модель №{} \n \n".format(number_string[model_number]))

description_file.write("Сетка задния границ: \n \n")

description_file.write("Узлы на оси X расположены от {} до {} м с шагом {} м. \n".format(X[0],
                                                                             X[- 1],
                                                                             X[1] - X[0]))
description_file.write("Узлы на оси Y расположены от {} до {} м с шагом {} м. \n \n".format(Y[0],
                                                                             Y[- 1],
                                                                             Y[1] - Y[0]))

description_file.write("Схема наблюдений: \n \n")

description_file.write("Источники расположены на оси X от {} до {} м с шагом {} м. \n".format(int(sou_line[0]),
                                                                                              int(sou_line[- 1]),
                                                                                              int(sou_line[1] -
                                                                                                  sou_line[0])))
description_file.write("Приёмники расположены на оси X от {} до {} м с шагом {} м. \n \n".format(int(rec_line[0]),
                                                                                                 int(rec_line[- 1]),
                                                                                                 int(rec_line[1] -
                                                                                                     rec_line[0])))

description_file.write("Функция источника: импульс Рикера с главной частотой {} Гц. \n \n".format(frequency_dom))

description_file.write("Для пар 'источник-приёмник', находящихся в закритической области,"
                       " лучи строиться не будут.\n\n")

description_file.write("Время записи на станции: {} с \n".format(max_time))
description_file.write("Длительность отсчётов: {} с \n \n".format(record_time[1] - record_time[0]))


description_file.write("============================================================ \n \n")

description_file.write("Время вычислений: \n \n")

description_file.write("Под обработкой луча понимается расчёт амплитуд и геометрического расхождения. \n")

description_file.write("Сетка задния границ: \n \n")

description_file.write("--- Конфигурации границ и схемы наблюдений заданы: %s секунд --- \n \n" % (time.time() -
                                                                                                   start_time))
print("\x1b[1;31m --- The configuration of interfaces and the observation system are defined: %s seconds ---" %
      (time.time() - start_time))

# Before we start processing rays, it will be convenient to create .txt files where we shall write values of the
# amplitude, traveltime and geometrical spreading.

geom_spread_1 = open("{}/Геометрическое расхождение в смысле 1.txt".format(dir_name), "w+")
geom_spread_2 = open("{}/Геометрическое расхождение в смысле 2.txt".format(dir_name), "w+")
seismogram_z = open("{}/Сейсмограммы. Z-компонента.txt".format(dir_name), "w+")
seismogram_x = open("{}/Сейсмограммы. X-компонента.txt".format(dir_name), "w+")
ray_amplitude = open("{}/Лучевые амплитуды.txt".format(dir_name), "w+")
hodograph = open("{}/Годограф.txt".format(dir_name), "w+")

# Let's form up heads for these files:
geom_spread_1.write("X, м\tС учётом кривизны границы, м^2\tБез учёта кривизны границ, м^2\n\n")
geom_spread_2.write("X, м\tС учётом кривизны границы, м^2/с\tБез учёта кривизны границ, м^2/с\n\n")

seismogram_z.write("Z-компонента вектора смещений\n\n")
seismogram_x.write("X-компонента вектора смещений\n\n")

seismogram_z.write("T, с\t")
seismogram_x.write("T, с\t")

ray_amplitude.write("Лучевые амплитуды")

ray_amplitude.write("\n\n")

ray_amplitude.write("Координата приёмника, м\tС учётом кривизны границ\t\tБез учёта кривизны границы")

ray_amplitude.write("\n\n")

for i in range(receivers.shape[0]):

    seismogram_z.write("Тр. {}\t".format(i))
    seismogram_x.write("Тр. {}\t".format(i))

seismogram_z.write("\n\n")
seismogram_x.write("\n\n")

hodograph.write("Годограф первых вступлений\n\n")
hodograph.write("X, м\tT, с\n\n")

# Constructing rays:

for i in range(sources.shape[0]):

    print("------------------------------------")
    rays[i] = Ray(sources[- 1 - i], receivers[- 1 - i], current_mod, current_raycode)
    rays[i].optimize(penalty=True)

    description_file.write("--- %s луч создан и оптимизирован: %s секунд --- \n" % (i + 1, time.time() - start_time))
    print("\x1b[1;31m --- %s ray constructed: %s seconds ---" % (i + 1, time.time() - start_time))

    # Check if we are in post-critical zone:

    if np.linalg.norm(np.imag(rays[i].amplitude_fr_dom(curv_bool = 1)))!= 0:

        rays = np.delete(rays, np.arange(i, rays.shape[0], 1), axis = 0)

        geom_spread_curv_1 = np.delete(geom_spread_curv_1, np.arange(i, geom_spread_curv_1.shape[0], 1), axis = 0)
        geom_spread_plane_1 = np.delete(geom_spread_plane_1, np.arange(i, geom_spread_plane_1.shape[0], 1), axis = 0)

        geom_spread_curv_2 = np.delete(geom_spread_curv_2, np.arange(i, geom_spread_curv_2.shape[0], 1), axis = 0)
        geom_spread_plane_2 = np.delete(geom_spread_plane_2, np.arange(i, geom_spread_plane_2.shape[0], 1), axis = 0)

        travel_time = np.delete(travel_time, np.arange(i, travel_time.shape[0], 1), axis = 0)

        gathers_x = np.delete(gathers_x, np.arange(i, gathers_x.shape[0], 1), axis = 0)
        gathers_y = np.delete(gathers_y, np.arange(i, gathers_y.shape[0], 1), axis = 0)
        gathers_z = np.delete(gathers_z, np.arange(i, gathers_z.shape[0], 1), axis = 0)

        description_file.write("--- На %s луче был достигнут критический угол. Процесс остановлен: %s секунд --- \n\n" %
                               (i + 1, time.time() - start_time))
        print("\x1b[1;31m --- On the %s-th ray critical angle has been reached. The process has been terminated:"
              " %s seconds ---" % (i + 1, time.time() - start_time))

        break

    rays[i].amplitude_fun = rays[i].amplitude_fr_dom(curv_bool = 1) # rewrite the amplitude field.

    travel_time[i] = rays[i].travel_time()

    ray_amplitude.write("{}\t{}\t\t{}\n".format(rec_line[i],
                                                np.linalg.norm(rays[i].amplitude_fr_dom(curv_bool = 1)),
                                                np.linalg.norm(rays[i].amplitude_fr_dom(curv_bool = 0))))

    geom_spread_curv_1[i], geom_spread_curv_2[i] = rays[i].spreading(1)
    geom_spread_plane_1[i], geom_spread_plane_2[i] = rays[i].spreading(0)

    for j in range(record_time.shape[0]):

        gathers_x[i, j], gathers_y[i, j], gathers_z[i, j] = np.real(rays[i].amplitude_t_dom(record_time[j]))

    description_file.write("--- %s луч обработан: %s секунд --- \n \n" % (i + 1, time.time() - start_time))
    print("\x1b[1;31m --- %s ray processed: %s seconds ---" % (i + 1, time.time() - start_time))

# Now, let's save seismograms:

max_amplitude = max(np.max(abs(gathers_z)), np.max(abs(gathers_x)))
accuracy = int(round(abs(np.log10(max_amplitude))) * 5)

for i in range(record_time.shape[0]):

    seismogram_z.write("{}\t".format(round(record_time[i], 3)))
    seismogram_x.write("{}\t".format(round(record_time[i], 3)))

    for j in range(rays.shape[0]):

        seismogram_z.write("{}\t".format(round(gathers_z[j, i], accuracy)))
        seismogram_x.write("{}\t".format(round(gathers_x[j, i], accuracy)))

    seismogram_z.write("\n")
    seismogram_x.write("\n")

# And other data:
for i in range(rays.shape[0]):

    geom_spread_1.write("{}\t{}\t{}\n".format(rec_line[i], geom_spread_curv_1[i], geom_spread_plane_1[i]))
    geom_spread_2.write("{}\t{}\t{}\n".format(rec_line[i], geom_spread_curv_2[i], geom_spread_plane_2[i]))

    hodograph.write("{}\t{}\n".format(rec_line[i], travel_time[i]))

# Plot rays and the medium:

fig = plt.figure()
ax = Axes3D(fig)
ax.invert_zaxis()

ax.view_init(0, - 90)

for l in current_mod.layers[:-1]:
    l.bottom.plot(ax=ax)

for i in range(rays.shape[0]):

    sources[- 1 - i].plot(ax=ax, color='r', marker='p', s=50)
    receivers[- 1 - i].plot(ax=ax, color='k', marker='^', s=50)

    rays[i].plot(ax=ax)

ax.set_xlabel("Расстояние по оси x, м")
ax.set_ylabel("Расстояние по оси y, м",)
ax.set_zlabel("Глубина, м")

plt.savefig("{}/Лучи.png".format(dir_name), dpi = 400)
print("\nЛучевая схема сохранена: {} секунд".format(time.time() - start_time))

plt.close(fig)

fig2 = plt.figure()

plt.title("Геометрическое расхождение в смысле 1. Модель №{}".format(number_string[model_number]))

plt.plot(rec_line[0 : geom_spread_curv_1.shape[0]], geom_spread_curv_1 / 1000000, 'r-', label = "С учётом кривизны границы")
plt.plot(rec_line[0 : geom_spread_plane_1.shape[0]], geom_spread_plane_1 / 1000000, 'r--', label = "Без учёта кривизны границы")

plt.legend()
plt.grid()

plt.xlabel("Координаты вдоль профиля, м")
plt.ylabel("Геометрическое расхождение, км^2")

plt.savefig("{}/Геометрическое расхождение в смысле 1.png".format(dir_name), dpi = 400)
print("График геометрического расхождения в смысле 1 сохранён: {} секунд".format(time.time() - start_time))

plt.close(fig2)

fig3 = plt.figure()

plt.title("Геометрическое расхождение в смысле 2. Модель №{}".format(number_string[model_number]))

plt.plot(rec_line[0 : geom_spread_curv_2.shape[0]], geom_spread_curv_2 / 1000000, 'r-', label = "С учётом кривизны границы")
plt.plot(rec_line[0 : geom_spread_plane_2.shape[0]], geom_spread_plane_2 / 1000000, 'r--', label = "Без учёта кривизны границы")
plt.legend()
plt.grid()

plt.xlabel("Координаты вдоль профиля, м")
plt.ylabel("Геометрическое расхождение, км^2 / c")

plt.savefig("{}/Геометрическое расхождение в смысле 2.png".format(dir_name), dpi = 400)
print("График геометрического расхождения в смысле 2 сохранён: {} секунд".format(time.time() - start_time))

plt.close(fig3)

fig4 = plt.figure()

plt.title("Годограф. Модель №{}".format(number_string[model_number]))

plt.plot(rec_line[0 : travel_time.shape[0]], travel_time, 'r-', label = "Годограф отражённой PP-волны")

plt.legend()
plt.grid()

plt.xlabel("Координаты вдоль профиля, м")
plt.ylabel("Время первых вступлений, с")

plt.savefig("{}/Годограф ОСТ.png".format(dir_name), dpi = 400)
print("Годограф сохранён: {} секунд".format(time.time() - start_time))

plt.close(fig4)

fig5 = plt.figure()

plt.title("Сейсмограмма (Z-комонента). Модель №{}".format(number_string[model_number]))
plt.gca().invert_yaxis()

for i in range(rays.shape[0]):

    plt.fill_between(record_time, gathers_z[i, :] / np.max(abs(gathers_z)) / 1.5 + i,
                     np.ones(record_time.shape) * i,
                     linewidth = 0.3, color = 'b', alpha = 0.5)

plt.yticks(np.arange(0, rays.shape[0], 1), ["{}".format(int(j)) for j in rec_line[0 : rays.shape[0]]],
           fontsize = - 5 / 16 * rays.shape[0] + 11 + 21 * 5 / 16)
# plt.axes().yaxis.grid(which = "major", linewidth = 0.1)

plt.ylabel("Координаты приёмника, м")
plt.xlabel("Время, с")

plt.savefig("{}/Сейсмограмма. Z-компонента.png".format(dir_name), dpi = 400)
print("Сейсмограмма (Z-компонента) сохранена: {} секунд".format(time.time() - start_time))

plt.close(fig5)

fig6 = plt.figure()

plt.title("Сейсмограмма (X-комонента). Модель №{}".format(number_string[model_number]))
plt.gca().invert_yaxis()

for i in range(rays.shape[0]):

    plt.fill_between(record_time, gathers_x[i, :] / np.max(abs(gathers_z)) / 1.5 + i,
                     np.ones(record_time.shape) * i,
                     linewidth = 0.3, color = 'r', alpha = 0.5)

plt.yticks(np.arange(0, rays.shape[0], 1), ["{}".format(int(j)) for j in rec_line[0 : rays.shape[0]]],
           fontsize = - 5 / 16 * rays.shape[0] + 11 + 21 * 5 / 16)
# plt.axes().yaxis.grid(which = "major", linewidth = 0.1)

plt.ylabel("Координаты приёмника, м")
plt.xlabel("Время, с")

plt.savefig("{}/Сейсмограмма. X-компонента.png".format(dir_name), dpi = 400)
print("Сейсмограмма (X-компонента) сохранена: {} секунд".format(time.time() - start_time))

plt.close(fig6)

description_file.write("Графики построены и сохранены: %s секунд" % (time.time() - start_time))

print("------------------------------------")
print("\x1b[1;31m max x-displacement = ", np.max(abs(gathers_x)))
print("\x1b[1;32m max y-displacement = ", np.max(abs(gathers_y)))
print("\x1b[1;34m max z-displacement = ", np.max(abs(gathers_z)))

# And finally, let's close all .txt files:

description_file.close()

geom_spread_1.close()
geom_spread_2.close()
hodograph.close()
seismogram_z.close()
seismogram_x.close()
ray_amplitude.close()