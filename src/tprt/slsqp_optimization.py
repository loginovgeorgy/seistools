import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so
from pylab import *
from mpl_toolkits.mplot3d import Axes3D as plt3d
from .ray import Segment

class SnelliusError(Exception):
    pass;

def _constraint(c, i):
    def con(x):
        return c[0] * x[3 * i] + c[1] * x[3 * i + 1] + c[2] * x[3 * i + 2] + c[3]

    return con

def get_constraints_plane(borders):
    borders = np.array(borders, ndmin=2)
    cons = []
    for i in range(len(borders)):
        cons.append({'type': 'eq', 'fun': _constraint(borders[i], i)})
    return cons

def get_ray_xyz(Ray, x0=None, vtype='vp', method="SLSQP", tol=1e-32, *args, **kwargs):
    if x0==None:
        x0 = Ray._trajectory[1:-1]
    amount = len(Ray.segments)
    sou = Ray._trajectory[0]
    receiver = Ray._trajectory[-1]
    v = np.array([Ray.segments[k].velocity(*args)[vtype] for k in range(amount)])

    if amount == 1: return traveltime(np.array([]),sou, receiver, v)
    normal = np.array([Ray.segments[k].horizon_normal for k in range(amount - 1)])
    depth = np.array([Ray.segments[k].horizon(np.array([0,0]))*(normal[k,-1]+1e-16) for k in range(amount -1)])
    planes = np.hstack((-normal,depth.reshape((len(depth),1))))

    cons = get_constraints_plane(planes)

    solution = so.minimize(traveltime, x0, args=(sou, receiver, v),
                    method=method, constraints=cons, tol=tol)

    new_segments = []
    x0 = (solution.x).reshape((len(cons),len(sou)))
    for segment, rec in zip(Ray.segments, x0):
        new_segments.append(Segment(sou, rec, segment.velocity, segment.horizon, segment.horizon_normal))
        sou = rec
    new_segments.append(Segment(sou, receiver, Ray.segments[-1].velocity, Ray.segments[-1].horizon, Ray.segments[-1].horizon_normal))
    Ray.segments = new_segments
    return solution.fun

def get_ray_xy(Ray, x0=None, vtype='vp', method="SLSQP", tol=1e-32, *args, **kwargs):
    if x0==None:
        x0 = Ray._trajectory[1:-1,:2]
    if not np.any(x0):
        return Ray.travel_time(vtype=vtype)
    solution = so.minimize(Ray.travel_time, x0.ravel(), args=(vtype),
                    method=method, tol=tol)
    return solution.fun

def traveltime(x0, sou, rec, v):
    dim = len(sou)
    if len(x0) > dim: x0 = x0.reshape(((int32)(len(x0) / dim), dim))
    if (len(x0.shape) == 1 and x0.shape[0] != 0): x0 = x0.reshape((1, len(x0)));
    p = []
    p.append(sou)
    if len(x0) > 0:
        for i in range(len(x0)):
            p.append(x0[i])
    p.append(rec)
    p = np.array(p)
    T = 0
    n = len(x0)
    vel = v[:n + 1]
    for i in range(len(p) - 1):
        T += np.sqrt(sum((p[i] - p[i + 1]) ** 2)) / vel[i]
    return T