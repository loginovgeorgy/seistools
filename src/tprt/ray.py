import numpy as np
from .utils import plot_line_3d
from scipy.optimize import minimize
from .rt_coefficients import iso_rt_coefficients
from .segment import Segment
from .. import seislet

import warnings

WAVECODE = {0: 'vp', 1: 'vs',
            'vp': 0, 'vs': 1}


########################### RAYCODE #######################################
# In order to initialize a ray with a raycode, you should know:
# 1. We define the raycode between source and receiver
# 2. Each element of raycode describes the changing of direction (depth)
#    between local source and receiver
# 3. An element is list as [direction, № of Layer from depth=0, wavecode]
###########################################################################

class Ray(object):
    def __init__(self, sou, rec, vel_mod, raycode=None,
                 init_trajectory=None, auto_optimize=False, vtype='vp'):
        """
        :param sou: object of type of Source
        :param rec: object of type of Receiver
        :param vel_mod: object of type of Velocity_model
        :param raycode: np.array([[+1 (down) or -1 (up), number of a layer, type of wave 0,1]])
        :param init_trajectory: is a array of initial coordinates of Ray [x, y]
        :param auto_optimize: True if you need optimize Ray immediately, False if you don't
        """

        self.source = sou
        self.receiver = rec
        self.velmod = vel_mod

        try:
            self.check_raycode(raycode)
        except RaycodeError:
            raise RaycodeError('Raycode is initialized not correctly!')

        self.raycode = raycode

        self.segments = self._initial_ray(init_trajectory)

        self.ray_amplitude = np.array([1, 0, 0])  # this initial amplitude should be replaced by the right one in the
        # optimize method.

        self.traveltime = self.get_travel_time()  # initial value

        if auto_optimize:
            self.optimize()

    def check_raycode(self, raycode):
        if np.any(raycode[:,0]) ==0:
            return True

        # 1. Check the source layer
        sou_layer = self._get_location_layer(self.source.location, self.velmod)
        sou_corresponding = (raycode[0, 1] == sou_layer.number)
        if not sou_corresponding:
            raise RaycodeError('Source layer does not correspond to the first layer of raycode!')

        # 2. Check the receiver layer
        rec_layer = self._get_location_layer(self.receiver.location, self.velmod)
        rec_corresponding = (raycode[-1, 1] == rec_layer.number)
        if not rec_corresponding:
            raise RaycodeError('Receiver layer does not correspond to the last layer of raycode!')

        # 3. Check the sequence of directions and layers
        # Changing of directions is occurred inside the same layer only.
        # If layer number is changing,
        # then it is head wave (for FlatHorizon) or penetrated wave (for GridHorizon).
        # So, if direction is not changing inside the same layer, then it is Error
        diff = np.diff(raycode[:, :-1], axis=0)

        for i, d in enumerate(diff):
            if d[0] == 0 and d[1] != raycode[i, 0]:
                raise RaycodeError('Raycode has incorrect sequence of layers and directions!')

            # if v2 > v1, then head wave can be an exception

            v1 = self.velmod.layers[raycode[i, 1]].get_velocity(0)[WAVECODE[raycode[i, -1]]]
            v2 = self.velmod.layers[raycode[i+1, 1]].get_velocity(0)[WAVECODE[raycode[i+1, -1]]]

            if d[0] != 0 and d[1] != 0 and v2 < v1:
                raise RaycodeError('Wave direction must be changed inside the same layer!')

        return True

    @staticmethod
    def _get_single_segment(sou, raycode, vel_mod, x=None):
        (sign, l, vtype) = raycode
        layer = vel_mod.layers[l]
        end_horizon = layer.code_horizon[sign]
        start_horizon = layer.code_horizon[-sign]
        if np.any(x):
            rec = np.array([x[0], x[1], end_horizon.get_depth(x)])
        else:
            rec = sou + 5*np.random.random(len(sou))
            rec = np.array([rec[0], rec[1], end_horizon.get_depth(rec[:2])])

        seg = Segment(sou, rec, layer, start_horizon, end_horizon, vtype=WAVECODE[vtype])
        return seg

    @staticmethod
    def get_raycode(sou, rec, reflect_horizon, vel_mod, vtype='vp', forward=False):
        # Only for reflected waves. Source and receiver layers must coincide
        if forward:
            return Ray._raycode_forward(sou, rec, vel_mod, vtype)
        else:
            return Ray._raycode_reflect(sou, rec, reflect_horizon, vel_mod, vtype)
    
    @staticmethod
    def _raycode_forward(sou, rec, vel_mod, vtype):
        sou_layer = Ray._get_location_layer(sou.location, vel_mod).number
        rec_layer = Ray._get_location_layer(rec.location, vel_mod).number

        d = np.sign(rec_layer - sou_layer)
        raycode = np.empty(shape=(0, 3), dtype=int)
        i_layer = sou_layer
        if d!=0:
            while rec_layer != i_layer-1:
                raycode = np.concatenate((raycode, np.array([[d, i_layer, WAVECODE[vtype]]], dtype=int),), axis=0)
                i_layer += d
        if d==0:
            raycode = np.concatenate((raycode, np.array([[d, i_layer, WAVECODE[vtype]]], dtype=int),), axis=0)
        
        return raycode
            
    @staticmethod
    def _raycode_reflect(sou, rec, reflect_horizon, vel_mod, vtype):
        sou_layer = Ray._get_location_layer(sou.location, vel_mod).number
        rec_layer = Ray._get_location_layer(rec.location, vel_mod).number

        raycode = np.empty(shape=(reflect_horizon, 3), dtype=int)
        for i in range(reflect_horizon):
            raycode[i] = np.array([+1, sou_layer + i, WAVECODE[vtype]], dtype=int)

        i_layer = reflect_horizon + sou_layer - 1
        while rec_layer != i_layer+1:
            raycode = np.concatenate((raycode, np.array([[-1, i_layer, WAVECODE[vtype]]], dtype=int),), axis=0)
            i_layer -= 1
            
        return raycode
        
    def _initial_ray(self, init_trj):

        sou = np.array(self.source.location, ndmin=1)
        receiver = np.array(self.receiver.location, ndmin=1)
        raycode = self.raycode
        vel_mod = self.velmod

        segments = []
        
        for k in range(raycode.shape[0]-1):
            if not np.any(init_trj):
                x = None
            else:
                x = init_trj[k]

            seg = self._get_single_segment(sou, raycode[k], vel_mod, x)
            segments.append(seg)
            sou = seg.receiver

        last_layer = self._get_location_layer(receiver, vel_mod)
        segments.append(Segment(sou, receiver, last_layer, start_horizon=last_layer.code_horizon[raycode[-1, 0]],
                                end_horizon=None, vtype=WAVECODE[raycode[-1, -1]]))
        segments[0].start_horizon = None
        
        return segments

    def _forward_ray(self, vel_mod, vtype='vp'):
        # This method is outdated and unused
        source = np.array(self.source.location, ndmin=1)
        receiver = np.array(self.receiver.location, ndmin=1)
        array = []
        for hor in vel_mod.horizons:
            rec = hor.intersect(source, receiver)
            if len(rec) == 0:
                continue
            dist = np.linalg.norm(rec - source)
            array.append([dist, rec, hor])

        intersections, horizons = [], []
        for _, x, y in sorted(array, key=lambda a: a[0]):
            intersections.append(x)
            horizons.append(y)

        segments = []
        sou = np.array(self.source.location, ndmin=1)
        end_horizon = None
        for i, rec in enumerate(intersections):
            layer = self._get_location_layer(sou/2 + rec/2, vel_mod)
            end_horizon = horizons[i]
            start_horizon = horizons[i-1]
            segments.append(Segment(sou, rec, layer, start_horizon, end_horizon, vtype=vtype))
            sou = rec

        layer = self._get_location_layer(sou/2 + receiver/2, vel_mod)
        start_horizon = end_horizon
        segments.append(Segment(sou, receiver, layer, start_horizon, None, vtype=vtype))
        segments[0].start_horizon = None
        return segments

    def _get_trajectory(self):

        trj = np.array(self.source.location, ndmin=1)
        for seg in self.segments:
            trj = np.vstack((trj, seg.receiver))
        return trj

    @staticmethod
    def _get_location_layer(x, vel_mod):
        for i, l in enumerate(vel_mod.layers):
            if l.bottom.get_depth(x[:2]) > x[2] > l.top.get_depth(x[:2]):
                return l

    def get_travel_time(self):
        time = 0.0
        for segment in self.segments:
            time += segment.get_traveltime()
        return time

    def update_segments(self, x):

        sou = self.segments[0].source

        for seg, rec in zip(self.segments, np.reshape(x, (-1, 2))):
            receiver = np.array([rec[0], rec[1], seg.end_horizon.get_depth(rec)])
            seg.receiver = receiver
            seg.source = sou

            sou = receiver

        self.segments[-1].source = sou

    def dtravel(self, survey2D=False):
        def _f(x):
            amount_of_borders = len(self.segments) - 1
            dt = np.zeros((amount_of_borders, 2))             # Производные по dx & dy соответственно,
                                                              # на каждой пересекающей луч границе

            for ind_border in range(amount_of_borders):
                seg1 = self.segments[ind_border]              # Соседние 2 сегмента, около точки на границе
                seg2 = self.segments[ind_border + 1]

                dist1, dist2 = seg1.get_distance(), seg2.get_distance()
                vec1, vec2 = seg1.get_vector(), seg2.get_vector()
                gradient = seg1.end_horizon.get_gradient(seg1.receiver[:-1])

                v1 = seg1.layer.get_velocity(vec1)[seg1.vtype]
                v2 = seg2.layer.get_velocity(vec2)[seg2.vtype]

                dv1 = seg1.layer.get_dv(vec1)[seg1.vtype]
                dv2 = seg2.layer.get_dv(vec2)[seg2.vtype]

                dt[ind_border] += (seg1.receiver[:-1] - seg1.source[:-1] +
                                   (seg1.receiver[-1] - seg1.source[-1])*gradient)/dist1/v1
                dt[ind_border] -= dist1 * dv1 / (v1 ** 2)

                dt[ind_border] -= (seg2.receiver[:-1] - seg2.source[:-1] +
                                   (seg2.receiver[-1] - seg2.source[-1])*gradient)/dist2/v2
                dt[ind_border] += dist2 * dv2 / (v2 ** 2)

            if survey2D:
                dt[:, 1] *= self._2d_parametrization(self.source.location, self.receiver.location, dy=survey2D)
                dt = dt.sum(axis=1)

            return dt.ravel()
        return _f

    @staticmethod
    def _2d_parametrization(sou, rec, dy=False):
        x0, x1 = sou[0], rec[0]
        y0, y1 = sou[1], rec[1]

        def y(x):
            return y0 + (y1-y0)/(x1-x0+1e-16)*(x-x0)
        if dy:
            return (y1-y0)/(x1-x0+1e-16)
        return y

    @staticmethod
    def to_3d(x, y):
        x_3d = np.array([[xi, yi] for xi, yi in zip(x, y(x))])
        return x_3d.ravel()

    def optimize(self, method='BFGS', tol=1e-32, Ferma=True,
                 snells_law=False, projection=False, dtravel=False, survey2D=False):

        if not Ferma and snells_law and dtravel:
            print('ERROR: Here must be at least 1 term, for example, Ferma=True')

        x0 = self._get_trajectory()[1:-1, :-1-survey2D]

        if not np.any(x0):
            return

        # Определим функционал для минимизации, он будет состоять из слагаемых:
        # 1. Время вдоль пути (которое мы минимизируем согласно принципу Ферма)
        # 2. Выполнение закона Снеллиуса (либо через синусы, либо через проекции)
        # 3. Производная вдоль луча. Там где время минимально, она должна быть = 0

        def _fun(x):
            if survey2D:
                y = self._2d_parametrization(self.source.location, self.receiver.location)
                x = self.to_3d(x, y)

            self.update_segments(x)

            f1 = 0.0
            if Ferma:
                f1 += self.get_travel_time()
            f2 = 0.0
            if snells_law:
                f2 += (self.snells_law(projection=projection)**2).mean()
            f3 = 0.0
            if dtravel:
                f3 += (self.dtravel(survey2D=survey2D)(0)**2).mean()

            return f1 + f2 + f3

        # Minimization and its options
        jac = None
        ops = None
        if method.casefold()=='bfgs':
            jac = self.dtravel(survey2D=survey2D)
        elif method.casefold()=='nelder-mead':
            ops = {'adaptive': True}

        xs = minimize(_fun, x0.ravel(), method=method, jac=jac, tol=tol, options=ops)
        self.traveltime = xs.fun
        # self.amplitude_fun = self.amplitude_fr_dom() # rewrite the amplitude field.

        return

    def plot(self, **kwargs):
        plot_line_3d(self._get_trajectory().T, **kwargs)
        return

    def snells_law(self, projection=True):
        if not projection: return self._snells_law_by_sin()
        return self._snells_law_by_projection()

    def _snells_law_by_sin(self):
        amount = len(self.segments) - 1  # Amount of boundaries

        # critic = []
        snell = []

        for i in range(amount):
            r1 = self.segments[i].get_vector()      # vector before boundary
            r2 = self.segments[i + 1].get_vector()  # vector after boundary

            normal = self.segments[i].end_horizon.get_normal()

            sin_r1 = np.sqrt(1 - r1.dot(normal) ** 2)  # -//- and r
            sin_r2 = np.sqrt(1 - r2.dot(normal) ** 2)  # sin of angle between normal and r_1

            v1 = self.segments[i].layer.get_velocity(r1)[self.segments[i].vtype]
            v2 = self.segments[i + 1].layer.get_velocity(r2)[self.segments[i + 1].vtype]

            # if v1 < v2:
            #     critic.append(sin_r1 >= v1 / v2)  # checking of critic angle
            # else:
            #     critic.append(False)

            # if np.array(critic).any():
            #     raise SnelliusError('На границе {} достигнут критический угол'.format(i + 1))

            snell.append(abs(sin_r1 * v2 - v1 * sin_r2))

        return np.array(snell)

    def _snells_law_by_projection(self):
        amount = len(self.segments) - 1  # Amount of boundaries

        snell = []

        for i in range(amount):
            r1 = self.segments[i].get_vector()  # vector before boundary
            r2 = self.segments[i + 1].get_vector()

            normal = self.segments[i].end_horizon.get_normal(self.segments[i].receiver[0:2])

            v1 = self.segments[i].layer.get_velocity(r1)[self.segments[i].vtype]
            v2 = self.segments[i + 1].layer.get_velocity(r2)[self.segments[i + 1].vtype]

            r2 = r1 - np.dot(normal, r1) * normal * (1 - (v2 / v1))

            pr_r1 = np.dot(normal, r1 / v1)

            pr_r2 = np.dot(normal, r2 / v2)

            snell.append(abs(pr_r1 - pr_r2))

        return np.array(snell)

    def compute_ray_amplitude(self, survey2D=False):
        """Computes vector of the ray's amplitude

        :param survey2D: boolean variable which indicates if ray amplitude should be computed as in 2.5D problem
        :return: displacement vector in the receiver point (in global coordinate system)
        """

        # This method computes amplitude vector in frequency domain. Assuming that the ray passes through N + 1 layers
        # amplitude of P-wave in the observation point can be found using formula:
        #
        # U = psi0 * 1 / v_1**3 / rho_1
        # = t / (4 * np.pi * s*_1) *
        # * Product(sqrt(|det M(i, s*_i) / det M(i, s*_i-1) )|, i = 2, i = N) *
        # * sqrt( |det M(N + 1, s) / det M(N + 1, s*_N)| ) *
        # * Product( k_i )
        #
        # Here s*_i denote i-th point of intersection of the ray with a boundary, k_i is either reflection of
        # transmission coefficient and M(k, s) is matrix M (explained below) in k-th segment evaluated at point s.
        # t is a unit vector pointed in the direction of the ray in the observation point. psi0 represents radiation
        # function of the source. It's worth to note that all "s" variables are actually arc lengths of the ray.
        # v_1 and rho_1 are wave velocity and density in the very first segment respectively.
        # Amplitdue of S has similar form.

        # Components of symmetric matrix M are solutions of a system of nonlinear differential equations. Physically
        # they represent second derivatives of the eikonal with respect to ray-centered coordinates q1 and q2. They take
        # form:

        # M[0, 0] = (v*s + c_22) / ( (v*s + c_11)*(v*s + c_22) - c_12**2 ),
        # M[1, 1] = (v*s + c_11) / ( (v*s + c_11)*(v*s + c_22) - c_12**2 ),
        # M[0, 1] = - c_12 / ( (v*s + c_11)*(v*s + c_22) - c_12**2 ),
        # M[1, 0] = - c_12 / ( (v*s + c_11)*(v*s + c_22) - c_12**2 ), where s is the ray's path length, v is velocity in
        # the medium, c_ij are unknown constants.

        # For more information see Červený V., Hron F. The ray series method and dynamic ray tracing system for
        # three-dimensional inhomogeneous media //Bulletin of the Seismological Society of America. – 1980. – vol. 70. –
        # №. 1. – pp. 47-77.

        # If survey2D is True all computations are performed as in 2.5D case.

        # So, let's get started:

        # Set the first segment and layer:
        first_segment = self.segments[0]
        first_layer = self.segments[0].layer

        # Compute amplitude in the end point of the first segment:
        U = self.source.source_radiation(first_segment.receiver, first_segment.vtype)

        # Now there are two variants:
        if len(self.segments) == 1:  # if there is only one segment, return amplitude

            return U

        else:  # if there are more segments, we have to perform all computations according to the formula above

            dist = first_segment.get_distance()  # distance along the ray, s in all commented formulae

            M = np.zeros((2, 2))
            M[0, 0] = 1 / (first_layer.get_velocity(1)[first_segment.vtype] * dist)  # matrix M in the end of the first
            M[1, 1] = 1 / (first_layer.get_velocity(1)[first_segment.vtype] * dist)  # segment

            # Ray-centered triplet:
            t = first_segment.get_vector()  # unit vector pointed in the direction of wave's propagation

            if t[2] == 1:  # if t is strictly vertical we cannot distinguish SV- and SH-polarized waves analytically:
                e1 = np.array([1, 0, 0])
                e2 = np.array([0, 1, 0])

            else:  # if not, we can:
                e2 = np.array([t[1], - t[0], 0]) / np.sqrt(t[1] ** 2 + t[0] ** 2)  # SH-polarized unit vector
                e1 = np.cross(e2, t)  # SV-polarized unit vector

            for i in np.arange(1, len(self.segments), 1):  # cycle over all remaining segments

                if np.linalg.norm(U) == 0:  # If amplitude becomes zero, return it:

                    warnings.warn("Zero amplitude in the start point of segment №{}.\n U = {}".format(i + 1, U))
                    return U

                # If not, proceed further:
                # Set current and previous segments, layers and velocities:
                prev_segment = self.segments[i - 1]
                prev_layer = self.segments[i - 1].layer
                prev_vel =prev_layer.get_velocity(0)[prev_segment.vtype]

                curr_segment = self.segments[i]
                curr_layer = self.segments[i].layer
                curr_vel = curr_layer.get_velocity(0)[curr_segment.vtype]

                # At every boundary matrix M changes according to special boundary conditions which take form:
                # M' = S W.T M W S + u* G D G,
                # where ' indicates the next layer, S = ([-+cos(inc_angle)/cos(tr_angle), 0], [0, 1]),
                # u = cos(inc_angle)/v +- cos(tr_angle)/v', G = ([+- 1/cos(tr_angle), 0], [0, 1]) and D is curvature
                # matrix (i.e. Hessian matrix) of the boundary in the point of intersection.
                # inc_angle - angle of incidence, out_angle - angle of transmission or reflection - depending on the
                # particular raycode. Signs also depend on the raycode.
                # W is a rotation matrix described below.

                # Let's form up necessary entities.

                # First, let's understand what happens at the boundary: reflection or transmission.
                if prev_layer == curr_layer:
                    rt_sign = - 1

                else:
                    rt_sign = 1

                # Curvature matrix and matrix of transition to the local system:
                D, loc_sys = prev_segment.end_horizon.get_local_properties(
                    prev_segment.receiver[0:2],
                    prev_segment.get_vector(),
                    survey2D=survey2D
                )

                # cosines of inc_angle and tr_angle:
                cos_inc = abs(np.dot(prev_segment.get_vector(), loc_sys[:, 2]))
                cos_out = abs(np.dot(curr_segment.get_vector(), loc_sys[:, 2]))

                S = np.array([
                    [rt_sign * cos_inc / cos_out, 0],
                    [0, 1]
                ])
                G = np.array([
                    [- rt_sign * 1 / cos_out, 0],
                    [0, 1]
                ])
                u = cos_inc / prev_vel - rt_sign * cos_out / curr_vel

                # The incident ray's e2 vector can make angle with the d2 so that:
                # dot(e2, d2) = sin(omega)

                # The matrix W is a rotation matrix:
                # W = np.array([[ cos(omega), sin(omega)],
                #               [ - sin(omega), cos(omega)]])

                # sin_omega = np.dot(e1, transit_matr[:, 1])
                # cos_omega = np.dot(e2, transit_matr[:, 1])

                W = np.array([
                    [np.dot(e2, loc_sys[:, 1]), np.dot(e1, loc_sys[:, 1])],
                    [- np.dot(e1, loc_sys[:, 1]), np.dot(e2, loc_sys[:, 1])]
                ])

                # Since all layers are homogeneous and isotropic the general solution for matrix M (i.e. its form)
                # remains the same:
                # M[0, 0] = (v*s + c_22) / ( (v*s + c_11)*(v*s + c_22) - c_12**2 ),
                # M[1, 1] = (v*s + c_11) / ( (v*s + c_11)*(v*s + c_22) - c_12**2 ),
                # M[0, 1] = - c_12 / ( (v*s + c_11)*(v*s + c_22) - c_12**2 ),
                # M[1, 0] = - c_12 / ( (v*s + c_11)*(v*s + c_22) - c_12**2 )

                # We can compute constants c_i by in terms of N = M**(-1). But first of all let's find new value of M
                # (i.e. M'). Note that new matrix M is related to the new triplet t, e1, e2 where e2 is coincident with
                # d2.

                M = np.einsum("ij, jk, kl, lm, mn", S, np.transpose(W), M, W, S) + u * np.einsum("ik, kl, ln", G, D, G)
                N = np.linalg.inv(M)

                # Remember that:
                # N[0, 0] = v * s + c_11
                # N[1, 1] = v * s + c_22
                # N[0, 1] = c_12
                # N[1, 0] = c_12

                # So, all equations for c_ij are linear and they have explicit solution:
                c_11 = N[0, 0] - curr_vel * dist
                c_22 = N[1, 1] - curr_vel * dist
                c_12 = N[0, 1]

                # Now we have a new value of M. Let's introduce / re-evaluate ratio
                # det_rat = det M(i, s*_i) / det M(i, s*_i-1)

                det_rat = 1 / np.linalg.det(M)  # now we have only denominator

                # Reflection / transmission coefficients are computed in local coordinate system. We have to find
                # slowness and polarization of the incident wave in this system:
                inc_slowness = np.dot(
                    np.transpose(loc_sys),
                    t / prev_vel
                )
                inc_polariz = np.dot(np.transpose(loc_sys), U)

                ampl_coeff = iso_rt_coefficients(
                    inc_slowness=inc_slowness,
                    inc_polariz=inc_polariz,
                    rt_signum=rt_sign,
                    vp1=self.velmod.layers[self.raycode[i - 1][1]].get_velocity(0)["vp"],
                    vs1=self.velmod.layers[self.raycode[i - 1][1]].get_velocity(0)["vs"],
                    rho1=self.velmod.layers[self.raycode[i - 1][1]].get_density(),
                    vp2=self.velmod.layers[self.raycode[i - 1][1] + self.raycode[i - 1][0]].get_velocity(0)["vp"],
                    vs2=self.velmod.layers[self.raycode[i - 1][1] + self.raycode[i - 1][0]].get_velocity(0)["vs"],
                    rho2=self.velmod.layers[self.raycode[i - 1][1] + self.raycode[i - 1][0]].get_density()
                )

                # Let's go further, to the next boundary:
                dist = dist + curr_segment.get_distance()

                M[0, 0] = (curr_vel * dist + c_22) / ((curr_vel * dist + c_11) * (curr_vel * dist + c_22) - c_12**2)
                M[1, 1] = (curr_vel * dist + c_11) / ((curr_vel * dist + c_11) * (curr_vel * dist + c_22) - c_12**2)
                M[0, 1] = - c_12 / ((curr_vel * dist + c_11) * (curr_vel * dist + c_22) - c_12**2)
                M[1, 0] = M[0, 1]

                det_rat = det_rat * np.linalg.det(M)  # full value of the detRat

                # New ray-centered triplet. Note that e2 = d2 where d2 is a vector from the local coordinate system:
                t = curr_segment.get_vector()
                e2 = loc_sys[:, 1]
                e1 = np.cross(e2, t)

                # Compute amplitude in the end of current segment:
                if curr_segment.vtype == "vp":

                    U = np.linalg.norm(U) * ampl_coeff[0] * np.sqrt(abs(det_rat)) * t

                if curr_segment.vtype == "vs":

                    U = (np.linalg.norm(U) * ampl_coeff[1] * e1 +
                         np.linalg.norm(U) * ampl_coeff[2] * e2) * np.sqrt(abs(det_rat))

                # That's all for this layer.

            return U

    def compute_spreading(self, inv_bool, survey2D=False):
        # Computes only geometrical spreading along the ray in the observation point. All comments are above.

        # curv_factor is an array of two boolean variables. curv_factor[0] indicates if we have to consider curvature
        # in the transmission points. curv_factor[1] indicates the same for the reflection point.
        # 1 = consider, 0 = don't consider

        # inv_bool is a boolean variable. It indicates whether to take into account ratios J(x+) / J(x-) or not.
        # J(x+) is geometrical spreading just below the interface and J(x-) is geometrical spreading just above it.
        # Note that "inv" comes from "inversion" since all these ratios vanish in the expression for amplitude. So,
        # in AVO-inversion they will not be needed.

        first_segment = self.segments[0]
        first_layer = self.segments[0].layer

        M = np.zeros((2, 2))

        dist = first_segment.get_distance() # this will be distance along the ray

        M[0, 0] = 1 / (first_layer.get_velocity(1)[first_segment.vtype] * dist)
        M[1, 1] = 1 / (first_layer.layer.get_velocity(1)[first_segment.vtype] * dist)

        t = first_segment.get_vector()

        if t[2] == 1:

            e1 = np.array([1, 0, 0])
            e2 = np.array([0, 1, 0])

        else:

            e2 = np.array([t[1], - t[0], 0]) / np.sqrt(t[1] ** 2 + t[0] ** 2)
            e1 = np.cross(e2, t)

        J = dist**2

        if len(self.segments) == 1:

            return J

        else:

            for i in np.arange(1, len(self.segments), 1):

                # Set current and previous segments, layers and velocities:
                prev_segment = self.segments[i - 1]
                prev_layer = self.segments[i - 1].layer
                prev_vel = prev_layer.get_velocity(0)[prev_segment.vtype]

                curr_segment = self.segments[i]
                curr_layer = self.segments[i].layer
                curr_vel = curr_layer.get_velocity(0)[curr_segment.vtype]

                if prev_layer == curr_layer:
                    rt_sign = - 1

                else:
                    rt_sign = 1

                D, loc_sys = prev_segment.end_horizon.get_local_properties(
                    prev_segment.receiver[0:2],
                    prev_segment.get_vector(),
                    survey2D=survey2D
                )

                cos_inc = abs(np.dot(prev_segment.get_vector(), loc_sys[:, 2]))
                cos_out = abs(np.dot(curr_segment.get_vector(), loc_sys[:, 2]))

                S = np.array([
                    [rt_sign * cos_inc / cos_out, 0],
                    [0, 1]
                ])
                G = np.array([
                    [- rt_sign * 1 / cos_out, 0],
                    [0, 1]
                ])
                u = cos_inc / prev_vel - rt_sign * cos_out / curr_vel

                W = np.array([
                    [np.dot(e2, loc_sys[:, 1]), np.dot(e1, loc_sys[:, 1])],
                    [- np.dot(e1, loc_sys[:, 1]), np.dot(e2, loc_sys[:, 1])]
                ])

                M = np.einsum("ij, jk, kl, lm, mn", S, np.transpose(W), M, W, S) + u * np.einsum("ik, kl, ln", G, D, G)
                N = np.linalg.inv(M)

                c_11 = N[0, 0] - curr_vel * dist
                c_22 = N[1, 1] - curr_vel * dist
                c_12 = N[0, 1]

                det_rat = 1 / np.linalg.det(M)  # now we have only denominator

                dist = dist + curr_segment.get_distance()

                M[0, 0] = (curr_vel * dist + c_22) / ((curr_vel * dist + c_11) * (curr_vel * dist + c_22) - c_12 ** 2)
                M[1, 1] = (curr_vel * dist + c_11) / ((curr_vel * dist + c_11) * (curr_vel * dist + c_22) - c_12 ** 2)
                M[0, 1] = - c_12 / ((curr_vel * dist + c_11) * (curr_vel * dist + c_22) - c_12 ** 2)
                M[1, 0] = M[0, 1]

                det_rat = det_rat * np.linalg.det(M)  # full value of the detRat

                t = curr_segment.get_vector()
                e2 = loc_sys[:, 1]
                e1 = np.cross(e2, t)

                if inv_bool == 0:

                    J = J * cos_out / cos_inc

                J = J / abs(det_rat)

            return J

    def compute_coefficients(self, survey2D=False):

        first_segment = self.segments[0]
        U = self.source.source_radiation(first_segment.receiver, first_segment.vtype)

        t = first_segment.get_vector()  # unit vector pointed in the direction of wave's propagation

        if t[2] == 1:
            e1 = np.array([1, 0, 0])
            e2 = np.array([0, 1, 0])

        else:
            e2 = np.array([t[1], - t[0], 0]) / np.sqrt(t[1] ** 2 + t[0] ** 2)  # SH-polarized unit vector
            e1 = np.cross(e2, t)  # SV-polarized unit vector

        if len(self.segments) == 1:

            return np.array([])

        else:

            refl_trans_coeff = np.zeros(len(self.segments) - 1, dtype = complex)

            for i in np.arange(1, len(self.segments), 1):

                prev_segment = self.segments[i - 1]
                prev_layer = self.segments[i - 1].layer
                prev_vel = prev_layer.get_velocity(0)[prev_segment.vtype]

                curr_segment = self.segments[i]
                curr_layer = self.segments[i].layer
                curr_vel = curr_layer.get_velocity(0)[curr_segment.vtype]

                if prev_layer == curr_layer:
                    rt_sign = - 1

                else:
                    rt_sign = 1

                _, loc_sys = prev_segment.end_horizon.get_local_properties(
                    prev_segment.receiver[0:2],
                    prev_segment.get_vector(),
                    survey2D=survey2D
                )

                inc_slowness = np.dot(
                    np.transpose(loc_sys),
                    t / prev_vel
                )
                inc_polariz = np.dot(np.transpose(loc_sys), U)

                ampl_coeff = iso_rt_coefficients(
                    inc_slowness=inc_slowness,
                    inc_polariz=inc_polariz,
                    rt_signum=rt_sign,
                    vp1=self.velmod.layers[self.raycode[i - 1][1]].get_velocity(0)["vp"],
                    vs1=self.velmod.layers[self.raycode[i - 1][1]].get_velocity(0)["vs"],
                    rho1=self.velmod.layers[self.raycode[i - 1][1]].get_density(),
                    vp2=self.velmod.layers[self.raycode[i - 1][1] + self.raycode[i - 1][0]].get_velocity(0)["vp"],
                    vs2=self.velmod.layers[self.raycode[i - 1][1] + self.raycode[i - 1][0]].get_velocity(0)["vs"],
                    rho2=self.velmod.layers[self.raycode[i - 1][1] + self.raycode[i - 1][0]].get_density()
                )

                t = curr_segment.get_vector()
                e2 = loc_sys[:, 1]
                e1 = np.cross(e2, t)

                if curr_segment.vtype == "vp":

                    refl_trans_coeff[i - 1] = ampl_coeff[0]
                    U = np.linalg.norm(U) * ampl_coeff[0] * t

                if curr_segment.vtype == "vs":

                    refl_trans_coeff[i - 1] = np.sqrt(ampl_coeff[1] ** 2 + ampl_coeff[2] ** 2)
                    U = np.linalg.norm(U) * ampl_coeff[1] * e1 +\
                        np.linalg.norm(U) * ampl_coeff[2] * e2

            return refl_trans_coeff

    def get_inc_cosines(self, survey2D=False):

        if len(self.segments) == 1:

            return np.array([])

        else:

            inc_cosines = np.zeros(len(self.segments) - 1)

            for i in np.arange(1, len(self.segments), 1):

                prev_segment = self.segments[i - 1]

                _, loc_sys = prev_segment.end_horizon.get_local_properties(
                    prev_segment.receiver[0:2],
                    prev_segment.get_vector(),
                    survey2D=survey2D
                )

                inc_cosines[i - 1] = abs(np.dot(prev_segment.get_vector(), loc_sys[:, 2]))

            return inc_cosines

    def get_recorded_amplitude(self, times):

        # Let's generate our wavelet:
        wavelet = seislet.seismic_signal(
            signal=self.source.wavelet_name,
            t=times,
            tau=self.get_travel_time(),
            **self.source.wavelet_parameters
        )

        # Receiver's sensitivity axes may not correspond to those of the global coordinate system. So, ray amplitude
        # vector must be rotated:
        rec_amplitude = np.dot(self.receiver.orientation.T, self.ray_amplitude)
        rec_amplitude = np.real(rec_amplitude)  # it can be complex, but seismographs detect only real component

        # We want to see array with shape (3, len(times)) so that its zeroth component would correspond to X-axis
        # record, second component - to Y-axis record and the third component - to Z-axis record:
        return wavelet * np.reshape(rec_amplitude, (3, 1))


class RaycodeError(Exception):
    """Exception raised for errors in the input."""

    def __init__(self, msg):
        self.message = msg