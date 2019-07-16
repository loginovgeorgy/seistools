import numpy as np
from .utils import plot_line_3d
from scipy.optimize import minimize
from .rt_coefficients import iso_rt_coefficients
from .segment import Segment
from .. import seislet

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

    def compute_ray_amplitude(self):

        # This method computes amplitude vector in frequency domain. Assuming that the ray passes through N + 1 layers
        # amplitude of P-wave in the observation point can be found using formula:
        #
        # U = psi0 * 1 / v_1**3 / rho_1
        # = t / (4 * np.pi * s*_1) *
        # * П(sqrt(|det M(i, s*_i) / det M(i, s*_i-1) )|, i = 2, i = N) *
        # * sqrt( |det M(N + 1, s) / det M(N + 1, s*_N)| ) *
        # * П( k_i )
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

        # So, let's get started:

        # dist = 0  # arc length of the ray
        M = np.zeros((2, 2))  # matrix M

        # Set first segment and layer:

        first_segment = self.segments[0]
        first_layer = self.segments[0].layer

        # Let's evaluate dist and M in the end of the first segment assuming that c11 = c22 = c12 = 0:

        dist = first_segment.get_distance() # this will be distance along the ray

        M[0, 0] = 1 / (first_layer.get_velocity(1)[first_segment.vtype] * dist)
        M[1, 1] = 1 / (first_layer.get_velocity(1)[first_segment.vtype] * dist)

        # Now we are ready to write down ray amplitude in the end of the first segment. It will be written in terms of
        # ray-centered coordinates, so let's set corresponding unit vectors:

        t = first_segment.get_vector()  # unit vector pointed in the direction of wave's propagation

        # We cannot distinguish SV and SH polarization if t is strictly vertical. In that case we'll just set e1 and
        # e2 coincident with i ang j unit vectors of the global Cartesian coordinates:

        if t[2] == 1:

            e1 = np.array([1, 0, 0])
            e2 = np.array([0, 1, 0])

        else:

            e2 = np.array([t[1], - t[0], 0]) / np.sqrt(t[1] ** 2 + t[0] ** 2)  # SH-polarized unit vector
            e1 = np.cross(e2, t)  # SV-polarized unit vector

        # In the first segment the polarization vector depends only on the source. But we should also take into account
        # the raycode.

        if self.segments[0].vtype == 'vp':

            U = self.source.psi0(first_segment.receiver, t) * t / dist

        # if self.segments[0].vtype == 'vs':
        else:

            U = (self.source.psi0(first_segment.receiver, e1) * e1 +
                 self.source.psi0(first_segment.receiver, e2) * e2) / dist

        # Now there are two opportunities. First, the ray can consist of just one segment. Second, it can have multiple
        # segments.

        if len(self.segments) == 1:
            # If there is only one segment, calculate the amplitude in
            # the receiver and return it.

            U = U / (4 * np.pi * first_layer.get_velocity(0)[first_segment.vtype]**3 * first_layer.get_density())

            return U

        else:

            # If there are some boundaries on the ray's path, we shall need to carry out more complicated calculations.

            for i in np.arange(1, len(self.segments), 1):

                # Set current and previous segments and layers:
                prev_segment = self.segments[i - 1]
                prev_layer = self.segments[i - 1].layer

                curr_segment = self.segments[i]
                curr_layer = self.segments[i].layer

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
                    prev_segment.get_vector()
                )

                # cosines of inc_angle and tr_angle:
                cos_inc = abs(np.dot(prev_segment.get_vector(), loc_sys[:, 2]))
                cos_out = abs(np.dot(curr_segment.get_vector(), loc_sys[:, 2]))

                S = np.array([[rt_sign * cos_inc / cos_out, 0],
                              [0, 1]])
                G = np.array([[- rt_sign * 1 / cos_out, 0],
                              [0, 1]])

                u = cos_inc / prev_layer.get_velocity(0)[prev_segment.vtype] -\
                    rt_sign * cos_out / curr_layer.get_velocity(0)[curr_segment.vtype]

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

                M = np.dot(np.dot(S, np.dot(W.T, np.dot(M, W))), S) + u * np.dot(np.dot(G, D), G)
                N = np.linalg.inv(M)

                # Remember that:

                # N[0, 0] = v * s + c_11
                # N[1, 1] = v * s + c_22
                # N[0, 1] = c_12
                # N[1, 0] = c_12

                # So, all equations for c_ij are linear!
                # s = dist, v = v'.

                c_11 = N[0, 0] - curr_layer.get_velocity(1)[curr_segment.vtype] * dist
                c_22 = N[1, 1] - curr_layer.get_velocity(1)[curr_segment.vtype] * dist
                c_12 = N[0, 1]

                # Now we have a new value of M. Let's introduce / re-evaluate ratio
                # det_rat = det M(i, s*_i) / det M(i, s*_i-1)
                # Now we know only the denominator:

                det_rat = 1 / np.linalg.det(M)

                # And we have to not forget about transmission coefficient. In order to compute it we have to rewrite
                # existing vector of polarization U in the local coordinate system. Remember that we've already
                # found necessary transition matrix.

                # Reflection / transmission coefficients are computed in local coordinate system. We have to find
                # slowness and polarization of the incident wave in this system:

                inc_slowness = np.dot(
                    np.transpose(loc_sys),
                    t / prev_layer.get_velocity(0)[prev_segment.vtype]
                )
                inc_polariz = np.dot(np.transpose(loc_sys), U)

                # It would be convenient to give parameters of the layers explicitly:

                vp1 = self.velmod.layers[self.raycode[i - 1][1]].get_velocity(0)["vp"]
                vs1 = self.velmod.layers[self.raycode[i - 1][1]].get_velocity(0)["vs"]
                rho1 = self.velmod.layers[self.raycode[i - 1][1]].get_density()

                vp2 = self.velmod.layers[self.raycode[i - 1][1] + self.raycode[i - 1][0]].get_velocity(0)["vp"]
                vs2 = self.velmod.layers[self.raycode[i - 1][1] + self.raycode[i - 1][0]].get_velocity(0)["vs"]
                rho2 = self.velmod.layers[self.raycode[i - 1][1] + self.raycode[i - 1][0]].get_density()

                ampl_coeff = iso_rt_coefficients(
                    inc_slowness=inc_slowness,
                    inc_polariz=inc_polariz,
                    rt_signum=rt_sign,
                    vp1=vp1,
                    vs1=vs1,
                    rho1=rho1,
                    vp2=vp2,
                    vs2=vs2,
                    rho2=rho2
                )

                # Let's go further, to the next boundary:

                dist = dist + curr_segment.get_distance()

                M[0, 0] = (curr_layer.get_velocity(1)[curr_segment.vtype] * dist + c_22) / \
                          ((curr_layer.get_velocity(1)[curr_segment.vtype] * dist + c_11) *
                           (curr_layer.get_velocity(1)[curr_segment.vtype] * dist + c_22) - c_12**2)

                M[1, 1] = (curr_layer.get_velocity(1)[curr_segment.vtype] * dist + c_11) / \
                          ((curr_layer.get_velocity(1)[curr_segment.vtype] * dist + c_11) *
                           (curr_layer.get_velocity(1)[curr_segment.vtype] * dist + c_22) - c_12**2)

                M[0, 1] = - c_12 / \
                          ((curr_layer.get_velocity(1)[curr_segment.vtype] * dist + c_11) *
                           (curr_layer.get_velocity(1)[curr_segment.vtype] * dist + c_22) - c_12**2)

                M[1, 0 ] = M[0, 1]

                # Full value of the detRat:

                det_rat = det_rat * np.linalg.det(M)

                # That's all for this segment. We are almost ready to rewrite the value of amlitude. But first consider
                # the following.
                # ampl_coeff is a list of 6 coefficients which specify the amplitude changes of the wave. They are
                # related to t, e1, e2 triplets of the outgoing waves. But we work just with one of them.

                # Let's specify its triplet. We should take into account that e2 = d2 where d2 is a vector from the
                # local coordinate system.

                t = curr_segment.get_vector()
                e2 = loc_sys[:, 1]
                e1 = np.cross(e2, t)

                # Finally, everything depends on the current wavetype:

                if curr_segment.vtype == 'vp':

                    U = np.linalg.norm(U) * ampl_coeff[0] * np.sqrt(abs(det_rat)) * t

                if curr_segment.vtype == 'vs':

                    U = (np.linalg.norm(U) * ampl_coeff[1] * e1 +
                         np.linalg.norm(U) * ampl_coeff[2] * e2) * np.sqrt(abs(det_rat))

                # Let's go to the next layer!

            # We've computed the amplitude in the cycle above. Let's return it's value, but before that we have
            # to add some coefficients related to the source's layer:

            U = U / (4 * np.pi * first_layer.get_velocity(0)[first_segment.vtype]**3 * first_layer.get_density())

            return U

    def compute_coefficients(self):

        first_segment = self.segments[0]

        t = first_segment.get_vector()  # unit vector pointed in the direction of wave's propagation

        if t[2] == 1:

            e1 = np.array([1, 0, 0])
            e2 = np.array([0, 1, 0])

        else:

            e2 = np.array([t[1], - t[0], 0]) / np.sqrt(t[1] ** 2 + t[0] ** 2)  # SH-polarized unit vector
            e1 = np.cross(e2, t)  # SV-polarized unit vector

        if self.segments[0].vtype == 'vp':

            U = self.source.psi0(first_segment.receiver, t) * t

        else:

            U = self.source.psi0(first_segment.receiver, e1) * e1 +\
                self.source.psi0(first_segment.receiver, e2) * e2

        if len(self.segments) == 1:

            return np.array([])

        else:

            refl_trans_coeff = np.zeros(len(self.segments) - 1, dtype = complex)

            for i in np.arange(1, len(self.segments), 1):

                prev_segment = self.segments[i - 1]
                prev_layer = self.segments[i - 1].layer

                curr_segment = self.segments[i]
                curr_layer = self.segments[i].layer

                if prev_layer == curr_layer:

                    rt_sign = - 1

                else:

                    rt_sign = 1

                _, loc_sys = prev_segment.end_horizon.get_local_properties(
                    prev_segment.receiver[0:2],
                    prev_segment.get_vector()
                )

                inc_slowness = np.dot(
                    np.transpose(loc_sys),
                    t / prev_layer.get_velocity(0)[prev_segment.vtype]
                )
                inc_polariz = np.dot(np.transpose(loc_sys), U)

                vp1 = self.velmod.layers[self.raycode[i - 1][1]].get_velocity(0)["vp"]
                vs1 = self.velmod.layers[self.raycode[i - 1][1]].get_velocity(0)["vs"]
                rho1 = self.velmod.layers[self.raycode[i - 1][1]].get_density()

                vp2 = self.velmod.layers[self.raycode[i - 1][1] + self.raycode[i - 1][0]].get_velocity(0)["vp"]
                vs2 = self.velmod.layers[self.raycode[i - 1][1] + self.raycode[i - 1][0]].get_velocity(0)["vs"]
                rho2 = self.velmod.layers[self.raycode[i - 1][1] + self.raycode[i - 1][0]].get_density()

                ampl_coeff = iso_rt_coefficients(
                    inc_slowness=inc_slowness,
                    inc_polariz=inc_polariz,
                    rt_signum=rt_sign,
                    vp1=vp1,
                    vs1=vs1,
                    rho1=rho1,
                    vp2=vp2,
                    vs2=vs2,
                    rho2=rho2
                )

                t = curr_segment.get_vector()
                e2 = loc_sys[:, 1]
                e1 = np.cross(e2, t)

                if curr_segment.vtype == 'vp':

                    refl_trans_coeff[i - 1] = ampl_coeff[0]

                    U = np.linalg.norm(U) * ampl_coeff[0] * t

                if curr_segment.vtype == 'vs':

                    refl_trans_coeff[i - 1] = np.sqrt(ampl_coeff[1] ** 2 + ampl_coeff[2] ** 2)

                    U = np.linalg.norm(U) * ampl_coeff[1] * e1 +\
                        np.linalg.norm(U) * ampl_coeff[2] * e2

            return refl_trans_coeff

    def get_inc_cosines(self):

        if len(self.segments) == 1:

            return np.array([])

        else:

            inc_cosines = np.zeros(len(self.segments) - 1)

            for i in np.arange(1, len(self.segments), 1):

                prev_segment = self.segments[i - 1]

                normal = prev_segment.end_horizon.get_normal(prev_segment.receiver[0:2])

                inc_cosines[i - 1] = abs(np.dot(prev_segment.get_vector(), normal))

            return inc_cosines

    def get_recorded_amplitude(self, times):
        # returns amplitude vector in the receiver in a particular time moment t.
        # Here I use theory presented in: Popov, M.M. Ray theory and gaussian beam method for geophysicists /
        # M. M. Popov. - Salvador: EDUFBA, 2002. – 172 p.

        # We use formula: A = Ricker(t - tau) * U)
        # where tau is time of the first break (i.e. traveltime along the ray), Ricker is
        # the Ricker wavelet with dispersion given in the Source  and U is a constant vector: self.ray_amplitude.

        # return seislet.seismic_signal(signal="ricker",
        #                               t=times,
        #                               f=self.source.fr_dom) * np.dot(self.receiver.orientation.T, self.ray_amplitude)

        tau = self.get_travel_time()
        sigma = np.sqrt(2) / self.source.fr_dom / 2 / np.pi

        time_set = np.transpose(np.array([times - tau]))  # we assume that times can be either scalar or vector of time
        # moments; transposition is performed for sake of multiplication below.

        return np.transpose(2 / np.sqrt(3 * sigma) / np.pi ** (1 / 4) *\
                            (1 - (time_set/ sigma)**2) *\
                            np.exp(- time_set**2 / (2 * sigma**2)) *\
                            np.dot(self.receiver.orientation.T, self.ray_amplitude))  # we transpose the result so that
        # its zeroth component would correspond to x-component of recoded displacement at any time moment in times

    def spreading(self, curv_factor, inv_bool):
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

                prev_segment = self.segments[i - 1]
                prev_layer = self.segments[i - 1].layer

                curr_segment = self.segments[i]
                curr_layer = self.segments[i].layer

                if prev_layer == curr_layer:

                    rt_sign = - 1

                else:

                    rt_sign = 1

                D, loc_sys = prev_segment.end_horizon.get_local_properties(
                    prev_segment.receiver[0:2],
                    prev_segment.get_vector()
                )

                cos_inc = abs(np.dot(prev_segment.get_vector(), loc_sys[:, 2]))
                cos_out = abs(np.dot(curr_segment.get_vector(), loc_sys[:, 2]))

                S = np.array([[rt_sign * cos_inc / cos_out, 0],
                              [0, 1]])
                G = np.array([[- rt_sign * 1 / cos_out, 0],
                              [0, 1]])

                u = cos_inc / prev_layer.get_velocity(0)[prev_segment.vtype] - \
                    rt_sign * cos_out / curr_layer.get_velocity(0)[curr_segment.vtype]

                W = np.array([
                    [np.dot(e2, loc_sys[:, 1]), np.dot(e1, loc_sys[:, 1])],
                    [- np.dot(e1, loc_sys[:, 1]), np.dot(e2, loc_sys[:, 1])]
                ])

                M = np.dot(np.dot(S, np.dot(W.T, np.dot(M, W))), S) + u * np.dot(np.dot(G, D), G)
                N = np.linalg.inv(M)

                c_11 = N[0, 0] - curr_layer.get_velocity(1)[curr_segment.vtype] * dist
                c_22 = N[1, 1] - curr_layer.get_velocity(1)[curr_segment.vtype] * dist
                c_12 = N[0, 1]

                det_rat = 1 / np.linalg.det(M)

                dist = dist + curr_segment.get_distance()

                M[0, 0] = (curr_layer.get_velocity(1)[curr_segment.vtype] * dist + c_22) / \
                          ((curr_layer.get_velocity(1)[curr_segment.vtype] * dist + c_11) *
                           (curr_layer.get_velocity(1)[curr_segment.vtype] * dist + c_22) - c_12**2)

                M[1, 1] = (curr_layer.get_velocity(1)[curr_segment.vtype] * dist + c_11) / \
                          ((curr_layer.get_velocity(1)[curr_segment.vtype] * dist + c_11) *
                           (curr_layer.get_velocity(1)[curr_segment.vtype] * dist + c_22) - c_12**2)

                M[0, 1] = - c_12 / \
                          ((curr_layer.get_velocity(1)[curr_segment.vtype] * dist + c_11) *
                           (curr_layer.get_velocity(1)[curr_segment.vtype] * dist + c_22) - c_12**2)

                M[1, 0 ] = M[0, 1]

                det_rat = det_rat * np.linalg.det(M)

                t = curr_segment.get_vector()
                e2 = loc_sys[:, 1]
                e1 = np.cross(e2, t)

                if inv_bool == 0:

                    J = J * cos_out / cos_inc

                J = J / abs(det_rat)

            return J


class RaycodeError(Exception):
    """Exception raised for errors in the input."""

    def __init__(self, msg):
        self.message = msg