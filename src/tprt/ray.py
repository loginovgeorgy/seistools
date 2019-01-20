import numpy as np
from .utils import plot_line_3d
from scipy.optimize import minimize
from functools import partial
from .rt_coefficients import rt_coefficients

WAVECODE = {0: 'vp', 1: 'vs'}

########################### RAYCODE #######################################
## In order to initialize a ray with a raycode, you should know:
## 1. Source and receiver, between them we will define raycode
## 2. Each element of raycode describes the changing of direction (depth)
##    between local source and receiver
## 3. An element is list as [direction, № of Layer from depth=0, wavecode]
############################################################################

class Ray(object):
    def __init__(self, sou, rec, vel_mod, raycode=None):
        """

        :param sou: object of type of Source
        :param rec: object of type of Receiver
        :param vel_mod: object of type of Velocity_model
        :param raycode: np.array([[+1 (down) or -1 (up), number of a layer, type of wave 0,1]])
        """

        self.source = sou
        self.receiver = rec
        self.raycode = raycode
        self.velmod = vel_mod
        self.segments = self._get_init_segments(vel_mod, raycode)
        self.amplitude_fun = np.array([1, 0, 0]) # this initial amplitude will be replaced by the right one in the
        # optimize method. Actually this is the amplitude in the frequency domain. Amplitude in the time domain can be
        # found using a particular method.

    def _get_init_segments(self, vel_mod, raycode):
        if raycode==None: return self._get_forward(vel_mod)

        sou = np.array(self.source.location, ndmin=1)
        receiver = np.array(self.receiver.location, ndmin=1)

        ##############################################################################
        ##########################!!!!!!!!!ALARM!!!!!!!!!!!!##########################
        # HERE IS A TROUBLE WITH INITIAL RAY BEFORE OPTIMIZATION
        # THIS PROBLEM MUST BE SOLVED
        shift_dist = np.linalg.norm(sou - receiver)/20 + 10*np.random.random(len(sou))
        # Now, I have added random to the initialization
        # Otherwise, when the offset is equal to 0 optimization does not work
        # I think, that cause of this problem is related to properties of the optimization method
        ##############################################################################

        segments = []
        # МНОГО, ОЧЕНЬ МНОГО if'ов
        for k, (sign, i, vtype) in enumerate(raycode):
            # raycode: [вниз +1 или вверх -1, номер слоя, тип волны, см. WAVECODE]
            first = k==0
            last = k==len(raycode)-1
            layer = vel_mod.layers[i]

            shifted_sou = (sou + shift_dist)

            rec = np.array([shifted_sou[0], shifted_sou[1], layer.top.get_depth(shifted_sou[:2])])       # ЭТО ОСТАЕТСЯ ОТКРЫТЫМ ВОПРОСОМ
            end_horizon = layer.top
            start_horizon = layer.bottom
            if sign > 0 and not last:
                rec = np.array([shifted_sou[0], shifted_sou[1], layer.bottom.get_depth(shifted_sou[:2])])
                end_horizon = layer.bottom
                start_horizon = layer.top
            elif last:
                rec = receiver

            if first: start_horizon = None
            if last: end_horizon = None

            segments.append(Segment(sou, rec, layer, start_horizon, end_horizon, vtype=WAVECODE[vtype]))
            sou = rec

        return segments

    def _get_forward(self, vel_mod):
        # TODO: make more pythonic
        source = np.array(self.source.location, ndmin=1)
        receiver = np.array(self.receiver.location, ndmin=1)
        intersections = []
        distance = []
        horizons = []
        for hor in vel_mod.horizons:
            rec = hor.intersect(source, receiver)
            if len(rec) == 0:
                continue
            dist = np.sqrt(((rec - source) ** 2).sum())
            intersections.append(rec)
            distance.append(dist)
            horizons.append(hor)

        intersections = [x for _, x in sorted(zip(distance, intersections))]
        horizons = [x for _, x in sorted(zip(distance, horizons))]

        segments = []
        sou = np.array(self.source.location, ndmin=1)
        for i, rec in enumerate(intersections):
            first = (i == 0)
            layer = self._get_location_layer(sou/2 + rec/2, vel_mod)
            end_horizon = horizons[i]
            if first: start_horizon = None
            else: start_horizon = horizons[i-1]
            segments.append(Segment(sou, rec, layer, start_horizon, end_horizon, vtype='vp'))
            sou = rec

        layer = self._get_location_layer(receiver, vel_mod)
        if len(horizons)==0: start_horizon = None
        else: start_horizon = horizons[-1]
        segments.append(Segment(sou, receiver, layer, start_horizon, None, vtype='vp'))

        return segments

    def _get_trajectory(self):
        # TODO: make more "pythonic"
        trj = np.array(self.source.location, ndmin=1)
        for i, seg in enumerate(self.segments):
            trj = np.vstack((trj, seg.receiver))
        return trj

    @staticmethod
    def _get_location_layer(x, vel_mod): # ВАЖНО! ПОГРАНИЧНЫЕ СЛУЧАИ ЗАГЛУБЛЯЮТСЯ
        for l in vel_mod.layers:
            if (l.bottom.get_depth(x[:2]) > x[2] + 1e-8 > l.top.get_depth(x[:2])): return l

    def travel_time(self, x=None):
        # TODO: make more pythonic and faster
        # Если даны новые координаты траектории,
        # тогда обновляются сегменты и следовательно траектория

        if np.any(x):
            new_segments = []
            sou = self.segments[0].source

            for seg, rec in zip(self.segments, np.reshape(x, (-1, 2))):
                receiver = np.array([rec[0], rec[1], seg.end_horizon.get_depth(rec)])

                new_segments.append(Segment(sou, receiver, seg.layer,
                                            seg.start_horizon, seg.end_horizon,
                                            vtype = seg.vtype))
                sou = receiver

            new_segments.append(Segment(sou, self.segments[-1].receiver, self.segments[-1].layer,
                                        self.segments[-1].start_horizon, self.segments[-1].end_horizon,
                                        vtype = self.segments[-1].vtype))
            self.segments = new_segments

        time = 0.0
        for segment in self.segments:
            time += segment.time
        return time

    def dtravel(self):
        amount_of_borders = len(self.segments) - 1
        dt = np.zeros((amount_of_borders, 2))             # Производные по dx & dy соответственно, на каждой пересекающей луч границе

        for ind_border in range(amount_of_borders):
            seg1 = self.segments[ind_border]              # Соседние 2 сегмента, около точки на границе
            seg2 = self.segments[ind_border + 1]

            dist1, dist2 = seg1.get_distance(), seg2.get_distance()
            vec1, vec2 = seg1.vector, seg2.vector
            gradient = seg1.end_horizon.get_gradient(seg1.receiver[:-1])

            v1 = seg1.layer.get_velocity(vec1)[seg1.vtype]
            v2 = seg2.layer.get_velocity(vec2)[seg2.vtype]

            dv1 = seg1.layer.get_dv(vec1)[seg1.vtype]
            dv2 = seg2.layer.get_dv(vec2)[seg2.vtype]

            dt[ind_border] += (seg1.receiver[:-1] - seg1.source[:-1] + (seg1.receiver[-1] - seg1.source[-1])*gradient)/dist1/v1
            dt[ind_border] -= dist1 * dv1 / (v1 ** 2)
            dt[ind_border] -= (seg2.receiver[:-1] - seg2.source[:-1] + (seg2.receiver[-1] - seg2.source[-1])*gradient)/dist2/v2
            dt[ind_border] += dist2 * dv2 / (v2 ** 2)
        return dt

    def optimize(self, method='Nelder-Mead', tol=1e-32,
                 snells_law=False, projection=True, dtravel=True):
        # TODO: Add derivatives and Snels Law check

        x0 = self._get_trajectory()[1:-1, :-1]

        if not np.any(x0):
            return self.travel_time()

        # Определим функционал для минимизации, он будет состоять из слагаемых:
        # 1. Время вдоль пути (которое мы минимизируем солгасно принципу Ферма)
        # 2. Выполнение закона Снеллиуса (либо через синусы, либо через проекции)
        # 3. Производная вдоль луча. Там где время минимально, она должна быть = 0

        def _fun(x):
            f1 = self.travel_time(x)
            f2 = 0
            if snells_law:
                f2 += 100*(abs(self.snells_law(projection=projection))).mean()
            f3 = 0
            if dtravel:
                f3 += 100*(abs(self.dtravel())).mean()

            return f1 + f2 + f3

        # ops = {'adaptive': False}
        xs = minimize(_fun, x0.ravel(), method=method, tol=tol)
        time = xs.fun

        # self.amplitude_fun = self.amplitude_fr_dom() # rewrite the amplitude field.

        return time

    def plot(self, style='trj', **kwargs):
        if style == 'trj':
            plot_line_3d(self._get_trajectory().T, **kwargs)
            return
        for s in self.segments:
            plot_line_3d(s.segment.T, **kwargs)

    def snells_law(self, projection=True):
        if not projection: return self._snells_law_by_sin()
        return self._snells_law_by_projection()

    def _snells_law_by_sin(self):
        amount = len(self.segments) - 1  # Amount of boundaries

        critic = []
        snell = []

        for i in range(amount):
            r1 = self.segments[i].vector  # vector before boundary
            r2 = self.segments[i + 1].vector  # vector after boundary

            normal = self.segments[i].end_horizon.normal

            sin_r1 = np.sqrt(1 - r1.dot(normal) ** 2)  # -//- and r
            sin_r2 = np.sqrt(1 - r2.dot(normal) ** 2)  # sin of angle between normal and r_1

            v1 = self.segments[i].layer.get_velocity(r1)[self.segments[i].vtype]
            v2 = self.segments[i + 1].layer.get_velocity(r2)[self.segments[i + 1].vtype]

            if v1 < v2:
                critic.append(sin_r1 >= v1 / v2)  # checking of critic angle
            else:
                critic.append(False)

            # if np.array(critic).any():
            #     raise SnelliusError('На границе {} достигнут критический угол'.format(i + 1))

            snell.append(abs(sin_r1 / v1 - sin_r2 / v2))

        return np.array(snell)

    def _snells_law_by_projection(self):
        amount = len(self.segments) - 1  # Amount of boundaries

        snell = []

        for i in range(amount):
            r1 = self.segments[i].vector  # vector before boundary
            r2 = self.segments[i + 1].vector

            normal = self.segments[i].end_horizon.get_normal(self.segments[i].receiver[0:2])

            v1 = self.segments[i].layer.get_velocity(r1)[self.segments[i].vtype]
            v2 = self.segments[i + 1].layer.get_velocity(r2)[self.segments[i + 1].vtype]

            r2 = r1 - np.dot(normal, r1) * normal * (1 - (v2 / v1))

            pr_r1 = np.dot(normal, r1 / v1)

            pr_r2 = np.dot(normal, r2 / v2)

            snell.append(abs(pr_r1 - pr_r2))

        return np.array(snell)

    def amplitude_fr_dom(self, curv_bool = 1):

        # This method computes amplitude in the observation point in the frequency domain using formula
        # U = U_0 /sqrt(s0**2 * |П( det M(s*_i-1) / det M(s*_i))|) * П( k_i )
        # where U is the amplitude, U_0 is the some function depending on the source, П means "Product" with
        # respect to index i, k_i is an appropriate amplitude coefficient at the i-th boundary on a ray's path, s*_i are
        # points of incidence at the i-th boundary (if i != 0 and i != N). s*_0 equals to s0 and
        # s*_N equals to the ray's length in the observation point.

        # The direction of the displacement vector is described below.

        # s0 is some small distance which denotes a point on the ray in the vicinity of the source. Matrix M will be
        # introduced further.

        # The algorithm is based on the dynamic ray tracing theory presented in:
        # The ray series method and dynamic ray tracing system for three-dimensional inhomogeneous media V. Červený
        # F. Hron Bulletin of the Seismological Society of America (1980) 70 (1): 47-77

        # In order to clarify what is "vicinity", let us find vertical distance between the source and the closest
        # boundary and set the "vicinity" to be equal to 1/10 part of this distance.

        # DEFINE S0 PROPERLY!!!

        # s0 = np.min(abs(self.source.location[2] - self.source.layer.top.get_depth(self.source.location[0:2])),
        #             abs(self.source.location[2] - self.source.layer.bottom.get_depth(self.source.location[0:2])))
        s0 = 1 / 10

        # Great. Let us introduce a matrix 2x2 that is involved in the following procedures: M. It depends on the ray
        # and contains second-order partial derivatives of the eikonal with respect to ray coordinates q_i. Of course,
        # it will vary along the ray, but now we are in the first segment in vicinity of the source.

        M = np.zeros((2, 2)) # This is a matrix of the second derivatives of the eikonal with respect to ray coordinates
        # q_i. The geometrical spreading depends on this matrix. General solution for corresponding system of
        # differential equations in homogeneous isotropic medium reads:

        # M[0, 0] = (v*s + c_22) / ( (v*s + c_11)*(v*s + c_22) - c_12**2 ),
        # M[1, 1] = (v*s + c_11) / ( (v*s + c_11)*(v*s + c_22) - c_12**2 ),
        # M[0, 1] = - c_12 / ( (v*s + c_11)*(v*s + c_22) - c_12**2 ),
        # M[1, 0] = - c_12 / ( (v*s + c_11)*(v*s + c_22) - c_12**2 ), where s is the ray's path length, v is velocity in
        # the medium, c_ij are unknown constants.

        # In order to find c_ij we have to impose initial or boundary conditions on M. But it would be more
        # convenient to solve the occurring system of equations in terms of N = M**(-1) since in that case all equations
        # will be linear. Matrix N has the following form:

        # N[0, 0] = v * s + c_11
        # N[1, 1] = v * s + c_22
        # N[0, 1] = c_12
        # N[1, 0] = c_12

        # In case of point source (which we have in the first segment of any ray) the initial condition for N reads:
        # N(s = 0) = 0. Since that all coefficients c_ij are equal to 0. So, in the point s = s0 on the ray we have:

        M[0, 0] = 1 / (self.segments[0].layer.get_velocity(1)[self.segments[0].vtype] * s0)
        M[1, 1] = 1 / (self.segments[0].layer.get_velocity(1)[self.segments[0].vtype] * s0)

        # Non-diagonal elements are already equal to 0

        # We have to introduce an auxiliary item: detProd = det M(s*_i-1) / det M(s*_i). But since we don't have yet the
        # determinant in the denominator we'd just remember the determinant in the nominator:

        detRat = np.linalg.det(M)

        # Let's go to the end of the zero segment. Matrix M changes:

        dist = np.linalg.norm(self.segments[0].source - self.segments[0].receiver) # this will be distance along the ray

        M[0, 0] = 1 / (self.segments[0].layer.get_velocity(1)[self.segments[0].vtype] * dist)
        M[1, 1] = 1 / (self.segments[0].layer.get_velocity(1)[self.segments[0].vtype] * dist)

        #  We can compute value of the detProd:

        detRat = detRat / np.linalg.det(M)

        # Now let's compute amplitude int the end of zero segment.

        # But before doing that we should understand the polarization of our wave. For this purpose, let's introduce
        # unit vectors e1 and e2 which together with ray's tangent vector form up ray-centered coordinates. Initially we
        # can choose two arbitrary orthogonal unit vectors tangent to the wavefront, but we shall try to follow
        # traditional SV- and SH- notation. In any case these two vectors will depend on the vector t tangent to the
        # ray:

        t = self.segments[0].vector / np.linalg.norm(self.segments[0].vector)

        # We cannot distinguish SV and SH polarization if t is strictly vertical. In that case we'll just set e1 and
        # e2 coincident with i ang j unit vectors of the global Cartesian coordinates:

        if t[2] == 1:

            e1 = np.array([1, 0, 0])
            e2 = np.array([0, 1, 0])

        else:

            e2 = np.array([t[1], - t[0], 0]) / np.sqrt(t[1] ** 2 + t[0] ** 2) # SH-polarized unit vector
            e1 = np.cross(e2, t) # SV-polarized unit vector

        # In the first segment the polarization vector depends only on the source. But we should also take into account
        # the raycode.

        if self.segments[0].vtype == 'vp':

            U = self.source.psi0(self.segments[0].receiver, t) / np.sqrt(s0**2 * abs(detRat)) * t

        if self.segments[0].vtype == 'vs':

            U = (self.source.psi0(self.segments[0].receiver, e1) * e1 +
                 self.source.psi0(self.segments[0].receiver, e2) * e2 )/ np.sqrt(s0**2 * abs(detRat))

        # Further actions depend on number of segments in the ray.

        if len(self.segments) == 1 or np.linalg.norm(U) == 0:

            return U / np.sqrt(self.segments[0].layer.get_velocity(0)[self.segments[0].vtype] *
                               self.segments[0].layer.get_density())
            # if there is only one segment (or the amplitude is already equal to 0), calculate the amplitude in
            # the receiver and return it.

        else:

            # If there are some boundaries on the ray's path, we shall need to carry out more complicated calculations.

            for i in np.arange(1, len(self.segments), 1):

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

                rt_sign = 1 # initially we think about transition

                if self.segments[i - 1].layer == self.segments[i].layer:

                    rt_sign = - 1 # but then check whether we were right or not.

                # cosine of inc_angle
                cos_inc = abs(np.dot(self.segments[i - 1].vector,
                                     self.segments[i - 1].end_horizon.get_normal(self.segments[i - 1].receiver[0:2])))
                # cosine of tr_angle
                cos_out = abs(np.dot(self.segments[i].vector,
                                     self.segments[i - 1].end_horizon.get_normal(self.segments[i - 1].receiver[0:2])))

                S = np.array([[rt_sign * cos_inc / cos_out, 0],
                              [0, 1]])

                G = np.array([[- rt_sign * 1 / cos_out, 0],
                              [0, 1]])

                u = cos_inc/self.segments[i - 1].layer.get_velocity(1)[self.segments[i - 1].vtype] - \
                    rt_sign * cos_out/self.segments[i].layer.get_velocity(1)[self.segments[i].vtype]

                D = np.zeros((2,2))
                D[0, 0], D[0, 1], D[1, 1], transit_matr =\
                    self.segments[i - 1].end_horizon.get_sec_deriv(self.segments[i - 1].receiver[0:2],
                                                                   self.segments[i - 1].vector)
                D[1, 0] = D[0, 1]

                if curv_bool == 0:

                    D = np.zeros((2,2))

                # Here transit_matr is a transition matrix from global Cartesian coordinates to local ones which are
                # connected to the point of incidence ant the incident ray. Of course, columns of this matrix are
                # coordinate unit vectors of the local system d1, d2 and n.

                # The incident ray's e2 vector can make angle with the d2 so that:

                # dot(e2, d2) = sin(omega)

                # The matrix W is a rotation matrix:

                # W = np.array([[ cos(omega), sin(omega)],
                #               [ - sin(omega), cos(omega)]])

                # sin_omega = np.dot(e1, transit_matr[:, 1])
                # cos_omega = np.dot(e2, transit_matr[:, 1])

                W = np.array([[np.dot(e2, transit_matr[:, 1]), np.dot(e1, transit_matr[:, 1])],
                              [- np.dot(e1, transit_matr[:, 1]), np.dot(e2, transit_matr[:, 1])]])


                # Since all layers are homogeneous and isotropic the general solution for matrix M (i.e. its form)
                # remains the same:

                # M[0, 0] = (v*s + c_22) / ( (v*s + c_11)*(v*s + c_22) - c_12**2 ),
                # M[1, 1] = (v*s + c_11) / ( (v*s + c_11)*(v*s + c_22) - c_12**2 ),
                # M[0, 1] = - c_12 / ( (v*s + c_11)*(v*s + c_22) - c_12**2 ),
                # M[1, 0] = - c_12 / ( (v*s + c_11)*(v*s + c_22) - c_12**2 )

                # We can compute constants c_i by in terms of N = M**(-1). But first of all let's find new value of M
                # (i.e. M'):

                M = np.dot(np.dot(S, np.dot(W.T, np.dot(M, W))), S) + u * np.dot(np.dot(G, D), G)

                # Note that new matrix M is related to the new triplet t, e1, e2 where e2 is coincident with d2.

                N = np.linalg.inv(M)

                # Remember that:

                # N[0, 0] = v * s + c_11
                # N[1, 1] = v * s + c_22
                # N[0, 1] = c_12
                # N[1, 0] = c_12

                # So, all equations for c_ij are linear!
                # s = dist, v = v'.

                c_11 = N[0, 0] - self.segments[i].layer.get_velocity(1)[self.segments[i].vtype] * dist
                c_22 = N[1, 1] - self.segments[i].layer.get_velocity(1)[self.segments[i].vtype] * dist
                c_12 = N[0, 1]

                # Now we have a new value of M. Let's calculate new value of detProd. Once again we know just the
                # nominator:

                detRat = np.linalg.det(M)

                # And we have to not forget about transmission coefficient. In order to compute it we have to rewrite
                # existing vector of polarization U in the local coordinate system. Remember that we've already
                # found necessary transition matrix.

                ampl_coeff = rt_coefficients(self.velmod.layers[self.raycode[i - 1][1]],
                                             self.velmod.layers[self.raycode[i - 1][1] + self.raycode[i - 1][0]],
                                             cos_inc,
                                             np.dot(transit_matr.T, U),
                                             self.segments[i - 1].layer.get_velocity(0)[self.segments[i - 1].vtype],
                                             rt_sign)

                # Let's go further, to the next boundary:

                dist = dist + np.linalg.norm(self.segments[i].source - self.segments[i].receiver)

                M[0, 0] = (self.segments[i].layer.get_velocity(1)[self.segments[i].vtype] * dist + c_22) / \
                          ( (self.segments[i].layer.get_velocity(1)[self.segments[i].vtype] * dist + c_11) *
                            (self.segments[i].layer.get_velocity(1)[self.segments[i].vtype] * dist + c_22) - c_12**2)

                M[1, 1] = (self.segments[i].layer.get_velocity(1)[self.segments[i].vtype] * dist + c_11) / \
                          ( (self.segments[i].layer.get_velocity(1)[self.segments[i].vtype] * dist + c_11) *
                            (self.segments[i].layer.get_velocity(1)[self.segments[i].vtype] * dist + c_22) - c_12**2)

                M[0, 1] = - c_12 / \
                          ( (self.segments[i].layer.get_velocity(1)[self.segments[i].vtype] * dist + c_11) *
                            (self.segments[i].layer.get_velocity(1)[self.segments[i].vtype] * dist + c_22) - c_12**2)

                M[1, 0 ] = M[0, 1]

                # Full value of the detRat:

                detRat = detRat / np.linalg.det(M)

                # That's all for this segment. We are almost ready to rewrite the value of amlitude. But first consider
                # the following.
                # ampl_coeff is a list of 6 coefficients which specify the amplitude changes of the wave. They are
                # related to t, e1, e2 triplets of the outgoing waves. But we work just with one of them.

                # Let's specify its triplet. We should take into account that e2 = d2 where d2 is a vector from the
                # local coordinate system.

                t = self.segments[i].vector / np.linalg.norm(self.segments[i].vector)
                e2 = transit_matr[:, 1]
                e1 = np.cross(e2, t)

                # Finally, everything depends on the current wavetype:

                if self.segments[i].vtype == 'vp':

                    U = np.linalg.norm(U) * ampl_coeff[0] / np.sqrt(abs(detRat)) * t

                if self.segments[i].vtype == 'vs':

                    U = (np.linalg.norm(U) * ampl_coeff[1] * e1 +
                         np.linalg.norm(U) * ampl_coeff[2] * e2) / np.sqrt(abs(detRat))


                # Let's go to the next layer!

            # We've computed the amplitude in the cycle above. Let's return it's value, but before that we have
            # to add some coefficients related to the source's layer:

            return U / np.sqrt(self.segments[0].layer.get_velocity(0)[self.segments[0].vtype] *
                               self.segments[0].layer.get_density())

    def amplitude_t_dom(self, t):
        # returns amplitude vector in the receiver in a particular time moment t.
        # Here I use theory presented in: Popov, M.M. Ray theory and gaussian beam method for geophysicists /
        # M. M. Popov. - Salvador: EDUFBA, 2002. – 172 p.

        # We use formula: A = Ricker(t - tau) * U)
        # where tau is time of the first break (i.e. traveltime along the ray), Ricker is
        # the Ricker wavelet with dispersion given in the Source  and U is a constant vector: self.amplitude_fun.

        tau = self.travel_time()
        sigma = np.sqrt(2) / self.source.fr_dom

        return 2 / np.sqrt(3 * sigma) / np.pi ** (1 / 4) *\
               (1 - ((t - tau )/ sigma)**2) *\
               np.exp(- (t - tau)**2 / (2 * sigma**2)) *\
               self.amplitude_fun
        # return 2 * np.exp(- (t - tau)**2 / (2 * sigma**2)) * (-3 * sigma**2 + (t - tau)**2) * (t - tau) / \
        #        (np.sqrt(3) * np.pi**(1 / 4) * sigma**(9 / 2)) * self.amplitude_fun

    def spreading(self, curv_bool):
        # Computes only geometrical spreading along the ray in the observation point. All comments are above.
        # curv_bool is boolean variable. If it is 1 than curvature of boundaries is taken into account.

        s0 = 1 / 10

        M = np.zeros((2, 2))

        M[0, 0] = 1 / (self.segments[0].layer.get_velocity(1)[self.segments[0].vtype] * s0)
        M[1, 1] = 1 / (self.segments[0].layer.get_velocity(1)[self.segments[0].vtype] * s0)

        detRat = np.linalg.det(M)

        dist = np.linalg.norm(self.segments[0].source - self.segments[0].receiver) # this will be distance along the ray

        M[0, 0] = 1 / (self.segments[0].layer.get_velocity(1)[self.segments[0].vtype] * dist)
        M[1, 1] = 1 / (self.segments[0].layer.get_velocity(1)[self.segments[0].vtype] * dist)

        detRat = detRat / np.linalg.det(M)

        # Now let's compute amplitude int the end of zero segment.

        # We should understand the polarization of our wave. For this purpose, let's introduce
        # unit vectors e1 and e2 which together with ray's tangent vector form up ray-centered coordinates.

        t = self.segments[0].vector / np.linalg.norm(self.segments[0].vector)

        if t[2] == 1:

            e1 = np.array([1, 0, 0])
            e2 = np.array([0, 1, 0])

        else:

            e2 = np.array([t[1], - t[0], 0]) / np.sqrt(t[1] ** 2 + t[0] ** 2) # SH-polarized unit vector
            e1 = np.cross(e2, t) # SV-polarized unit vector

        J = s0**2 * detRat

        if len(self.segments) == 1:

            return J, dist

        else:

            for i in np.arange(1, len(self.segments), 1):

                cos_inc = abs(np.dot(self.segments[i - 1].vector,
                                     self.segments[i - 1].end_horizon.get_normal(self.segments[i - 1].receiver[0:2])))

                cos_out = abs(np.dot(self.segments[i].vector,
                                     self.segments[i - 1].end_horizon.get_normal(self.segments[i - 1].receiver[0:2])))

                rt_sign = 1

                if self.segments[i - 1].layer == self.segments[i].layer:

                    rt_sign = - 1

                S = np.array([[rt_sign * cos_inc / cos_out, 0],
                              [0, 1]])

                G = np.array([[- rt_sign * 1 / cos_out, 0],
                              [0, 1]])

                u = cos_inc/self.segments[i - 1].layer.get_velocity(1)[self.segments[i - 1].vtype] - \
                    rt_sign * cos_out/self.segments[i].layer.get_velocity(1)[self.segments[i].vtype]

                D = np.zeros((2,2))

                D[0, 0], D[0, 1], D[1, 1], transit_matr = \
                    self.segments[i - 1].end_horizon.get_sec_deriv(self.segments[i - 1].receiver[0:2],
                                                                   self.segments[i - 1].vector)
                D[1, 0] = D[0, 1]

                if curv_bool == 0:

                    D = np.zeros((2,2))

                # Here transit_matr is a transition matrix from global Cartesian coordinates to local ones which are
                # connected to the point of incidence ant the incident ray. Of course, columns of this matrix are
                # coordinate unit vectors of the local system d1, d2 and n.

                # The incident ray's e2 vector can make angle with the d2 so that:

                # dot(e2, d2) = sin(omega)

                # The matrix W is a rotation matrix:

                # W = np.array([[ cos(omega), sin(omega)],
                #               [ - sin(omega), cos(omega)]])

                # sin_omega = np.dot(e1, transit_matr[:, 1])
                # cos_omega = np.dot(e2, transit_matr[:, 1])

                W = np.array([[np.dot(e2, transit_matr[:, 1]), np.dot(e1, transit_matr[:, 1])],
                              [- np.dot(e1, transit_matr[:, 1]), np.dot(e2, transit_matr[:, 1])]])


                # Since all layers are homogeneous and isotropic the general solution for matrix M (i.e. its form)
                # remains the same:

                # M[0, 0] = (v*s + c_22) / ( (v*s + c_11)*(v*s + c_22) - c_12**2 ),
                # M[1, 1] = (v*s + c_11) / ( (v*s + c_11)*(v*s + c_22) - c_12**2 ),
                # M[0, 1] = - c_12 / ( (v*s + c_11)*(v*s + c_22) - c_12**2 ),
                # M[1, 0] = - c_12 / ( (v*s + c_11)*(v*s + c_22) - c_12**2 )

                # We can compute constants c_i by in terms of N = M**(-1). But first of all let's find new value of M
                # (i.e. M'):

                M = np.dot(np.dot(S, np.dot(W.T, np.dot(M, W))), S) + u * np.dot(np.dot(G, D), G)

                # Note that new matrix M is related to the new triplet t, e1, e2 where e2 is coincident with d2.

                N = np.linalg.inv(M)

                c_11 = N[0, 0] - self.segments[i].layer.get_velocity(1)[self.segments[i].vtype] * dist
                c_22 = N[1, 1] - self.segments[i].layer.get_velocity(1)[self.segments[i].vtype] * dist
                c_12 = N[0, 1]

                detRat = np.linalg.det(M)

                dist = dist + np.linalg.norm(self.segments[i].source - self.segments[i].receiver)

                M[0, 0] = (self.segments[i].layer.get_velocity(1)[self.segments[i].vtype] * dist + c_22) / \
                          ( (self.segments[i].layer.get_velocity(1)[self.segments[i].vtype] * dist + c_11) *
                            (self.segments[i].layer.get_velocity(1)[self.segments[i].vtype] * dist + c_22) - c_12**2)

                M[1, 1] = (self.segments[i].layer.get_velocity(1)[self.segments[i].vtype] * dist + c_11) / \
                          ( (self.segments[i].layer.get_velocity(1)[self.segments[i].vtype] * dist + c_11) *
                            (self.segments[i].layer.get_velocity(1)[self.segments[i].vtype] * dist + c_22) - c_12**2)

                M[0, 1] = - c_12 / \
                          ( (self.segments[i].layer.get_velocity(1)[self.segments[i].vtype] * dist + c_11) *
                            (self.segments[i].layer.get_velocity(1)[self.segments[i].vtype] * dist + c_22) - c_12**2)

                M[1, 0 ] = M[0, 1]

                detRat = detRat / np.linalg.det(M)

                t = self.segments[i].vector / np.linalg.norm(self.segments[i].vector)
                e2 = transit_matr[:, 1]
                e1 = np.cross(e2, t)

                J = J * cos_out / cos_inc

            J = J * abs(detRat)

            return J,\
                   np.sqrt(J) * np.sqrt(self.segments[0].layer.get_velocity(0)[self.segments[0].vtype]) * \
                   np.sqrt(self.segments[-1].layer.get_velocity(0)[self.segments[-1].vtype])


class Segment(object):
    def __init__(self, source, receiver, layer, start_horizon, end_horizon, vtype):
        self.source = source                # Just np.array([x0,y0,z0]), it's not Object of type Source
        self.receiver = receiver            # Just np.array([x1,y1,z1]), it's not Object of type Receivers
        self.vector = self.get_vector()
        self.vtype = vtype
        self.layer = layer
        self.time = self.get_time()
        self.start_horizon = start_horizon                          # object of type of Horizon
        self.end_horizon = end_horizon                              # object of type of Horizon

    def get_distance(self):
        return np.linalg.norm(self.receiver - self.source)

    def get_vector(self):
        dist = self.get_distance()
        vec = (self.receiver - self.source) / (dist+1e-16)
        return vec

    def get_time(self):
        dist = self.get_distance()
        time = dist/self.layer.get_velocity(self.vector)[self.vtype]
        return time

    def _what_horizon(self, point):
        top = self.layer.top
        bottom = self.layer.bottom
        if top == None and bottom != None: return bottom
        if bottom == None and top != None: return top
        if bottom == None and top == None: return None
        dist_top = abs(top.get_depth(point[:2]) - point[2])
        dist_bottom = abs(bottom.get_depth(point[:2]) - point[2])
        if dist_top<dist_bottom: return top
        else: return bottom

    def __repr__(self):
        return self.segment

    def __str__(self):
        return self.segment


class SnelliusError(Exception):
    pass;
