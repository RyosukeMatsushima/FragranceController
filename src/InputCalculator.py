import numpy as np
from copy import copy
import tensorflow as tf
from src.TopologicalSpace import TopologicalSpace

class InputCalculator:
    def __init__(self):
        self.t_s = TopologicalSpace()

        d = 1.
        self.u_set = np.arange(-2., 2. + d, d)
        self.moderate_u = 0
        u_P_list = np.full(self.u_set.shape, 1.)
        u_P_set = u_P_list/u_P_list.sum()
        print(u_P_set)

        self.target_coodinate = (0, 0)
        self._simulate_time = 0.

        self.astablishment_space = copy(self.t_s.astablishment_space)
        self.astablishment_space[self.t_s.coodinate2pos_AS(self.target_coodinate)] = 1.0 # set target coodinate
        self.astablishment_space_tf = tf.Variable(np.array([self.astablishment_space]).T, dtype=tf.float32)

        self.stochastic_matrix = [ u_P_set[i] * self.t_s.stochastic_matrix(u) for i, u in enumerate(self.u_set)]
        self.stochastic_matrix = sum(self.stochastic_matrix)

        self.stochastic_matrix_tf = tf.constant(self.stochastic_matrix, dtype=tf.float32)

    def simulate(self):
        self.astablishment_space_tf.assign(self.stochastic_matrix_tf @ self.astablishment_space_tf)
        self._simulate_time += self.t_s.delta_t

    def update_astablishment_space(self):
        self.t_s.astablishment_space = self.astablishment_space_tf.numpy()

    def norm_velosity(self, pos_TS, input):
        vel = self.t_s.model.dynamics(*pos_TS, input)
        norm_param = np.linalg.norm(vel)
        if norm_param == 0.0:
            return np.zeros(vel.shape)
        return vel/norm_param

    def method1(self):
        self.simTimeToLim()
        self.update_astablishment_space()
        gradient_matrix = self.t_s.gradient_matrix()
        input_space = np.zeros([len(axis.elements) for axis in self.t_s.axes])
        pos_TS_elements = self.t_s.pos_TS_elements()

        for pos_TS in pos_TS_elements:
            gradient = gradient_matrix[self.t_s.pos_TS2pos_AS(pos_TS)]
            dot_list = np.array([np.dot(gradient, self.norm_velosity(pos_TS, input)) for input in self.u_set])
            proposal_input = self.u_set[np.where(dot_list == max(dot_list))]
            if len(proposal_input) is not 1:
                print("sevral proposal inputs exist")
                print(self.t_s.pos_TS2coodinate(pos_TS))
                print(gradient) #TODO: remove print
                if gradient[0] != 0.0:
                    raise TypeError("omg")
                print(dot_list)
                print(np.where(dot_list == max(dot_list)))
                proposal_input = self.moderate_u
            input_space[pos_TS] = proposal_input

        for input in input_space:
            print(input)

    def simTimeToLim(self):
        # e, v = tf.linalg.eigh(self.stochastic_matrix)
        # print(e)
        #TODO: check all e element is under 1.0

        eye = np.eye(self.t_s.element_count, self.t_s.element_count, dtype=float)
        eye_tf = tf.constant(eye, dtype=tf.float32)
        val1 = tf.linalg.inv(self.stochastic_matrix_tf - eye_tf)
        self.astablishment_space_tf.assign(- self.stochastic_matrix_tf @ val1 @ self.astablishment_space_tf)
