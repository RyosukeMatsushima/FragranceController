import numpy as np
from copy import copy
import tensorflow as tf
from src.TopologicalSpace import TopologicalSpace

class InputCalculator:
    def __init__(self):
        self.t_s = TopologicalSpace()

        d = 1.
        u_set = np.arange(-2., 2. + d, d)
        u_P_list = np.full(u_set.shape, 1.)
        u_P_set = u_P_list/u_P_list.sum()

        self.target_coodinate = (0, 0)
        self._simulate_time = 0.

        self.astablishment_space = copy(self.t_s.astablishment_space)
        self.astablishment_space[self.t_s.coodinate2pos_AS(self.target_coodinate)] = 1.0 # set target coodinate
        self.astablishment_space_tf = tf.Variable(np.array([self.astablishment_space]).T, dtype=tf.float32)

        self.stochastic_matrix = [ u_P_set[i] * self.t_s.stochastic_matrix(u) for i, u in enumerate(u_set)]
        self.stochastic_matrix = sum(self.stochastic_matrix)

        self.stochastic_matrix_tf = tf.constant(self.stochastic_matrix, dtype=tf.float32)

    def simulate(self):
        self.astablishment_space_tf.assign(self.stochastic_matrix_tf @ self.astablishment_space_tf)
        self._simulate_time += self.t_s.delta_t

    def update_astablishment_space(self):
        self.t_s.astablishment_space = self.astablishment_space_tf.numpy()
