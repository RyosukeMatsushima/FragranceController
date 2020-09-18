import numpy as np
from copy import copy
import tensorflow as tf
from src.TopologicalSpace import TopologicalSpace

class InputCalculator:
    def __init__(self):
        self.t_s = TopologicalSpace()

        self.target_coodinate = (0, 0)
        self.simulate_time = 0

        self.astablishment_space = copy(self.t_s.astablishment_space)
        self.astablishment_space[self.t_s.coodinate2pos_AS(self.target_coodinate)] = 1.0 # set target coodinate
        self.astablishment_space_tf = tf.Variable(np.array([self.astablishment_space]).T, dtype=tf.float32)
        self.stochastic_matrix_tf = tf.constant(self.t_s.stochastic_matrix(0), dtype=tf.float32)

    @tf.function
    def simulate(self):
        self.astablishment_space_tf.assign(self.stochastic_matrix_tf @ self.astablishment_space_tf)

    def update_astablishment_space(self):
        self.t_s.astablishment_space = self.astablishment_space_tf.numpy()