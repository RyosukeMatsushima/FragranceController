import numpy as np
import tensorflow as tf
from tempfile import mkdtemp
from copy import copy
import os.path as path
from tqdm import tqdm

# for save input space
import os
import glob
import json
import datetime

from src.InputCalculator import InputCalculator
from src.TopologicalSpace import TopologicalSpace

class FuildSimulator(InputCalculator):
    def __init__(self, t_s: TopologicalSpace, target_coodinate, graph_arg, u_set, moderate_u, model, delta_t):
        super().__init__(t_s, target_coodinate, graph_arg, u_set, moderate_u, model)
        self.astablishment_space_tf = tf.Variable(self.astablishment_space, dtype=tf.float32)
        self.delta_t = delta_t
    
    def update_astablishment_space(self):
        self.t_s.astablishment_space = self.astablishment_space_tf.numpy()

    def init_stochastic_matrix(self, save: bool):
        print("\n init_stochastic_matrix \n")
        step_list = np.array([axis.min_step for axis in self.t_s.axes]) #TODO: applay changeable step
        courant_number_list = [np.apply_along_axis(lambda x: self.model.dynamics(*x, u) * self.delta_t / step_list, 1, self.t_s.coodinate_space) for u in self.u_set]
        positive_courant_number_list = np.array([np.where(courant_number > 0, courant_number, 0) for courant_number in courant_number_list])
        negative_courant_number_list = np.array([np.where(courant_number < 0, courant_number, 0) for courant_number in courant_number_list])
        abs_courant_number_list = np.array([np.abs(courant_number) for courant_number in courant_number_list])
        abs_courant_number_list = np.sum(abs_courant_number_list, axis=2)
        positive_gather = [np.roll(self.t_s.posTS_space, -1, axis=axis).reshape(self.t_s.element_count) for axis in range(len(self.t_s.axes))]
        negative_gather = [np.roll(self.t_s.posTS_space, 1, axis=axis).reshape(self.t_s.element_count) for axis in range(len(self.t_s.axes))]

        # check abs courant_number < 1
        for abs_courant_number in abs_courant_number_list:
            max_courant_number = np.max(abs_courant_number)
            if max_courant_number > 1:
                pos = np.where(abs_courant_number == max_courant_number)
                pos_TS = self.t_s.pos_AS2pos_TS(pos[0])
                coodinate = self.t_s.pos_TS2coodinate(pos_TS)
                axis_num = pos[1][0]
                axis = self.t_s.axes[axis_num]
                s = " coodinate: " + str(coodinate) + " axis: " + axis.name + " step" + str(axis.min_step) + " max_courant_number: " + str(max_courant_number)
                s = "p_remain_pos is under zero" + s
                raise ArithmeticError(s)

        positive_courant_number = np.sum(positive_courant_number_list * self.u_P, axis=0)
        negative_courant_number = np.sum(negative_courant_number_list * self.u_P, axis=0)
        abs_courant_number = np.sum(abs_courant_number_list * self.u_P, axis=0)

        self.positive_courant_number_tf = tf.constant(positive_courant_number, dtype=tf.float32)
        self.negative_courant_number_tf = tf.constant(negative_courant_number, dtype=tf.float32)
        self.abs_courant_number_tf = tf.constant(abs_courant_number, dtype=tf.float32)
        self.positive_gather_tf = tf.constant(positive_gather, dtype=tf.int32)
        self.negative_gather_tf = tf.constant(negative_gather, dtype=tf.int32)
        if save:
            #TODO: add save logic
            print(" ")

        print("\n init_stochastic_matrix end \n")

    def simulate(self):
        for i in range(self.stochastic_matrix_tf.shape[0]):
            self.gathered_matrix_tf[i].assign(tf.gather(self.astablishment_space_tf, self.gather_matrix_tf[i]))
        self.astablishment_space_tf.assign(tf.reduce_sum(self.stochastic_matrix_tf * self.gathered_matrix_tf, 0))
        self._simulate_time += self.delta_t

    def save_stochastic_matrix(self, stochastic_matrix, gather_matrix):
        file_list = glob.glob("./stochastic_matrix/*")

        n = 1
        while(True):
            path = "./stochastic_matrix/stochastic_matrix" + str(n)
            n += 1
            if not path in file_list:
                print(path)
                os.mkdir(path)
                os.chdir(path)

                np.save("stochastic_matrix", stochastic_matrix)
                np.save("gather_matrix", gather_matrix)

                self.write_param()
                break
        os.chdir("../../")

    def load_stochastic_matrix(self, num):
        print("load_stochastic_matrix")
        stochastic_matrix_path = "./stochastic_matrix/stochastic_matrix" + str(num)
        os.chdir(stochastic_matrix_path)
        stochastic_matrix = np.load("stochastic_matrix.npy")
        gather_matrix = np.load("gather_matrix.npy")
        os.chdir("../..")
        print("load_stochastic_matrix end")
        return stochastic_matrix, gather_matrix

    def set_stochastic_matrix(self, stochastic_matrix, gather_matrix):
        self.stochastic_matrix_tf = tf.constant(stochastic_matrix, dtype=tf.float32)
        self.gather_matrix_tf = tf.constant(gather_matrix, dtype=tf.int64)
        self.gathered_matrix_tf = tf.Variable(stochastic_matrix, dtype=tf.float32)
