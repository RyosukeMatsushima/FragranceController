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
    def __init__(self, t_s: TopologicalSpace, target_coodinate, graph_arg, u_set, moderate_u):
        self.t_s = t_s
        self.graph_arg = graph_arg
        self.is_time_reversal = True

        self.u_set = u_set
        self.moderate_u = moderate_u
        u_P_list = np.full(self.u_set.shape, 1.)
        self.u_P_set = u_P_list/u_P_list.sum()
        print("u_P_set")
        print(self.u_P_set)

        self.target_coodinate = target_coodinate
        self._simulate_time = 0.
        self.astablishment_space = copy(self.t_s.astablishment_space)
        self.astablishment_space[self.t_s.coodinate2pos_AS(self.target_coodinate)] = 1.0 # set target coodinate
        self.astablishment_space_tf = tf.Variable(self.astablishment_space, dtype=tf.float32)
    
    def update_astablishment_space(self):
        self.t_s.astablishment_space = self.astablishment_space_tf.numpy()

    def init_stochastic_matrix(self, save: bool):
        print("\n init_stochastic_matrix \n")
        filename = path.join(mkdtemp(), 'stochastic_matrix.dat')
        stochastic_matrix = np.memmap(filename, dtype='float32', mode='w+', shape=(len(self.t_s.axes) * 2 + 1, self.t_s.element_count))
        filename = path.join(mkdtemp(), 'gather_matrix.dat')
        gather_matrix = np.memmap(filename, dtype='int64', mode='w+', shape=(len(self.t_s.axes) * 2 + 1, self.t_s.element_count))
        pos_TS_elements = self.t_s.pos_TS_elements()
        time_direction = -1.0 if self.is_time_reversal else 1.0
        for pos_TS in tqdm(pos_TS_elements):
            pos_AS = self.t_s.pos_TS2pos_AS(pos_TS)
            gather_list = np.array([pos_AS])
            stochastic_list = np.array([0.0])

            for i, input in enumerate(self.u_set):
                input_P = self.u_P_set[i]
                if self.t_s.is_edge_of_TS(pos_TS):
                    continue
                velosites = time_direction * self.t_s.model.dynamics(*self.t_s.pos_TS2coodinate(pos_TS), input) # velosity as a vector

                p_remain_pos = 1.0    #P(pos_i, t + delta_t | pos_i, t)
                for i in range(len(pos_TS)):
                    velosity = velosites[i]
                    step = self.t_s.axes[i].get_step(pos_TS[i])
                    courant_number = velosity * self.t_s.delta_t / step

                    pos = pos_TS[:]
                    pos = list(pos)
                    if courant_number > 0:
                        pos[i] -= 1
                    else:
                        pos[i] += 1
                    if np.any( gather_list == self.t_s.pos_TS2pos_AS(pos) ):
                        stochastic_list[np.where(gather_list == self.t_s.pos_TS2pos_AS(pos))] += abs(courant_number) * input_P
                    else:
                        gather_list = np.append(gather_list, self.t_s.pos_TS2pos_AS(pos))
                        stochastic_list = np.append(stochastic_list, abs(courant_number) * input_P)
                    p_remain_pos -= abs(courant_number)

                if p_remain_pos < 0:
                    raise ArithmeticError("p_remain_pos is under zero")
                stochastic_list[np.where(gather_list == pos_AS)] += p_remain_pos * input_P

            for i, gather in enumerate(gather_list):
                gather_matrix[i, pos_AS] = gather
            for i, stochastic in enumerate(stochastic_list):
                stochastic_matrix[i, pos_AS] = stochastic

        self.stochastic_matrix_tf = tf.constant(stochastic_matrix, dtype=tf.float32)
        self.gather_matrix_tf = tf.constant(gather_matrix, dtype=tf.int64)
        self.gathered_matrix_tf = tf.Variable(stochastic_matrix, dtype=tf.float32)
        # tf.fill(self.gathered_matrix_tf, 0.0)
        if save:
            self.save_stochastic_matrix(stochastic_matrix, gather_matrix)
        print("\n init_stochastic_matrix end \n")

    def simulate(self):
        for i in range(self.stochastic_matrix_tf.shape[0]):
            self.gathered_matrix_tf[i].assign(tf.gather(self.astablishment_space_tf, self.gather_matrix_tf[i]))
        self.astablishment_space_tf.assign(tf.reduce_sum(self.stochastic_matrix_tf * self.gathered_matrix_tf, 0))
        self._simulate_time += self.t_s.delta_t

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
