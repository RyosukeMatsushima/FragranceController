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
        super().__init__(t_s, target_coodinate, graph_arg, u_set, moderate_u, model, delta_t)
        self.astablishment_space_tf = tf.Variable(self.t_s.astablishment_space, dtype=tf.float32)

    def get_boundary_condition(self):
        boundary_condition = np.ones(self.t_s.element_count, dtype=np.float32)
        min_edge = [axis.min + axis.min_step/2 for axis in self.t_s.axes]
        max_edge = [axis.max - axis.min_step/2 for axis in self.t_s.axes]
        boundary_condition[np.any((self.t_s.coodinate_space < min_edge), axis=1)] = 0.
        boundary_condition[np.any((self.t_s.coodinate_space > max_edge), axis=1)] = 0.

        self.boundary_condition_tf = tf.constant(boundary_condition, dtype=tf.float32)
        del boundary_condition, min_edge, max_edge

    def add_obstacle(self, min_range, max_range):
        is_obstacle_min = np.array([self.t_s.coodinate_space[:, i] >= min_range[i] for i in range(len(self.t_s.axes))])
        is_obstacle_max = np.array([self.t_s.coodinate_space[:, i] <= max_range[i] for i in range(len(self.t_s.axes))])

        is_obstacle = np.all(is_obstacle_min, axis=0) & np.all(is_obstacle_max, axis=0)
        self.boundary_condition_tf = tf.where(is_obstacle, 0.0, self.boundary_condition_tf)

    def init_velocity_space(self, save: bool):
        filename = path.join(mkdtemp(), 'velocity_space.dat')
        velocity_space = np.memmap(filename, dtype='float32', mode='w+', shape=(len(self.u_set), self.t_s.element_count, len(self.t_s.axes)))

        split = int(self.t_s.element_count / 100000) + 1
        print("split")
        print(split)
        splited_coodinate_space_list = np.array_split(self.t_s.coodinate_space, split, axis=0)
        num = 0
        for splited_coodinate_space in tqdm(splited_coodinate_space_list):
            splited_velocity_space = np.array([np.apply_along_axis(lambda x: -self.model.dynamics(*x, u), 1, splited_coodinate_space) for u in self.u_set])
            velocity_space[:, num: num + len(splited_coodinate_space)] = splited_velocity_space
            num += len(splited_coodinate_space)
        self.velocity_space_tf = tf.constant(velocity_space, dtype=tf.float32)
        if save:
            self.save_velocity_space()

    def update_astablishment_space(self):
        self.t_s.astablishment_space = self.astablishment_space_tf.numpy()

    def init_stochastic_matrix(self, save: bool):
        print("\n init_stochastic_matrix \n")
        step_list_tf = tf.constant([axis.min_step for axis in self.t_s.axes], dtype=tf.float32) #TODO: applay changeable step

        # calculate about advection term
        print("calculate about advection term")
        courant_number_tf = tf.reduce_sum(self.velocity_space_tf, 0) * self.u_P * self.delta_t / step_list_tf
        positive_courant_number_tf = tf.where(courant_number_tf > 0, courant_number_tf, 0)
        negative_courant_number_tf = tf.where(courant_number_tf < 0, -courant_number_tf, 0)
        abs_courant_number_tf = tf.abs(courant_number_tf)
        abs_courant_number_tf = tf.reduce_sum(abs_courant_number_tf, 1)
        positive_gather = np.array([np.roll(self.t_s.posTS_space, 1, axis=axis).reshape(self.t_s.element_count) for axis in range(len(self.t_s.axes))]).T
        negative_gather = np.array([np.roll(self.t_s.posTS_space, -1, axis=axis).reshape(self.t_s.element_count) for axis in range(len(self.t_s.axes))]).T

        # check abs courant_number < 1 to be stable
        if tf.size(tf.where(abs_courant_number_tf > 1)) != 0:
            pos = tf.where(abs_courant_number_tf == tf.reduce_max(abs_courant_number_tf)).numpy()
            print("pos")
            print(pos)
            pos = pos[0]
            pos_TS = self.t_s.pos_AS2pos_TS(pos)
            coodinate = self.t_s.pos_TS2coodinate(pos_TS)
            s = " coodinate: " + str(coodinate) + " max_courant_number: " + str(tf.reduce_max(abs_courant_number_tf).numpy())
            s = "p_remain_pos is under zero beacse of courant_number" + s
            raise ArithmeticError(s)

        # calculate about diffusion term
        print("calculate about diffusion term")
        diffusion_number_tf = tf.reduce_sum(self.velocity_space_tf ** 2, 0) * self.u_P / 2 * self.delta_t ** 2 / step_list_tf ** 2
        sum_diffusion_number_tf = tf.reduce_sum(diffusion_number_tf, 1)

        # check abs diffusion_number < 1 to be stable
        if tf.size(tf.where(sum_diffusion_number_tf > 1/2)) != 0:
            pos = tf.where(sum_diffusion_number_tf == tf.reduce_max(sum_diffusion_number_tf)).numpy()
            print("pos")
            print(pos)
            pos = pos[0]
            pos_TS = self.t_s.pos_AS2pos_TS(pos)
            coodinate = self.t_s.pos_TS2coodinate(pos_TS)
            s = " coodinate: " + str(coodinate) + " max_courant_number: " + str(tf.reduce_max(abs_courant_number_tf).numpy())
            s = "p_remain_pos is under zero becase of diffusion_number" + s
            raise ArithmeticError(s)

        self.positive_courant_number_tf = positive_courant_number_tf + diffusion_number_tf
        self.negative_courant_number_tf = negative_courant_number_tf + diffusion_number_tf
        self.abs_courant_number_tf = abs_courant_number_tf + 2 * sum_diffusion_number_tf
        self.positive_gather_tf = tf.constant(positive_gather, dtype=tf.int32)
        self.negative_gather_tf = tf.constant(negative_gather, dtype=tf.int32)

        self.get_boundary_condition()

        print("self.positive_courant_number_tf")
        print(self.positive_courant_number_tf)
        print("self.negative_courant_number_tf")
        print(self.negative_courant_number_tf)
        print("self.abs_courant_number_tf")
        print(self.abs_courant_number_tf)
        print("self.positive_gather_tf")
        print(self.positive_gather_tf)
        print("self.negative_gather_tf")
        print(self.negative_gather_tf)
        print("self.boundary_condition_tf")
        print(self.boundary_condition_tf)
        if save:
            self.save_stochastic_matrix()

        print("\n init_stochastic_matrix end \n")

    def simulate(self):
        self.astablishment_space_tf.assign(self.astablishment_space_tf * self.boundary_condition_tf)
        positive = tf.reduce_sum(self.positive_courant_number_tf * tf.gather(self.astablishment_space_tf, self.positive_gather_tf), 1)
        negative = tf.reduce_sum(self.negative_courant_number_tf * tf.gather(self.astablishment_space_tf, self.negative_gather_tf), 1)
        self.astablishment_space_tf.assign(positive + negative - self.abs_courant_number_tf * self.astablishment_space_tf + self.astablishment_space_tf)
        self._simulate_time += self.delta_t

    def save_velocity_space(self):
        path2dir = "./velocity_space/" + self.model.name
        os.makedirs(path2dir, exist_ok=True)
        file_list = glob.glob(path2dir + "/*")

        n = 1
        while(True):
            path = path2dir + "/velocity_space" + str(n)
            n += 1
            if not path in file_list:
                print(path)
                os.mkdir(path)
                os.chdir(path)
                np.save("velocity_space", self.velocity_space_tf.numpy())
                self.write_param()
                break
        os.chdir("../../../")

    def load_velocity_space(self, num):
        print("load_velocity_space")
        stochastic_matrix_path = "./velocity_space/" + self.model.name + "/velocity_space" + str(num)
        os.chdir(stochastic_matrix_path)

        self.velocity_space_tf = tf.constant(np.load("velocity_space.npy"), dtype=tf.float32)

        os.chdir("../../../")
        print("load_velocity_space end")

    def save_stochastic_matrix(self):
        path2dir = "./stochastic_matrix/" + self.model.name
        os.makedirs(path2dir, exist_ok=True)
        file_list = glob.glob(path2dir + "/*")

        n = 1
        while(True):
            path = path2dir + "/stochastic_matrix" + str(n)
            n += 1
            if not path in file_list:
                print(path)
                os.mkdir(path)
                os.chdir(path)

                np.save("positive_courant_number", self.positive_courant_number_tf.numpy())
                np.save("negative_courant_number", self.negative_courant_number_tf.numpy())
                np.save("abs_courant_number", self.abs_courant_number_tf.numpy())
                np.save("positive_gather", self.positive_gather_tf.numpy())
                np.save("negative_gather", self.negative_gather_tf.numpy())
                np.save("boundary_condition", self.boundary_condition_tf.numpy())

                self.write_param()
                break
        os.chdir("../../../")

    def load_stochastic_matrix(self, num):
        print("load_stochastic_matrix")
        stochastic_matrix_path = "./stochastic_matrix/" + self.model.name + "/stochastic_matrix" + str(num)
        os.chdir(stochastic_matrix_path)

        self.positive_courant_number_tf = tf.constant(np.load("positive_courant_number.npy"), dtype=tf.float32)
        self.negative_courant_number_tf = tf.constant(np.load("negative_courant_number.npy"), dtype=tf.float32)
        self.abs_courant_number_tf = tf.constant(np.load("abs_courant_number.npy"), dtype=tf.float32)
        self.positive_gather_tf = tf.constant(np.load("positive_gather.npy"), dtype=tf.int32)
        self.negative_gather_tf = tf.constant(np.load("negative_gather.npy"), dtype=tf.int32)
        self.boundary_condition_tf = tf.constant(np.load("boundary_condition.npy"), dtype=tf.float32)

        os.chdir("../../../")
        print("load_stochastic_matrix end")
