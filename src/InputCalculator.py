import numpy as np
import pandas as pd
from copy import copy
import tensorflow as tf
from tempfile import mkdtemp
import os.path as path
from tqdm import tqdm
from src.TopologicalSpace import TopologicalSpace

# for save input space
import os
import glob
import json
import datetime

class InputCalculator:
    def __init__(self, t_s: TopologicalSpace, target_coodinate, graph_arg, u_set, moderate_u, model, delta_t):
        self.t_s = t_s
        self.graph_arg = graph_arg
        self.model = model
        self.delta_t = delta_t
        self.is_time_reversal = True

        self.u_set = u_set
        self.moderate_u = moderate_u
        u_P_list = np.full(len(u_set), 1.)
        self.u_P = 1/u_P_list.sum()
        self.u_P_set = u_P_list/u_P_list.sum()
        print("u_P_set")
        print(self.u_P_set)

        self.target_coodinate = target_coodinate
        self._simulate_time = 0.
        self.astablishment_space = copy(self.t_s.astablishment_space)
        self.t_s.astablishment_space[self.t_s.coodinate2pos_AS(self.target_coodinate)] = 1.0 # set target coodinate
        self.astablishment_space_tf = tf.Variable(np.array([self.t_s.astablishment_space]).T, dtype=tf.float32)

    def init_stochastic_matrix(self, save: bool):
        stochastic_matrix = self.t_s.stochastic_matrix(self.is_time_reversal, self.u_set, self.u_P_set)
        self.stochastic_matrix_tf = tf.constant(stochastic_matrix, dtype=tf.float32)
        if save:
            self.save_stochastic_matrix(stochastic_matrix)

    def init_eye(self):
        print("init_eye")
        filename = path.join(mkdtemp(), 'eye_mat.dat')
        eye_mat = np.memmap(filename, dtype='float32', mode='w+', shape=(self.t_s.element_count, self.t_s.element_count))
        for i in tqdm(range(self.t_s.element_count)):
            eye_mat[i][i] = 1.
        self.eye_tf = tf.constant(eye_mat, dtype=tf.float32)

    def set_stochastic_matrix(self, stochastic_matrix):
        self.stochastic_matrix_tf = tf.constant(stochastic_matrix, dtype=tf.float32)

    def simulate(self):
        self.astablishment_space_tf.assign(tf.matmul(self.stochastic_matrix_tf, self.astablishment_space_tf))
        self._simulate_time += self.t_s.delta_t

    def update_astablishment_space(self):
        self.t_s.astablishment_space = self.astablishment_space_tf.numpy().T[0]

    def norm_velosity(self, coodinate, u):
        vel = self.model.dynamics(*coodinate, u)
        norm_param = np.linalg.norm(vel)
        if norm_param == 0.0:
            return np.zeros(vel.shape)
        return vel/norm_param

    def method0(self):

        self.update_astablishment_space()
        self.t_s.show_astablishment_space(*self.graph_arg)
        self.model.state = self.target_coodinate
        columns = ["time"] + [axis.name for axis in self.t_s.axes]
        df = pd.DataFrame(columns=columns)
        for i in range(10):
            for j in range(1000):
                self.simulate()
                self.model.step(self.t_s.delta_t)
                tmp_data = tuple([self._simulate_time]) + self.model.state
                tmp_se = pd.Series(tmp_data, index=df.columns)
                df = df.append(tmp_se, ignore_index=True)

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
        self.save_input_space(input_space)
        self.t_s.show_concentration_img(self.t_s.axes[0], self.t_s.axes[1], input_space)

    def method1(self):
        self.simTimeToLim()
        self.update_astablishment_space()
        self.save_astablishment_space(self.t_s.astablishment_space)

    def method1_5(self, trains_num, graph: bool):
        for i in tqdm(range(trains_num)):
            self.update_astablishment_space()
            self.astablishment_space += self.t_s.astablishment_space
            self.simulate()
            if i % int(trains_num/10) == 0 and graph:
                self.t_s.show_astablishment_space(*self.graph_arg)
                save = self.t_s.astablishment_space
                self.t_s.astablishment_space = self.astablishment_space
                self.t_s.show_astablishment_space(*self.graph_arg)
                self.t_s.astablishment_space = save
        self.t_s.astablishment_space = self.astablishment_space
        self.t_s.show_astablishment_space(*self.graph_arg)
        self.save_astablishment_space(self.t_s.astablishment_space)

    def method2(self, threshold_param, trains_num, graph: bool):
        # self.t_s.astablishment_space.fill(0.0)
        # self.astablishment_space.fill(0.0)

        for i in tqdm(range(trains_num)):
            self.update_astablishment_space()
            threshold = self.t_s.astablishment_space.max() * threshold_param
            self.astablishment_space[np.where((self.t_s.astablishment_space > threshold) & (self.astablishment_space == 0.0))] = self._simulate_time
            self.simulate()
            if i % int(trains_num/10) == 0 and graph:
                save = self.t_s.astablishment_space
                self.t_s.astablishment_space = self.astablishment_space
                self.t_s.show_astablishment_space(*self.graph_arg)
                self.t_s.astablishment_space = save
        self.t_s.astablishment_space = self.astablishment_space
        self.t_s.show_astablishment_space(*self.graph_arg)
        self.save_astablishment_space(self.t_s.astablishment_space)

    def getInputSpace(self, to_high: bool):
        direction = -1.0 if to_high else 1.0
        gradient_matrix = self.t_s.gradient_matrix()
        shape = [len(axis.elements) for axis in self.t_s.axes]
        if type(self.moderate_u) == list:
          shape += [len([self.moderate_u])]
        input_space = np.zeros(shape)
        pos_TS_elements = self.t_s.pos_TS_elements()

        for pos_TS in tqdm(pos_TS_elements):
            gradient = direction * gradient_matrix[self.t_s.pos_TS2pos_AS(pos_TS)]
            dot_list = np.array([np.dot(gradient, self.norm_velosity(self.t_s.pos_TS2coodinate(pos_TS), u)) for u in self.u_set])
            proposal_input = np.array(self.u_set)[np.where(dot_list == max(dot_list))]
            if len(proposal_input) is not 1:
                if gradient[0] != 0.0:
                    print("sevral proposal inputs exist")
                    print("gradient")
                    print(gradient)
                    print("pos_TS")
                    print(pos_TS)
                    print("coodinate")
                    print(self.t_s.pos_TS2coodinate(pos_TS))
                    # raise TypeError("omg")
                proposal_input = self.moderate_u
            input_space[pos_TS] = proposal_input

        self.save_input_space(input_space)
        self.t_s.show_concentration_img(self.t_s.axes[0], self.t_s.axes[1], input_space)

    def get_input(self, coodinate, to_high: bool):
        direction = -1.0 if to_high else 1.0
        gradient = direction * self.t_s.get_gradient(coodinate)
        dot_list = np.array([np.dot(gradient, self.norm_velosity(coodinate, u)) for u in self.u_set])
        proposal_input = np.array(self.u_set)[np.where(dot_list == max(dot_list))]
        if len(proposal_input) is not 1:
            if gradient[0] != 0.0:
                print("sevral proposal inputs exist")
                print("gradient")
                print(gradient)
                print("coodinate")
                print(coodinate)
                # raise TypeError("omg")
            proposal_input = self.moderate_u
        else:
            proposal_input = proposal_input[0]
        return proposal_input

    def getInputSpace2(self, to_high: bool):
        direction = -1.0 if to_high else 1.0
        gradient_matrix = self.t_s.gradient_matrix2()
        size_list = [len(axis.elements) for axis in self.t_s.axes]
        if type(self.moderate_u) is list:
            size_list += [len(self.moderate_u)]
        input_space = np.zeros(size_list)
        pos_TS_elements = self.t_s.pos_TS_elements()
        if len(self.t_s.axes) <= 2:
            self.t_s.show_quiver(gradient_matrix)

        for pos_TS in tqdm(pos_TS_elements):
            gradient = direction * gradient_matrix[self.t_s.pos_TS2pos_AS(pos_TS)]
            dot_list = np.array([np.dot(gradient, self.norm_velosity(pos_TS, input)) for input in self.u_set])
            proposal_input = np.array(self.u_set)[np.where(dot_list == max(dot_list))]
            if len(proposal_input) is not 1:
                print("sevral proposal inputs exist")
                print(self.t_s.pos_TS2coodinate(pos_TS))
                print(gradient) #TODO: remove print
                if gradient[0] != 0.0:
                    print("coodinate")
                    print(self.t_s.pos_TS2coodinate(pos_TS))
                    # raise TypeError("omg")
                print(dot_list)
                print(np.where(dot_list == max(dot_list)))
                proposal_input = self.moderate_u
            input_space[pos_TS] = proposal_input

        for input in input_space:
            print(input)
        self.save_input_space(input_space)
        self.t_s.show_concentration_img(self.t_s.axes[0], self.t_s.axes[1], input_space)

    def method_save_astablishment_space(self):
        self.simTimeToLim()
        self.update_astablishment_space()
        self.save_astablishment_space(self.t_s.astablishment_space)

    @tf.function
    def simTimeToLim(self):
        # e, v = tf.linalg.eigh(self.stochastic_matrix)
        # print(e)
        #TODO: check all e element is under 1.0
        val1 = tf.linalg.inv(self.stochastic_matrix_tf - self.eye_tf)
        self.astablishment_space_tf.assign(- self.stochastic_matrix_tf @ val1 @ self.astablishment_space_tf)

    def save_input_space(self, input_space):
        path2dir = "./input_space/" + self.model.name
        os.makedirs(path2dir, exist_ok=True)
        file_list = glob.glob(path2dir + "/*")

        n = 1
        while(True):
            path = path2dir + "/input_space" + str(n)
            n += 1
            if not path in file_list:
                print(path)
                os.mkdir(path)
                os.chdir(path)
                np.save("input_space", input_space)
                self.write_param()
                break
        os.chdir("../../../")

    def save_astablishment_space(self, astablishment_space):
        path2dir = "./astablishment_space/" + self.model.name
        os.makedirs(path2dir, exist_ok=True)
        file_list = glob.glob(path2dir + "/*")

        n = 1
        while(True):
            path = path2dir + "/astablishment_space" + str(n)
            n += 1
            if not path in file_list:
                print(path)
                os.mkdir(path)
                os.chdir(path)
                np.save("astablishment_space", astablishment_space)
                self.write_param()
                break
        os.chdir("../../../")

    def save_stochastic_matrix(self, stochastic_matrix):
        file_list = glob.glob("./stochastic_matrix/*")

        n = 1
        while(True):
            path = "./stochastic_matrix/stochastic_matrix" + str(n)
            n += 1
            if not path in file_list:
                print(path)
                os.mkdir(path)
                os.chdir(path)

                for i in range(self.t_s.element_count):
                    space_name = "stochastic_matrix" + str(i)
                    np.save(space_name, stochastic_matrix[i])
                self.write_param()
                break
        os.chdir("../../")

    def write_param(self):
        dt_now = datetime.datetime.now()
        model_param = self.model.get_param()
        axes = [axis.get_param() for axis in self.t_s.axes]

        param = {
                "model_name": self.model.name,
                "datetime": str(dt_now),
                "axes": axes,
                "model_param": model_param,
                "u_set": self.u_set,
                "moderate_u": self.moderate_u,
                "delta_t": self.delta_t
                }

        with open('param.json', 'w') as json_file:
            json.dump(param, json_file)

    def load_stochastic_matrix(self, num):
        print("load_stochastic_matrix")
        stochastic_matrix_path = "./stochastic_matrix/stochastic_matrix" + str(num)
        os.chdir(stochastic_matrix_path)
        filename = path.join(mkdtemp(), 'stochastic_matrix.dat')
        stochastic_matrix = np.memmap(filename, dtype='float32', mode='w+', shape=(self.t_s.element_count, self.t_s.element_count))

        for i in tqdm(range(self.t_s.element_count)):
            name = "stochastic_matrix" + str(i) + ".npy"
            stochastic_matrix[i] = np.load(name)
        print("load_stochastic_matrix end")
        return stochastic_matrix