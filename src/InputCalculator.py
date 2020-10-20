import numpy as np
import pandas as pd
from copy import copy
import tensorflow as tf
from tqdm import tqdm
from src.TopologicalSpace import TopologicalSpace

# for save input space
import os
import glob
import json
import datetime

class InputCalculator:
    def __init__(self, t_s: TopologicalSpace):
        self.t_s = t_s
        is_time_reversal = True

        d = 1.
        self.u_set = np.arange(-2., 2. + d, d)
        # self.u_set = np.array([0.])
        self.moderate_u = 0
        u_P_list = np.full(self.u_set.shape, 1.)
        u_P_set = u_P_list/u_P_list.sum()
        print("u_P_set")
        print(u_P_set)

        self.target_coodinate = (0., 0.)
        self._simulate_time = 0.

        self.astablishment_space = copy(self.t_s.astablishment_space)
        self.astablishment_space[self.t_s.coodinate2pos_AS(self.target_coodinate)] = 1.0 # set target coodinate
        self.astablishment_space_tf = tf.Variable(np.array([self.astablishment_space]).T, dtype=tf.float32)

        self.stochastic_matrix = np.zeros(self.t_s.stochastic_matrix(u, is_time_reversal, 1).shape)
        for i, u in enumerate(self.u_set):
            self.stochastic_matrix += self.t_s.stochastic_matrix(u, is_time_reversal, u_P_set[i])

        self.stochastic_matrix_tf = tf.constant(self.stochastic_matrix, dtype=tf.float32)

    def simulate(self):
        self.astablishment_space_tf.assign(tf.matmul(self.stochastic_matrix_tf, self.astablishment_space_tf))
        self._simulate_time += self.t_s.delta_t

    def update_astablishment_space(self):
        self.t_s.astablishment_space = self.astablishment_space_tf.numpy().T[0]

    def norm_velosity(self, pos_TS, input):
        vel = self.t_s.model.dynamics(*pos_TS, input)
        norm_param = np.linalg.norm(vel)
        if norm_param == 0.0:
            return np.zeros(vel.shape)
        return vel/norm_param

    def method0(self):

        self.update_astablishment_space()
        self.t_s.show_astablishment_space("theta", "theta_dot", [0, 0])
        self.t_s.model.state = self.target_coodinate
        columns = ["time"] + [axis.name for axis in self.t_s.axes]
        df = pd.DataFrame(columns=columns)
        for i in range(10):
            for j in range(1000):
                self.simulate()
                self.t_s.model.step(self.t_s.delta_t)
                tmp_data = tuple([self._simulate_time]) + self.t_s.model.state
                tmp_se = pd.Series(tmp_data, index=df.columns)
                df = df.append(tmp_se, ignore_index=True)

            self.update_astablishment_space()
            self.t_s.show_astablishment_space_with_tragectory("theta", "theta_dot", [0, 0], df["theta"], df["theta_dot"])
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

    def method2(self):
        threshold_param = 0.2
        self.t_s.astablishment_space.fill(0.0)
        self.astablishment_space.fill(0.0)

        for i in tqdm(range(10000)):
            self.update_astablishment_space()
            threshold = self.t_s.astablishment_space.max() * threshold_param
            self.astablishment_space[np.where((self.t_s.astablishment_space > threshold) & (self.astablishment_space == 0.0))] = self._simulate_time
            self.simulate()
            # if i % 1000 == 0:
            #     save = self.t_s.astablishment_space
            #     self.t_s.astablishment_space = self.astablishment_space
            #     self.t_s.show_astablishment_space("theta", "theta_dot", [0, 0])
            #     self.t_s.astablishment_space = save
        self.t_s.astablishment_space = self.astablishment_space
        self.t_s.show_astablishment_space("theta", "theta_dot", [0, 0])
        self.save_astablishment_space(self.t_s.astablishment_space)

    def getInputSpace(self, to_high: bool):
        direction = -1.0 if to_high else 1.0
        gradient_matrix = self.t_s.gradient_matrix()
        input_space = np.zeros([len(axis.elements) for axis in self.t_s.axes])
        pos_TS_elements = self.t_s.pos_TS_elements()
        self.t_s.show_quiver(gradient_matrix)

        for pos_TS in tqdm(pos_TS_elements):
            gradient = direction * gradient_matrix[self.t_s.pos_TS2pos_AS(pos_TS)]
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

    def method_save_astablishment_space(self):
        self.simTimeToLim()
        self.update_astablishment_space()
        self.save_astablishment_space(self.t_s.astablishment_space)

    def simTimeToLim(self):
        # e, v = tf.linalg.eigh(self.stochastic_matrix)
        # print(e)
        #TODO: check all e element is under 1.0

        eye = np.eye(self.t_s.element_count, self.t_s.element_count, dtype=float)
        eye_tf = tf.constant(eye, dtype=tf.float32)
        val1 = tf.linalg.inv(self.stochastic_matrix_tf - eye_tf)
        self.astablishment_space_tf.assign(- self.stochastic_matrix_tf @ val1 @ self.astablishment_space_tf)

    def save_input_space(self, input_space):
        dt_now = datetime.datetime.now()

        file_list = glob.glob("./input_space/*")

        n = 1
        while(True):
            path = "./input_space/input_space" + str(n)
            n += 1
            if not path in file_list:
                print(path)
                os.mkdir(path)
                os.chdir(path)
                np.save("input_space", input_space)
                model_param = self.t_s.model.get_param()
                axes = [axis.get_param() for axis in self.t_s.axes]

                param = {
                        "datetime": str(dt_now),
                        "axes": axes,
                        "model_param": model_param
                        }

                with open('param.json', 'w') as json_file:
                    json.dump(param, json_file)
                break


    def save_astablishment_space(self, astablishment_space):
        dt_now = datetime.datetime.now()

        file_list = glob.glob("./astablishment_space/*")

        n = 1
        while(True):
            path = "./astablishment_space/astablishment_space" + str(n)
            n += 1
            if not path in file_list:
                print(path)
                os.mkdir(path)
                os.chdir(path)
                np.save("astablishment_space", astablishment_space)
                model_param = self.t_s.model.get_param()
                axes = [axis.get_param() for axis in self.t_s.axes]

                param = {
                        "datetime": str(dt_now),
                        "axes": axes,
                        "model_param": model_param
                        }

                with open('param.json', 'w') as json_file:
                    json.dump(param, json_file)
                break
