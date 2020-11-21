import numpy as np
import matplotlib.pyplot as plt
import itertools
from tempfile import mkdtemp
import os.path as path
from tqdm import tqdm

from src.Axis import Axis
from src.submodule.PhysicsSimulator.SinglePendulum.SinglePendulum import SinglePendulum

'''
# name space
pos_in_astablishment_space pos_AS
pos_in_topological_space pos_TS
coodinate
'''

class TopologicalSpace:
    def __init__(self, *axes: Axis):
        self.model = SinglePendulum(3, 0, mass=0.6, length=2, drag=0.1)
        self.axes = axes
        self.element_count = self._element_count(self.axes)
        filename = path.join(mkdtemp(), 'astablishment_space.dat')
        self.astablishment_space = np.memmap(filename, dtype='float32', mode='w+', shape=self.element_count)

        # parameter set
        self.delta_t = 0.001

    def _element_count(self, axes):
        val = 1
        for axis in axes:
            val *= len(axis.elements)
        return val

    def get_val(self, coodinate):
        pos_TS = self.coodinate2pos_TS(coodinate)
        return self.astablishment_space[self.pos_TS2pos_AS(pos_TS)]

    def write_val(self, coodinate, val):
        pos_TS = self.coodinate2pos_TS(coodinate)
        self.astablishment_space[self.pos_TS2pos_AS(pos_TS)] = val

    def pos_TS2pos_AS(self, pos_TS):
        if len(pos_TS) is not len(self.axes):
            raise TypeError("size of pos_TS and number of axes is not same")
        axis_list = [len(axis.elements) - 1 for axis in self.axes]
        pos = 0
        for i in range(len(self.axes)):
            del axis_list[0]
            step = self._times_all(axis_list)
            pos += pos_TS[i] * step
        return pos

    def pos_AS2pos_TS(self, pos_AS):    #TODO: recheck
        pos = pos_AS
        pos_TS = []
        for axis in self.axes:
            pos_TS.append(int(pos/len(axis.elements)))
            pos = pos % len(axis.elements)
        return pos_TS

    def _times_all(self, l):
        val = 1
        for i in l:
            val *= i
        return val

    def pos_TS2coodinate(self, pos_TS):
        return [self.axes[i].num2val(pos_TS[i]) for i in range(len(self.axes))]

    def coodinate2pos_TS(self, coodinate):
        if len(coodinate) is not len(self.axes):
            raise TypeError("size of coodinate and number of axes is not same")
        pos_TS = []
        for i in range(len(self.axes)):
            pos_TS.append(self.axes[i].val2num(coodinate[i]))
        return pos_TS

    def coodinate2pos_AS(self, coodinate):
        return self.pos_TS2pos_AS(self.coodinate2pos_TS(coodinate))

    def is_edge_of_TS(self, pos_TS):
        if len(pos_TS) is not len(self.axes):
            raise TypeError("size of element_num and number of axes is not same")
        pos = np.array(pos_TS)
        min_edge = np.full(len(self.axes), 0, dtype=int)
        max_edge = np.array([len(axis.elements) - 1 for axis in self.axes])
        return True in (pos == min_edge) or True in (pos == max_edge)

    def pos_TS_elements(self):
        return list(itertools.product(*[list(range(len(axis.elements))) for axis in self.axes]))


# calcurate stochastic_matrix using windward difference method
# TODO: find refarence
    def stochastic_matrix(self, is_time_reversal: bool, input_set, input_P_set):
        print("calculate stochastic_matrix")
        filename = path.join(mkdtemp(), 'stochastic_matrix.dat')
        stochastic_matrix = np.memmap(filename, dtype='float32', mode='w+', shape=(self.element_count, self.element_count))
        pos_TS_elements = self.pos_TS_elements()
        time_direction = -1.0 if is_time_reversal else 1.0
        for pos_TS in tqdm(pos_TS_elements):
            for i, input in enumerate(input_set):
                input_P = input_P_set[i]
                if self.is_edge_of_TS(pos_TS):
                    continue
                velosites = time_direction * self.model.dynamics(*self.pos_TS2coodinate(pos_TS), input) # velosity as a vector

                p_remain_pos = 1.0    #P(pos_i, t + delta_t | pos_i, t)
                for i in range(len(pos_TS)):
                    velosity = velosites[i]
                    step = self.axes[i].get_step(pos_TS[i])
                    courant_number = velosity * self.delta_t / step

                    pos = pos_TS[:]
                    pos = list(pos)
                    if courant_number > 0:
                        pos[i] += 1
                    else:
                        pos[i] -= 1
                    stochastic_matrix[self.pos_TS2pos_AS(pos)][self.pos_TS2pos_AS(pos_TS)] += abs(courant_number) * input_P
                    p_remain_pos -= abs(courant_number)

                if p_remain_pos < 0:
                    raise ArithmeticError("p_remain_pos is under zero")
                stochastic_matrix[self.pos_TS2pos_AS(pos_TS)][self.pos_TS2pos_AS(pos_TS)] += p_remain_pos * input_P

        return stochastic_matrix

    def gradient_matrix(self):
        def differential_calculus(pre, follow, step):
            return (pre - follow)/(2*step)

        gradient_matrix = np.zeros((self.element_count, len(self.axes)))
        pos_TS_elements = self.pos_TS_elements()

        for pos_TS in pos_TS_elements:
            if self.is_edge_of_TS(pos_TS):
                continue

            for i in range(len(pos_TS)): #TODO: turn to list first
                pre_pos = list(pos_TS[:])
                pre_pos[i] -= 1
                follow_pos = list(pos_TS[:])
                follow_pos[i] += 1

                if (follow_pos[i] - pre_pos[i]) is not 2:
                    ArithmeticError("gradient_matrix is wrong") #TODO: remove sometimes

                pre_val = self.astablishment_space[self.pos_TS2pos_AS(pre_pos)]
                follow_val = self.astablishment_space[self.pos_TS2pos_AS(follow_pos)]
                step = self.axes[i].get_step(pos_TS[i])
                gradient_matrix[self.pos_TS2pos_AS(pos_TS), i] = differential_calculus(pre_val, follow_val, step)

        return gradient_matrix

    def gradient_matrix2(self):
        gradient_matrix = np.zeros((self.element_count, len(self.axes)))
        pos_TS_elements = self.pos_TS_elements()

        for pos_TS in pos_TS_elements:
            if self.is_edge_of_TS(pos_TS):
                continue

            time_list = np.zeros(len(pos_TS))
            for i in range(len(pos_TS)): #TODO: turn to list first
                pre_pos = list(pos_TS[:])
                pre_pos[i] -= 1
                follow_pos = list(pos_TS[:])
                follow_pos[i] += 1

                if (follow_pos[i] - pre_pos[i]) is not 2:
                    ArithmeticError("gradient_matrix is wrong") #TODO: remove sometimes

                pre_val = self.astablishment_space[self.pos_TS2pos_AS(pre_pos)]
                follow_val = self.astablishment_space[self.pos_TS2pos_AS(follow_pos)]
                val = 0.0
                direction = 0.0
                if pre_val < follow_val:
                    val = pre_val
                    direction = -1.0
                else:
                    val = follow_val
                    direction = 1.0
                if val == 0:
                    continue
                time_list[i] = val
                gradient_matrix[self.pos_TS2pos_AS(pos_TS), i] = direction

            min_time = np.min(time_list)
            if min_time == 0.0:
                continue
            gradient_matrix[self.pos_TS2pos_AS(pos_TS), np.where( time_list != min_time )] = 0.0

        return gradient_matrix

    def axis_name2axis(self, name):
        axis_index_list = [i for i, axis in enumerate(self.axes) if axis.name == name]
        if len(axis_index_list) is not 1:
            raise TypeError(f"wrong axis name {name}")
        axis_index = axis_index_list[0]
        return self.axes[axis_index], axis_index

    def get_concentration(self, axis1: Axis, axis2: Axis, axis1_index, axis2_index, coodinate):
        concentration = np.zeros((len(axis1.elements), len(axis2.elements)))

        for i, axis1_val in enumerate(axis1.elements):
            for j, axis2_val in enumerate(axis2.elements):
                coodinate[axis1_index] = axis1_val
                coodinate[axis2_index] = axis2_val
                concentration[i][j] = self.astablishment_space[self.coodinate2pos_AS(coodinate)]

        return concentration

    def show_astablishment_space(self, axis1_name, axis2_name, coodinate):
        axis1, axis1_index = self.axis_name2axis(axis1_name)
        axis2, axis2_index = self.axis_name2axis(axis2_name)
        concentration = self.get_concentration(axis1, axis2, axis1_index, axis2_index, coodinate)
        plt.figure(figsize=(7,5))
        ex = [axis1.min, axis1.max, axis2.min, axis2.max]
        plt.imshow(np.flip(concentration.T, 0),extent=ex,interpolation='nearest',cmap='Blues',aspect=(axis1.max-axis1.min)/(axis2.max-axis2.min),alpha=1)

        plt.colorbar()
        plt.show()

    def show_astablishment_space_in_range(self, axis1_name, axis2_name, coodinate, max_value):
        axis1, axis1_index = self.axis_name2axis(axis1_name)
        axis2, axis2_index = self.axis_name2axis(axis2_name)
        concentration = self.get_concentration(axis1, axis2, axis1_index, axis2_index, coodinate)
        plt.figure(figsize=(7,5))
        ex = [axis1.min, axis1.max, axis2.min, axis2.max]
        plt.imshow(np.flip(np.where(concentration > max_value, max_value, concentration).T, 0),extent=ex,interpolation='nearest',cmap='Blues',aspect=(axis1.max-axis1.min)/(axis2.max-axis2.min),alpha=1)

        plt.colorbar()
        plt.show()

    def show_astablishment_space_with_tragectory(self, axis1_name, axis2_name, coodinate, data1, data2):
        axis1, axis1_index = self.axis_name2axis(axis1_name)
        axis2, axis2_index = self.axis_name2axis(axis2_name)
        concentration = self.get_concentration(axis1, axis2, axis1_index, axis2_index, coodinate)
        fig = plt.figure(figsize=(7,5))
        ax = fig.subplots()
        ex = [axis1.min, axis1.max, axis2.min, axis2.max]
        # TODO: add color bar
        ax.imshow(np.flip(concentration.T, 0),extent=ex,interpolation='nearest',cmap='Blues',aspect=(axis1.max-axis1.min)/(axis2.max-axis2.min),alpha=1)

        ax.plot(data1, data2, linewidth=1, color="crimson")
        ax.set_xlabel(r"$\theta$ [rad]") #TODO: appley other axis
        ax.set_ylabel(r"$\dot \theta$ [rad/s]")
        plt.show()

    def show_concentration_img(self, axis1: Axis, axis2: Axis, concentration):
        plt.figure(figsize=(7,5))
        ex = [axis1.min, axis1.max, axis2.min, axis2.max]
        plt.imshow(np.flip(concentration.T, 0),extent=ex,interpolation='nearest',cmap='Blues',aspect=(axis1.max-axis1.min)/(axis2.max-axis2.min),alpha=1)

        plt.colorbar()
        plt.show()

    def show_concentration_img_with_tragectory(self, axis1: Axis, axis2: Axis, concentration, data1, data2):
        fig = plt.figure(figsize=(7,5))
        ax = fig.subplots()
        ex = [axis1.min, axis1.max, axis2.min, axis2.max]
        ax.imshow(np.flip(concentration.T, 0),extent=ex,interpolation='nearest',cmap='Blues',aspect=(axis1.max-axis1.min)/(axis2.max-axis2.min),alpha=1)

        ax.plot(data1, data2, linewidth=1, color="crimson")
        ax.set_xlabel(r"$\theta$ [rad]") #TODO: appley other axis
        ax.set_ylabel(r"$\dot \theta$ [rad/s]")
        plt.show()

    def show_quiver(self, gradient_matrix):
        fig, ax = plt.subplots()
        x1_n = int(len(self.axes[0].elements)/40)
        x2_n = int(len(self.axes[1].elements)/40)
        fig_x1_set = self.axes[0].elements[::x1_n]
        fig_x2_set = self.axes[1].elements[::x2_n]
        fig_velocoty_x1_dot_set = np.zeros((len(fig_x1_set), len(fig_x2_set)))
        fig_velocoty_x2_dot_set = np.zeros((len(fig_x1_set), len(fig_x2_set)))
        for i, x1 in enumerate(fig_x1_set):
            for j, x2 in enumerate(fig_x2_set):
                pos_TS = [x1, x2]
                fig_velocoty_x1_dot_set[i][j] = gradient_matrix[self.coodinate2pos_AS(pos_TS), 0]
                fig_velocoty_x2_dot_set[i][j] = gradient_matrix[self.coodinate2pos_AS(pos_TS), 1]

        q = ax.quiver(fig_x1_set, fig_x2_set, fig_velocoty_x1_dot_set.T, fig_velocoty_x2_dot_set.T)
        ax.quiverkey(q, X=0.3, Y=1.1, U=10, label='Quiver key, length = 10', labelpos='E')

        plt.show()
