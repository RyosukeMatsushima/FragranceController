import numpy as np
import matplotlib.pyplot as plt
import itertools
from src.Axis import Axis
from src.submodule.PhysicsSimulator.SinglePendulum.SinglePendulum import SinglePendulum

'''
# name space
pos_in_astablishment_space pos_AS
pos_in_topological_space pos_TS
coodinate
'''

class TopologicalSpace:
    def __init__(self):
        self.model = SinglePendulum(0, 0, mass=10, length=2, drag=4)

        self.theta_axis = Axis("theta", -1.0, 1.0, 0.1)
        self.theta_dot_axis = Axis("theta_dot", -1.0, 1.0, 0.1)

        self.axes = (
            self.theta_axis,
            self.theta_dot_axis
        )
        self.element_count = self._element_count(self.axes)
        self.astablishment_space = np.zeros(self.element_count)

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


# calcurate stochastic_matrix using windward difference method
# TODO: find refarence

    def stochastic_matrix(self, input):
        stochastic_matrix = np.zeros((self.element_count, self.element_count))
        pos_TS_elements = itertools.product(*[list(range(len(axis.elements))) for axis in self.axes])
        for pos_TS in pos_TS_elements:
            if self.is_edge_of_TS(pos_TS):
                continue
            velosites = self.model.dynamics(*self.pos_TS2coodinate(pos_TS), input) # velosity as a vector

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
                stochastic_matrix[self.pos_TS2pos_AS(pos)][self.pos_TS2pos_AS(pos_TS)] = abs(courant_number)
                p_remain_pos -= abs(courant_number)

            if p_remain_pos < 0:
                raise ArithmeticError("p_remain_pos is under zero")
            stochastic_matrix[self.pos_TS2pos_AS(pos_TS)][self.pos_TS2pos_AS(pos_TS)] = p_remain_pos

        return stochastic_matrix

    def show_plot(self, axis1_name, axis2_name, coodinate):

        axis1_index_list = [i for i, axis in enumerate(self.axes) if axis.name == axis1_name]
        axis2_index_list = [i for i, axis in enumerate(self.axes) if axis.name == axis2_name]

        if len(axis1_index_list) is not 1:
            raise TypeError(f"wrong axis name {axis1_name}")
        if len(axis2_index_list) is not 1:
            raise TypeError(f"wrong axis name {axis2_name}")

        axis1_index = axis1_index_list[0]
        axis2_index = axis2_index_list[0]
        axis1 = self.axes[axis1_index]
        axis2 = self.axes[axis2_index]

        concentration = np.zeros((len(axis1.elements), len(axis2.elements)))

        for i, axis1_val in enumerate(axis1.elements):
            for j, axis2_val in enumerate(axis2.elements):
                coodinate[axis1_index] = axis1_val
                coodinate[axis2_index] = axis2_val
                concentration[i][j] = self.astablishment_space[self.coodinate2pos_AS(coodinate)]

        plt.figure(figsize=(7,5))
        ex = [axis1.min, axis1.max, axis2.min, axis2.max]
        plt.imshow(np.flip(concentration.T, 0),extent=ex,interpolation='nearest',cmap='Blues',aspect=(axis1.max-axis1.min)/(axis2.max-axis2.min),alpha=1)

        plt.colorbar()
        plt.show()