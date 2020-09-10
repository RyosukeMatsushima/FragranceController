import numpy as np
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

        self.theta_axis = Axis("theta", -5.0, 5.0, 0.1)
        self.theta_dot_axis = Axis("theta_dot", -5.0, 5.0, 0.1)

        self.axes = (
            self.theta_axis,
            self.theta_dot_axis
        )
        self.element_count = self._element_count(self.axes)
        self.astablishment_space = np.zeros(self.element_count)

        # parameter set
        self.delta_t = 0.001

    def _element_count(self, axes):
        val = 0
        for axis in axes:
            val += len(axis.elements)
        return val

    def get_val(self, pos_TS):
        return self.astablishment_space[self.pos_TS2pos_AS(pos_TS)]

    def write_val(self, pos_TS, val):
        self.astablishment_space[self.pos_TS2pos_AS(pos_TS)] = val

    def pos_TS2pos_AS(self, pos_TS):
        if len(pos_TS) is not len(self.axes):
            raise TypeError("size of element_num and number of axes is not same")
        num = 0
        step = 1
        for i in range(len(self.axes)):
            step *= len(self.axes[i].elements)
            self.axes[i].check_num_range(pos_TS[i])
            num += pos_TS[i] * int(self.element_count/step)
        return num

    def pos_TS2coodinate(self, pos_TS):
        return [self.axes[i].num2val(pos_TS[i]) for i in range(len(self.axes))]

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
                val = self.axes[i].num2val(pos_TS[i])
                step = self.axes[i].get_step(val)
                courant_number = velosity * self.delta_t / step

                pos = pos_TS[:]
                pos = list(pos)
                if courant_number > 0:
                    pos[i] += 1
                else:
                    pos[i] -= 1
                stochastic_matrix[self.pos_TS2pos_AS(pos)][self.pos_TS2pos_AS(pos_TS)] = courant_number
                p_remain_pos -= abs(courant_number)

            if p_remain_pos < 0:
                raise ArithmeticError("p_remain_pos is under zero")
            stochastic_matrix[self.pos_TS2pos_AS(pos_TS)][self.pos_TS2pos_AS(pos_TS)] = p_remain_pos

        return stochastic_matrix
