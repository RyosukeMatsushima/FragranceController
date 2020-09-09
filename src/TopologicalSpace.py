import numpy as np
from src.Axis import Axis

class TopologicalSpace:
    def __init__(self):
        self.theta_axis = Axis("theta", -10.0, 10.0, 0.01)
        self.theta_dot_axis = Axis("theta_dot", -10.0, 10.0, 0.01)

        self.axes = (
            self.theta_axis,
            self.theta_dot_axis
        )
        self.element_count = self._element_count(self.axes)
        self.astablishment_space = np.zeros(self.element_count)

        print(self.element_count)

    def _element_count(self, axes):
        val = 0
        for axis in axes:
            val += axis.element_count
        return val

    def get_val(self, coodinate):
        return self.astablishment_space[self.coodinate2num(coodinate)]

    def write_val(self, coodinate, val):
        self.astablishment_space[self.coodinate2num(coodinate)] = val

    def coodinate2num(self, coodinate):
        if len(coodinate) is not len(self.axes):
            raise TypeError("size of coodinate and number of axes is not same")
        num = 0
        step = 1
        for i in range(len(self.axes)):
            step *= self.axes[i].element_count
            self.axes[i].check_num_range(coodinate[i])
            num += coodinate[i] * int(self.element_count/step)
        return num

    def stochastic_matrix(self):
        a = np.arange(0, 4, 1)
        b = np.arange(0, 4*4, 1)
        a = a.reshape(2, 2)
        b = b.reshape(2, 2, 2, 2)
        print(np.dot(a, b))
