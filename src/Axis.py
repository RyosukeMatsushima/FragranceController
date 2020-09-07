import numpy as np

class Axis:
    def __init__(self, name, min, max, min_step):
        self.name = name
        self.min = min
        self.max =max
        self.min_step = min_step
        self.elements = np.arange(min, max + min_step, min_step)
        self.element_count = self.elements.shape[0]

    def num2val(self, num):
        if num < 0 and self.element_count + 1 < num:
            ArithmeticError("num is not in range")
        return self.elements[num]

    def val2num(self, val):
        if val < self.min and self.max < val:
            ArithmeticError("val is not in range")
        return int(val / self.min_step)

