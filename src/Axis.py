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
        self.check_num_range(num)
        return self.elements[num]

    def val2num(self, val):
        self.check_val_range(val)
        return int(val / self.min_step)

    def check_num_range(self, num):
        if num < 0 or self.element_count + 1 < num:
            raise ArithmeticError("num is not in range")


    def check_val_range(self, val):
        if val < self.min or self.max < val:
            raise ArithmeticError("val is not in range")
