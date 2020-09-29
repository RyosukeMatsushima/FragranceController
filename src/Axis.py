import numpy as np

class Axis:
    def __init__(self, name, min, max, min_step):
        self.name = name
        self.min = min
        self.max =max
        self.min_step = min_step
        self.elements = np.arange(min, max + min_step, min_step)
        self.steps = np.full(self.elements.shape[0], min_step, dtype=float)

    def num2val(self, num):
        self.check_num_range(num)
        return self.elements[num]

    def val2num(self, val):
        self.check_val_range(val)
        return np.abs(self.elements - val).argmin()

    def check_num_range(self, num):
        if num < 0 or len(self.elements) < num:
            raise ArithmeticError("num is not in range")

    def check_val_range(self, val):
        if val < self.min or self.max < val:
            raise ArithmeticError("val is not in range")

    def get_step(self, num):
        return self.steps[num]

    def get_param(self):
        return {"name": self.name, "min": self.min, "max": self.max, "min_step": self.min_step, "elements": self.elements.tolist()}

    def set_param(self, **kwargs):
        for key in kwargs:
            if key == "name":
                self.name = kwargs[key]
                continue
            if key == "min":
                self.min = kwargs[key]
                continue
            if key == "max":
                self.max = kwargs[key]
                continue
            if key == "min_step":
                self.min_step = kwargs[key]
                continue
            if key == "elements":
                self.elements = np.array(kwargs[key])
                continue
            raise TypeError("The required key {key!r} ""are not in kwargs".format(key=key))