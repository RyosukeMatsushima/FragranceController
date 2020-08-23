import numpy as np

class Axis:
    def __init__(self, name, min, max, min_step):
        self.name = name
        self.min = min
        self.max =max
        self. min_step = min_step
        self.elements = np.arange(min, max + min_step, min_step)
        self.element_count = self.elements.shape[0]
