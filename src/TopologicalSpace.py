import numpy as np
from src.Axis import Axis

class TopologicalSpace:
    def __init__(self):
        self.theta_axis = Axis("theta", -10.0, 10.0, 0.01)
        self.theta_dot_axis = Axis("theta_dot", -10.0, 10.0, 0.01)

        axes = (self.theta_axis.element_count,
                self.theta_dot_axis.element_count)
        self.scalar_grid = np.zeros(axes)
        self.bool_grid = np.zeros(axes, dtype=bool)
        
        print(self.scalar_grid)
        print(self.bool_grid)
