from unittest import TestCase
from src.TimeToPosition import TimeToPosition
from src.TopologicalSpace import TopologicalSpace
from src.Axis import Axis

import numpy as np

axes = (Axis("theta", -7.0, 7.0, 0.1), Axis("theta_dot", -10.0, 10.0, 0.1))
t_s = TopologicalSpace(*axes)
graph_center = ["theta", "theta_dot", [0, 0]]

d = 1.
u_set = np.arange(-2., 2. + d, d)
moderate_u = 0
inputCalculator = TimeToPosition(t_s, (0, 0), graph_center, u_set, moderate_u)

inputCalculator.method3()

