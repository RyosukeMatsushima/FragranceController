import unittest
import numpy as np

from src.TopologicalSpace import TopologicalSpace
from src.FuildSimulator import FuildSimulator
from src.Axis import Axis
from src.submodule.PhysicsSimulator.SinglePendulum.SinglePendulum import SinglePendulum

class FuildSimulatorTest(unittest.TestCase):
    def test_init_stochastic_matrix(self):
        axes = (Axis("theta", -10.0, 10.0, 0.1), Axis("theta_dot", - 10.0, 10.0, 0.1))
        self.topologicalSpace = TopologicalSpace(*axes)
        self.model = SinglePendulum(0, 0)
        graph_center = ["theta", "theta_dot", [0, 0]]
        self.fuildSimulator = FuildSimulator(self.topologicalSpace, [0, 0], graph_center, [-1, 0, 1], 0, self.model, 0.00001)
        self.fuildSimulator.init_stochastic_matrix(True)
        # print("self.fuildSimulator.vel_space_tf")
        # print(self.fuildSimulator.vel_space_tf)

if __name__ == "__main__":
    unittest.main()