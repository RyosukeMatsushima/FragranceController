import unittest
import numpy as np

from src.TopologicalSpace import TopologicalSpace
from src.FuildSimulator import FuildSimulator
from src.Axis import Axis
from src.submodule.PhysicsSimulator.SinglePendulum.SinglePendulum import SinglePendulum

class FuildSimulatorTest(unittest.TestCase):
    def test_save_velcity_space(self):
        print("test_save_velcity_space")
        axes = (Axis("theta", -7.0, 7.0, 0.1), Axis("theta_dot", - 7.0, 7.0, 0.1))
        self.topologicalSpace = TopologicalSpace(*axes)
        self.model = SinglePendulum(0, 0, mass=0.6, length=2, drag=0.1)
        graph_center = ["theta", "theta_dot", [0, 0]]
        self.fuildSimulator = FuildSimulator(self.topologicalSpace, [1, 1], graph_center, [-1, 0, 1], 0, self.model, 0.00001)
        self.fuildSimulator.init_velocity_space(True)

    def test_load_velcity_space(self):
        print("test_load_velcity_space")
        axes = (Axis("theta", -7.0, 7.0, 0.1), Axis("theta_dot", - 7.0, 7.0, 0.1))
        self.topologicalSpace = TopologicalSpace(*axes)
        self.model = SinglePendulum(0, 0, mass=0.6, length=2, drag=0.1)
        graph_center = ["theta", "theta_dot", [0, 0]]
        self.fuildSimulator = FuildSimulator(self.topologicalSpace, [1, 1], graph_center, [-1, 0, 1], 0, self.model, 0.00001)
        self.fuildSimulator.load_velocity_space(1)
        self.fuildSimulator.init_stochastic_matrix(True)
        self.fuildSimulator.add_obstacle([-4., -2.], [-3., -1.5])
        self.fuildSimulator.method1_5(100000, True)

    def test_init_stochastic_matrix(self):
        axes = (Axis("theta", -7.0, 7.0, 0.1), Axis("theta_dot", - 7.0, 7.0, 0.1))
        self.topologicalSpace = TopologicalSpace(*axes)
        self.model = SinglePendulum(0, 0, mass=0.6, length=2, drag=0.1)
        graph_center = ["theta", "theta_dot", [0, 0]]
        self.fuildSimulator = FuildSimulator(self.topologicalSpace, [1, 1], graph_center, [-1, 0, 1], 0, self.model, 0.00001)
        self.fuildSimulator.init_velocity_space(False)
        self.fuildSimulator.init_stochastic_matrix(True)
        self.fuildSimulator.add_obstacle([-4., -2.], [-3., -1.5])
        self.fuildSimulator.method1_5(100000, True)
        # self.fuildSimulator.method2(0.0024, 10000, True)
        # print("self.fuildSimulator.vel_space_tf")
        # print(self.fuildSimulator.vel_space_tf)

    def test_get_boundary_condition(self):
        axes = (Axis("theta", -2.0, 2.0, 0.1), Axis("theta_dot", - 1.0, 1.0, 0.1))
        self.topologicalSpace = TopologicalSpace(*axes)
        self.model = SinglePendulum(0, 0, mass=0.6, length=2, drag=0.1)
        graph_center = ["theta", "theta_dot", [0, 0]]
        self.fuildSimulator = FuildSimulator(self.topologicalSpace, [0, 0], graph_center, [-1, 0, 1], 0, self.model, 0.001)
        self.fuildSimulator.get_boundary_condition()

if __name__ == "__main__":
    unittest.main()