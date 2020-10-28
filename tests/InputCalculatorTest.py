from unittest import TestCase
from src.InputCalculator import InputCalculator
from src.TopologicalSpace import TopologicalSpace
from src.Axis import Axis

axes = (Axis("theta", -1.0, 7.0, 0.1), Axis("theta_dot", -10.0, 10.0, 0.05))
t_s = TopologicalSpace(*axes)
inputCalculator = InputCalculator(t_s)
# inputCalculator.method0()
inputCalculator.method2(0.024, 100)
# inputCalculator.update_astablishment_space()
# inputCalculator.t_s.show_plot("theta", "theta_dot", [0, 0])

# while(True):
#     for i in range(1000):
#         inputCalculator.simulate()
#     inputCalculator.update_astablishment_space()
#     inputCalculator.t_s.show_plot("theta", "theta_dot", [0, 0])
#     print(inputCalculator._simulate_time)
