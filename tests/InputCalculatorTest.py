from unittest import TestCase
from src.InputCalculator import InputCalculator

inputCalculator = InputCalculator()
inputCalculator.method1()
inputCalculator.update_astablishment_space()
inputCalculator.t_s.show_plot("theta", "theta_dot", [0, 0])

# while(True):
#     for i in range(1000):
#         inputCalculator.simulate()
#     inputCalculator.update_astablishment_space()
#     inputCalculator.t_s.show_plot("theta", "theta_dot", [0, 0])
#     print(inputCalculator._simulate_time)
