from unittest import TestCase
from src.InputCalculator import InputCalculator

inputCalculator = InputCalculator()
inputCalculator.t_s.show_plot("theta", "theta_dot", [0, 0])
inputCalculator.simulate(50)
inputCalculator.t_s.show_plot("theta", "theta_dot", [0, 0])