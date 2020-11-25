from unittest import TestCase
import numpy as np

from src.FuildSimulator import FuildSimulator
from src.TopologicalSpace import TopologicalSpace
from src.Axis import Axis

import os
import json

init_dir = os.getcwd()
num = 14
path = "./stochastic_matrix/stochastic_matrix" + str(num)
os.chdir(path)

with open('param.json', 'r') as json_file:
    json_data = json.load(json_file)
os.chdir("../../")

axes_param_list = [list(param_dict.values()) for param_dict in json_data["axes"]]
print(axes_param_list)
axes = [Axis(*param) for param in axes_param_list]
print(axes)
t_s = TopologicalSpace(*axes)
graph_center = ["theta", "theta_dot", [0, 0]]
u_set = json_data["u_set"]
moderate_u = json_data["moderate_u"]
inputCalculator = FuildSimulator(t_s, (0, 0), graph_center, u_set, moderate_u)
stochastic_matrix, gather_matrix = inputCalculator.load_stochastic_matrix(num)
inputCalculator.set_stochastic_matrix(stochastic_matrix, gather_matrix)

# inputCalculator.init_eye()
# inputCalculator.method1()
inputCalculator.method2(0.024, 100)