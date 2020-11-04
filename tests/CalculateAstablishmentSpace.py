from unittest import TestCase
import numpy as np

from src.InputCalculator import InputCalculator
from src.TopologicalSpace import TopologicalSpace
from src.Axis import Axis

import os
import json

init_dir = os.getcwd()
num = 1
path = "./stochastic_matrix/stochastic_matrix" + str(num)
os.chdir(path)

stochastic_matrix = np.load("stochastic_matrix.npy")
with open('param.json', 'r') as json_file:
    json_data = json.load(json_file)
os.chdir("../../")

axes_param_list = [list(param_dict.values()) for param_dict in json_data["axes"]]
print(axes_param_list)
axes = [Axis(*param) for param in axes_param_list]
print(axes)
t_s = TopologicalSpace(*axes)
graph_center = ["theta", "theta_dot", [0, 0]]
inputCalculator = InputCalculator(t_s, (0, 0), graph_center)
inputCalculator.set_stochastic_matrix(stochastic_matrix)

# inputCalculator.method0()
inputCalculator.method2(0.024, 100)