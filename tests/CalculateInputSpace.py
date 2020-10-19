from unittest import TestCase
import numpy as np

from src.InputCalculator import InputCalculator
from src.TopologicalSpace import TopologicalSpace
from src.Axis import Axis

import os
import json

init_dir = os.getcwd()
num = 2
path = "./astablishment_space/astablishment_space" + str(num)
os.chdir(path)

astablishment_space = np.load("astablishment_space.npy")
with open('param.json', 'r') as json_file:
    json_data = json.load(json_file)

axes_param_list = [list(param_dict.values()) for param_dict in json_data["axes"]]
print(axes_param_list)
axes = [Axis(*param) for param in axes_param_list]
print(axes)
t_s = TopologicalSpace(*axes)
t_s.astablishment_space = astablishment_space

for i in range(1):
    max_val = 1.7 ** (- i)
    t_s.show_astablishment_space_in_range("theta", "theta_dot", [0, 0], max_val)

os.chdir(init_dir)
inputCalculator = InputCalculator(t_s)
inputCalculator.getInputSpace()