from unittest import TestCase
import numpy as np

from src.InputCalculator import InputCalculator
from src.TopologicalSpace import TopologicalSpace
from src.Axis import Axis
from src.submodule.PhysicsSimulator.SinglePendulum.SinglePendulum import SinglePendulum

import os
import json

init_dir = os.getcwd()
num = 1
path = "./astablishment_space/SinglePendulum/astablishment_space" + str(num)
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
model = SinglePendulum(0, 0, **json_data["model_param"])
graph_center = ["theta", "theta_dot", [0, 0]]
d = 1.
u_set = np.arange(-2., 2. + d, d).tolist()
moderate_u = 0
inputCalculator = InputCalculator(t_s, (0, 0), graph_center, u_set, moderate_u, model, json_data["delta_t"])
inputCalculator.getInputSpace(True)