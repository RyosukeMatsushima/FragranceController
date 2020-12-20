from unittest import TestCase
import numpy as np

from src.Simulator import Simulator
from src.InputCalculator import InputCalculator
from src.TopologicalSpace import TopologicalSpace
from src.Axis import Axis

import os
import json

num = 48
path = "./input_space/input_space" + str(num)
os.chdir(path)

input_space = np.load("input_space.npy")
with open('param.json', 'r') as json_file:
    json_data = json.load(json_file)

axes_param_list = [list(param_dict.values()) for param_dict in json_data["axes"]]
print(axes_param_list)
axes = [Axis(*param) for param in axes_param_list]
print(axes)
t_s = TopologicalSpace(*axes)
t_s.model.state = (-t_s.model.state[0]-0.1, t_s.model.state[1])

t_s.show_concentration_img(axes[0], axes[1], input_space)

simulator = Simulator(t_s, input_space)
df = simulator.do()
simulator.t_s.show_concentration_img_with_tragectory(simulator.t_s.axes[0], simulator.t_s.axes[1], simulator.input_space, df["theta"], df["theta_dot"])
