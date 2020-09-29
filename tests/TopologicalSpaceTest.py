from unittest import TestCase
from src.TopologicalSpace import TopologicalSpace
from src.Axis import Axis
import numpy as np

axes = (Axis("theta", -1.0, 7.0, 0.1), Axis("theta_dot", -7.0, 7.0, 0.1))
topologicalSpace = TopologicalSpace(*axes)
print(topologicalSpace.astablishment_space)
stochastic_matrix = topologicalSpace.stochastic_matrix(0)
# for lists in stochastic_matrix:
#     print(lists)

print("stochastic_matrix_5")

stochastic_matrix_5 = np.linalg.matrix_power(stochastic_matrix, 5)
for lists in stochastic_matrix_5:
    print(lists)
coodinate = [1, 3]
topologicalSpace.write_val(coodinate, 1.2)
print(topologicalSpace.get_val(coodinate))