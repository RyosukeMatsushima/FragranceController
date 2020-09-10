from unittest import TestCase
from src.TopologicalSpace import TopologicalSpace
import numpy as np

topologicalSpace = TopologicalSpace()
# print(topologicalSpace.stochastic_matrix(1))
stochastic_matrix = topologicalSpace.stochastic_matrix(0)
for lists in stochastic_matrix:
    print(lists)

stochastic_matrix_5 = np.linalg.matrix_power(stochastic_matrix, 5)
for lists in stochastic_matrix_5:
    print(lists)
coodinate = [1, 3]
topologicalSpace.write_val(coodinate, 1.2)
print(topologicalSpace.get_val(coodinate))