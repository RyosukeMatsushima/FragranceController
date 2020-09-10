from unittest import TestCase
from src.TopologicalSpace import TopologicalSpace

topologicalSpace = TopologicalSpace()
# print(topologicalSpace.stochastic_matrix(1))
for lists in topologicalSpace.stochastic_matrix(1):
    print(lists)
coodinate = [1, 3]
topologicalSpace.write_val(coodinate, 1.2)
print(topologicalSpace.get_val(coodinate))