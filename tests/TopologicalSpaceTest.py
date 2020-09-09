from unittest import TestCase
from src.TopologicalSpace import TopologicalSpace

topologicalSpace = TopologicalSpace()
topologicalSpace.stochastic_matrix()
coodinate = [100, 13]
topologicalSpace.write_val(coodinate, 1.2)
print(topologicalSpace.get_val(coodinate))