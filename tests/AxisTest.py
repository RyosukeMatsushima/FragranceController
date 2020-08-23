from unittest import TestCase
from src.Axis import Axis

axis = Axis("test", 0.0, 10.2, 0.3)
print(axis.name)
print(axis.elements)
print(axis.element_count)
