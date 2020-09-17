import numpy as np
from src.TopologicalSpace import TopologicalSpace

class InputCalculator:
    def __init__(self):
        self.t_s = TopologicalSpace()

        self.target_coodinate = (0, 0)
        self.simulate_time = 0
        self.set_target_coodinate(self.target_coodinate)
    
    def set_target_coodinate(self, coodinate):
        self.t_s.write_val(coodinate, 1.0)

    def simulate(self, times):
        st_matrix = np.linalg.matrix_power(self.t_s.stochastic_matrix(0), times)
        self.t_s.astablishment_space = np.dot(st_matrix.T, self.t_s.astablishment_space.T) #TODO: verification of accounts
