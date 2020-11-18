import numpy as np
from copy import copy

from src.InputCalculator import InputCalculator
from src.TopologicalSpace import TopologicalSpace

class TimeToPosition(InputCalculator):
    def __init__(self, t_s: TopologicalSpace, target_coodinate, graph_arg, u_set, moderate_u):
        self.t_s = t_s
        self.graph_arg = graph_arg

        self.u_set = u_set
        self.moderate_u = moderate_u

        self.target_coodinate = target_coodinate
        self._simulate_time = 0.

        self.time_list = np.array([0.0])
        self.pos_list = np.array([self.t_s.coodinate2pos_TS(self.target_coodinate)])
        self.last_plot_time = 0.

        self.max_simulation_time = 10

    def method3(self):
        while(len(self.time_list) > 0):
            element_num = np.argmin(self.time_list, axis=0)
            self._simulate_time = self.time_list[element_num]
            pos_TS = self.pos_list[element_num]
            self.delete_element(pos_TS)

            if self.t_s.is_edge_of_TS(pos_TS):
                continue
            if self.write_time(pos_TS, self._simulate_time):
                self.next_elements(pos_TS)

            if (self._simulate_time - self.last_plot_time) > 1:
                self.t_s.show_astablishment_space(*self.graph_arg)
                self.last_plot_time = self._simulate_time

            if self._simulate_time > self.max_simulation_time:
                break
        self.t_s.show_astablishment_space(*self.graph_arg)
        self.save_astablishment_space(self.t_s.astablishment_space)

    def write_time(self, pos_TS, time) -> bool:
        pos_AS = self.t_s.pos_TS2pos_AS(pos_TS)
        if self.t_s.astablishment_space[pos_AS] != 0:
            return False
        self.t_s.astablishment_space[pos_AS] = time
        return True
    
    def add_element(self, pos_TS, time):
        self.time_list = np.append(self.time_list, time)
        self.pos_list = np.append(self.pos_list, [pos_TS], axis=0)
    
    def delete_element(self, pos_TS):
        num = np.where((self.pos_list == pos_TS).all(axis=1))
        self.time_list = np.delete(self.time_list, num, 0)
        self.pos_list = np.delete(self.pos_list, num, 0)

    def next_elements(self, pos_TS):
        coodinate = self.t_s.pos_TS2coodinate(pos_TS)
        print("coodinate")
        print(coodinate)
        print("self._simulate_time")
        print(self._simulate_time)
        vel_list = np.array([-self.t_s.model.dynamics(*coodinate, u) for u in self.u_set])
        max_vel = np.max(vel_list, axis=0)
        min_vel = np.min(vel_list, axis=0)
        max_vel[np.where(max_vel < 0)] = 0
        min_vel[np.where(min_vel > 0)] = 0
        self.set_element(pos_TS, max_vel)
        self.set_element(pos_TS, min_vel)

    def set_element(self, pos_TS, vel):
        for i, v in enumerate(vel):
            if v == 0:
                continue
            pos = copy(pos_TS)
            step = self.t_s.axes[i].get_step(pos[i])    #TODO: applay variable step
            if v > 0:
                pos[i] += 1
            else:
                pos[i] -= 1
            self.add_element(pos, self._simulate_time + abs(step/v))
