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
        self.last_direction_list = np.array([0])
        self.continuous_time_list = np.array([0.0])

        self.last_plot_time = 0.
        self.max_simulation_time = 10

    def method3(self):
        while(len(self.time_list) > 0):
            element_num = np.argmin(self.time_list, axis=0)
            self._simulate_time = self.time_list[element_num]
            pos_TS = self.pos_list[element_num]
            last_direction = self.last_direction_list[element_num]
            continuous_time = self.continuous_time_list[element_num]
            self.delete_element(pos_TS)

            if self.t_s.is_edge_of_TS(pos_TS):
                continue
            if self.write_time(pos_TS, self._simulate_time):
                self.next_elements(pos_TS, last_direction, continuous_time)

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
    
    def add_element(self, time, pos_TS, last_direction, continuous_time):
        # if self.t_s.astablishment_space[self.t_s.pos_TS2pos_AS(pos_TS)] != 0:
        #     return
        self.time_list = np.append(self.time_list, time)
        self.pos_list = np.append(self.pos_list, [pos_TS], axis=0)
        self.last_direction_list = np.append(self.last_direction_list, last_direction)
        self.continuous_time_list = np.append(self.continuous_time_list, continuous_time)
    
    def delete_element(self, pos_TS):
        num = np.where((self.pos_list == pos_TS).all(axis=1))
        self.time_list = np.delete(self.time_list, num, 0)
        self.pos_list = np.delete(self.pos_list, num, 0)
        self.last_direction_list = np.delete(self.last_direction_list, num, 0)
        self.continuous_time_list = np.delete(self.continuous_time_list, num, 0)

    def next_elements(self, pos_TS, last_direction, continuous_time):
        coodinate = self.t_s.pos_TS2coodinate(pos_TS)
        print("coodinate")
        print(coodinate)
        print("self._simulate_time")
        print(self._simulate_time)
        vel_list = np.array([-self.t_s.model.dynamics(*coodinate, u) for u in self.u_set])
        # TODO: select important vel
        for vel in vel_list:
            self.set_element(pos_TS, vel, last_direction, continuous_time)

    def set_element(self, pos_TS, vel, last_direction, continuous_time):
        steps = [self.t_s.axes[i].get_step(pos) for i, pos in enumerate(pos_TS)]
        is_stop = False
        for i, step in enumerate(steps):
            if last_direction == i:
                continue
            if step < continuous_time * abs(vel[i]):
                is_stop = True

        for i, v in enumerate(vel):
            if i == last_direction and is_stop:
                continue
            if v == 0:
                continue
            pos = copy(pos_TS)
            step = steps[i]    #TODO: applay variable step
            if v > 0:
                pos[i] += 1
            else:
                pos[i] -= 1

            time2next = abs(step/v)
            time = time2next
            if i == last_direction:
                time = continuous_time + time2next

            self.add_element(self._simulate_time + time2next, pos, i, time)
