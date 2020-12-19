import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.TopologicalSpace import TopologicalSpace
from src.submodule.PhysicsSimulator.SinglePendulum.SinglePendulum import SinglePendulum

class Simulator:
    def __init__(self, t_s: TopologicalSpace, input_space):
        self.t_s = t_s
        self.input_space = input_space
        self.time_range = 20

    # start simulation
    def do(self):
        model = self.t_s.model

        time = 0.
        dt = 10**(-2)
        max_step = int(self.time_range * 10**(2) + 1)
        columns = ["time"] + [axis.name for axis in self.t_s.axes] + ["input"]

        df = pd.DataFrame(columns=columns)

        print("self.input_space")
        print(self.input_space.shape)
        print("pos")
        pos = tuple(self.t_s.coodinate2pos_TS(model.state))
        print(pos)
        print("input")
        print(self.input_space[pos])

        for s in range(0, max_step):
            time = s * dt
            if self.t_s.is_edge_of_TS(self.t_s.coodinate2pos_TS(model.state)):
                print("out of range")
                return df
            model.input = self.input_space[tuple(self.t_s.coodinate2pos_TS(model.state))]
            # singlePendulum.input = 4
            tmp_data = tuple([time]) + model.state + tuple([model.input])
            print(time)
            print(model.state)
            tmp_se = pd.Series(tmp_data, index=df.columns)
            df = df.append(tmp_se, ignore_index=True)
            model.step(dt)

        return df
