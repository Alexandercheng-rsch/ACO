import math
import numpy as np
import numba as nb


spec = [
    ('edge_coord', nb.float64[:,:]),  # 2D array of float64
    ('demand', nb.int64[:]),
    ('drivers', nb.int64),
    ('capacity', nb.int64),
    ('customers', nb.int64)
]
@nb.experimental.jitclass(spec)
class SweepAlgorithm:
    def __init__(self, edge_coord, demand, capacity, drivers):
        self.edge_coord = edge_coord
        self.demand = demand[1:]
        self.capacity = capacity
        self.drivers = drivers
        self.customers = demand.shape[0]

    def cartesian2polar(self, edge_coord):
        x_coord = self.edge_coord[:, 0]
        y_coord = self.edge_coord[:, 1]

        depot_x, depot_y = self.edge_coord[0, 0], self.edge_coord[0, 1]
        polar_coord = np.array([math.atan2(y_coord[i] - depot_y, x_coord[i] - depot_x) for i in range(1, len(x_coord))])

        return polar_coord

    def cluster(self):
        polar_coord = self.cartesian2polar(self.edge_coord)  #  convert to polar coordinates
        idx_coordinate = np.argsort(polar_coord)[::-1]  #  sorting

        new_demand = self.demand[idx_coordinate]
        path = np.full((self.drivers, self.customers + 3), dtype=np.float64, fill_value=np.nan)
        path[:, -2] = 0
        path[:, 0] = 0
        path[:, -1] = 0
        path[:, -3] = 0
        driver_idx = 0
        served = 0
        idx = 1

        while served <= self.customers - 2:
            if path[driver_idx, -2] + new_demand[served] <= self.capacity:
                path[driver_idx, idx] = idx_coordinate[served] + 1
                path[driver_idx, -2] += new_demand[served]
                path[driver_idx, -3] += 1

                served += 1
                idx += 1
            else:
                loc = path[driver_idx, -3]
                path[driver_idx, int(loc) + 1] = 0
                driver_idx += 1
                idx = 1

            loc = path[driver_idx, -3]
            path[driver_idx, int(loc) + 1] = 0

        return path
