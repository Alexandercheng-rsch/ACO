# import math
# import numpy as np
# import numba as nb
# import time
#
# spec = [
#     ('edge_coord', nb.float64[:,:]),  # 2D array of float64
#     ('demand', nb.int64[:]),
#     ('drivers', nb.int64),
#     ('capacity', nb.int64),
#     ('customers', nb.int64),
#     ('distance', nb.float64[:, :]),
# ]
# @nb.experimental.jitclass(spec)
# class SweepAlgorithm:
#     def __init__(self, edge_coord, demand, capacity, drivers, distance):
#         self.edge_coord = edge_coord
#         self.demand = demand[1:]
#         self.capacity = capacity
#         self.drivers = drivers
#         self.customers = demand.shape[0]
#         self.distance = distance.astype(np.float64)
#
#
#     def cartesian2polar(self, edge_coord):
#         x_coord = self.edge_coord[:, 0]
#         y_coord = self.edge_coord[:, 1]
#
#         depot_x, depot_y = self.edge_coord[0, 0], self.edge_coord[0, 1]
#         polar_coord = np.array([math.atan2(y_coord[i] - depot_y, x_coord[i] - depot_x) for i in range(1, len(x_coord))])
#
#         return polar_coord
#
#     def path_dist(self, path):
#         total = 0.0
#         for i in range(len(path) - 1):
#             a = int(path[i])
#             b = int(path[i + 1])
#             total += self.distance[a, b]
#         return total
#
#     def cluster(self):
#         polar_coord = self.cartesian2polar(self.edge_coord)  #  convert to polar coordinates
#         idx_coordinate = np.argsort(polar_coord)[::-1]  #  sorting
#
#         new_demand = self.demand[idx_coordinate]
#         path = np.full((self.drivers, self.customers + 3, 1), dtype=np.float64, fill_value=np.nan)
#         path[:, -2, 0] = 0
#         path[:, 0, 0] = 0
#         path[:, -1, 0] = 0
#         path[:, -3, 0] = 0
#         driver_idx = 0
#         served = 0
#         idx = 1
#
#         while served <= self.customers - 2:
#             if path[driver_idx, -2, 0] + new_demand[served] <= self.capacity:
#                 path[driver_idx, idx, 0] = idx_coordinate[served] + 1
#                 path[driver_idx, -2, 0] += new_demand[served]
#                 path[driver_idx, -3, 0] += 1
#                 prev = path[driver_idx, idx - 1, 0]
#                 cur = path[driver_idx, idx, 0]
#
#                 path[driver_idx, -1, 0] += self.path_dist([prev, cur])
#
#                 served += 1
#                 idx += 1
#             else:
#                 loc = path[driver_idx, -3, 0]
#                 path[driver_idx, int(loc) + 1, 0] = 0
#                 path[driver_idx, -1, 0] += self.path_dist([cur, 0])
#                 driver_idx += 1
#                 idx = 1
#
#             loc = path[driver_idx, -3, 0]
#             path[driver_idx, int(loc) + 1, 0] = 0
#
#         return path
#


import math
import numpy as np
import numba as nb
import time

spec = [
    ('edge_coord', nb.float64[:, :]),  # 2D array of float64
    ('demand', nb.int64[:]),
    ('drivers', nb.int64),
    ('capacity', nb.int64),
    ('customers', nb.int64),
    ('distance', nb.float64[:, :]),
]


@nb.experimental.jitclass(spec)
class SweepAlgorithm:
    def __init__(self, edge_coord, demand, capacity, drivers, distance):
        self.edge_coord = edge_coord
        self.demand = demand[1:]
        self.capacity = capacity
        self.drivers = drivers
        self.customers = demand.shape[0]
        self.distance = distance.astype(np.float64)

    def get_centre(self):
        x_coord = self.edge_coord[:, 0]
        y_coord = self.edge_coord[:, 1]

        x_max = max(x_coord)
        x_min = min(x_coord)

        y_max = max(y_coord)
        y_min = min(y_coord)
        x_centre = (x_max + x_min) / 2
        y_centre = (y_max + y_min) / 2

        return x_centre, y_centre

    def cartesian2polar(self, edge_coord):
        x_coord = self.edge_coord[:, 0]
        y_coord = self.edge_coord[:, 1]
        centre_x, centre_y = self.get_centre()

        polar_coord = np.array(
            [math.atan2(y_coord[i] - centre_y, x_coord[i] - centre_x) for i in range(1, len(x_coord))])

        return polar_coord

    def path_dist(self, path):
        total = 0.0
        for i in range(len(path) - 1):
            a = int(path[i])
            b = int(path[i + 1])
            total += self.distance[a, b]
        return total

    def cluster(self):
        polar_coord = self.cartesian2polar(self.edge_coord)  #  convert to polar coordinates
        idx_coordinate = np.argsort(polar_coord)[::-1]  #  sorting

        new_demand = self.demand[idx_coordinate]
        path = np.full((self.drivers, self.customers + 3, 1), dtype=np.float64, fill_value=np.nan)
        path[:, -2, 0] = 0
        path[:, 0, 0] = 0
        path[:, -1, 0] = 0
        path[:, -3, 0] = 0
        driver_idx = 0
        served = 0
        idx = 1

        while served <= self.customers - 2:
            if path[driver_idx, -2, 0] + new_demand[served] <= self.capacity:
                path[driver_idx, idx, 0] = idx_coordinate[served] + 1
                path[driver_idx, -2, 0] += new_demand[served]
                path[driver_idx, -3, 0] += 1
                prev = path[driver_idx, idx - 1, 0]
                cur = path[driver_idx, idx, 0]

                path[driver_idx, -1, 0] += self.path_dist([prev, cur])

                served += 1
                idx += 1
            else:
                loc = path[driver_idx, -3, 0]
                path[driver_idx, int(loc) + 1, 0] = 0
                path[driver_idx, -1, 0] += self.path_dist([cur, 0])
                driver_idx += 1
                idx = 1

            loc = path[driver_idx, -3, 0]
            path[driver_idx, int(loc) + 1, 0] = 0
        path = path[:, :, :, np.newaxis]
        return path
