import numpy as np
import numba as nb
import random
from ACO.read_file import tsplib_distance_matrix
from ACO.sweep import SweepAlgorithm
import time

np.set_printoptions(precision=20, suppress=True, threshold=np.inf)
spec = [
    ('distances', nb.float64[:,:]),
    ('n_ants', nb.int64),
    ('n_iterations', nb.int64),
    ('alpha', nb.float64),
    ('beta', nb.float64),
    ('rho', nb.float64),
    ('Q', nb.float64),
    ('initial_pheromone', nb.float64),
    ('pheromones', nb.float64[:,:]),
    ('customers', nb.int64),
    ('demand', nb.int64[:]),
    ('drivers', nb.int64),
    ('capacity', nb.int64),
    ('edge_coord', nb.float64[:,:]),
    ('sweep', nb.float64[:,:]),
    ('tau_max', nb.int64)
]

@nb.njit  # credit to some reddit user
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    a = np.searchsorted(np.cumsum(prob), np.random.random(), side="right")
    return arr[a]


@nb.experimental.jitclass(spec)
class AntColony:
    """
    :param n_ants: Number of ants.
    :param alpha: Alpha value.
    :param beta: Beta value.
    :param rho: Rho value.
    :param Q: Q value.
    :param initial_pheromone: Initial pheromone value.
    :param distances: Distance matrix.
    :return shortest, shortest_route: Shortest path from start to end and distance.
    """
    def __init__(self,
                 distances,
                 edge_coord,
                 demand,
                 drivers,
                 capacity,
                 n_ants,
                 n_iterations,
                 alpha,
                 beta,
                 rho,
                 Q,
                 initial_pheromone):
        self.distances = distances.astype(np.float64)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.initial_pheromone = initial_pheromone
        self.customers = distances.shape[0]  # assumption of symmetric distance matrix
        self.pheromones = np.full((self.customers, self.customers), initial_pheromone)
        self.demand = demand.astype(np.int64)
        self.drivers = drivers
        self.capacity = capacity
        self.edge_coord = edge_coord
        self.sweep = SweepAlgorithm(self.edge_coord, self.demand, self.capacity, self.drivers).cluster()
        self.tau_max = 1

    def gen_paths(self):
        paths = paths = np.zeros((self.drivers, self.customers + 3, self.n_ants), dtype=np.float64)  # extra entries for the distance column and capacity
        for i in range(0, self.n_ants):
            path = self.construct_path()

            paths[:, :, i] = path

        return paths

    def construct_path(self):
        sweep = self.sweep
        path = np.full((self.drivers, self.customers + 3), np.nan)
        initial_point = 0
        unvisited = np.ones(len(self.distances), dtype=np.bool_)
        unvisited[initial_point] = False


        for d in range(self.drivers):
            path[d, -3] = 1  # route length
            path[d, -2] = 0  # current capacity
            path[d, -1] = 0  # total distance
            path[d, 0] = initial_point

        while np.any(unvisited):

            for driver_idx in range(self.drivers):
                driver_idx = np.random.randint(0, self.drivers)
                route_len = int(path[driver_idx, -3])
                prev = int(path[driver_idx, route_len - 1])

                feasibility1 = np.array([(path[driver_idx, -2] + self.demand[idx]) <= self.capacity
                                         for idx in range(len(self.distances))])
                feasibility2 = np.zeros(len(self.distances), dtype=np.bool_)

                sweep_indices = np.zeros(3, dtype=np.int64)
                if 0 < driver_idx < self.drivers - 1:
                    sweep_indices = np.array([driver_idx - 1, driver_idx, driver_idx + 1])
                elif driver_idx == 0:
                    sweep_indices = np.array([driver_idx, driver_idx + 1, self.drivers - 1])
                else:  # driver_idx == self.drivers - 1
                    sweep_indices = np.array([driver_idx - 1, 0, driver_idx])

                # Shuffle all indices
                for i in range(len(sweep_indices)):
                    j = np.random.randint(i, len(sweep_indices))
                    sweep_indices[i], sweep_indices[j] = sweep_indices[j], sweep_indices[i]

                for sweep_idx in sweep_indices:
                    for j in range(1, int(sweep[sweep_idx, -3] + 1)):
                        city = int(sweep[sweep_idx, j])
                        if city != 0:
                            feasibility2[city] = True

                cluster = np.where(np.logical_and(feasibility1, feasibility2))[0]
                cluster = cluster[cluster != 0]

                if len(cluster) == 0:
                    continue

                choose_city = self.choose_node(unvisited, self.pheromones[prev],
                                               self.distances[prev], feasibility1)

                if choose_city == -1:
                    if np.any(unvisited):
                        continue
                    break

                path[driver_idx, route_len] = choose_city
                path[driver_idx, -2] += self.demand[choose_city]
                path[driver_idx, -3] += 1
                unvisited[choose_city] = False
                path[driver_idx, -1] += self.distances[prev, choose_city]


        for i in range(self.drivers):
            idx = int(path[i, -3])
            path[i, idx] = initial_point
            path[i, -1] += self.distances[int(path[i, idx - 1]), initial_point]



        return path
    def choose_node(self, unvisited, pheromones, distance, feasibility1):
        feasible_unvisited = np.logical_and(unvisited, feasibility1)

        if not np.any(feasible_unvisited):
            return -1

        probabilities = np.zeros_like(pheromones)
        feasible_indices = np.where(feasible_unvisited)[0]

        for i in feasible_indices:
            probabilities[i] = pheromones[i] ** self.alpha * (1.0 / distance[i]) ** self.beta

        total = np.sum(probabilities)
        if total == 0:
            return -1

        probabilities /= total

        r = np.random.random()
        cum_prob = 0
        for i in feasible_indices:
            cum_prob += probabilities[i]
            if r <= cum_prob:
                return i

        return feasible_indices[-1]

    def update_pheromones(self, ants_paths):

        self.pheromones *= (1 - self.rho)  # decay
        for k in range(0, ants_paths.shape[2]):
            for i in range(0, ants_paths.shape[0]):
                for j in range(0, int(ants_paths[i, -3, k] - 1) + 1):

                    path_current = int(ants_paths[i, j + 1, k])
                    path_prev = int(ants_paths[i, j, k])
                    self.pheromones[path_prev, path_current] +=  np.exp(self.pheromones[path_prev, path_current] / self.tau_max)/ self.path_dist([path_prev, path_current])  # update pheromone




    def path_dist(self, path):
        total = 0.0
        for i in range(len(path) - 1):
            a = int(path[i])
            b = int(path[i + 1])
            total += self.distances[a, b]

        return total

    def run(self):

        shortest = np.inf
        shortest_route = None

        for i in range(self.n_iterations):
            all_paths = self.gen_paths()
            summed = np.sum(all_paths[:, -1, :], axis=0)
            idx = np.argsort(summed)[:2]

            self.update_pheromones(all_paths[:, :, idx])
            idxx = idx[0]
            if summed[idxx] < shortest:
                shortest = summed[idxx]
                shortest_route = all_paths[:, :, idxx]
                print(f'Current shortest path:', shortest_route)
                print(f'Current shortest distance:', shortest)
                self.tau_max = 1/(shortest * self.rho)
                print(i)

        return shortest, shortest_route


tsplib_file = "ACO/att48.tsp"

import vrplib
file = 'ACO/A-n32-k5.vrp'
re = vrplib.read_instance(file)
demand = re['demand']
distance_matrix = re['edge_weight']
capacity = re['capacity']
edge_coord = np.array(re['node_coord'], dtype=np.float64)

aco = AntColony(
    distances=distance_matrix,
    edge_coord=edge_coord,
    demand=demand,
    capacity=capacity,
    drivers=5,
    n_ants=200,
    n_iterations=10000,
    alpha=1,
    beta=3,
    rho=0.01,
    Q=1,
    initial_pheromone=1/32
)

best_solution = aco.run()


#
# sweep = SweepAlgorithm(re['node_coord'], demand)
# sweep.cluster()
#
