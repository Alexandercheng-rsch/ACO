import numpy as np
import numba as nb
import random
from ACO.read_file import tsplib_distance_matrix
import time
from ACO.path_construction import gen_paths, twoopt, intraswap
from ACO.sweep import SweepAlgorithm

np.set_printoptions(precision=20, suppress=True, threshold=np.inf)
spec = [
    ('distances', nb.float64[:, :]),
    ('n_ants', nb.int64),
    ('n_iterations', nb.int64),
    ('alpha', nb.float64),
    ('beta', nb.float64),
    ('rho', nb.float64),
    ('Q', nb.float64),
    ('initial_pheromone', nb.float64),
    ('pheromones', nb.float64[:, :, :]),
    ('customers', nb.int64),
    ('demand', nb.int64[:]),
    ('drivers', nb.int64),
    ('capacity', nb.int64),
    ('colonies', nb.int64),
    ('exchange_rate', nb.int64)

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


@nb.njit
def rand_choice_nb_multiple(arr, num_samples):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param num_samples: Number of samples to choose.
    :return: An array of random samples chosen without replacement.
    """
    result = np.empty(num_samples, dtype=arr.dtype)
    temp_arr = arr.copy()
    for i in range(num_samples):
        idx = np.random.randint(0, len(temp_arr))
        result[i] = temp_arr[idx]
        temp_arr[idx] = temp_arr[-1]
        temp_arr = temp_arr[:-1]
    return result



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
                 demand,
                 drivers,
                 capacity,
                 n_ants,
                 n_iterations,
                 alpha,
                 beta,
                 rho,
                 Q,
                 colonies,
                 exchange_rate,
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
        self.demand = demand.astype(np.int64)
        self.drivers = drivers
        self.capacity = capacity
        self.colonies = colonies
        self.exchange_rate = exchange_rate
        self.pheromones = np.full((self.customers, self.customers, self.colonies + 1), initial_pheromone)
    def update_pheromones(self, ants_paths, no_colonie):
        if ants_paths.ndim == 2:
            ants_paths = ants_paths[:, :, np.newaxis]

        self.pheromones *= (1 - self.rho)  # decay
        for k in range(0, ants_paths.shape[2]):
            for i in range(0, ants_paths.shape[0]):
                for j in range(0, int(ants_paths[i, -3, k])):
                    path_current = int(ants_paths[i, j + 1, k])
                    path_prev = int(ants_paths[i, j, k])
                    self.pheromones[path_prev, path_current, no_colonie] += self.Q / ants_paths[i, -1, k]  # update
                    self.pheromones[path_current, path_prev, no_colonie] += self.Q / ants_paths[i, -1, k]
                    # pheromone

    def update_all_colonies(self, paths, same):
        if same:
            for i in range(self.colonies - 1):
                self.update_pheromones(paths, i)
        else:
            for i in range(self.colonies - 1):
                path = paths[:, :, i]
                self.update_pheromones(path, i)
    def get_colony_best_fitness(self, paths):
        colony_best_idx = np.zeros((1, self.colonies + 1))
        colony_best_fitness = np.zeros((1, self.colonies + 1))
        best_colony_route = np.zeros((self.drivers, self.customers + 3, self.colonies + 1), dtype=np.float64)

        for colony in range(self.colonies + 1):
            summed = np.sum(paths[:, -1, :, colony], axis=0)
            smallest = np.argsort(summed)[0]
            colony_best_idx[:, colony] = smallest
            best_colony_route[:, :, colony] = paths[:, :, smallest, colony]
            colony_best_fitness[:, colony] = summed[smallest]

        return colony_best_fitness, best_colony_route
    def exchange_information(self, best_paths):
        res = [[i, i + 1 % self.colonies]
               for i in range(self.colonies-1)]
        for i in res:
            self.update_pheromones(best_paths[:, :, i[1]], i[0])
            self.update_pheromones(best_paths[:, :, i[0]], i[1])

    def update_master_colony(self, colony_fitness):
        total = np.sum(colony_fitness[:, :-1])
        self.pheromones[:, :, -1] = 0

        for col in range(self.colonies):
            self.pheromones[:, :, -1] += np.clip((self.pheromones[:, :, col] * colony_fitness[:, col])/total, 0, 1)
        min_ = np.min(self.pheromones[:, :, -1])
        max_ = np.max(self.pheromones[:, :, -1])
        self.pheromones[:, :, -1] = (self.pheromones[:, :, -1] - min_) / (max_ - min_)



    def run(self):
        shortest = np.full((1, self.colonies + 1), np.inf)
        shortest_route = None
        # for i in range(10):
        #     sweep = twoopt(sweep, self.distances)
        #     sweep = intraswap(sweep, self.demand, self.capacity, self.drivers, self.distances, False)
        # self.update_all_colonies(sweep[:,:,:,0], True)
        print('Started')
        for i in range(self.n_iterations):
            print(i)

            if i > 2:
                self.update_master_colony(shortest)
            all_paths = gen_paths(self.drivers, self.customers, self.pheromones, self.alpha, self.beta, self.rho,
                                  self.n_ants, self.demand, self.capacity, self.distances, self.colonies + 1,False)
            colony_fitness, colony_routes = self.get_colony_best_fitness(all_paths)
            if i % self.exchange_rate == 0:
                self.exchange_information(colony_routes)
            else:
                self.update_all_colonies(colony_routes, False)
            for i in range(self.colonies + 1):
                if colony_fitness[:, i].item() < shortest[:, i].item():
                    shortest[:, i] = colony_fitness[:, i]
                    print(shortest)

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
    demand=demand,
    capacity=capacity,
    drivers=10,
    n_ants=20,
    n_iterations=10000,
    alpha=2,
    beta=3,
    rho=0.8,
    Q=1,
    colonies=2,
    exchange_rate=50,
    initial_pheromone=1.0
)

best_solution = aco.run()
