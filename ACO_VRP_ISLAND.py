import numba
import numpy as np
from numba import njit
import random
from ACO.read_file import tsplib_distance_matrix
import time
from ACO.path_construction import gen_paths, twoopt, intraswap
from numba import prange

np.set_printoptions(precision=20, suppress=True, threshold=np.inf)


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
                 gamma,
                 Q,
                 colonies,
                 exchange_rate,
                 master_colony_update_rate,
                 initial_pheromone,
                 shuffle_it):
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
        self.gamma = gamma
        self.exchange_rate = exchange_rate
        self.master_colony_update_rate = master_colony_update_rate
        self.pheromones = np.full((self.customers, self.customers, self.colonies + 1), initial_pheromone)
        self.shuffle_it = shuffle_it
        self.exchanging = False

    def update_pheromones(self, ants_paths, no_colonie):
        self.pheromones[:, :, no_colonie] *= (1 - self.rho)

        if ants_paths.ndim == 2:
            ants_paths = ants_paths[:, :, np.newaxis]

        for k in range(0, ants_paths.shape[2]):
            for i in prange(0, ants_paths.shape[0]):
                for j in range(0, int(ants_paths[i, -3, k])):
                    path_current = int(ants_paths[i, j + 1, k])
                    path_prev = int(ants_paths[i, j, k])
                    self.pheromones[path_prev, path_current, no_colonie] += self.Q / ants_paths[i, -1, k]  # update
    def decay_pheromones(self, no_colonie):
        self.pheromones[:, :, no_colonie] *= (1 - self.rho)


    def update_all_colonies(self, paths):
        for i in range(self.colonies + 1):
            # self.decay_pheromones(i)
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
               for i in range(self.colonies)]
        res.append([self.colonies, 0])
        for i in res:
            self.update_pheromones(best_paths[:, :, i[1]], i[0])
            self.update_pheromones(best_paths[:, :, i[0]], i[1])

    def update_master_colony(self, colony_fitness):
        total = np.sum(colony_fitness[:, :-1])
        self.pheromones[:, :, -1] = 0  # Start from zero
        for col in range(self.colonies):
            self.pheromones[:, :, -1] += (self.pheromones[:, :, col] * ((1 / (colony_fitness[:, col])) ** self.gamma) /
                                          total)

        min_val = np.min(self.pheromones[:, :, -1])
        max_val = np.max(self.pheromones[:, :, -1])
        self.pheromones[:, :, -1] = (self.pheromones[:, :, -1] - min_val) / (max_val - min_val)

    def shuffle(self):
        indices_to_shuffle = np.arange(self.colonies)
        shuffled_indices = np.random.permutation(indices_to_shuffle)
        temp = np.copy(self.pheromones)
        self.pheromones[:, :, indices_to_shuffle] = temp[:, :, shuffled_indices]
        return shuffled_indices

    def run(self):

        shortest = np.full((1, self.colonies + 1), np.inf)
        shortest_route = np.zeros((self.drivers, self.customers + 3, self.colonies + 1), dtype=np.float64)
        limit = 500
        rho = self.rho
        print('Started')
        for i in range(self.n_iterations):
            print(i)
            if i > 1:
                if i % self.master_colony_update_rate == 0 or i < 40:
                    self.update_master_colony(shortest)

            all_paths = gen_paths(self.drivers, self.customers, self.pheromones, self.alpha, self.beta, self.rho,
                                  self.n_ants, self.demand, self.capacity, self.distances, self.colonies + 1, limit,
                                  True)

            colony_fitness, colony_routes = self.get_colony_best_fitness(all_paths)

            for j in range(self.colonies + 1):
                if colony_fitness[:, j].item() < shortest[:, j].item():
                    shortest[:, j] = colony_fitness[:, j]
                    shortest_route[:, :, j] = colony_routes[:, :, j]
                    print(shortest)
            if i % (self.exchanging - 15) == 0:
                self.rho = 0.01
            if i % self.exchange_rate == 0 and i > 1:
                self.exchanging = True
                self.exchange_information(shortest_route)
                self.exchanging = False
                self.rho = rho
            else:
                self.update_all_colonies(colony_routes)
            if i % self.shuffle_it == 0:
                shuffled_idx = self.shuffle()
                shortest[0, :self.colonies] = shortest[0, shuffled_idx]
        shortest_idx = np.argsort(shortest)
        shortest_route = shortest_route[:, :, shortest_idx]
        print(shortest_route)
        return shortest, shortest_route


tsplib_file = "ACO/att48.tsp"

import vrplib

file = 'ACO/A-n80-k10.vrp'
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
    n_ants=150,
    n_iterations=5000,
    alpha=2,
    beta=3,
    gamma=1.35,
    rho=0.05,
    Q=1,
    colonies=8,
    exchange_rate=20,
    master_colony_update_rate=150,
    initial_pheromone=1.0,
    shuffle_it=2500
)

best_solution = aco.run()