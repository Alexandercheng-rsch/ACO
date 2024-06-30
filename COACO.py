import threading

import numpy as np
import numba as nb
import random
from ACO.read_file import tsplib_distance_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

np.set_printoptions(precision=20, suppress=True, threshold=np.inf)
spec = [
    ('distances', nb.float64[:, :]),
    ('n_ants', nb.int64),
    ('n_colony', nb.int64),
    ('n_iterations', nb.int64),
    ('alpha', nb.float64),
    ('beta', nb.float64),
    ('rho', nb.float64),
    ('Q', nb.float64),
    ('initial_pheromone', nb.float64),
    ('pheromones', nb.float64[:, :, :]),
    ('n_cities', nb.int64),
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


# @nb.experimental.jitclass(spec)
class CoAntColony():
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

    def __init__(self, distances, n_ants, n_colony, n_iterations, alpha, beta, rho, Q, initial_pheromone):
        self.distances = distances
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.n_colony = n_colony
        self.initial_pheromone = initial_pheromone
        self.n_cities = distances.shape[0]  # assumption of symmetric distance matrix
        self.pheromones = np.full((n_colony, self.n_cities, self.n_cities), initial_pheromone)
        self.probability = np.zeros((1, self.n_cities))

        # precompute inverse distance for better efficiency
        self.inverse_distances = np.zeros_like(self.distances)
        mask = self.distances != 0
        self.inverse_distances[mask] = 1.0 / self.distances[mask]

    def gen_paths(self, colony_idx):
        paths = np.zeros((self.n_ants, self.n_cities + 2))  # extra entry for the distance column
        for i in range(0, self.n_ants):
            path = self.construct_path(colony_idx)
            paths[i, :-1] = path
            paths[i, -1] = self.path_dist(path)

        return paths

    def construct_path(self, colony_idx):

        path = np.zeros((1, self.n_cities + 1), dtype=np.int64)

        initial_point = random.randint(0, len(self.distances) - 1)

        unvisited = np.ones((1, len(self.distances)))
        unvisited[:, initial_point] = 0
        path[:, 0] = initial_point
        for i in range(1, self.n_cities):
            prev = path[:, i - 1]

            get_path = self.choose_node(unvisited, self.pheromones[colony_idx, prev, :],
                                        self.inverse_distances[prev, :])

            unvisited[:, get_path] = 0
            path[:, i] = get_path

        path[:, -1] = initial_point  # return back to the starting city

        return path

    def choose_node(self, unvisited, pheromones, inverse_distances):

        probability = (pheromones ** self.alpha) * (inverse_distances ** self.beta)
        probability = np.where(unvisited != 0, probability, 0)
        probability /= probability.sum()
        return rand_choice_nb(np.arange(len(unvisited[0])), probability[0])

    def update_pheromones(self, ants_paths, colony_idx):
        self.pheromones[colony_idx, :, :] *= (1 - self.rho)  # decay

        for i in range(0, ants_paths.shape[0]):
            for j in range(0, ants_paths.shape[1] - 2):
                path_current = int(ants_paths[i, j + 1])
                path_prev = int(ants_paths[i, j])

                self.pheromones[colony_idx, path_prev, path_current] += self.Q / ants_paths[i, -1]  # update pheromone

    def path_dist(self, path):
        total = 0
        for i in range(path.shape[1] - 1):
            a = path[0, i]
            b = path[0, i + 1]
            total += self.distances[a, b]

        return total

    def get_colony_fitness(self, colony_paths):
        fitness_list = [fit for fit in colony_paths[:, -1]]

        return sum(fitness_list) / len(fitness_list)

    def run(self):
        shortest = np.full((1, self.n_colony), np.inf)
        avg_colony_fitness = np.zeros((1, self.n_colony))

        for i in range(self.n_iterations):
            for j in range(self.n_colony):
                all_paths = self.gen_paths(j)
                self.update_pheromones(all_paths, j)
                current_shortest = np.argsort(all_paths[:, -1])[0]
                if all_paths[current_shortest, -1] < shortest[:, j]:
                    shortest[:, j] = all_paths[current_shortest, -1]
                    # print(all_paths[:, -1])
                avg_colony_fitness[:, j] = self.get_colony_fitness(all_paths)
            print('Average colony fitness: ', avg_colony_fitness)

        return avg_colony_fitness


tsplib_file = "ACO/att48.tsp"

distance_matrix = tsplib_distance_matrix(tsplib_file)
distance_matrix = np.array(distance_matrix, dtype=np.float64)

aco = CoAntColony(
    distances=distance_matrix,
    n_ants=20,
    n_iterations=5000,
    alpha=1,
    n_colony=10,
    beta=3,
    rho=0.1,
    Q=0.5,
    initial_pheromone=0.1
)

best_solution = aco.run()

