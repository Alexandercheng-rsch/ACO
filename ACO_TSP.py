import numpy as np
import numba as nb
import random
from ACO_TSP.read_file import tsplib_distance_matrix
import time

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
    ('pheromones', nb.float64[:, :]),
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
    def __init__(self, distances, n_ants, n_iterations, alpha, beta, rho, Q, initial_pheromone):
        self.distances = distances
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.initial_pheromone = initial_pheromone
        self.n_cities = distances.shape[0]  # assumption of symmetric distance matrix
        self.pheromones = np.full((self.n_cities, self.n_cities), initial_pheromone)

    def gen_paths(self):
        paths = np.zeros((self.n_ants, self.n_cities + 2))  # extra entry for the distance column
        for i in range(0, self.n_ants):
            path = self.construct_path()
            paths[i, :-1] = path
            paths[i, -1] = self.path_dist(path)

        return paths

    def construct_path(self):
        path = np.zeros((1, self.n_cities + 1), dtype=np.int64)
        initial_point = random.randint(0, len(self.distances) - 1)
        unvisited = np.ones((1, len(self.distances)))
        unvisited[:, initial_point] = 0
        path[:, 0] = initial_point
        for i in range(1, len(self.distances)):
            prev = path[:, i - 1]

            get_path = self.choose_node(unvisited, self.pheromones[prev], self.distances[prev, :])
            unvisited[:, get_path] = 0
            path[:, i] = get_path

        path[:, -1] = initial_point  #return back to the starting city
        return path

    def choose_node(self, unvisited, pheromones, distance):

        probability = np.where((distance != 0) & (unvisited != 0),
                               (pheromones ** self.alpha) * ((1 / distance) ** self.beta),
                               0)

        probability /= probability.sum()
        return rand_choice_nb(np.arange(len(unvisited[0])), probability[0])

    def update_pheromones(self, ants_paths):
        self.pheromones *= (1 - self.rho)  # decay
        for i in range(0, ants_paths.shape[0]):
            for j in range(0, ants_paths.shape[1] - 2):
                path_current = int(ants_paths[i, j + 1])
                path_prev = int(ants_paths[i, j])

                self.pheromones[path_prev, path_current] += self.Q / ants_paths[i, -1]  # update pheremone

    def path_dist(self, path):
        total = 0
        for i in range(path.shape[1] - 1):
            a = path[0, i]
            b = path[0, i + 1]
            total += self.distances[a, b]

        return total

    def run(self):
        shortest = np.inf
        shortest_route = None

        for i in range(self.n_iterations):
            all_paths = self.gen_paths()
            self.update_pheromones(all_paths)

            current_shortest = np.argsort(all_paths[:, -1])[0]

            if all_paths[current_shortest, -1] < shortest:
                shortest = all_paths[current_shortest, -1]
                shortest_route = all_paths[current_shortest, :]
                print(f'Current shortest path:', shortest_route)
                print(f'Current shortest distance:', shortest)

        return shortest, shortest_route


tsplib_file = "ACO/att48.tsp"

distance_matrix = tsplib_distance_matrix(tsplib_file)
distance_matrix = np.array(distance_matrix, dtype=np.float64)

aco = AntColony(
    distances=distance_matrix,
    n_ants=100,
    n_iterations=10000,
    alpha=1,
    beta=3,
    rho=0.1,
    Q=0.5,
    initial_pheromone=0.1
)

best_solution = aco.run()
