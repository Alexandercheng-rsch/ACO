import numpy as np
import numba as nb
import random
from read_file import tsplib_distance_matrix

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
    ('n_cities', nb.int64),
]


@nb.njit
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

@nb.experimental.jitclass(spec)
class AntColony:
    def __init__(self, distances, n_ants, n_iterations, alpha, beta, rho, Q, initial_pheromone):
        self.distances = distances
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.initial_pheromone = initial_pheromone
        self.n_cities = distances.shape[0]
        self.pheromones = np.full((self.n_cities, self.n_cities), initial_pheromone)

    def gen_paths(self):
        paths = []
        for _ in range(self.n_ants):
            path = self.construct_path()
            paths.append((path, self.path_dist(path)))
        return paths


    def construct_path(self):
        path = []
        initial_point = random.randint(0, len(self.distances) - 1)
        unvisited = np.ones((1, len(self.distances)))
        unvisited[:, initial_point] = 0
        prev = initial_point
        for _ in range(len(self.distances) - 1):
            get_path = self.choose_node(unvisited, self.pheromones[prev], self.distances[prev])
            path.append((prev, get_path))
            prev = get_path

            unvisited[:, prev] = 0

        path.append((prev, initial_point))

        return path

    def choose_node(self, unvisited, pheromones, distance):
        collection = []
        for loc, i in enumerate(unvisited[0]):
            if i == 1:
                prob = pheromones[loc] ** self.alpha * (1/distance[loc]) ** self.beta
                collection.append(prob)
            else:
                collection.append(0)
        collection = np.array(collection)
        collection /= collection.sum()

        decision = rand_choice_nb(list(range(len(unvisited[0]))), collection)
        return decision

    def update_pheromones(self, ants_paths):
        # Evaporation step
        self.pheromones *= (1 - self.rho)

        # Pheromone deposit step
        for path in ants_paths:
            for edge in path[0]:
                self.pheromones[edge] += self.Q / path[1]

    def path_dist(self, path):
        total = 0
        for edge in path:
            total += self.distances[edge]

        return total

    def run(self):
        shortest = np.inf

        for i in range(self.n_iterations):
            all_paths = self.gen_paths()

            self.update_pheromones(all_paths)

            current_shortest = sorted(all_paths, key=lambda x: x[1])[0]

            if current_shortest[1] < shortest:
                shortest = current_shortest[1]
                print(f'Current shortest path:', shortest)

        return shortest




tsplib_file = "att48.tsp"
distance_matrix = tsplib_distance_matrix(tsplib_file)
distance_matrix = np.array(distance_matrix, dtype=np.float64)

aco = AntColony(
    distances=distance_matrix,
    n_ants=48,
    n_iterations=5000,
    alpha=2,
    beta=3,
    rho=0.1,
    Q=0.5,
    initial_pheromone=0.1
)

best_solution = aco.run()





