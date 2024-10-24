import numpy as np
import numba as nb

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
    ('customers', nb.int64),
    ('demand', nb.int64[:]),
    ('drivers', nb.int64),
    ('capacity', nb.int64),
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

    def gen_paths(self):
        paths = np.zeros((self.drivers, self.customers + 3, self.n_ants), dtype=np.float64)  # extra entries for the distance column and capacity
        for i in range(0, self.n_ants):
            path = self.construct_path()
            paths[:, :, i] = path
        return paths

    def construct_path(self):
        path = np.full((self.drivers, self.customers + 3), dtype=np.float64, fill_value=np.nan)
        initial_point = 0
        unvisited = np.ones((1, len(self.distances)))
        unvisited[:, initial_point] = 0
        path[:, -3] = 1  # this is the length of the driver route
        path[:, -2] = 0  # this is demand of the driver
        path[:, -1] = 0  # this is the distance ok?
        path[:, 0] = initial_point
        patience = 0
        error = 0
        error_counter = 0
        while np.any(unvisited == 1):
            for driver_idx in range(self.drivers):
                if np.all(unvisited == 0):  # all customers have been assigned, horray!
                    break
                route_len = int(
                    path[driver_idx, -3])  # the last 3rd entry always stores the the length of the driver route
                prev = int(path[driver_idx, route_len - 1])

                feasibility = np.array([(path[driver_idx, -2] + demand[idx]) <= capacity  # some feasibility check array
                                        for idx in range(0, len(self.distances))]).reshape(1, -1)
                choose_city = self.choose_node(unvisited, self.pheromones[prev, :], self.distances[prev, :],feasibility)
                if not np.isnan(choose_city):
                    patience = 0
                if np.isnan(choose_city):
                    if patience > 40:  # before it gives up and resets
                        if np.any(unvisited == 1):

                            path[:, :] = np.nan
                            initial_point = 0
                            unvisited = np.ones((1, len(self.distances)))
                            unvisited[:, initial_point] = 0
                            path[:, -3] = 1
                            path[:, -2] = 0
                            path[:, -1] = 0
                            path[:, 0] = initial_point
                            patience = 0
                            error += 1
                        else:
                            break
                    if error > 100000:  # this is just really bad, worst case scenario, if this pops up then the algorithm is
                        # dead, throw your computer away
                        print('Error:', which)
                        error_counter += 1
                        error = 0
                    elif np.any(unvisited == 1):
                        patience += 1
                        continue
                    continue

                choose_city = int(
                    choose_city)  # this part is just entering ths stuff into the array, demand, distance, unvisited, whatever you name it
                path[driver_idx, route_len] = choose_city
                path[driver_idx, -2] += demand[choose_city]
                path[driver_idx, -3] += 1
                unvisited[:, choose_city] = 0
                path[driver_idx, -1] += int(self.path_dist([prev, choose_city]))
        for i in range(self.drivers):
            idx = int(path[i, -3])
            path[i, idx] = initial_point
            path[i, -1] += self.path_dist([int(path[i, idx - 1]), initial_point])
        return path

    def choose_node(self, unvisited, pheromones, distance, feasibility):

        probability = np.where((distance != 0) & (unvisited != 0) & (feasibility != 0),
                               (pheromones ** self.alpha) * ((1 / distance) ** self.beta),
                               0)

        if np.sum(probability) == 0:
            return np.NAN
        else:
            probability /= probability.sum()
            return int(rand_choice_nb(np.arange(len(unvisited[0])), probability[0]))

    def update_pheromones(self, ants_paths):

        self.pheromones[:, :] *= (1 - self.rho)  # decay
        for k in range(0, ants_paths.shape[2]):
            for i in range(0, ants_paths.shape[0]):
                for j in range(0, int(ants_paths[i, -3, k] - 1)):

                    path_current = int(ants_paths[i, j + 1, k])
                    path_prev = int(ants_paths[i, j, k])
                    self.pheromones[path_prev, path_current] += self.Q / ants_paths[i, -1, k]  # update pheromone

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
            idx = np.argsort(summed)[:3]

            self.update_pheromones(all_paths[:, :, idx])
            idxx = idx[0]
            if summed[idxx] < shortest:
                print(i)
                shortest = summed[idxx]
                shortest_route = all_paths[:, :, idxx]
                print(f'Current shortest path:', shortest_route)
                print(f'Current shortest distance:', shortest)

        return shortest, shortest_route



import vrplib
file = 'ACO/instances/A-n60-k9.vrp'
re = vrplib.read_instance(file)
demand = re['demand']
distance_matrix = re['edge_weight']
capacity = re['capacity']

aco = AntColony(
    distances=distance_matrix,
    demand=demand,
    capacity=capacity,
    drivers=9,
    n_ants=1,
    n_iterations=3000,
    alpha=1,
    beta=3,
    rho=0.01,
    Q=1,
    initial_pheromone=1.0
)

best_solution = aco.run()
