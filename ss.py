import numpy as np
import numba as nb
import random
from ACO.read_file import tsplib_distance_matrix
import time
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
    ('pheromones', nb.float64[:, :]),
    ('edge_coord', nb.float64[:, :]),
    ('sweep', nb.float64[:, :, :]),
    ('customers', nb.int64),
    ('demand', nb.int64[:]),
    ('drivers', nb.int64),
    ('capacity', nb.int64),
    ('tau_max', nb.float64),
    ('tau_min', nb.float64),
    ('p_best', nb.float64),
    ('s_gb', nb.float64[:, :]),
    ('L_nn', nb.float64)

]
@nb.njit
def nearest_neighbor_heuristic(distances, demand, capacity, drivers):
    n = distances.shape[0]
    unvisited = np.ones(n, dtype=np.bool_)
    unvisited[0] = False  # depot is not a customer
    total_distance = 0
    for _ in range(drivers):
        current = 0  # start at depot
        route_demand = 0
        while True:
            nearest = -1
            min_dist = np.inf
            for j in range(1, n):
                if unvisited[j] and route_demand + demand[j] <= capacity:
                    dist = distances[current, j]
                    if dist < min_dist:
                        min_dist = dist
                        nearest = j
            if nearest == -1:
                total_distance += distances[current, 0]  # return to depot
                break
            total_distance += min_dist
            current = nearest
            unvisited[current] = False
            route_demand += demand[current]
    return total_distance

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
                 demand,
                 drivers,
                 capacity,
                 n_ants,
                 n_iterations,
                 alpha,
                 beta,
                 rho,
                 Q,
                 initial_pheromone,
                 edge_coord,
                 p_best):
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
        self.edge_coord = edge_coord

        self.sweep = SweepAlgorithm(self.edge_coord, self.demand, self.capacity, self.drivers, self.distances).cluster()
        self.L_nn = nearest_neighbor_heuristic(self.distances, self.demand, self.capacity, self.drivers)
        self.s_gb = np.zeros((self.drivers, self.customers + 3))
        self.p_best = p_best
        self.tau_max = 1.0
        self.tau_min = 0.0
        self.pheromones = np.full((self.customers, self.customers), self.tau_max)
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
        path[:, -3] = 1
        path[:, -2] = 0
        path[:, -1] = 0
        path[:, 0] = initial_point
        test = 0
        while np.any(unvisited == 1):
            for driver_idx in range(self.drivers):

                rand = [random.randint(0, self.drivers-1) for _ in range(self.drivers)]
                driver_idx = rand[driver_idx]
                route_len = int(path[driver_idx, -3])
                prev = int(path[driver_idx, route_len - 1])

                feasibility = np.array([(path[driver_idx, -2] + self.demand[idx]) <= self.capacity
                               for idx in range(0, len(self.distances))]).reshape(1, -1)

                choose_city = self.choose_node(unvisited, self.pheromones[prev, :],
                                               self.distances[prev, :], feasibility)
                if np.isnan(choose_city):
                    if np.any(unvisited == 1):
                        test += 1
                        continue
                    break
                if choose_city != np.nan:
                    test = 0
                choose_city = int(choose_city)
                path[driver_idx, route_len] = choose_city
                path[driver_idx, -2] += self.demand[choose_city]
                path[driver_idx, -3] += 1
                unvisited[:, choose_city] = 0
                path[driver_idx, -1] += int(self.path_dist([prev, choose_city]))

            if test > 100:
                path = np.full((self.drivers, self.customers + 3), dtype=np.float64, fill_value=np.nan)
                initial_point = 0
                unvisited = np.ones((1, len(self.distances)))
                unvisited[:, initial_point] = 0
                path[:, -3] = 1
                path[:, -2] = 0
                path[:, -1] = 0
                path[:, 0] = initial_point
                test = 0

        for i in range(self.drivers):
            idx = int(path[i, -3])
            path[i, idx] = initial_point
            path[i, -1] += self.path_dist([int(path[i, idx - 1]), initial_point])

        return path

    def choose_node(self,  unvisited, pheromones, distance, feasibility):
        probability = np.where((distance != 0) & (unvisited != 0) & (feasibility != 0),
                               (pheromones ** self.alpha) * ((1 / distance) ** self.beta),
                               0)

        if np.sum(probability) == 0:
            return np.NAN
        else:
            probability /= probability.sum()
            return int(rand_choice_nb(np.arange(len(unvisited[0])), probability[0]))

    def decay_pheromones(self):
        self.pheromones *= (1 - self.rho)  # decay

    def update_pheromones(self, ants_paths, s_best):
        for i in range(0, ants_paths.shape[0]):
            for j in range(0, int(ants_paths[i, -3].item())):
                path_current = int(ants_paths[i, j + 1].item())
                path_prev = int(ants_paths[i, j].item())
                dist = self.path_dist([path_prev, path_current])
                self.pheromones[path_prev, path_current] += self.Q / dist
                self.pheromones[path_current, path_prev] += self.Q / dist



    def update_tau(self):
        n = self.distances.shape[0]
        avg = n/self.drivers
        best_so_far = np.sum(self.s_gb[:, -1])
        self.tau_max = 1 / ((self.rho * best_so_far) * (1 - self.rho))
        self.tau_min = self.tau_max * (1 - pow(self.p_best, 1/n))/((avg - 1) * pow(self.p_best, 1/n))



    def path_dist(self, path):
        total = 0.0
        for i in range(len(path) - 1):
            a = int(path[i])
            b = int(path[i + 1])
            total += self.distances[a, b]
        return total

    def intraswap(self, ants_paths):
        for i in range(ants_paths.shape[2]):
            ant_path = ants_paths[:, : , i].copy()
            timeout = 0
            chosen_drivers = rand_choice_nb_multiple(np.arange(1, self.drivers), 2)
            driver1, driver2, = chosen_drivers[0], chosen_drivers[1]
            old = ants_paths[driver1, -1 , i] + ants_paths[driver2, -1 , i]
            while timeout < 500:
                driverlen1 = int(ant_path[driver1, -3])
                choose_cust1 = rand_choice_nb_multiple(np.arange(1, driverlen1), 1)
                idx1 = int(ant_path[driver1, choose_cust1[0]])
                path_demand1 = ant_path[driver1, -2]
                demand1 = self.demand[idx1]

                driverlen2 = int(ant_path[driver2, -3])
                choose_cust2 = rand_choice_nb_multiple(np.arange(1, driverlen2), 1)
                idx2 = int(ant_path[driver2, choose_cust2[0]])
                path_demand2 = ant_path[driver2, -2]
                demand2 = self.demand[idx2]

                timeout +=1

                if ((path_demand1 - demand1 + demand2 <= self.capacity) &
                        (path_demand2 - demand2 + demand1 <= self.capacity)):
                    ant_path[driver1, choose_cust1] = idx2
                    ant_path[driver2, choose_cust2] = idx1

                    ant_path[driver1, -1] = self.path_dist(ant_path[driver1, 0:driverlen1 + 1])
                    ant_path[driver2, -1] = self.path_dist(ant_path[driver2, 0:driverlen2 + 1])
                    ant_path[driver1, -2] = path_demand1 - demand1 + demand2
                    ant_path[driver2, -2] = path_demand2 - demand2 + demand1

                    if ant_path[driver1, -1] + ant_path[driver2, -1] < old:
                        ants_paths[driver1, :, i] = ant_path[driver1, :]
                        ants_paths[driver2, :, i] = ant_path[driver2, :]
                        old = ant_path[driver1, -1] + ant_path[driver2, -1]
                        timeout = 0
                else:
                    continue

        return ants_paths

    def twoopt(self, path):
        for i in range(path.shape[2]):
            for j in range(path.shape[0]):
                length = int(path[j, -3, i])
                pathz = path[j, :, i].copy()
                limit = 0
                old = path[j, -3, i]
                if length <= 3:  # Skip routes that are too short for 2-opt
                    continue
                while limit < 500:
                    idx = rand_choice_nb_multiple(np.arange(1, length), 2)
                    idx1, idx2 = int(min(idx)), int(max(idx))

                    # Reverse the segment
                    pathz[idx1:idx2 + 1] = pathz[idx1:idx2 + 1][::-1]

                    # Recalculate the route distance

                    pathz[-1] = self.path_dist(pathz[:length + 1])
                    if pathz[-1] < old:
                        path[j, :, i] = pathz
                        path[j, -1, i] = pathz[-1]
                        old = pathz[-1]
                        limit = 0
                    limit += 1
        return path

    def run(self):
        b = None
        stagnation = 0
        shortest = np.inf
        all_paths = self.sweep
        for i in range(10):
            all_paths = self.twoopt(all_paths)
            all_paths = self.intraswap(all_paths)
        summed = np.sum(all_paths[:, -1, :], axis=0)
        self.update_pheromones(all_paths, summed[0])

        for i in range(self.n_iterations):
            all_paths = self.gen_paths()
            summed = np.sum(all_paths[:, -1, :], axis=0)
            idxx = np.argsort(summed)[:10]
            all_paths = self.twoopt(all_paths[:,:,idxx])
            all_paths = self.intraswap(all_paths)
            summed = np.sum(all_paths[:, -1, :], axis=0)
            idxx = np.argmin(summed)
            if summed[idxx] < shortest:
                shortest = summed[idxx]
                self.s_gb = all_paths[:, :, idxx]
                print(f'Current shortest distance:', shortest, i)
                stagnation = 0
                b = [shortest, i]
            else:
                stagnation += 1

            self.decay_pheromones()
            self.update_pheromones(self.s_gb, shortest)





        return b


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
    edge_coord=edge_coord,
    capacity=capacity,
    drivers=10,
    n_ants=1000,
    n_iterations=20000,
    alpha=1,
    beta=3,
    rho=0.99,
    Q=1,
    initial_pheromone=0.99,
    p_best=0.01
)

best_solution = aco.run()

# rho = np.linspace(0.01,0.99,10)
# p_best = np.linspace(0.1,0.99,10)
# alpha = [1,2,3,4,5]
# beta = [1,2,3,4,5]
# l = []
# for i in rho:
#     for j in p_best:
#         for k in alpha:
#             for p in beta:
#
#                 aco = AntColony(
#                     distances=distance_matrix,
#                     demand=demand,
#                     edge_coord=edge_coord,
#                     capacity=capacity,
#                     drivers=5,
#                     n_ants=200,
#                     n_iterations=1000,
#                     alpha=k,
#                     beta=p,
#                     rho=i,
#                     Q=1,
#                     initial_pheromone=0.99,
#                     p_best=j
#                 )
#                 best_solution = aco.run()
#                 l.append([best_solution, [i, j, k, p]])
#                 print(l)



# l = []
# for i in range(30):
#     aco = AntColony(
#         distances=distance_matrix,
#         demand=demand,
#         edge_coord=edge_coord,
#         capacity=capacity,
#         drivers=5,
#         n_ants=200,
#         n_iterations=4000,
#         alpha=2,
#         beta=3,
#         rho=0.99,
#         Q=1,
#         initial_pheromone=0.99,
#         p_best=0.01
#     )
#
#     best_solution = aco.run()
#     l.append(best_solution)
#     print(l)