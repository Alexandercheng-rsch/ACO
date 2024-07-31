import time

from numba import njit, prange
import numba as nb
import numpy as np
import random


@njit
def rand_choice_nb(arr, prob):
    a = np.searchsorted(np.cumsum(prob), np.random.random(), side="right")
    return arr[a]


@njit
def rand_choice_nb_multiple(arr, num_samples):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param num_samples: Number of samples to choose.
    :return: An array of random samples chosen without replacement.
    """
    result = np.empty(num_samples, dtype=arr.dtype)
    temp_arr = arr.copy()
    for i in prange(num_samples):
        idx = np.random.randint(0, len(temp_arr))
        result[i] = temp_arr[idx]
        temp_arr[idx] = temp_arr[-1]
        temp_arr = temp_arr[:-1]
    return result


@njit
def path_dist(path, distances):
    total = 0.0
    for i in prange(len(path) - 1):
        a, b = int(path[i]), int(path[i + 1])
        total += distances[a, b]
    return total

@njit(parallel=True)
def intraswap(ants_paths, demand, capacity, drivers, distances, limit): #  just some swap algorithm which swaps to customers
    for k in range(ants_paths.shape[3]):
        for i in prange(ants_paths.shape[2]):
            ant_path = ants_paths[:, :, i, k].copy()
            timeout = 0
            chosen_drivers = rand_choice_nb_multiple(np.arange(0, drivers), 2)
            driver1, driver2, = chosen_drivers[0], chosen_drivers[1]
            old = ants_paths[driver1, -1, i, k] + ants_paths[driver2, -1, i, k]
            while timeout < limit:
                driverlen1 = int(ant_path[driver1, -3])
                choose_cust1 = rand_choice_nb_multiple(np.arange(1, driverlen1), 1)
                idx1 = int(ant_path[driver1, choose_cust1[0]])
                path_demand1 = ant_path[driver1, -2]
                demand1 = demand[idx1]

                driverlen2 = int(ant_path[driver2, -3])
                choose_cust2 = rand_choice_nb_multiple(np.arange(1, driverlen2), 1)
                idx2 = int(ant_path[driver2, choose_cust2[0]])
                path_demand2 = ant_path[driver2, -2]
                demand2 = demand[idx2]

                timeout += 1
                if ((path_demand1 - demand1 + demand2 <= capacity) &
                        (path_demand2 - demand2 + demand1 <= capacity)):
                    ant_path[driver1, choose_cust1] = idx2
                    ant_path[driver2, choose_cust2] = idx1

                    ant_path[driver1, -1] = path_dist(ant_path[driver1, 0:driverlen1 + 1], distances)
                    ant_path[driver2, -1] = path_dist(ant_path[driver2, 0:driverlen2 + 1], distances)
                    ant_path[driver1, -2] = path_demand1 - demand1 + demand2
                    ant_path[driver2, -2] = path_demand2 - demand2 + demand1

                    if ant_path[driver1, -1] + ant_path[driver2, -1] < old:
                        ants_paths[driver1, :, i, k] = ant_path[driver1, :]  # new path
                        ants_paths[driver2, :, i, k] = ant_path[driver2, :]

                        ants_paths[driver1, -1, i, k] = ant_path[driver1, -1]  # change to new distance
                        ants_paths[driver2, -1, i, k] = ant_path[driver2, -1]

                        ants_paths[driver1, -2, i, k] = ant_path[driver1, -2]  # new demand
                        ants_paths[driver2, -2, i, k] = ant_path[driver2, -2]
                        old = ant_path[driver1, -1] + ant_path[driver2, -1]
                        timeout = 0
                else:
                    continue

    return ants_paths


@njit(parallel=True)
def twoopt(path, distances, limit):
    for k in range(path.shape[3]):
        for i in prange(path.shape[2]):
            for j in range(path.shape[0]):
                length = int(path[j, -3, i, k])
                pathz = path[j, :, i, k].copy()
                timeout = 0
                old = path[j, -3, i, k]
                if length <= 3:  # Skip routes that are too short for 2-opt
                    continue
                while timeout < limit:
                    idx = rand_choice_nb_multiple(np.arange(1, length), 2)
                    idx1, idx2 = int(min(idx)), int(max(idx))

                    # Reverse the segment
                    pathz[idx1:idx2 + 1] = pathz[idx1:idx2 + 1][::-1]

                    # Recalculate the route distance

                    pathz[-1] = path_dist(pathz[:length + 1], distances)
                    if pathz[-1] < old:
                        path[j, :, i, k] = pathz
                        path[j, -1, i, k] = pathz[-1]
                        old = pathz[-1]
                        timeout = 0
                    timeout += 1
    return path


@njit(parallel=True)
def gen_paths(drivers, customers, pheromone, alpha, beta, rho, n_ants, demand, capacity, distance, colonies, limit, optimise):
    paths = np.zeros((drivers, customers + 3, n_ants, colonies), dtype=np.float64)
    for k in range(colonies):
        for i in prange(n_ants):
            path = construct_path(drivers, customers, pheromone[:, :, k], alpha, beta, rho, n_ants, distance, demand,
                                  capacity)
            paths[:, :, i, k] = path
    if optimise:
        paths = twoopt(paths, distance, limit)
        paths = intraswap(paths, demand, capacity, drivers, distance, limit)
    return paths

@njit
def construct_path(drivers, customers, pheromone, alpha, beta, rho, n_ants, distances, demand, capacity):
    path = np.full((drivers, customers + 3), dtype=np.float64, fill_value=np.nan)
    initial_point = 0
    unvisited = np.ones((1, len(distances)))
    unvisited[:, initial_point] = 0
    path[:, -3] = 1
    path[:, -2] = 0
    path[:, -1] = 0
    path[:, 0] = initial_point
    patience = 0
    while np.any(unvisited == 1):

        for driver_idx in range(drivers):
            rand = [random.randint(0, drivers - 1) for _ in range(drivers)]
            driver_idx = rand[driver_idx]
            route_len = int(path[driver_idx, -3])
            prev = int(path[driver_idx, route_len - 1])

            feasibility = np.array([(path[driver_idx, -2] + demand[idx]) <= capacity
                           for idx in range(0, len(distances))]).reshape(1, -1)

            choose_city = choose_node(unvisited, pheromone[prev, :],
                                      distances[prev, :], feasibility, alpha, beta)
            if not np.isnan(choose_city):
                patience = 0
            if np.isnan(choose_city):
                if patience > 50:
                    path[:, :] = np.nan
                    initial_point = 0
                    unvisited = np.ones((1, len(distances)))
                    unvisited[:, initial_point] = 0
                    path[:, -3] = 1
                    path[:, -2] = 0
                    path[:, -1] = 0
                    path[:, 0] = initial_point
                    patience = 0
                elif np.any(unvisited == 1):
                    patience += 1
                    continue

                break

            choose_city = int(choose_city)
            path[driver_idx, route_len] = choose_city
            path[driver_idx, -2] += demand[choose_city]
            path[driver_idx, -3] += 1
            unvisited[:, choose_city] = 0
            path[driver_idx, -1] += int(path_dist([prev, choose_city], distances))
    for i in range(drivers):
        idx = int(path[i, -3])
        path[i, idx] = initial_point
        path[i, -1] += path_dist([int(path[i, idx - 1]), initial_point], distances)

    return path


@njit
def choose_node(unvisited, pheromones, distance, feasibility, alpha, beta):
    eps = 0.1
    probability = np.where((distance != 0) & (unvisited != 0) & (feasibility != 0),
                           (pheromones ** alpha) * ((1 / (distance + eps)) ** beta),
                           0)

    if np.sum(probability) == 0:
        return np.nan
    else:
        probability /= probability.sum()
        return int(rand_choice_nb(np.arange(len(unvisited[0])), probability[0]))

