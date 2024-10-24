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
    for i in range(len(path) - 1):
        a, b = int(path[i]), int(path[i + 1])
        total += distances[a, b]
    return total


@njit
def interswap_single_ant(ant_path, demand, capacity, drivers, distances, limit):
    aa = list(range(drivers))
    kek = [(a, b) for idx, a in enumerate(aa) for b in aa[idx + 1:]]

    for driver1, driver2 in kek:
        timeout = 0
        while timeout < limit:
            placeholder = ant_path.copy()
            old = np.sum(ant_path[:, -1])
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

            placeholder[driver1, choose_cust1] = idx2
            placeholder[driver2, choose_cust2] = idx1

            placeholder[driver1, -1] = path_dist(placeholder[driver1, 0:driverlen1 + 1], distances)
            placeholder[driver2, -1] = path_dist(placeholder[driver2, 0:driverlen2 + 1], distances)

            timeout += 1
            if ((path_demand1 - demand1 + demand2 <= capacity) &
                    (path_demand2 - demand2 + demand1 <= capacity)):
                if np.sum(placeholder[:, -1]) < old:
                    ant_path[driver1, choose_cust1] = idx2
                    ant_path[driver2, choose_cust2] = idx1

                    ant_path[driver1, -1] = path_dist(ant_path[driver1, 0:driverlen1 + 1], distances)
                    ant_path[driver2, -1] = path_dist(ant_path[driver2, 0:driverlen2 + 1], distances)
                    ant_path[driver1, -2] = path_demand1 - demand1 + demand2
                    ant_path[driver2, -2] = path_demand2 - demand2 + demand1
                    timeout = 0
            else:
                continue
    return ant_path


@njit(parallel=True)
def interswap(ants_paths, demand, capacity, drivers, distances,
              limit):  #  just some swap algorithm which swaps to customers
    for k in range(ants_paths.shape[3]):
        for i in prange(ants_paths.shape[2]):
            ant_path = ants_paths[:, :, i, k]
            ants_paths[:, :, i, k] = interswap_single_ant(ant_path, demand, capacity, drivers, distances,
                                                          limit)
    return ants_paths


@njit
def twoopt_single_path(path, distances, limit):
    length = int(path[-3])
    if length <= 3:  # skip routes that are too short for 2-opt
        return path

    best_path = path.copy()
    best_distance = path[-1]
    timeout = 0

    while timeout < limit:
        idx = np.random.choice(np.arange(1, length), 2, replace=False)
        idx1, idx2 = min(idx), max(idx)

        new_path = best_path.copy()
        new_path[idx1:idx2 + 1] = new_path[idx1:idx2 + 1][::-1]

        new_distance = path_dist(new_path[:length + 1], distances)

        if new_distance < best_distance:
            best_path = new_path
            best_distance = new_distance
            timeout = 0
        else:
            timeout += 1

    best_path[-1] = best_distance
    return best_path


@njit(parallel=True)
def twoopt(paths, distances, limit):
    result = np.empty_like(paths)
    for k in range(paths.shape[3]):
        for i in prange(paths.shape[2]):
            for j in range(paths.shape[0]):
                result[j, :, i, k] = twoopt_single_path(paths[j, :, i, k], distances, limit)
    return result


@njit(parallel=True)
def gen_paths(drivers, customers, pheromone, alpha, beta, rho, n_ants, demand, capacity, distance,
              limit_1, limit_2, boost, randomness, optimise):
    old_limit_1 = limit_1
    old_limit_2 = limit_2
    paths = np.zeros((drivers, customers + 3, n_ants, pheromone.shape[2]), dtype=np.float64)
    for k in range(pheromone.shape[2]):
        for i in prange(n_ants):
            path = construct_path(drivers, customers, pheromone[:, :, k], alpha,
                                  beta, rho, n_ants, distance, demand, capacity, randomness)
            paths[:, :, i, k] = path

    if optimise:
        if boost:
            limit_1 = 500
            limit_2 = 1000
        else:
            limit_1 = old_limit_1
            limit_2 = old_limit_2
        paths = interswap(paths, demand, capacity, drivers, distance, limit_2)
        paths = twoopt(paths, distance, limit_1)
    return paths


@njit
def repair(path, drivers, distances, demand, unvisited, capacity):
    changed = False
    for i in range(drivers):
        for k in range(distances.shape[0]):
            feasibility = np.array([(path[i, -2] + demand[idx]) <= capacity
                                    for idx in range(0, len(distances))]).reshape(1, -1)
            good = np.logical_and(feasibility, unvisited.astype(np.bool_))
            idx = int(path[i, -3]) - 1
            if good[0, k]:
                path[i, idx + 1] = k
                path[i, -2] += demand[k]
                path[i, -3] += 1
                path[i, -1] += path_dist([idx, k], distances)
                unvisited[:, k] = 0
                changed = True
                feasibilityz = np.array([(path[i, -2] + demand[idx]) <= capacity
                                         for idx in range(0, len(distances))]).reshape(1, -1)
                if np.all(feasibilityz == False):
                    path[i, -1] += path_dist([path[i, idx + 1], 0], distances)
                    path[i, -3] += 1
                    path[i, idx + 2] = 0

    return unvisited, path, changed


@njit
def construct_path(drivers, customers, pheromone, alpha, beta, rho, n_ants, distances, demand, capacity, randomness):
    path = np.full((drivers, customers + 3), dtype=np.float64, fill_value=np.nan)
    initial_point = 0
    unvisited = np.ones((1, len(distances)))
    unvisited[:, initial_point] = 0
    path[:, -3] = 1  # this is the length of the driver route
    path[:, -2] = 0  # this is demand of the driver
    path[:, -1] = 0  # this is the distance
    path[:, 0] = initial_point
    patience = 0
    error = 0
    error_counter = 0
    while np.any(unvisited == 1):
        for driver_idx in range(drivers):
            if np.all(unvisited == 0):  # all customers have been assigned
                break
            if randomness:
                driver_idx = random.randint(0, drivers - 1)
            route_len = int(
                path[driver_idx, -3])  #  the last 3rd entry always stores the the length of the driver route
            prev = int(path[driver_idx, route_len - 1])

            feasibility = np.array([(path[driver_idx, -2] + demand[idx]) <= capacity  #  feasibility check array
                                    for idx in range(0, len(distances))]).reshape(1, -1)

            choose_city = choose_node(unvisited, pheromone[prev, :],
                                      distances[prev, :], feasibility, alpha, beta)
            if not np.isnan(choose_city):
                patience = 0
            if np.isnan(choose_city):
                if patience > 40:  #  before it gives up and resets
                    # if fail_safe and which == colonies - 1:
                    #     unvisited, path, changed = repair(path, drivers, distances, demand, unvisited, capacity)
                    if np.any(unvisited == 1):
                        path[:, :] = np.nan
                        initial_point = 0
                        unvisited = np.ones((1, len(distances)))
                        unvisited[:, initial_point] = 0
                        path[:, -3] = 1
                        path[:, -2] = 0
                        path[:, -1] = 0
                        path[:, 0] = initial_point
                        patience = 0
                        error += 1
                    else:
                        break
                if error > 200000:
                    print('Error')
                    error_counter += 1
                    error = 0
                elif np.any(unvisited == 1):
                    patience += 1
                    continue
                continue

            choose_city = int(
                choose_city)
            path[driver_idx, route_len] = choose_city
            path[driver_idx, -2] += demand[choose_city]
            path[driver_idx, -3] += 1
            unvisited[:, choose_city] = 0
            path[driver_idx, -1] += path_dist([prev, choose_city], distances)
    for i in range(drivers):
        idx = int(path[i, -3])
        path[i, idx] = initial_point
        path[i, -1] += path_dist([int(path[i, idx - 1]), initial_point], distances)

    return path


@njit
def choose_node(unvisited, pheromones, distance, feasibility, alpha, beta):
    eps = 0.1
    probability = np.where((unvisited != 0) & (feasibility != 0),
                           (pheromones ** alpha) * ((1 / (distance + eps)) ** beta),
                           0)

    if np.sum(probability) == 0:  #  either it's an unfeasible solution or the construction of the route is done
        return np.nan
    else:
        probability /= probability.sum()
        return int(rand_choice_nb(np.arange(len(unvisited[0])), probability[0]))
