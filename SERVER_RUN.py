import numpy as np
import numba as nb
import random
from read_file import tsplib_distance_matrix
import time
from path_construction import gen_paths, twoopt, intraswap
from numba import njit, prange
import optuna

np.set_printoptions(precision=20, suppress=True, threshold=np.inf)


@njit
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
                 gamma,
                 Q,
                 colonies,
                 exchange_rate,
                 master_colony_update_rate,
                 colony_rebirth_limit,
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
        self.gamma = gamma
        self.exchange_rate = exchange_rate
        self.colony_rebirth_limit = colony_rebirth_limit
        self.master_colony_update_rate = master_colony_update_rate
        self.pheromones = np.full((self.customers, self.customers, self.colonies + 1), initial_pheromone)
        self.iter_local_optima = np.full((1, self.colonies), 0)

    def update_pheromones(self, ants_paths, no_colonie, best_routes):
        if best_routes is None:
            best_routes = np.full((1, self.colonies + 1), 1)
        else:
            pass
        if ants_paths.ndim == 2:
            ants_paths = ants_paths[:, :, np.newaxis]

        self.pheromones *= (1 - self.rho)  # decay
        for k in range(0, ants_paths.shape[2]):
            for i in range(0, ants_paths.shape[0]):
                for j in range(0, int(ants_paths[i, -3, k])):
                    path_current = int(ants_paths[i, j + 1, k])
                    path_prev = int(ants_paths[i, j, k])
                    self.pheromones[path_prev, path_current, no_colonie] += (
                            self.Q / (ants_paths[i, -1, k] * 1/(best_routes[:, no_colonie])))  # update

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def update_all_colonies(self, paths, best_routes):
        for i in range(self.colonies + 1):
            path = paths[:, :, i]
            self.update_pheromones(path, i, best_routes)

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

    def exchange_information(self, best_paths, best_routes):
        colonies = list(range(self.colonies))
        res = []

        # Exchange information between random pairs
        while len(colonies) > 1:
            i = colonies.pop(random.randint(0, len(colonies) - 1))
            j = colonies.pop(random.randint(0, len(colonies) - 1))
            res.append([i, j])

        # Add exchanges between the last index and all others
        last_index = self.colonies
        for i in range(self.colonies):
            res.append([i, last_index])

        # Perform the information exchange
        for i, j in res:
            self.update_pheromones(best_paths[:, :, j], i, best_routes)
            self.update_pheromones(best_paths[:, :, i], j, best_routes)

    def update_master_colony(self, colony_fitness):
        total = np.sum(colony_fitness[:, :-1])
        self.pheromones[:, :, -1] = 0  # Start from zero
        for col in range(self.colonies):
            self.pheromones[:, :, -1] += (self.pheromones[:, :, col] * ((1 / (colony_fitness[:, col])) ** self.gamma) /
                                          total)

        sum_ = np.sum(self.pheromones[:, :, -1])
        if sum_ > 0:
            self.pheromones[:, :, -1] /= sum_

    def colony_rebirth(self, colony_fitness):
        colony_fitness = colony_fitness.squeeze()
        sigma = np.std(colony_fitness[:-1])
        max_ = max(colony_fitness[:-1])
        min_ = min(colony_fitness[:-1])
        diff = max_ - min_
        idx = np.argsort(colony_fitness[:-1])
        worst_colony = idx[-1]
        observation = np.exp(-(abs(diff * sigma) / (self.iter_local_optima[:, worst_colony] *
                                                    colony_fitness[worst_colony])))
        if observation >= random.uniform(0, 1) and self.colonies >= 2:
            print('rebirthed')
            pick_colonies = rand_choice_nb_multiple(idx[:-1], 2)
            self.pheromones[:, :,  worst_colony] = 0  #execution
            total = np.sum(colony_fitness[pick_colonies])
            for colony in pick_colonies:
                self.pheromones[:, :, worst_colony] += (self.pheromones[:, :, colony] *
                                                  ((1 / (colony_fitness[colony])) ** self.gamma) / total)

            if total > 0:
                self.pheromones[:, :, worst_colony] /= total
            self.iter_local_optima[:, worst_colony] = 0


    def run(self):
        shortest = np.full((1, self.colonies + 1), np.inf)
        shortest_route = None
        iter_local_optima = np.full((1, self.colonies + 1), 0)  # iterations stuck in local optima
        # for i in range(10):
        #     sweep = twoopt(sweep, self.distances)
        #     sweep = intraswap(sweep, self.demand, self.capacity, self.drivers, self.distances, False)
        # self.update_all_colonies(sweep[:,:,:,0], True)
        print('Started')
        for i in range(self.n_iterations):
            if i > 1:
                if i % self.master_colony_update_rate == 0 or i == 2:
                    self.update_master_colony(shortest)

            all_paths = gen_paths(self.drivers, self.customers, self.pheromones, self.alpha, self.beta, self.rho,
                                  self.n_ants, self.demand, self.capacity, self.distances, self.colonies + 1, True)

            colony_fitness, colony_routes = self.get_colony_best_fitness(all_paths)

            if i % self.exchange_rate == 0 and i != 0:
                self.exchange_information(colony_routes, shortest_route)
            else:
                self.update_all_colonies(colony_routes, shortest_route)

            self.iter_local_optima += 1

            if i % self.colony_rebirth_limit == 0 and i != 0:
                self.colony_rebirth(colony_fitness)
            for j in range(self.colonies + 1):
                if colony_fitness[:, j].item() < shortest[:, j].item():
                    shortest[:, j] = colony_fitness[:, j]
                    if j != self.colonies:
                        self.iter_local_optima[:, j] = 0
        a = min(shortest.squeeze())
        return a, shortest_route


tsplib_file = "ACO/att48.tsp"

import vrplib

file = 'ACO/A-n80-k10.vrp'
re = vrplib.read_instance(file)
demand = re['demand']
distance_matrix = re['edge_weight']
capacity = re['capacity']
edge_coord = np.array(re['node_coord'], dtype=np.float64)

# aco = AntColony(
#     distances=distance_matrix,
#     demand=demand,
#     capacity=capacity,
#     drivers=10,
#     n_ants=150,
#     n_iterations=5,
#     alpha=2,
#     beta=3,
#     gamma=2,
#     rho=0.05,
#     Q=1,
#     colonies=10,
#     exchange_rate=100,
#     master_colony_update_rate=200,
#     initial_pheromone=1.0,
#     colony_rebirth_limit=1500
# )

# best_solution = aco.run()


def objective(trial):
    # Define the parameters to optimize
    n_ants = trial.suggest_int('n_ants', 100, 100)
    alpha = trial.suggest_float('alpha', 0.5, 5)
    beta = trial.suggest_float('beta', 1, 5)
    gamma = trial.suggest_float('gamma', 1, 5)
    rho = trial.suggest_float('rho', 0.01, 0.06)
    colonies = trial.suggest_int('colonies', 5, 20)
    exchange_rate = trial.suggest_int('exchange_rate', 20, 200)
    master_colony_update_rate = trial.suggest_int('master_colony_update_rate', 1, 500)
    colony_rebirth_limit = trial.suggest_int('colony_rebirth_limit', 20, 2000)
    l = []
    trials = 5
    for _ in range(trials):
        print(_)
        aco = AntColony(
            distances=distance_matrix,
            demand=demand,
            capacity=capacity,
            drivers=10,
            n_ants=n_ants,
            n_iterations=3500,  # You might want to reduce this for optimization
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            rho=rho,
            Q=1,
            colonies=colonies,
            exchange_rate=exchange_rate,
            master_colony_update_rate=master_colony_update_rate,
            initial_pheromone=1.0,
            colony_rebirth_limit=colony_rebirth_limit
        )
        best_solution, _ = aco.run()
        l.append(best_solution)
    std = np.std(l)
    mean = np.mean(l)
    return mean + std

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1000, n_jobs=5)  # -1 uses all available cores

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("\nParameter Performance:")
    for param_name in study.best_params.keys():
        values = []
        scores = []
        for trial in study.trials:
            if param_name in trial.params:
                values.append(trial.params[param_name])
                scores.append(trial.value)

        param_values, param_scores = zip(*sorted(zip(values, scores)))

        print("\n  {}:".format(param_name))
        print("    Best value: {}".format(study.best_params[param_name]))
        print("    Value range: {} to {}".format(min(param_values), max(param_values)))
        print("    Best 3 values: {}".format(param_values[:3]))
        print("    Corresponding scores: {}".format(param_scores[:3]))
