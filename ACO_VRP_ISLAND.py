from ACO.path_construction import gen_paths, path_dist
import numpy as np
import vrplib
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
                 alpha_c,
                 beta_c,
                 rho,
                 rho_c,
                 gamma,
                 Q,
                 colonies,
                 balance,
                 exchange_rate,
                 initial_pheromone,
                 lemon,
                 p_best,
                 initialise_master,
                 optimal_result
                 ):
        self.alpha_c = alpha_c
        self.beta_c = beta_c
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
        self.pheromones = np.full((self.customers, self.customers, self.colonies + 1), initial_pheromone)
        self.lemon = lemon
        self.exchange = False
        self.rho_c = rho_c
        self.tau_max = np.ones((1, self.colonies + 1))
        self.tau_min = np.zeros((1, self.colonies + 1))
        self.p_best = p_best
        self.s_best = np.zeros((self.drivers, self.customers + 3, self.colonies + 1), dtype=np.float64)
        self.c_best = np.full((1, self.colonies + 1), np.inf)
        self.k_best = np.full((1, self.colonies + 1), np.inf)
        self.true_dis = np.zeros(self.colonies + 1)
        self.counter = 0
        self.trigger = np.zeros((1, 2))
        self.pheromones_archive = np.full((self.customers, self.customers, self.colonies + 1), initial_pheromone)
        self.fitness_archive = np.full((1, self.colonies + 1), np.inf)
        self.tau_min_archive = np.zeros((1, self.colonies + 1))
        self.updated = np.full((1, self.colonies + 1), 1)
        self.special_rho = 0.99
        self.balance = balance
        self.initialise_master = initialise_master
        self.exchanging = False
        self.boost = False
        self.optimal_result = optimal_result

    def true_dist(self):
        for k in range(0, self.s_best.shape[2]):
            total = 0
            for i in range(0, self.s_best.shape[0]):
                length = int(self.s_best[i, -3, k])
                total += path_dist(self.s_best[i, :length + 1, k], np.round(self.distances))
            self.true_dis[k] = total

    def update_pheromones(self, ants_paths, no_colonie, consts=1.0):
        if no_colonie != self.colonies:  # evap for worker colony
            if no_colonie > self.balance:
                self.pheromones[:, :, no_colonie] *= (1 - self.special_rho)
            else:
                self.pheromones[:, :, no_colonie] *= (1 - self.rho)
        else:
            self.pheromones[:, :, -1] *= (1 - self.rho_c)  #evap for master colony

        if ants_paths.ndim == 2:
            ants_paths = ants_paths[:, :, np.newaxis]

        for k in range(0, ants_paths.shape[2]):
            for i in range(0, ants_paths.shape[0]):
                for j in range(0, int(ants_paths[i, -3, k])):
                    path_current = int(ants_paths[i, j + 1, k])
                    path_prev = int(ants_paths[i, j, k])
                    if no_colonie != self.colonies:
                        if self.exchanging:
                            self.pheromones[path_prev, path_current, no_colonie] += (
                                    1 / (ants_paths[i, -1, k]))
                        else:
                            self.pheromones[path_prev, path_current, no_colonie] += (
                                    1 / (ants_paths[i, -1, k]))
                    else:
                        self.pheromones[path_prev, path_current, no_colonie] += (consts / ants_paths[i, -1, k])

        if np.any(self.updated == 1):
            which = np.where(self.updated == 1)
            which = which[1]
            for j in which:
                self.pheromones_archive[:, :, j] = self.pheromones[:, :, j]  # save the newly update phermones
                self.fitness_archive[:, j] = self.c_best[:, j]
        self.updated[:, :] = 0
        if no_colonie != self.colonies and not self.exchanging:  # master colony never gets clipped
            self.pheromones[:, :, no_colonie] = np.clip(self.pheromones[:, :, no_colonie], self.tau_min[:, no_colonie],
                                                        self.tau_max[:, no_colonie])

    def update_all_colonies(self, paths, indexes):
        for i in range(self.colonies + 1):
            if i != self.colonies:
                idx = indexes[i][:self.lemon]
                path = paths[:, :, idx, i]
                self.update_pheromones(path, i)
            else:
                idx = indexes[i][0]
                path = paths[:, :, idx, i]
                self.update_pheromones(path, i, 1)

    def update_tau(self):
        for i in range(self.colonies + 1):
            n = self.customers
            avg = n / 2
            if i > self.balance:
                self.tau_max[:, i] = 1 / (self.special_rho * self.c_best[:, i].squeeze())
                self.tau_min[:, i] = (self.tau_max[:, i] * (1 - pow(self.p_best, 1 / n))) / (
                        (avg - 1) * pow(self.p_best, 1 / n))
            else:
                self.tau_max[:, i] = 1 / (self.rho * self.c_best[:, i].squeeze())
                self.tau_min[:, i] = (self.tau_max[:, i] * (1 - pow(0.95, 1 / n))) / (
                        (avg - 1) * pow(0.95, 1 / n))

    def get_colony_best_fitness(self, paths):
        colony_best_idx = np.zeros((1, paths.shape[3]))
        colony_best_fitness = np.zeros((1, paths.shape[3]))
        best_colony_route = np.zeros((self.drivers, self.customers + 3, paths.shape[3]), dtype=np.float64)
        best_list = []

        for colony in range(paths.shape[3]):
            summed = np.sum(paths[:, -1, :, colony], axis=0)
            smallest = np.argsort(summed)
            best_list.append(smallest)
            smallest = smallest[0]
            colony_best_idx[:, colony] = smallest
            best_colony_route[:, :, colony] = paths[:, :, smallest, colony]
            colony_best_fitness[:, colony] = summed[smallest]

        return colony_best_fitness, best_colony_route, best_list

    def exchange_information(self, best_paths, paths):
        self.exchanging = True
        aa = list(range(0, self.colonies))

        res = [(a, b) for idx, a in enumerate(aa) for b in aa[idx + 1:]]  # fully connected sharing
        second = [(self.colonies, idx) for idx in aa]
        res.extend(second)

        i, j = res[self.counter]
        if i != 0:  # i to j
            path = best_paths[:, :, i][:, :, np.newaxis]
            self.update_pheromones(path, j)
        if i != self.colonies:  # j to i
            path = best_paths[:, :, j][:, :, np.newaxis]
            self.update_pheromones(path, i)

        for k in range(self.colonies + 1):
            if k != i or k != j or k == self.colonies:
                path = paths[:, :, k]
                self.update_pheromones(path, k)

        self.exchanging = False

        self.counter += 1  # counter so next iter it migrates the next sol

        if self.counter >= len(res):
            self.counter = 0
            self.boost = True
            self.update_master_colony(self.c_best)
            self.trigger[:, 0] = 1
            print('Migration event has ended')

    def update_master_colony(self, colony_fitness):
        self.pheromones[:, :, -1] = 0
        rank = np.argsort(colony_fitness[:, :-1]).squeeze()
        num_colonies = len(rank)
        for i, col in enumerate(rank):
            if col != 0:
                a = self.pheromones_archive[:, :, col] - np.min(self.pheromones_archive[:, :, col])
                self.pheromones[:, :, -1] += (a * (((num_colonies - i) / num_colonies) ** self.gamma))
            else:
                a = self.pheromones_archive[:, :, col] - np.min(self.pheromones_archive[:, :, col])
                self.pheromones[:, :, -1] += (a * (((num_colonies - i) / num_colonies) ** 10))
        sum_ = np.sum(self.pheromones[:, :, -1])
        self.pheromones[:, :, -1] = self.pheromones[:, :, -1] / sum_

    def real_gen_path(self):
        limit_1 = 150
        limit_2 = 100
        limitz_1 = 150
        limitz_2 = 300
        all_paths1 = gen_paths(self.drivers, self.customers, self.pheromones[:, :, 0][:, :, np.newaxis], 1,
                               2.7
                               , self.rho, self.n_ants, self.demand, self.capacity,
                               self.distances, limit_1, limit_2, self.boost, True, True)
        all_paths2 = gen_paths(self.drivers, self.customers, self.pheromones[:, :, 1:-1], self.alpha, self.beta,
                               self.rho, self.n_ants, self.demand, self.capacity,
                               self.distances,
                               limit_1, limit_2, self.boost, True, True)
        all_paths3 = gen_paths(self.drivers, self.customers, self.pheromones[:, :, -1][:, :, np.newaxis], self.alpha_c,
                               self.beta_c,
                               self.rho, self.n_ants, self.demand, self.capacity,
                               self.distances, limitz_1, limitz_2, self.boost, True, True)
        all_paths = np.concatenate([all_paths1, all_paths2, all_paths3], axis=3)
        return all_paths

    def run(self):
        print('Started')
        saved_best = None
        for i in range(self.n_iterations):
            print(i)
            if i == self.initialise_master and i > 0:
                self.boost = True
                self.update_master_colony(self.c_best)
            if self.trigger[:, 0] != 0:
                self.trigger[:, 1] += 1
                if self.trigger[:, 1] >= 5:
                    self.trigger[:, :] = 0
                    self.boost = False
            if i % (self.initialise_master + 5) == 0 and i > 0:
                self.boost = False

            all_paths = self.real_gen_path()
            self.k_best, colony_routes, best_list = self.get_colony_best_fitness(all_paths)

            for j in range(self.colonies + 1):
                if self.k_best[:, j].item() < self.c_best[:, j].item():
                    self.c_best[:, j] = self.k_best[:, j]
                    self.s_best[:, :, j] = colony_routes[:, :, j]
                    self.updated[0, j] = 1
                    self.true_dist()
                    print(self.true_dis)
            if (i % self.exchange_rate == 0) and i > 0 or (
                    self.counter > 0 and (self.exchange_rate - i) % 2 == 0 and i > 0):
                if (i % self.exchange_rate == 0) and i > 0:
                    saved_best = self.s_best
                self.update_tau()
                self.exchange_information(saved_best, colony_routes)
            else:
                self.update_tau()
                self.update_all_colonies(all_paths, best_list)

        aaa = np.argsort(self.c_best.squeeze())[0]
        best_path = self.s_best[:,:, aaa].squeeze()
        return best_path




file = 'ACO/instances/A-n80-k10.vrp' # select the instance
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
    n_ants=300,
    n_iterations=600,
    alpha=1.4,
    beta=2.7,
    alpha_c=1.1,
    beta_c=0.89,
    gamma=2.6,  #5.2
    rho=0.65,
    rho_c=0.36,  # 34
    Q=1,
    colonies=8,
    exchange_rate=400,
    initial_pheromone=1.0,
    lemon=1,
    p_best=0.05,
    initialise_master=50,
    balance=0,
    optimal_result=None
)


run = aco.run()
