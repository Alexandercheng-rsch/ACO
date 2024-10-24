import vrplib
import numpy as np
import os
import time
from ACO_VRP_ISLAND import AntColony
from multiprocessing import Pool, cpu_count, Manager

instances = [
    'A-n33-k5', 'A-n33-k6', 'A-n46-k7', 'A-n48-k7', 'A-n55-k9', 'A-n60-k9', 'A-n63-k10', 'A-n69-k9', 'A-n80-k10'
]
base_path = '../axc1353/instances/'
results_path = 'results/'

optimal_results_map = {
    'A-n33-k5': 661,
    'A-n33-k6': 724,
    'A-n46-k7': 914,
    'A-n48-k7': 1073,
    'A-n55-k9': 1073,
    'A-n60-k9': 1354,
    'A-n63-k10': 1314,
    'A-n69-k9': 1159,
    'A-n80-k10': 1763
}


def run_aco_single(args):
    instance, optimal_result, run_number, shared_results = args
    file = f"{base_path}{instance}.vrp"
    re = vrplib.read_instance(file)
    num_vehicles = int(instance.split('-')[-1][1:])

    aco = AntColony(
        distances=re['edge_weight'],
        demand=re['demand'],
        capacity=re['capacity'],
        drivers=num_vehicles,
        n_ants=300,
        n_iterations=600,
        alpha=1.1,
        beta=2.7,
        alpha_c=1.1,
        beta_c=0.89,
        gamma=2.7,
        rho=0.58,
        rho_c=0.03,
        Q=1,
        colonies=1,
        exchange_rate=400,
        master_colony_update_rate=320,
        initial_pheromone=1.0,
        lemon=1,
        p_best=0.05,
        initialise_master=50,
        balance=0,
        optimal_result=optimal_result
    )

    start_time = time.time()
    best_fitness, best_route = aco.run()
    end_time = time.time()
    execution_time = end_time - start_time

    result = (best_fitness, best_route, execution_time)
    shared_results[run_number] = result

    save_partial_results(instance, shared_results)

    print(f'Completed {instance}, run {run_number + 1}/30')
    return result


def save_partial_results(instance, shared_results):
    partial_results = [r for r in shared_results if r is not None]
    if partial_results:
        best_results, best_routes, execution_times = zip(*partial_results)

        results_file = f"{results_path}{instance}_ACO_best_results_partial.npy"
        routes_file = f"{results_path}{instance}_ACO_best_routes_partial.npy"
        times_file = f"{results_path}{instance}_ACO_execution_times_partial.npy"

        np.save(results_file, best_results)
        np.save(routes_file, best_routes)
        np.save(times_file, execution_times)


def run_aco_parallel(instance, optimal_result):
    manager = Manager()
    shared_results = manager.list([None] * 30)

    with Pool(processes=3) as pool:
        results = pool.map(run_aco_single, [(instance, optimal_result, i, shared_results) for i in range(10)])

    best_results, best_routes, execution_times = zip(*results)
    return np.array(best_results), np.array(best_routes), np.array(execution_times)


def process_instance(instance):
    optimal_result = optimal_results_map.get(instance, None)
    if optimal_result is None:
        print(f"No optimal result found for {instance}. Skipping.")
        return

    print(f"Starting optimization for instance: {instance}")
    instance_start_time = time.time()

    best_results, best_routes, execution_times = run_aco_parallel(instance, optimal_result)

    instance_end_time = time.time()
    total_instance_time = instance_end_time - instance_start_time

    results_file = f"{results_path}{instance}_ACO_best_results.npy"
    routes_file = f"{results_path}{instance}_ACO_best_routes.npy"
    times_file = f"{results_path}{instance}_ACO_execution_times.npy"

    np.save(results_file, best_results)
    np.save(routes_file, best_routes)
    np.save(times_file, execution_times)

    print(f"Final results for {instance} saved as:")
    print(f"{results_file}")
    print(f"{routes_file}")
    print(f"{times_file}")
    print(f"Completed optimization for instance: {instance}")
    print(f"Total time for instance: {total_instance_time:.2f} seconds")
    print(f"Average time per run: {np.mean(execution_times):.2f} seconds")
    print(f"Best result: {np.min(best_results)}")
    print(f"Average result: {np.mean(best_results):.2f}")
    print()


def main():
    os.makedirs(results_path, exist_ok=True)
    for instance in instances:
        process_instance(instance)


if __name__ == '__main__':
    main()