import sys
sys.path.append('../')

import time
import multiprocessing
import json
import numpy as np
from optm_algorithms.opt_ai_net import OptAiNet
from fitness_function import RocketFitness, bound_values, fitness_func
from tqdm import tqdm

# Inicializando função de fitness
rocket_fitness = RocketFitness(bound_values, num_workers=4)
random_values = np.random.rand(10, 10)
fitness_func_class = rocket_fitness.calc_fitness

def execute_boxplot(path, n_exec):
    # cria uma lista de sementes sequenciais com base no número de execuções
    seed_list = np.arange(0, n_exec)
    
    # cria uma lista para armazenar os resultados
    results = []

    def run_opt_ai_net(seed, return_dict):
        nc = int(4)
        beta = 10.0
        clone_threshold = 0.05
        supression_threshold = 2.0
        newcomers_percentage = 0.1

        def save_partial_results(opt_ai_net_instance):
            dict_save = {
                'nc': nc,
                'beta': beta,
                'clone_threshold': clone_threshold,
                'supression_threshold': supression_threshold,
                'newcomers_percentage': newcomers_percentage,
                'fitness_calls': opt_ai_net_instance.fitness_calls_list.tolist(),
                'best_ind_list': opt_ai_net_instance.best_ind_list.tolist(),
                'avg_ind_list': opt_ai_net_instance.avg_ind_list.tolist(),
                'exec_time_list': opt_ai_net_instance.exec_time_list.tolist(),
            }
            return_dict.update(dict_save)

        opt_ai_net = OptAiNet(
            num_epochs=100,
            pop_size=20,
            Nc=nc,
            chrom_length=10,
            clone_threshold=clone_threshold,
            supression_threshold=supression_threshold,
            newcomers_percentage=newcomers_percentage,
            beta=beta,
            value_ranges=bound_values,
            fitness_func=fitness_func_class,
            verbose=True,
            eval_every=10,
            limit_fitness_calls=np.inf,
            seed=seed,
            callback=save_partial_results
        )

        try:
            start_time = time.time()
            best_solutions = opt_ai_net.fit()
            total_time = time.time() - start_time

            dict_save = {
                'nc': nc,
                'beta': beta,
                'clone_threshold': clone_threshold,
                'supression_threshold': supression_threshold,
                'newcomers_percentage': newcomers_percentage,
                'fitness_calls': opt_ai_net.fitness_calls_list.tolist(),
                'best_ind_list': opt_ai_net.best_ind_list.tolist(),
                'avg_ind_list': opt_ai_net.avg_ind_list.tolist(),
                'best_solutions': np.array(best_solutions).tolist(),
                'total_time': total_time,
                'exec_time_list': opt_ai_net.exec_time_list.tolist(),
            }
            return_dict.update(dict_save)
        except Exception as e:
            # Handle timeout or any other exception
            total_time = time.time() - start_time
            dict_save = {
                'nc': nc,
                'beta': beta,
                'clone_threshold': clone_threshold,
                'supression_threshold': supression_threshold,
                'newcomers_percentage': newcomers_percentage,
                'fitness_calls': opt_ai_net.fitness_calls_list.tolist(),
                'best_ind_list': opt_ai_net.best_ind_list.tolist(),
                'avg_ind_list': opt_ai_net.avg_ind_list.tolist(),
                'best_solutions': [],
                'total_time': total_time,
                'exec_time_list': opt_ai_net.exec_time_list.tolist(),
                'error': str(e)
            }
            return_dict.update(dict_save)

    # itera n_exec vezes
    for seed in tqdm(seed_list):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        process = multiprocessing.Process(target=run_opt_ai_net, args=(seed, return_dict))
        process.start()
        process.join(timeout=2)
        if process.is_alive():
            print(f"Execução com seed {seed} excedeu o limite de tempo e foi abortada.")
            process.terminate()
            process.join()
        results.append(return_dict.copy())

    # salva os resultados em um json
    with open(path, 'w') as f:
        json.dump(results, f)
    return

path = './results/optainet_boxplot_red.json'

execute_boxplot(path, 20)
