import sys
sys.path.append('../')
from optm_algorithms.pso import PSO
from optm_algorithms.differential_evolution import DifferentialEvolutionAlgorithm
from optm_algorithms.depso import DEPSO
from optm_algorithms.cma_es import CMA_ES
from optm_algorithms.opt_ai_net import OptAiNet


from fitness_function import RocketFitness, bound_values, fitness_func
import numpy as np
import warnings
from tqdm import tqdm
import json

# Inicializando função de fitness
rocket_fitness = RocketFitness(bound_values, num_workers=4)
random_values = np.random.rand(10,10)
fitness_func_class = rocket_fitness.calc_fitness



def execute_boxplot(path, n_exec):
    # cria uma lista de sementes sequenciais com base no número de execuções
    seed_list = np.arange(0, n_exec)
    
    # cria uma lista para armazenar os resultados
    results = []

    # itera n_exec vezes
    for seed in tqdm(seed_list):


        global_factor = 2.25
        local_factor = 2.25
        v_max = 7.75


        pso_alg = PSO(
            num_epochs=50,
            pop_size=2000,
            chrom_length=10,
            n_best=2,
            global_factor = global_factor,
            local_factor = local_factor,
            speed_factor = 1,
            v_max=v_max,
            value_ranges=bound_values,
            fitness_func=fitness_func_class,
            neighborhood_mode='self',
            verbose=True,
            eval_every=10,
            seed=seed
            )

        best_solutions =  pso_alg.fit()
        
        dict_save = {'global_factor': global_factor,
                     'local_factor': local_factor,
                     'v_max': v_max,
                     'fitness_calls': pso_alg.fitness_calls_list.tolist(),
                     'best_ind_list': pso_alg.best_ind_list.tolist(),
                     'avg_ind_list': pso_alg.avg_ind_list.tolist(),
                     'best_solutions': best_solutions.tolist(),
                     'total_time': pso_alg.total_exec_time,
                     'exec_time_list': pso_alg.exec_time_list.tolist(),
                     }


        results.append(dict_save)
    
    # salva os resultados em um json
    with open(path, 'w') as f:
        json.dump(results, f)   
    return

path = './results/pso_boxplot.json'


execute_boxplot(path, 20)
