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

        mi = 10
        sigma = 0.6

        print(f"mi: {mi}, sigma: {sigma}")
        
        cmaes = CMA_ES(
            num_epochs=50,
            lamb=2000,
            mi=mi,
            chrom_length=10,
            value_ranges=bound_values,
            fitness_func=fitness_func_class,
            eval_every=99,
            verbose=True,
            sigma=sigma,
            seed = seed,
        )
        best_solutions = cmaes.fit()

        dict_save = {'mi': mi,
                    'sigma': sigma,
                    'fitness_calls': cmaes.fitness_calls_list.tolist(),
                    'best_ind_list': cmaes.best_ind_list.tolist(),
                    'avg_ind_list': cmaes.avg_ind_list.tolist(),
                    'best_solutions': best_solutions.tolist(),
                    'total_time': cmaes.total_exec_time,
                    'exec_time_list': cmaes.exec_time_list.tolist(),
                    }
        
        results.append(dict_save)
    
    # salva os resultados em um json
    with open(path, 'w') as f:
        json.dump(results, f)   
    return

path = './results/cmaes_boxplot.json'


execute_boxplot(path, 20)
