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

def execute_boxplot(alg, path, n_exec, params_dict):
    # cria uma lista de sementes sequenciais com base no número de execuções
    seed_list = np.arange(0, n_exec)
    # faz uma lista com n_exec cópias independentes do objeto alg
    alg_list = [alg.copy() for _ in range(n_exec)]
    
    # cria uma lista para armazenar os resultados
    results = []

    # itera n_exec vezes
    for seed, alg in tqdm(seed_list, alg_list):
        # define a semente
        np.random.seed(seed)
        alg.seed = seed
        # reseta o objeto alg toda iteraçao
        alg.reset()

        best_solutions =  alg.fit()
        
        dict_result = {
            "seed": seed,
            "best_solutions": best_solutions.tolist(),
            "total_time": alg.total_exec_time,
            'avg_ind_list': alg.avg_ind_list.tolist(),
            'best_ind_list': alg.best_ind_list.tolist(),
            'fitness_calls': alg.fitness_calls_list.tolist(),
        }

        for key, value in params_dict.items():
            dict_result[key] = value
        
        results.append(dict_result)
    
    # salva os resultados em um json
    with open(path, 'w') as f:
        json.dump(results, f)   
    return

