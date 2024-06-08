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

        nc = int(4)
        beta = 10
        clone_threshold = 0.3
        supression_threshold = 0.8
        newcomers_percentage = 0.4
        
        opt_ai_net = OptAiNet( 
                        num_epochs=50,
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
                        #limit_fitness_calls= 1000 * 2 * 100
                        limit_fitness_calls = np.inf,
                        seed=seed * 2
        )
        best_solutions = opt_ai_net.fit()
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
            'total_time': opt_ai_net.total_exec_time,
            'exec_time_list': opt_ai_net.exec_time_list.tolist(),
        }

        results.append(dict_save)
    
    # salva os resultados em um json
    with open(path, 'w') as f:
        json.dump(results, f)   
    return

path = './results/optainet_boxplot.json'


execute_boxplot(path, 20)
