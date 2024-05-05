import sys

sys.path.append('../')

from optm_algorithms.pso import PSO
from optm_algorithms.differential_evolution import DifferentialEvolutionAlgorithm
from optm_algorithms.depso import DEPSO
from optm_algorithms.cma_es import CMA_ES

from fitness_function import RocketFitness, bound_values, fitness_func
import numpy as np
import warnings
from tqdm import tqdm
import json

rocket_fitness = RocketFitness(bound_values, num_workers=4)
random_values = np.random.rand(10,10)
fitness_func_class = rocket_fitness.calc_fitness


#mi_list = [ 20,  60, 100, 140, 180, 220, 260, 300, 340, 380]
#sigma_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.55, 0.6]

mi_list = [1,2,3,4,5,10, 20, 100, 200, 300]
sigma_list = [0.2, 0.25, 0.3, 0.35, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]


grid1, grid2 = np.meshgrid(mi_list, sigma_list)
combinations = np.vstack((grid1.ravel(), grid2.ravel())).T

print(combinations)

def execute_sensitivity_analysis_cmaes(combinations, filename):
    
    simulations_list = []
    for row in tqdm(combinations):
        #try:
        mi = int(row[0])
        sigma = row[1] 

        print(f"mi: {mi}, sigma: {sigma}")
        
        cmaes = CMA_ES(
            num_epochs=50,
            lamb=4000,
            mi=mi,
            chrom_length=10,
            value_ranges=bound_values,
            fitness_func=fitness_func_class,
            eval_every=99,
            verbose=True,
            sigma=sigma,
            seed = 1,
        )
        best_solutions = cmaes.fit()

        dict_save = {'mi': mi,
                    'sigma': sigma,
                    'fitness_calls': cmaes.fitness_calls_list.tolist(),
                    'best_ind_list': cmaes.best_ind_list.tolist(),
                    'avg_ind_list': cmaes.avg_ind_list.tolist(),
                    'best_solutions': best_solutions.tolist(),
                    'total_time': cmaes.total_exec_time,
                    }
        simulations_list.append(dict_save)
        # except Exception as e:
        #     print(f"Erro -> {e}")
        #     dict_save = {'mi': mi,
        #                 'sigma': sigma,
        #                 'fitness_calls': "ERRO",
        #                 'best_ind_list': "ERRO",
        #                 'avg_ind_list': "ERRO",
        #                 'best_solutions': "ERRO",
        #                 'total_time': "ERRO",
        #                 }
        #     simulations_list.append(dict_save)
        
    with open(filename, 'w') as fout:
        json.dump(simulations_list, fout)
        ...

    return

if __name__ == '__main__':
    execute_sensitivity_analysis_cmaes(combinations, 'simulations/cmaes_sensitivity.json')
