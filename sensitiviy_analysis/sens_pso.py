import sys
sys.path.append('../')
from optm_algorithms.pso import PSO
from optm_algorithms.differential_evolution import DifferentialEvolutionAlgorithm
from optm_algorithms.depso import DEPSO

from fitness_function import RocketFitness, bound_values, fitness_func
import numpy as np
import warnings
from tqdm import tqdm
import json

# Inicializando função de fitness
rocket_fitness = RocketFitness(bound_values, num_workers=4)
random_values = np.random.rand(10,10)
fitness_func_class = rocket_fitness.calc_fitness

global_factor_list = np.linspace(1, 6, 5)
local_factor_list = np.linspace(1, 6, 5)
v_max = np.linspace(1,10,5)
# Printa hiperparâmetros
print("Global factor list: ", global_factor_list)
print("Local factor list: ", local_factor_list)
print("V_max: ", v_max)

grid1, grid2, grid3 = np.meshgrid(global_factor_list, local_factor_list, v_max)

combinations = np.vstack((grid1.ravel(), grid2.ravel(), grid3.ravel())).T
sum_of_columns = combinations[:, 0] + combinations[:, 1]

# Use logical indexing to select rows where the sum of the first and second columns is greater than 4
combinations = combinations[sum_of_columns > 4]

def execute_sensitivity_analysis_pso(combinations, filename, seed=42):
    
    #combinations = [[2.05, 2.05, 1],[2.05, 2.05, 1]]

    simulations_list = []
    for row in tqdm(combinations):
        
        global_factor = row[0]
        local_factor = row[1]
        v_max = row[2]
        
        pso_alg = PSO(
            num_epochs=100,
            pop_size=1000,
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

        best_solutions = pso_alg.fit()

        dict_save = {'global_factor': global_factor,
                     'local_factor': local_factor,
                     'v_max': v_max,
                     'fitness_calls': pso_alg.fitness_calls_list.tolist(),
                     'best_ind_list': pso_alg.best_ind_list.tolist(),
                     'avg_ind_list': pso_alg.avg_ind_list.tolist(),
                     'best_solutions': best_solutions.tolist(),
                     'total_time': pso_alg.total_exec_time,
                     }
        simulations_list.append(dict_save)
        
    with open(filename, 'w') as fout:
        json.dump(simulations_list, fout)

    return
        
if __name__ == '__main__':
    execute_sensitivity_analysis_pso(combinations, 'simulations/pso_sensitivity_red.json')