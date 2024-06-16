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

rocket_fitness = RocketFitness(bound_values, num_workers=4)
random_values = np.random.rand(10,10)
fitness_func_class = rocket_fitness.calc_fitness

# global_factor_list = np.linspace(1.05, 3.05, 3)
# local_factor_list = np.linspace(1.05, 3.05, 3)
# v_max = np.linspace(1,8,3)

# mutation_rate = np.linspace(0.2, 0.9, 3)
# crossover_rate = np.linspace(0.2, 1, 3)

global_factor_list = np.array([1.0, 2.25, 3.5])
local_factor_list = np.array([1.0, 2.25, 3.5])
v_max = np.array([5.5, 7.75, 10])

mutation_rate = np.array([0.2, 0.4, 0.6, 0.8])
crossover_rate = np.array([0.5, 0.7, 0.9, 1])

grid1, grid2, grid3, grid4, grid5 = np.meshgrid(global_factor_list, local_factor_list, v_max, mutation_rate, crossover_rate)

combinations = np.vstack((grid1.ravel(), grid2.ravel(), grid3.ravel(), grid4.ravel(), grid5.ravel())).T
sum_of_columns = combinations[:, 0] + combinations[:, 1]

# Use logical indexing to select rows where the sum of the first and second columns is greater than 4
combinations = combinations[sum_of_columns > 4]

def execute_sensitivity_analysis_depso(combinations, filename):
    
    simulations_list = []
    for row in tqdm(combinations):
        
        global_factor = row[0]
        local_factor = row[1]
        v_max = row[2]
        mutation_rate = row[3]
        crossover_rate = row[4]
        
        depso = DEPSO(
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
            crossover_rate = crossover_rate,
            mutation_rate = mutation_rate,
            seed=1
            )

        best_solutions = depso.fit()

        dict_save = {'global_factor': global_factor,
                     'local_factor': local_factor,
                     'v_max': v_max,
                     'mutation_rate': mutation_rate,
                     'crossover_rate': crossover_rate,
                     'fitness_calls': depso.fitness_calls_list.tolist(),
                     'best_ind_list': depso.best_ind_list.tolist(),
                     'avg_ind_list': depso.avg_ind_list.tolist(),
                     'best_solutions': best_solutions.tolist(),
                     'total_time': depso.total_exec_time,
                     }
        simulations_list.append(dict_save)
        
    with open(filename, 'w') as fout:
        json.dump(simulations_list, fout)

    return


if __name__ == '__main__':
    execute_sensitivity_analysis_depso(combinations, 'simulations/depso_sensitivity_red.json')
    