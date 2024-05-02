import sys
sys.path.append('../')
from optm_algorithms.pso import PSO
from optm_algorithms.differential_evolution import DifferentialEvolutionAlgorithm

from fitness_function import RocketFitness, bound_values, fitness_func
import numpy as np
import warnings
from tqdm import tqdm
import json


# Inicializando função de fitness
rocket_fitness = RocketFitness(bound_values, num_workers=4)
random_values = np.random.rand(10,10)
fitness_func_class = rocket_fitness.calc_fitness

mutation_rate = np.linspace(0.1, 1, 10)
crossover_rate = np.linspace(0.1, 1,10)
print("Mutation rate: ", mutation_rate)
print("Crossover rate: ", crossover_rate)
grid1, grid2 = np.meshgrid(mutation_rate, crossover_rate)
combinations = np.vstack((grid1.ravel(), grid2.ravel())).T



def execute_sensitivity_analysis_de(combinations, filename, seed=42):
    
    #combinations = [[0.5, 0,5]]

    simulations_list = []
    for row in tqdm(combinations):
        
        mutation_rate = row[0]
        crossover_rate = row[1]
        
        de_alg = DifferentialEvolutionAlgorithm(
                                            num_epochs=100,
                                            pop_size=1000,
                                            chrom_length=10,
                                            value_ranges=bound_values,
                                            mutation_rate=mutation_rate,
                                            crossover_rate=crossover_rate,
                                            fitness_func=fitness_func_class,
                                            verbose=True,
                                            eval_every=10,
                                            seed=seed
                                            )

        best_solutions = de_alg.fit()

        dict_save = {'mutation_rate': mutation_rate,
                     'crossover_rate': crossover_rate,
                     'fitness_calls': de_alg.fitness_calls_list.tolist(),
                     'best_ind_list': de_alg.best_ind_list.tolist(),
                     'avg_ind_list': de_alg.avg_ind_list.tolist(),
                     'best_solutions': best_solutions.tolist(),
                     'total_time': de_alg.total_exec_time,
                     }
        simulations_list.append(dict_save)
        
    with open(filename, 'w') as fout:
        json.dump(simulations_list, fout)

    return
                
if __name__ == '__main__':
    execute_sensitivity_analysis_de(combinations, 'simulations/de_sensitivity.json')