import sys
sys.path.append('../')
from optm_algorithms.opt_ai_net import OptAiNet
from multiprocessing import Process, Manager
from fitness_function import RocketFitness, bound_values, fitness_func
import numpy as np
import warnings
from tqdm import tqdm
import json



rocket_fitness = RocketFitness(bound_values, num_workers=4)
random_values = np.random.rand(10,10)
fitness_func_class = rocket_fitness.calc_fitness

nc_list = np.array([2, 4, 8])
beta_list = np.array([10,100,1000])
clone_threshold_list = np.array([0.05, 0.15, 0.3])
supression_threshold_list = np.array([0.4, 0.8, 2])
newcomers_percentage_list = np.array([0.1, 0.2, 0.4])

# nc_list = np.array([4])
# beta_list = np.array([100])
# clone_threshold_list = np.array([0.05])
# supression_threshold_list = np.array([0.8])
# newcomers_percentage_list = np.array([0.2])

grid1, grid2, grid3, grid4, grid5 = np.meshgrid(nc_list, beta_list, clone_threshold_list, supression_threshold_list, newcomers_percentage_list)
combinations = np.vstack((grid1.ravel(), grid2.ravel(), grid3.ravel(), grid4.ravel(), grid5.ravel())).T

print(combinations)


def optimization_task(opt_ai_net, return_dict):
    best_solutions = opt_ai_net.fit()
    # Armazenar os resultados necessários em um dicionário para ser retornado
    return_dict['fitness_calls'] = opt_ai_net.fitness_calls_list
    return_dict['best_ind_list'] = opt_ai_net.best_ind_list
    return_dict['avg_ind_list'] = opt_ai_net.avg_ind_list
    return_dict['total_time'] = opt_ai_net.total_exec_time
    return_dict['best_solutions'] = best_solutions

def run_optimization(opt_ai_net, best_solutions_container, time_limit):
    manager = Manager()
    return_dict = manager.dict()
    
    process = Process(target=optimization_task, args=(opt_ai_net, return_dict))
    process.start()
    process.join(timeout=time_limit)
    
    if process.is_alive():
        process.terminate()
        process.join()
        print("Processo terminado devido a timeout. Possíveis subprocessos não foram encerrados corretamente.")
        return False
    else:
        # Transferir resultados do dicionário gerenciado para o contêiner fornecido
        best_solutions_container['fitness_calls'] = return_dict['fitness_calls']
        best_solutions_container['best_ind_list'] = return_dict['best_ind_list']
        best_solutions_container['avg_ind_list'] = return_dict['avg_ind_list']
        best_solutions_container['total_time'] = return_dict['total_time']
        best_solutions_container['best_solutions'] = return_dict['best_solutions']
        return True

def execute_sensitivity_analysis_optainet(combinations, filename):
    simulations_list = []
    for row in tqdm(combinations):
        
        nc = int(row[0])
        beta = row[1]
        clone_threshold = row[2]
        supression_threshold = row[3]
        newcomers_percentage = row[4]
        
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
                        limit_fitness_calls=np.inf
        )

        best_solutions_container = {}
        success = run_optimization(opt_ai_net, best_solutions_container, 500)

        if not success:
            dict_save = {
                'nc': nc,
                'beta': beta,
                'clone_threshold': clone_threshold,
                'supression_threshold': supression_threshold,
                'newcomers_percentage': newcomers_percentage,
                'fitness_calls': 0,
                'best_ind_list': 0,
                'avg_ind_list': 0,
                'best_solutions': 0,
                'total_time': 0,
            }
        else:
            dict_save = {
                'nc': nc,
                'beta': beta,
                'clone_threshold': clone_threshold,
                'supression_threshold': supression_threshold,
                'newcomers_percentage': newcomers_percentage,
                'fitness_calls': best_solutions_container['fitness_calls'].tolist(),
                'best_ind_list': best_solutions_container['best_ind_list'].tolist(),
                'avg_ind_list': best_solutions_container['avg_ind_list'].tolist(),
                'best_solutions': np.array(best_solutions_container['best_solutions']).tolist(),
                'total_time': best_solutions_container['total_time'],
            }

        simulations_list.append(dict_save)

        
    with open(filename, 'w') as fout:
        json.dump(simulations_list, fout)

if __name__ == '__main__':
    execute_sensitivity_analysis_optainet(combinations, 'simulations/optainet_new_sensitivity.json')
