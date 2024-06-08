import numpy as np
from model.build_rocket import RocketModel
import sys, os
import math
import concurrent.futures
import timeit
import multiprocessing
from functools import partial
import joblib
import time
from rocketcea.cea_obj_w_units import CEA_Obj



#### GLOBAL PARAMETERS ####
#reg_path = '/Users/pdcos/Documents/Estudos/Mestrado/Tese/Implementação da Tese do Jentzsch/rocket_optimization_implementation/model/engines/decision_tree_model.pkl'
#reg_path = '/home/ubuntu/Mestrado/modelo_foguete/model/engines/decision_tree_model.pkl'
reg_path = '/home/ubuntu/Mestrado/modelo_foguete_corrigido/improve_exec_speed/data/DecisionTreeRegressor_score_1.0.joblib'

reg_model = joblib.load(reg_path)
#reg_model = False
cea_obj = ceaObj = CEA_Obj( oxName='LOX', fuelName='RP-1', pressure_units='Pa', cstar_units='m/s', temperature_units='K')



# bound_values = np.array([[0.1e6, 12e6], [1.5, 3.5], [0.2, 0.3], [2, 200],
#                 [0.1e6, 12e6], [1.5, 3.5], [0.2, 0.3], [2, 200],
#                 [1, 6],
#                 [1, 6]
#                 ])


bound_values = np.array([[7e6, 12e6], [1.5, 2.5], [0.2, 0.3], [30, 200],
                [7e6, 12e6], [1.5, 2.5], [0.2, 0.3], [2, 70],
                [1.5, 4.5],
                [1.5, 4.5]
                ])


verbose=False   

###########################




def fitness_func(parameters_list):
    #parameters_list = denormalize(parameters_list, bounds)
    engineParams = {"oxName": "LOX",
                    "fuelName": "RP-1",
                    "combPressure": parameters_list[0],
                    "MR": parameters_list[1],
                    "nozzleDiam": parameters_list[2],
                    "eps": parameters_list[3]}

    engineParamsFirst = {"oxName": "LOX",
                    "fuelName": "RP-1",
                    "combPressure": parameters_list[4],
                    "MR": parameters_list[5],
                    "nozzleDiam": parameters_list[6],
                    "eps": parameters_list[7]}

    upperStageStructureParams = {"oxName": "LOX",
                                 "fuelName": "RP1",
                                 "MR": parameters_list[1],
                                 "tankPressure": 0.1,
                                 "radius": parameters_list[8],
                                } # 0 porque ainda nao temos esse valor
    firstStageStructureParams = {"oxName": "LOX",
                                "fuelName": "RP1",
                                "MR": parameters_list[5],
                                "tankPressure": 0.1,
                                "radius": parameters_list[9],
                            } # 0 porque ainda nao temos esse valor
    payloadBayParams = {"payloadHeight": 6.7,
                    "payloadRadius": 4.6/2,
                    "payloadMass": 4850,
                    "lowerStageRadius": parameters_list[8],
                    "lowerRocketSurfaceArea": 0} # 0 porque ainda nao temos esse valor

    rocket_model = RocketModel(upperEngineParams=engineParams,
                               firstEngineParams=engineParamsFirst,
                               payloadBayParams=payloadBayParams,
                               upperStageStructureParams=upperStageStructureParams,
                               firstStageStructureParams = firstStageStructureParams,
                               deltaV_upperStage=9000,
                               deltaV_landing=2000,
                               deltaV_firstStage=3000,
                               nEnginesUpperStage=1,
                               nEnignesFirstStage=9,
                               reg_model=reg_model,
                               cea_obj=cea_obj,
                               bound_values=bound_values)

    try:
        rocket_model.build_all()
        glow = rocket_model.glow
        if verbose:
            rocket_model.print_all_parameters()

    except:
        return -1
    
    
    
    neg_value = 0
    if math.isnan(glow):
        return -1
    if math.isnan(rocket_model.m_0_1):
        return -1
    if math.isnan(rocket_model.m_0_2):
        return -1
    if rocket_model.diff_raio_inf<=0:
        neg_value += rocket_model.diff_raio_inf
    if rocket_model.diff_raio_sup <= 0:
        neg_value += rocket_model.diff_raio_sup
    if rocket_model.m_0_1 <= 0:
        neg_value += rocket_model.m_0_1
    if rocket_model.m_0_2 <= 0:
        neg_value += rocket_model.m_0_2
    if rocket_model.m_p_1 <= 0:
        neg_value += rocket_model.m_p_1
    if rocket_model.m_p_2 <= 0:
        neg_value += rocket_model.m_p_2
    if rocket_model.upperStageStructure.oxTankCylHeight < 0:
        neg_value += rocket_model.upperStageStructure.oxTankCylHeight
    if rocket_model.upperStageStructure.fuelTankCylHeight < 0:
        neg_value += rocket_model.upperStageStructure.fuelTankCylHeight
    if rocket_model.firstStageStructure.oxTankCylHeight < 0:
        neg_value += rocket_model.firstStageStructure.oxTankCylHeight
    if rocket_model.firstStageStructure.fuelTankCylHeight  < 0:
        neg_value += rocket_model.firstStageStructure.fuelTankCylHeight
    if rocket_model.upperStageEngine.thrustVac * 1<= 0.8 * rocket_model.m_0_2 * 9.81:
        return 0
    if rocket_model.firstStageEngine.thrustSea * 9 <= 1.3 * rocket_model.glow * 9.81: # Multiplicar pelo número de motores (9)
        return 0
    # Caso o estágio inferior seja menor que o superior retorna 0
    if rocket_model.firstStageStructure.radius < rocket_model.upperStageStructure.radius:
        return 0
    if neg_value < 0:
        if neg_value > -100:
            return neg_value/100
        else:
            return -1
    fitness = 1.0/glow * 100000
    if math.isnan(fitness):
        return -1 
    return fitness

class RocketFitness():
    def __init__(self, bound_values, num_workers=1):
        self.best_fitness = 1
        self.worst_fitness = -1

        self.lowest_glow = 0

        self.bound_values = bound_values

        self.min_mat = bound_values.T[0, :]
        self.max_mat = bound_values.T[1,:]

        self.num_workers = num_workers  




    def calc_fitness(self, params_matrix, values_ranges=False):
        pop = params_matrix * (self.max_mat - self.min_mat) + self.min_mat
        # with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
        #     futures = [executor.submit(fitness_func, x) for x in pop]
        #     results = np.array([future.result() for future in concurrent.futures.as_completed(futures)])
        #partial_fitness_function = partial(fitness_func, verbose=verbose, reg_model=reg_model)
        pool = multiprocessing.Pool(processes=self.num_workers)
        results = pool.map(fitness_func, pop)
        pool.close()
        pool.join()

        
        results = np.array(results)
        # normalizacao
        results = (results + 1)/2         
        # best_curr_fitness = results.max()
        # worst_curr_fitness = results.min()

        # if best_curr_fitness > self.best_fitness:
        #     self.best_fitness = best_curr_fitness
        #     self.best_ind = params_matrix[results.argmax()]
        #     self.best_ind_denorm = pop[results.argmax()]
        # if worst_curr_fitness < self.worst_fitness:
        #     self.worst_fitness = worst_curr_fitness
        #     self.worst_ind = params_matrix[results.argmin()]
        #     self.worst_ind_denorm = pop[results.argmax()]

        return results
    
    def calc_fit_sequential(self, params_matrix):
        pop = params_matrix * (self.max_mat - self.min_mat) + self.min_mat
        results = []
        for x in pop:
            results.append(fitness_func(x))
        results = np.array(results)
        results = (results + 1)/2
        return results
        


if __name__ == "__main__":
    import joblib
    np.random.seed(1)
    params_list = np.array([9.7 * 1e6, 2.9, 0.23125, 165, 
                9.7 * 1e6, 2.9, 0.23125, 16,
                2,
                2.6])
    
    #print(params_list)
    #reg_path = '/Users/pdcos/Documents/Estudos/Mestrado/Tese/Implementação da Tese do Jentzsch/rocket_optimization_implementation/model/engines/decision_tree_model.pkl'
    #reg_model = joblib.load(reg_path)
    #reg_model = False
    #fit = fitness_func(params_list)
    #print(fit)



    # fit_class = RocketFitness(bound_values, num_workers=4)
    # np.random.seed(1)
    # random_values = np.random.rand(1,10)
    # #random_values = np.array([random_values[6]])
    # start_time = time.time()
    # x = fit_class.calc_fitness(random_values)
    # #x = fitness_func(random_values[0])
    # print("--- %s seconds ---" % (time.time() - start_time))
    # #print(x)


    fit_class = RocketFitness(bound_values, num_workers=4)
    np.random.seed(20)
    random_values = np.random.rand(1000,10)
    # start_time = time.time()
    #x = fit_class.calc_fit_sequential(random_values)
    #print(x)
    x = fit_class.calc_fitness(random_values)
    print(x.sum())
    print(x.max())
    # #x = fitness_func(random_values[0])
    # print("--- %s seconds ---" % (time.time() - start_time))
    # #print(x)


