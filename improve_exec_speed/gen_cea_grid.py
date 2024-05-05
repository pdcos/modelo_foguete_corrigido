# Importação de pacotes

from rocketcea.cea_obj_w_units import CEA_Obj
from proptools import nozzle  
import numpy as np
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import os
import multiprocessing
import pandas as pd


import joblib
import matplotlib.pyplot as plt


def calculate_cea_outputs(params):
    oxName='LOX'
    fuelName='RP-1'
    ceaObj = CEA_Obj( oxName=oxName, fuelName=fuelName, pressure_units='Pa', cstar_units='m/s', temperature_units='K')

    #min_mat = bound_values.T[0, :]
    #max_mat = bound_values.T[1,:]
    #denorm_values = params * (max_mat - min_mat) + min_mat
    
    IspVac, Cstar, Tc, mw, gamma = ceaObj.get_IvacCstrTc_ChmMwGam(Pc=params[0], MR=params[1], eps=params[2])
    IspSea = ceaObj.estimate_Ambient_Isp(Pc=params[0], MR=params[1], eps=params[2], Pamb=1e5)[0]

    valores_calculados = [IspSea, IspVac, Cstar, mw, Tc, gamma]
    return valores_calculados



def grid_search(N, grid_resolution):
    #N = 10  # Number of parameters
    #grid_resolution = 3  # Granularity of the grid

    param_ranges = [np.linspace(0, 1, grid_resolution) for _ in range(N)]
    grid_points = np.array(list(product(*param_ranges)))

    return grid_points


# Processamento paralelo
bound_values = np.array([[0.1e6, 12e6], [1.5, 3.5], [2, 200]])

def calculate_cea_params_single(params):
    oxName='LOX'
    fuelName='RP-1'
    ceaObj = CEA_Obj( oxName=oxName, fuelName=fuelName, pressure_units='Pa', cstar_units='m/s', temperature_units='K')

    #min_mat = bound_values.T[0, :]
    #max_mat = bound_values.T[1,:]
    #denorm_values = params * (max_mat - min_mat) + min_mat
    
    #IspVac, Cstar, Tc, mw, gamma = ceaObj.get_IvacCstrTc_ChmMwGam(Pc=params[0], MR=params[1], eps=params[2])
    #valores_calculados = [IspVac, Cstar, Tc, gamma]
    valores_calculados = params
    print(params)
    return valores_calculados


def calc_fitness(params_matrix, bound_values, fitness_func, parallel=True):
    min_mat = bound_values.T[0, :]
    max_mat = bound_values.T[1,:]
    denorm_matrix = params_matrix * (max_mat - min_mat) + min_mat

    if parallel:
        num_workers = os.cpu_count()
    else:
        num_workers = 1

    pool = multiprocessing.Pool(processes=num_workers)
    results = pool.map(fitness_func, denorm_matrix)
    pool.close()
    pool.join()
    results = np.array(results)
    #print(denorm_matrix)

    return results



class CalculaCEA():
    def __init__(self, bound_values, parallel=True):
        self.bound_values = bound_values

        self.min_mat = bound_values.T[0, :]
        self.max_mat = bound_values.T[1,:]

        if parallel:
            self.num_workers = os.cpu_count()
        else:
            self.num_workers = 1


    def calculate_cea_single(self, params):
        oxName='LOX'
        fuelName='RP-1'
        self.ceaObj = CEA_Obj( oxName=oxName, fuelName=fuelName, pressure_units='Pa', cstar_units='m/s', temperature_units='K')
        IspVac, Cstar, Tc, mw, gamma = self.ceaObj.get_IvacCstrTc_ChmMwGam(Pc=params[0], MR=params[1], eps=params[2])

        valores_calculados = [IspVac, Cstar, Tc, mw, gamma]
        return valores_calculados

    
    def calc_fitness(self, params_matrix):
        denorm_matrix = params_matrix * (self.max_mat - self.min_mat) + self.min_mat
        pool = multiprocessing.Pool(processes=self.num_workers)
        results = pool.map(calculate_cea_outputs, denorm_matrix)
        pool.close()
        pool.join()
        results = np.array(results)
        return results


def main():

    # Encontra o diretório em que o arquivo atual está:
    current_dir = os.path.dirname(os.path.realpath(__file__))

    grid_points = grid_search(N=3, grid_resolution=100)
    # Printa o número total de dados a serem calculados
    print('Total number of data points: ', grid_points.shape)
    cea_fitness = CalculaCEA(bound_values, parallel=True)
    y = cea_fitness.calc_fitness(grid_points)

    df_grid_points = pd.DataFrame(grid_points, columns=['P_c', 'MR', 'eps'])
    df_target = pd.DataFrame(y, columns=['IspSea','IspVac','Cstar', 'Tc', 'mw', 'gamma', ])
    df_grid_target = pd.concat([df_grid_points, df_target], axis=1)
    # salva os dados num csv dentro da pasta data no diretório atual
    df_grid_target.to_csv( current_dir + '/data/grid_target.csv', index=False)
    return

if __name__ == '__main__':
    main()  