import sys
# Faz este append para poder importar o fitness_function que está na pasta acima
sys.path.append('../')
from fitness_function import RocketFitness, bound_values, fitness_func
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from itertools import product
from tqdm import tqdm

rocket_fitness = RocketFitness(bound_values, num_workers=4)
random_values = np.random.rand(100,10)
fitness_func_class = rocket_fitness.calc_fitness

def grid_search_rocket(fitness_fiunction):
    N = 10  # Number of parameters
    grid_resolution = 5  # Granularity of the grid

    param_ranges = [np.linspace(0, 1, grid_resolution) for _ in range(N)]
    grid_points = list(product(*param_ranges))

    return grid_points


grid_points = grid_search_rocket(fitness_func_class)

grid_points = np.array(grid_points)

fitness_scores = fitness_func_class(grid_points)

# Faz o merge dos pontos com os scores para armazenar em um dataframe e salvar, para não precisar rodar de novo
grid_points_scores = np.hstack((grid_points, fitness_scores.reshape(-1,1)))

# Salva o grid search
np.savetxt('grid_search_rocket.csv', grid_points_scores, delimiter=',')