import sys
sys.path.append('../')
from fitness_function import RocketFitness, bound_values, fitness_func
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import umap.umap_ as umap  # Importação corrigida
from itertools import product
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

rocket_fitness = RocketFitness(bound_values, num_workers=4)
random_values = np.random.rand(100,10)
fitness_func_class = rocket_fitness.calc_fitness

# Carrega o grid search
grid_points_scores = np.loadtxt('grid_search_rocket.csv', delimiter=',')
grid_points = grid_points_scores[:, :-1]
fitness_scores = grid_points_scores[:, -1]

# Deleta o grid_points_scores para liberar memória
del grid_points_scores

# Remove aleatoriamente pontos de fitness = 0.5 e fitness = 0 até que tenham apenas 1.000.000 pontos
np.random.seed(42)
zero_points = np.where(fitness_scores == 0)[0]
zero_points = np.random.choice(zero_points, zero_points.shape[0] - 500000, replace=False)
print(zero_points.shape)

half_points = np.where(fitness_scores == 0.5)[0]
half_points = np.random.choice(half_points, half_points.shape[0] - 500000, replace=False)
print(half_points.shape)

points_to_remove = np.concatenate((zero_points, half_points))
grid_points = np.delete(grid_points, points_to_remove, axis=0)
fitness_scores = np.delete(fitness_scores, points_to_remove, axis=0)
print(grid_points.shape)


# Definir um conjunto de hiperparâmetros para testar no UMAP
param_grid = {
    'n_neighbors': [5, 15, 30],
    'min_dist': [0.1, 0.25, 0.5],
    'metric': ['euclidean']
}

# Testar todas as combinações de hiperparâmetros
for config in tqdm(ParameterGrid(param_grid)):
    umap_model = umap.UMAP(n_components=2, n_neighbors=config['n_neighbors'],
                           min_dist=config['min_dist'], metric=config['metric'],
                           random_state=42)
    embedding = umap_model.fit_transform(grid_points)

    file_name = f'./embeddings/umap_n_neighbors_{config["n_neighbors"]}_min_dist_{config["min_dist"]}_metric_{config["metric"]}.csv'
    np.savetxt(file_name, embedding, delimiter=',')
