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

# Remove os pontos correspondentes de 

# Treina o t-SNE para diferentes valores de perplexidade e salva cada um dos resultados
# em um arquivo diferente dentro da pasta ./embeddings

# Definir um conjunto de hiperparâmetros para testar
param_grid = {
    'perplexity': [5, 30, 50, 100],
    'n_iter': [1000, 5000],

}

# Testar todas as combinações de hiperparâmetros
for config in tqdm(ParameterGrid(param_grid)):
    tsne = TSNE(n_components=2, perplexity=config['perplexity'], n_iter=config['n_iter'],
                n_jobs=4, random_state=42)
    embedding = tsne.fit_transform(grid_points)

    file_name = f'./embeddings/tsne_perplexity_{config["perplexity"]}_n_inter{config["n_iter"]}.csv'
    np.savetxt(file_name, embedding, delimiter=',')

