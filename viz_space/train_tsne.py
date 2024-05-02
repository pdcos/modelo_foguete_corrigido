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

