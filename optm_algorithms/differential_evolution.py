import numpy as np
import random
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from tqdm import tqdm
import time

class DifferentialEvolutionAlgorithm():
    def __init__(self,
                 num_epochs:int,
                 pop_size:int,
                 chrom_length:int,
                 value_ranges:list,
                 mutation_rate:float,
                 fitness_func, # Function Type,
                 crossover_rate = 0.8,
                 seed=42,
                 eval_every=100,
                 verbose = 0,
                ):
        
        self.num_epochs = num_epochs
        self.pop_size = pop_size
        self.chrom_length = chrom_length
        self.value_ranges = np.array(value_ranges)
        self.mutation_rate = mutation_rate
        self.fitness_func = fitness_func
        self.seed = seed    
        self.crossover_rate = crossover_rate
        self.best_ind_list = np.zeros(self.num_epochs)
        self.avg_ind_list = np.zeros(self.num_epochs)
        self.eval_every = eval_every
        self.verbose = verbose
        self.best_solution_fitness = 0
        self.best_solution = 0

        self.fitness_calls_counter = 0
        self.fitness_calls_list = np.zeros(self.num_epochs)

        # Inicializa lista de tempos de execução
        self.exec_time_list = np.zeros(self.num_epochs)

        np.random.seed(seed=seed)

    def init_pop(self):
        """
        Initializes a matrix with random values from an uniform distribution
        """
        self.x_g = np.random.rand(self.pop_size, self.chrom_length)
        # Denormalization process
        min_mat = self.value_ranges.T[0, :]
        max_mat = self.value_ranges.T[1,:]
        #self.x_g = self.x_g * (max_mat - min_mat) + min_mat
        return

    def mutation(self):
        mutation_ind_indices_1 = np.random.randint(low=0, high=self.pop_size, size=self.pop_size)
        mutation_ind_indices_2 = np.random.randint(low=0, high=self.pop_size, size=self.pop_size)
        self.v_g = self.x_g + self.mutation_rate * \
              (self.x_g[mutation_ind_indices_1] - self.x_g[mutation_ind_indices_2])
        mask = self.v_g > 1
        self.v_g[mask] = 1
        mask = self.v_g < 0
        self.v_g[mask] = 0
        return
                
    def crossover(self):
        crossover_prob = np.random.rand(self.pop_size, self.chrom_length)
        aleat_index = np.random.randint(low=0, high=self.chrom_length, size=self.pop_size)
        #aleat_index_ohe = np.zeros((aleat_index.size, aleat_index.max() + 1))
        aleat_index_ohe = np.full((aleat_index.size, self.chrom_length), fill_value=False, dtype=bool)
        aleat_index_ohe[np.arange(aleat_index.size, dtype=int), aleat_index] = True
        self.u_g = self.x_g.copy()
        self.u_g[crossover_prob >= self.crossover_rate] = self.v_g[crossover_prob >= self.crossover_rate]
        self.u_g[aleat_index_ohe] = self.v_g[aleat_index_ohe]
        return

    def selection(self):
        self.fitness_x_g = self.fitness_func(self.x_g, self.value_ranges)
        self.fitness_u_g = self.fitness_func(self.u_g, self.value_ranges)
        self.fitness_calls_counter += 2

        replacement_indices = self.fitness_u_g > self.fitness_x_g
        self.x_g[replacement_indices] = self.u_g[replacement_indices]
        self.fitness_x_g[replacement_indices] = self.fitness_u_g[replacement_indices]

        curr_max_fitness = self.fitness_x_g.max()
        if self.best_solution_fitness < curr_max_fitness:
            self.best_solution = self.x_g[self.fitness_x_g.argmax()]
            self.best_solution_fitness = curr_max_fitness
        return

    def callback(self):
        max_val = np.max(self.fitness_x_g)
        mean_val = np.mean(self.fitness_x_g)
        self.best_ind_list[self.curr_epoch] = max_val
        self.avg_ind_list[self.curr_epoch] = mean_val
        self.fitness_calls_list[self.curr_epoch] = self.fitness_calls_counter
        if (self.curr_epoch % self.eval_every == 0) and self.verbose != 0 :
            print(f"Epoch {self.curr_epoch}: Best: {max_val}, Average: {mean_val}")
    
    def fit(self):
        start_time = time.time()
        self.init_pop()
        for epoch in tqdm(range(self.num_epochs)):
            self.curr_epoch = epoch
            self.mutation()
            self.crossover()
            self.selection()
            self.callback()
            # Atualiza tempo de execução
            exec_time = time.time() - start_time
            self.exec_time_list[epoch] = exec_time
        self.total_exec_time = time.time() - start_time
        print("--- %s seconds ---" % (self.total_exec_time))
        return self.x_g

    def plot(self):
        plt.plot(self.best_ind_list, label="Best")
        plt.plot(self.avg_ind_list, label="Average")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Fitness Value")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    def schaffer_function(mat_x_y):
        x = mat_x_y[:, 0]
        y = mat_x_y[:, 1]
        g = 0.5 + (np.power((np.sin( np.sqrt( np.power(x, 2) + np.power(y, 2)))), 2) - 0.5)/ \
            (1 + 0.001 * (np.power(x, 2) + np.power(y, 2)))
        return g
    de_alg = DifferentialEvolutionAlgorithm(
                                            num_epochs=200,
                                            pop_size=100,
                                            chrom_length=2,
                                            value_ranges=[(-10,10), (-10,10)],
                                            mutation_rate=0.8,
                                            fitness_func=schaffer_function
                                            )
    best_solutions = de_alg.fit()
    def schaffer_function_plot(x,y):
        g = 0.5 + (np.power((np.sin( np.sqrt( np.power(x, 2) + np.power(y, 2)))), 2) - 0.5)/ \
            (1 + 0.001 * (np.power(x, 2) + np.power(y, 2)))
        return g

    x_data = best_solutions[:, 0]
    y_data = best_solutions[:, 1]
    z_data = schaffer_function_plot(x_data, y_data)

    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)

    X, Y = np.meshgrid(x, y)
    Z = schaffer_function_plot(X, Y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='jet', alpha=0.2)
    ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap='binary', alpha=1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.show()

    print(de_alg.best_solution)
    print(de_alg.best_solution_fitness)
