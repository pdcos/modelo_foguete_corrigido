import numpy as np
import random
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from tqdm import tqdm
import time


class DEPSO():
    def __init__(self,
                 num_epochs:int,
                 pop_size:int,
                 chrom_length:int,
                 n_best:int,
                 local_factor:float,
                 global_factor:float,
                 speed_factor:float,
                 v_max: float,
                 value_ranges:list,
                 crossover_rate:float,
                 mutation_rate: float,
                 fitness_func, # Function Type,
                 seed=42,
                 eval_every=100,
                 verbose = 0,
                 neighborhood_mode = "self",
                 maintain_history=False,
                 early_stopping=False,
                ):
        
        self.num_epochs = num_epochs
        self.pop_size = pop_size
        self.n_best = n_best
        self.chrom_length = chrom_length
        self.local_factor = local_factor
        self.global_factor = global_factor
        self.speed_factor = speed_factor
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.v_max = v_max
        self.value_ranges = np.array(value_ranges)
        self.fitness_func = fitness_func
        self.seed = seed    
        self.best_ind_list = np.zeros(self.num_epochs)
        self.avg_ind_list = np.zeros(self.num_epochs)
        self.eval_every = eval_every
        self.verbose = verbose
        self.f_gbest_alltime = 0
        self.neighborhood_mode = neighborhood_mode
        self.maintain_history=maintain_history
        self.particle_history = []
        self.pbest_history = []
        self.gbest_history = []
        self.speed_history = []
        self.best_solution_fitness = 0
        self.best_solution = 0

        self.fitness_calls_counter = 0
        self.fitness_calls_list = np.zeros(self.num_epochs)

        self.min_mat = self.value_ranges.T[0, :]
        self.max_mat = self.value_ranges.T[1,:]

        self.phi = self.global_factor + self.local_factor
        self.K = 2.0/(np.abs(2-self.phi - np.sqrt((self.phi**2)-(4*self.phi))))

        # Initialize execution time list
        self.exec_time_list = np.zeros(self.num_epochs)

        # Inicializa o early stopping. Se False, não será utilizado, caso contrário deve ser
        # um inteiro que representa o número de épocas sem melhora para parar o algoritmo
        self.early_stopping = early_stopping

        self.best_fit_alltime = 0

        np.random.seed(seed=seed)

        # self.init_pop()
        # self.calculate_fitness()
        # self.update_gbest()
        # self.update_pbest()
        # self.update_speed()
        # self.update_position()
        # self.mutation()
        # self.crossover()
        # self.calculate_fitness()
        # self.selection()


    def init_pop(self):
        self.x_i = np.random.rand(self.pop_size, self.chrom_length)
        self.v_i = np.random.rand(self.pop_size, self.chrom_length)
        self.v_i = self.v_i * (2 * self.v_max) - self.v_max
        self.f_x_i = self.fitness_func(self.x_i, self.value_ranges)
        self.fitness_calls_counter += 1
        self.f_gbest = self.f_x_i.max()
        #self.gbest = self.x_i[self.f_x_i == self.f_x_i.max()].squeeze(axis=0)
        self.gbest = self.x_i[self.f_x_i == self.f_x_i.max()]
        self.pbest = self.x_i.copy()
        self.f_pbest = self.f_x_i.copy()
        self.u_g = self.x_i.copy()
    
    def calculate_fitness(self):
        self.f_x_i = self.fitness_func(self.x_i, self.value_ranges).copy()
        #self.f_x_i = self.fitness_func(self.x_i, self.value_ranges)
        self.f_u_g = self.fitness_func(self.u_g, self.value_ranges).copy()
        self.fitness_calls_counter += 2

        curr_max_fitness = self.f_x_i.max()
        if self.best_solution_fitness < curr_max_fitness:
            self.best_solution = self.x_i[self.f_x_i.argmax()]
            self.best_solution_fitness = curr_max_fitness

        if self.maintain_history:
            particle = self.x_i * (self.max_mat - self.min_mat) + self.min_mat
            self.particle_history.append(particle)


    def update_gbest(self):
        curr_max = self.f_x_i.max()
        curr_max = 1
        if curr_max > self.f_gbest:
            mask = self.f_x_i.argmax()
            self.gbest = self.x_i[mask,:]
            self.f_gbest = self.f_x_i[mask]
        if self.maintain_history:
            particle = self.gbest * (self.max_mat - self.min_mat) + self.min_mat
            self.gbest_history.append(particle)
        return
    
    def update_pbest_self(self):
        mask = self.f_x_i > self.f_pbest
        self.pbest[mask,:] = self.x_i[mask,:]
        self.f_pbest[mask] = self.f_x_i[mask]
        return

    def update_pbest_ring(self):
        arr = self.f_x_i
        n = len(arr)
        arr_circular = np.concatenate((arr[n - 1:], arr, arr[:1]))
        n_matrix = np.vstack((arr_circular[:-2], arr_circular[1:-1], arr_circular[2:])).T
        self.pbest_aux = n_matrix.max(axis=1)
        self.pbest_aux_indices = np.where(self.f_x_i[None,:] == self.pbest_aux[:,None])[1]

        self.l_best = self.x_i[self.pbest_aux_indices]
        mask = self.f_pbest < self.f_x_i

        self.f_pbest[mask] = self.f_x_i[mask]
        self.pbest[mask,:] = self.x_i[mask,:]


    def update_pbest(self):
        if self.neighborhood_mode == 'self':
            self.update_pbest_self()
        elif self.neighborhood_mode == 'ring':
            self.update_pbest_ring()

        if self.maintain_history:
            particle = self.pbest * (self.max_mat - self.min_mat) + self.min_mat
            self.pbest_history.append(particle)

    
    def update_speed(self):
        rand_factor_1, random_factor_2 = np.random.rand(2)

        self.v_i = self.K * ((self.v_i * self.speed_factor) + \
                    (rand_factor_1 * self.global_factor * (self.gbest - self.x_i)) + \
                    (random_factor_2 * self.local_factor * (self.pbest - self.x_i)))
        
        self.v_i[self.v_i > self.v_max] = self.v_max
        self.v_i[self.v_i < -self.v_max] = -self.v_max

        if self.maintain_history:
            speeds = self.v_i * (self.max_mat - self.min_mat) + self.min_mat
            self.speed_history.append(self.v_i)
    
    def update_position(self):
        self.x_i = self.x_i + self.v_i
        mask = self.x_i > 1
        self.x_i[mask] = 1
        mask = self.x_i < 0
        self.x_i[mask] = 0
    
    def mutation(self):
        mut_idx_1 = np.random.randint(low=0, high=self.pop_size, size=self.pop_size)
        mut_idx_2 = np.random.randint(low=0, high=self.pop_size, size=self.pop_size)
        mut_idx_3 = np.random.randint(low=0, high=self.pop_size, size=self.pop_size)
        mut_idx_4 = np.random.randint(low=0, high=self.pop_size, size=self.pop_size)
        self.delta = ((self.x_i[mut_idx_1] - self.x_i[mut_idx_2]) + (self.x_i[mut_idx_3] - self.x_i[mut_idx_4]) )/2
        self.v_g = self.x_i + (self.delta * self.mutation_rate )
        mask = self.v_g > 1
        self.v_g[mask] = 1
        mask = self.v_g < 0
        self.v_g[mask] = 0


    def crossover(self):
        crossover_prob = np.random.rand(self.pop_size, self.chrom_length)
        aleat_index = np.random.randint(low=0, high=self.chrom_length, size=self.pop_size)
        aleat_index_ohe = np.full((aleat_index.size, self.chrom_length), fill_value=False, dtype=bool)
        aleat_index_ohe[np.arange(aleat_index.size, dtype=int), aleat_index] = True
        self.u_g = self.x_i.copy()
        self.u_g[crossover_prob >= self.crossover_rate] = self.v_g[crossover_prob >= self.crossover_rate]
        self.u_g[aleat_index_ohe] = self.v_g[aleat_index_ohe]

    def selection(self):
        #self.f_x_i = self.fitness_func(self.x_i, self.value_ranges).copy()
        #self.f_u_g = self.fitness_func(self.u_g, self.value_ranges).copy()

        replacement_indices = self.f_u_g > self.f_x_i
        self.x_i[replacement_indices] = self.u_g[replacement_indices]
        self.f_x_i[replacement_indices] = self.f_u_g[replacement_indices]


    def callback(self):
        max_val = self.f_gbest
        mean_val = np.mean(self.f_x_i)
        self.best_ind_list[self.curr_epoch] = max_val
        self.avg_ind_list[self.curr_epoch] = mean_val
        self.fitness_calls_list[self.curr_epoch] = self.fitness_calls_counter
        if (self.curr_epoch % self.eval_every == 0) and self.verbose != 0 :
            print(f"Epoch {self.curr_epoch}: Best: {max_val}, Average: {mean_val}")

    def fit(self):
        start_time = time.time()
        self.init_pop()
        self.calculate_fitness()
        # Inicializa o número de épocas sem melhora
        no_improvement = 0
        for epoch in tqdm(range(self.num_epochs)):
            self.curr_epoch = epoch
            self.update_gbest()
            self.update_pbest()
            self.update_speed()
            self.update_position()
            #self.calculate_fitness()
            self.mutation()
            self.crossover()
            self.selection()
            self.calculate_fitness()
            self.callback()
            # Atualiza o tempo de execução
            exec_time = time.time() - start_time
            self.exec_time_list[epoch] = exec_time
            # Verifica se o early stopping está ativo
            if self.early_stopping:
                # Encontra o melhor indivíduo da época
                self.best_fit = self.f_x_i.max()

                # Verifica se houve melhora
                if self.best_fit > self.best_fit_alltime:
                    self.best_fit_alltime = self.best_fit
                    no_improvement = 0
                else:
                    no_improvement += 1
                # Verifica se o número de épocas sem melhora é maior que o limite
                if no_improvement >= self.early_stopping:
                    print(f"Early stopping at epoch {epoch}")
                    print(f"Best fitness: {self.best_fit_alltime}")
                    break
        self.total_exec_time = time.time() - start_time
        print("--- %s seconds ---" % (self.total_exec_time))
        return self.pbest

    def plot(self):
        plt.plot(self.best_ind_list, label="Best")
        plt.plot(self.avg_ind_list, label="Average")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Fitness Value")
        plt.legend()
        plt.show()