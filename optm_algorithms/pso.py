import numpy as np
import random
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import time

class PSO():
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
                 fitness_func, # Function Type,
                 seed=42,
                 eval_every=100,
                 verbose = 0,
                 neighborhood_mode = "self"
                ):
        
        self.num_epochs = num_epochs
        self.pop_size = pop_size
        self.n_best = n_best
        self.chrom_length = chrom_length
        self.local_factor = local_factor
        self.global_factor = global_factor
        self.speed_factor = speed_factor
        self.v_max = v_max
        self.value_ranges = np.array(value_ranges)
        self.fitness_func = fitness_func
        self.seed = seed    
        self.best_ind_list = np.zeros(self.num_epochs)
        self.avg_ind_list = np.zeros(self.num_epochs)
        self.eval_every = eval_every
        self.verbose = verbose
        self.f_g_best_alltime = 0
        self.neighborhood_mode = neighborhood_mode

        self.fitness_calls_counter = 0
        self.fitness_calls_list = np.zeros(self.num_epochs)

        self.phi = self.global_factor + self.local_factor
        self.K = 2.0/(np.abs(2-self.phi - np.sqrt((self.phi**2)-(4*self.phi))))

        # Inicializa lista de tempos de execução
        self.exec_time_list = np.zeros(self.num_epochs)

        np.random.seed(seed=seed)

        #self.init_pop()
        #self.calculate_fitness()
        #self.update_gbest()
        #self.update_pbest()
        #self.update_speed()
        #self.update_position()


    def init_pop(self):
        self.x_i = np.random.rand(self.pop_size, self.chrom_length)
        self.v_i = np.random.rand(self.pop_size, self.chrom_length)
        self.v_i = self.v_i * (2 * self.v_max) - self.v_max
        self.f_x_i = self.fitness_func(self.x_i)
        self.f_gbest = self.f_x_i.max()
        #self.gbest = self.x_i[self.f_x_i == self.f_x_i.max()].squeeze(axis=0)
        self.gbest = self.x_i[self.f_x_i == self.f_x_i.max()]
        self.pbest = self.x_i.copy()
        self.f_pbest = self.f_x_i.copy()
    
    def calculate_fitness(self):
        self.f_x_i = self.fitness_func(self.x_i).copy()
        # Increment fitness calls counter
        self.fitness_calls_counter += 1

    def update_gbest(self):
        curr_max = self.f_x_i.max()
        curr_max = 1
        if curr_max > self.f_gbest:
            mask = self.f_x_i.argmax()
            self.gbest = self.x_i[mask,:]
            self.f_gbest = self.f_x_i[mask]
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
    
    def update_speed(self):
        rand_factor_1, random_factor_2 = np.random.rand(2)

        self.v_i = self.K*((self.v_i * self.speed_factor) + \
                    (rand_factor_1 * self.global_factor * (self.gbest - self.x_i)) + \
                    (random_factor_2 * self.local_factor * (self.pbest - self.x_i)))
        
        self.v_i[self.v_i > self.v_max] = self.v_max
        self.v_i[self.v_i < -self.v_max] = -self.v_max
    
    def update_position(self):
        self.x_i = self.x_i + self.v_i
        mask = self.x_i > 1
        self.x_i[mask] = 1
        mask = self.x_i < 0
        self.x_i[mask] = 0

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
        for epoch in tqdm(range(self.num_epochs)):
            self.curr_epoch = epoch
            self.calculate_fitness()
            self.update_gbest()
            self.update_pbest()
            self.update_speed()
            self.update_position()
            self.callback()
            # Atualiza tempo de execução
            exec_time = time.time() - start_time
            self.exec_time_list[epoch] = exec_time
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