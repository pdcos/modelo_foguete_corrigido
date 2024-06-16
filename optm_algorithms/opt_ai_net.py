import numpy as np
import random
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import time
import scipy
from scipy.spatial import distance_matrix
import math


class OptAiNet():
    def __init__(self,
                 num_epochs:int,
                 pop_size:int,
                 Nc:int,
                 chrom_length:int,
                 value_ranges:list,
                 fitness_func, # Function Type,
                 beta=100,
                 clone_threshold = 0.01,
                 supression_threshold=0.2,
                 newcomers_percentage = 0.4,
                 seed=42,
                 eval_every=100,
                 verbose = 0,
                 maintain_history = False,
                 limit_fitness_calls = np.inf,
                 callback=None
                ):
        
        self.num_epochs = num_epochs
        self.pop_size = pop_size
        self.value_ranges = np.array(value_ranges)
        self.fitness_func = fitness_func
        self.chrom_length = chrom_length
        self.Nc = Nc
        self.beta = beta
        self.clone_threshold = clone_threshold
        self.supression_threshold = supression_threshold
        self.newcomers_percentage = newcomers_percentage
        self.callback_func = callback

        self.f_pop_avg_previous = 0
        self.continue_clone = True

        self.seed = seed    
        self.best_ind_list = np.zeros(self.num_epochs)
        self.avg_ind_list = np.zeros(self.num_epochs)
        self.eval_every = eval_every
        self.verbose = verbose
        np.random.seed(seed=seed)


        # Problem in max_fitness inicialization due to high incidence of zeros in rocket fitness
        self.max_fitness = 0.1
        self.min_fitness = 0

        self.best_solution_fitness = 0
        self.best_solution = 0

        self.maintain_history = maintain_history
        self.memory_cell_history = []
        self.clone_history = []

        self.min_mat = self.value_ranges.T[0, :]
        self.max_mat = self.value_ranges.T[1,:]
        self.previous_memory_cells_quantity = 0
        self.has_converged = False

        self.fitness_calls_counter = 0
        self.fitness_calls_list = np.zeros(self.num_epochs)
        self.limit_fitness_calls = limit_fitness_calls

        # inicializa total_exec_time
        self.total_exec_time = 0

        # Inicializa lista de tempos de execução
        self.exec_time_list = np.zeros(self.num_epochs)

        # self.init_pop()
        # self.fitness_evaluation()
        # self.clone()
        # self.mutation()
        # self.fitness_evaluation()
        # self.evaluation()
        # #self.supress_cells()
        # self.supress_cells_2()
        # #self.add_newcomers()


    
    def init_pop(self):
        self.pop = np.random.rand(self.pop_size, self.chrom_length)
        self.memory_cells = self.pop.copy()
        #self.min_mat = self.value_ranges.T[0, :]
        #self.max_mat = self.value_ranges.T[1,:]
        #self.pop = self.pop * (self.max_mat - self.min_mat) + self.min_mat
        #self.f_pop = self.fitness_func(self.pop)
    
    def fitness_evaluation(self):
        self.f_pop = self.fitness_func(self.pop)
        self.fitness_calls_counter += 1 * len(self.pop) 
        self.curr_f_max = self.f_pop.max()
        self.curr_f_min = self.f_pop.min()

        self.max_fitness = self.curr_f_max
        self.min_fitness = self.curr_f_min
        if self.max_fitness == 0:
            self.max_fitness = 0.01

        # if self.max_fitness < self.curr_f_max:
        #     self.max_fitness = self.curr_f_max
        # if self.min_fitness > self.curr_f_min:
        #     self.min_fitness = self.curr_f_min

        self.f_pop_norm = (self.f_pop - self.min_fitness)/(self.max_fitness - self.min_fitness)
        curr_max_fitness = self.f_pop.max()
        if self.best_solution_fitness < curr_max_fitness:
            self.best_solution = self.pop[self.f_pop.argmax()]
            self.best_solution_fitness = curr_max_fitness

        return
    
    def clone(self):
        self.pop = np.repeat(self.pop, repeats=self.Nc + 1, axis=0)
        self.f_pop_norm = np.repeat(self.f_pop_norm, repeats=self.Nc + 1, axis=0)
        memory_denorm = self.memory_cells * (self.max_mat - self.min_mat) + self.min_mat
        self.memory_cell_history.append(memory_denorm)
        clone_denorm = self.pop * (self.max_mat - self.min_mat) + self.min_mat
        self.clone_history.append(clone_denorm)
        print(f"Tamanho da populacao: {self.pop.shape[0]}")
        print(f"Iterações: {self.fitness_calls_counter} / {self.limit_fitness_calls}")



    def mutation(self):
        self.alpha = (1/self.beta) * np.exp(-self.f_pop_norm)
        self.random_mutation = np.random.normal(0, 1, size=self.pop.shape[0] * self.pop.shape[1])
        self.random_mutation = self.random_mutation.reshape(self.pop.shape[0], self.pop.shape[1])
        mask = np.zeros(self.random_mutation.shape[0], dtype=bool)
        mask[::self.Nc + 1] = True
        self.random_mutation[mask,:] = 0
        self.alpha = np.repeat(self.alpha, self.chrom_length)

        self.alpha = self.alpha.reshape(self.random_mutation.shape)

        self.pop = self.pop + self.alpha * self.random_mutation
        mask = self.pop > 1

        # self.pop[mask] = 1
        # print(mask)
        # mask = self.pop < 0
        # print(mask)
        # self.pop[mask] = 0
        rows_to_delete = np.any(self.pop > 1, axis=1)
        self.pop = self.pop[~rows_to_delete]
        rows_to_delete = np.any(self.pop < 0, axis=1)
        self.pop = self.pop[~rows_to_delete]

        # Still needs to add a way to invalidate a individual in a positsion outside of the searhc spacie

    def evaluation(self):
        self.f_pop_avg = self.f_pop.mean()
        mean_error = np.abs(self.f_pop_avg - self.f_pop_avg_previous)
        if  mean_error < self.clone_threshold:
            self.continue_clone = False
            #if self.maintain_history:
                #clone_denorm = self.pop * (self.max_mat - self.min_mat) + self.min_mat
                #self.clone_history.append(clone_denorm)
        else:
            self.continue_clone = True
        self.f_pop_avg_previous = self.f_pop_avg
    
    def supress_cells(self):
        #print(self.pop)
        distances = distance_matrix(self.pop, self.pop) 
        #print(distances)       
        f_pop_matrix = np.tile(self.f_pop, (distances.shape[0], 1)) 
        #print(f_pop_matrix)
        masked_f = f_pop_matrix * (distances<self.supression_threshold)
        best_indices = np.where(masked_f == masked_f.max(axis=1).T)[1]
        best_indices = np.unique(best_indices)
        self.pop = self.pop[best_indices]
        self.f_pop = self.f_pop[best_indices]
        self.best_ind = self.pop.copy()
        self.best_fits = self.f_pop.copy()
        self.memory_cells = self.pop.copy()

    def supress_cells_2(self):
        #print(self.pop)
        #print(self.f_pop)
        #distances = distance_matrix(self.pop, self.pop) 
        #print(distances)

        i = 0
        j = 0
        inter_count = 0
        # print(self.pop)
        # print("###")
        # print(self.f_pop)
        # print("###")
        while True:
            #if inter_count > 5:
            #    return
            #inter_count += 1
            if i == j:
                j+=1
            if j >= self.pop.shape[0]:
                i = i + 1 
                j = 0
            if i >= self.pop.shape[0]:
                self.best_ind = self.pop.copy()
                self.best_fits = self.f_pop.copy()
                self.memory_cells = self.pop.copy()
                break

            dist = np.linalg.norm(self.pop[i] - self.pop[j])
            is_near = dist < self.supression_threshold
            if is_near:
                if self.f_pop[i] > self.f_pop[j]:
                    self.f_pop = np.delete(self.f_pop, j)
                    self.pop = np.delete(self.pop, j, 0)
                    #j+=1
                else:
                    self.f_pop = np.delete(self.f_pop, i)
                    self.pop = np.delete(self.pop, i, 0)
                    #i+=1
            else:
                j+=1

        # Check if algorithm has converged
        curr_memory_cells_quantity = self.pop.shape[0]
        if self.previous_memory_cells_quantity == curr_memory_cells_quantity:
            self.has_converged=True
        else:
            self.previous_memory_cells_quantity = curr_memory_cells_quantity
        # print(self.pop)
        # print("###")
        # print(self.f_pop)
                
    def add_newcomers(self):
        n_new_ind = int(len(self.pop) * self.newcomers_percentage)
        if n_new_ind == 0:
            n_new_ind = 1
        newcomers = np.random.rand(n_new_ind, self.chrom_length)
        #newcomers = newcomers * (self.max_mat - self.min_mat) + self.min_mat
        self.pop = np.append(self.pop, newcomers, axis=0)

    def callback(self):
        max_val = np.max(self.f_pop)
        mean_val = np.mean(self.f_pop)
        self.best_ind_list[self.curr_epoch] = max_val
        self.avg_ind_list[self.curr_epoch] = mean_val
        self.fitness_calls_list[self.curr_epoch] = self.fitness_calls_counter
        if (self.curr_epoch % self.eval_every == 0) and self.verbose != 0 :
            print(f"Epoch {self.curr_epoch}: Best: {max_val}, Average: {mean_val}")

    def fit(self):
        start_time = time.time()
        self.init_pop()
        self.fitness_evaluation()
        for epoch in tqdm(range(self.num_epochs)):
            self.curr_epoch = epoch
            ###
            ###
            while self.continue_clone:
                self.clone()
                self.mutation()
                self.fitness_evaluation()
                self.evaluation()
                if self.fitness_calls_counter >= self.limit_fitness_calls:
                    break
            self.continue_clone = True
            self.f_pop_avg_previous = 0
            self.supress_cells_2()
            self.callback()
            # print(self.previous_memory_cells_quantity)
            # if self.has_converged:
            #     print(f"Algorthm has converged in generation {epoch}")
            #     break
            self.add_newcomers()
            self.fitness_evaluation()
            # Atualiza lista de tempos de execução
            self.exec_time_list[self.curr_epoch] = time.time() - start_time
            if self.fitness_calls_counter >= self.limit_fitness_calls:
                break

            if self.callback_func is not None:
                self.callback_func(self)
                
        self.total_exec_time = time.time() - start_time
        print("--- %s seconds ---" % (self.total_exec_time))
        return self.best_ind

    def plot(self):
        plt.plot(self.best_ind_list, label="Best")
        plt.plot(self.avg_ind_list, label="Average")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Fitness Value")
        plt.legend()
        plt.show()
