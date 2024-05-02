import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


class CMA_ES():
    def __init__(self,
                 num_epochs:int,
                 lamb:int,
                 mi:int,
                 chrom_length:int,
                 value_ranges:list,
                 fitness_func,
                 seed=1,    
                 eval_every = 10,
                 verbose=0,
                 maintain_history=False,
                 sigma=0.0444,
                ):
        
        np.random.seed(seed=seed)
        self.num_epochs = num_epochs
        self.N = chrom_length
        self.x_mean = np.array([0.5] * self.N)
        self.lamb = lamb
        self.sigma = sigma  # Step size
        self.mi = mi  # Number of best candidates
        self.C = np.identity(self.N)
        self.fitness_func = fitness_func
        self.value_ranges = value_ranges
        self.weights = np.log(self.mi + 0.5) - np.log(np.arange(1, self.mi + 1))
        self.weights = self.weights / self.weights.sum()
        self.maintain_history = maintain_history
        self.x_i_history = []

        # Adicionando o caminho de evolução
        self.p_sigma = np.zeros(self.N)  # Caminho de evolução para o tamanho do passo

        # Parâmetros para adaptação do tamanho do passo
        self.c_sigma = 0.3 / self.N  # Taxa de mudança para p_sigma
        self.d_sigma = 1  # Fator de ajuste para o tamanho do passo

        # Outras variáveis
        self.best_ind_list = np.zeros(self.num_epochs)
        self.avg_ind_list = np.zeros(self.num_epochs)
        self.eval_every = eval_every
        self.verbose = verbose
        self.fitness_calls_counter = 0
        self.fitness_calls_list = np.zeros(self.num_epochs)
        self.min_mat = self.value_ranges.T[0, :]
        self.max_mat = self.value_ranges.T[1,:]

        # Inicializa lista de tempos de execução
        self.exec_time_list = np.zeros(self.num_epochs)


    def step(self):
        # Amostragem da população
        self.x_i = np.random.multivariate_normal(self.x_mean, (self.sigma ** 2) * self.C, size=self.lamb)

        self.x_i = np.clip(self.x_i, 0, 1)
        self.f_x_i = self.fitness_func(self.x_i, self.value_ranges)
        self.fitness_calls_counter += 1

        mask = (-self.f_x_i).argsort()
        self.f_x_i = self.f_x_i[mask]
        self.x_i = self.x_i[mask]
        self.best_indvs = self.x_i[0:self.mi]
        self.cov_mat = np.cov(self.best_indvs.T)
        self.x_mean = np.dot(self.weights, self.best_indvs)
        #print(self.cov_mat)

        # Update step size (sigma) 
        # Need to improve this part
        #p_sigma = np.zeros(self.N)
        #print(p_sigma)
        #for i in range(self.mi):
        #    p_sigma += self.weights[i] * (self.best_indvs[i] - self.x_mean) / self.sigma
        #print(p_sigma)
        # Atualização de p_sigma
        weighted_sum = np.sum(self.weights[:, np.newaxis] * (self.best_indvs - self.x_mean) / self.sigma, axis=0)
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + np.sqrt(1 - (1 - self.c_sigma) ** 2) * np.sqrt(self.mi) * weighted_sum
        self.x_mean = np.dot(self.weights, self.best_indvs)
        
        # Adaptação do tamanho do passo
        self.sigma *= np.exp((np.linalg.norm(self.p_sigma) - np.sqrt(self.N)) / (self.d_sigma * np.sqrt(self.N)))

        if self.maintain_history:
            particle = self.x_i * (self.max_mat - self.min_mat) + self.min_mat
            self.x_i_history.append(particle)

    def callback(self):
        max_val = self.f_x_i.max()
        mean_val = np.mean(self.f_x_i)
        self.best_ind_list[self.curr_epoch] = max_val
        self.avg_ind_list[self.curr_epoch] = mean_val
        self.fitness_calls_list[self.curr_epoch] = self.fitness_calls_counter
        if (self.curr_epoch % self.eval_every == 0) and self.verbose != 0 :
            print(f"Epoch {self.curr_epoch}: Best: {max_val}, Average: {mean_val}")

    def fit(self):
        start_time = time.time()
        for epoch in tqdm(range(self.num_epochs)):
            self.curr_epoch = epoch
            self.step()
            self.callback()
            # Atualiza tempo de execução
            exec_time = time.time() - start_time
            self.exec_time_list[epoch] = exec_time
        self.total_exec_time = time.time() - start_time
        print("--- %s seconds ---" % (self.total_exec_time))
        return self.best_indvs

    def plot(self):
        plt.plot(self.best_ind_list, label="Best")
        plt.plot(self.avg_ind_list, label="Average")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Fitness Value")
        plt.legend()
        plt.show()


