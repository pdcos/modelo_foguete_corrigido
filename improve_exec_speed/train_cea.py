import numpy as np
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

import os
import multiprocessing
import pandas as pd
import time

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.linear_model import Ridge  # Exemplo usando Ridge, pode ser substituído por LinearRegression ou outro modelo
from sklearn.metrics import mean_squared_error
from joblib import dump
import json

from sklearn.multioutput import MultiOutputRegressor


class Trainer:
    def __init__(self, model_name, data_path):
        self.model_name = model_name
        self.data_path = data_path
        self.train_time = None
        self.inference_time = None
        self.inference_samples = None
        self.hiperparametros = None

    def read_data(self):
        # Leitura dos dados
        df_grid_target = pd.read_csv(self.data_path)

        # Processa e converte para numpy
        X_y = df_grid_target.to_numpy()
        X = X_y[:,0:3]
        y = X_y[:,3:]

        # Deleta o dataframe para liberar espaco na RAM
        del df_grid_target
        
        # Divide os dados em treino e teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def linear_regression(self):
        # Treina o modelo
        # Não possui hiperparâmetros porque é apenas uma regressão linear
        self.model = LinearRegression().fit(self.X_train, self.y_train)

    def regression_tree(self):
        # Treina o modelo com grid search
        self.model = DecisionTreeRegressor(random_state=42)
        parameters = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 10, 20],
            'min_samples_leaf': [1, 5, 10]
        }

        # Grid Search com validação cruzada
        np.random.seed(42)
        grid_search = GridSearchCV(self.model, parameters, cv=5, scoring='neg_mean_squared_error', n_jobs=4, verbose=3)
        grid_search.fit(self.X_train, self.y_train)

        # Melhores hiperparâmetros\
        self.hiperparametros = grid_search.best_params_
        self.model = grid_search.best_estimator_

    def mlp_regressor(self):
        # Treina o modelo com grid search
        self.model = MLPRegressor(random_state=42)
        parameters = {
            'hidden_layer_sizes': [(64,64,64), (64,128,64), (256,)],
            #'activation': ['tanh', 'relu'],
            #'solver': ['sgd'],
            #'alpha': [0.0001, 0.05],
            #'learning_rate': ['constant','adaptive'],.
            #'max_iter': [200]
        }

        # Grid Search com validação cruzada
        np.random.seed(42)
        grid_search = GridSearchCV(self.model, parameters, cv=5, scoring='neg_mean_squared_error', n_jobs=4, verbose=3)
        grid_search.fit(self.X_train, self.y_train)

        # Melhores hiperparâmetros
        self.hiperparametros = grid_search.best_params_
        self.model = grid_search.best_estimator_

    def svr_regressor(self):
        # Treina o modelo com grid search
        self.model = MultiOutputRegressor(SVR())
        parameters = {
            'estimator__kernel': ['rbf', 'linear'],
            'estimator__C': [1, 10, 100],
            'estimator__gamma': ['scale', 'auto'],
            'estimator__max_iter': [10000]  # Limita o número de iterações
        }

        # Grid Search com validação cruzada
        np.random.seed(42)
        grid_search = GridSearchCV(self.model, parameters, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=3)
        grid_search.fit(self.X_train, self.y_train)

        # Melhores hiperparâmetros
        self.hiperparametros = grid_search.best_params_
        self.model = grid_search.best_estimator_


    def test_model(self):
        # Calcula o erro
        self.y_pred = self.model.predict(self.X_test)
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.score = self.model.score(self.X_test, self.y_test)
        print('MSE: ', self.mse)
        print("Score: ", self.score)

    def save_model(self):
        # Salva o modelo
        dump(self.model, 'model.joblib')    
        model_path = os.path.join('./data', f'{type(self.model).__name__}_score_{round(self.score, 2)}.joblib')
        dump(self.model, model_path)

    def save_results(self):
        # Salva os resultados
        results = {
            'model_name': self.model_name,
            'train_time': self.train_time,
            'inference_time': self.inference_time,
            'inference_samples': self.inference_samples,
            'hiperparametros': self.hiperparametros,
            'mse': self.mse,
            'score': self.score
        }
        results_path = os.path.join('./logs', f'{type(self.model).__name__}_score_{round(self.score, 2)}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f)

    def execute(self):
        self.read_data()
        self.train_time = time.time()
        print("Modelo: ", self.model_name)
        print("Inicio do treino")
        if self.model_name == 'linear_regression':
            self.linear_regression()
        elif self.model_name == 'regression_tree':
            self.regression_tree()
        elif self.model_name == 'mlp_regressor':
            self.mlp_regressor()
        elif self.model_name == 'svr_regressor':
            self.svr_regressor()
        self.train_time = time.time() - self.train_time
        print("Tempo de treino: ", self.train_time)
        self.inference_time = time.time()
        self.inference_samples = self.X_test.shape[0]
        self.test_model()
        self.inference_time = time.time() - self.inference_time
        print("Tempo de inferencia: ", self.inference_time)
        print("Numero de amostras: ", self.inference_samples)

        self.save_model()
        self.save_results()

if __name__ == '__main__':
    # Lê o argumento passado pelo terminal e joga no nome do modelo
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='linear_regression')
    args = parser.parse_args()
    model_name = args.model_name
    data_path = '/home/ubuntu/Mestrado/modelo_foguete/improve_exec_speed/data/grid_target.csv'
    trainer = Trainer(model_name=model_name, data_path=data_path)
    trainer.execute()