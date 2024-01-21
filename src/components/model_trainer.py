import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:

    def __init__(self, config:ModelTrainerConfig=ModelTrainerConfig()):
        self.model_trainer_config = config
    
    def initiate_model_training(self, train_array, test_array):
        logging.info("Starting the Model training stage")

        try:
            

            X_train = train_array[:,:-1]
            X_test = test_array[:,:-1]
            y_train = train_array[:,-1]
            y_test = test_array[:,-1]
            logging.info("Done Spiliting independant and dependant feature from data")
            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet(),
            'DecisionTree':DecisionTreeRegressor(),
            'RandomForest': RandomForestRegressor(),
            'KNeighborsRegressor': KNeighborsRegressor(),
            }
            report:dict = evaluate_model(X_train, X_test, y_train, y_test, models)
            logging.info(f'Model Report : {report}')

            best_score = max(sorted(report.values()))

            best_model_name = list(report.keys())[list(report.values()).index(best_score)]

            best_model = models[best_model_name]
            logging.info("found best model")
            best_model.fit(X_train, y_train)

            logging.info("trained best model")
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_score}')
            save_object(self.model_trainer_config.trained_model_path, best_model)
            logging.info("saved the trained model")
            return self.model_trainer_config.trained_model_path

        except Exception as e:
            logging.info("An error occur at model training stage")
            raise CustomException(e, sys)
