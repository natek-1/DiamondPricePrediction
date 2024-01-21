import pickle
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path:str, obj):
    logging.info(f"trying to save object to {file_path}")
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)  
        
    except Exception as e:
        logging.info("an error occured while saving the object")
        raise CustomException(e, sys)

def evaluate_model(X_train, X_test, y_train, y_test, models):
    '''
    a function used to train and subsequently test the models.

    '''
    try:
        logging.info("starting the model evaluation")
        report = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            score = r2_score(y_test, y_pred)

            report[model_name] = score
        logging.info("done evaluating the different models")
        return report

    except Exception as e:
        logging.info("An error occured while evaluating the models")
        raise CustomException(e, sys)
