import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

import pandas as pd


if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initial_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_obj_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    model_trainer = ModelTrainer()
    trained_model_path = model_trainer.initiate_model_training(train_arr, test_arr)
    print(trained_model_path)