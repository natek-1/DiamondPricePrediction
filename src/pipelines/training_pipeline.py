import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

import pandas as pd


if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initial_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_obj_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    print(preprocessor_obj_path)