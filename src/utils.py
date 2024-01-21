import pickle
import os
import sys

import numpy as np
import pandas as pd

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

