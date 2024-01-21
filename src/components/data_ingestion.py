import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split


## ingestion configuration
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


# Data Ingestion class
class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.ingestion_config = config

    def initial_data_ingestion(self):
        logging.info("Data Ingestion Starts")
        try:
            df = pd.read_csv(os.path.join("notebooks/data", "gemstone.csv"))
            logging.info("dataset read as pandas dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("raw data saved")

            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion is complete")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("an error occured at data ingestion stage")
            raise CustomException(e, sys)
