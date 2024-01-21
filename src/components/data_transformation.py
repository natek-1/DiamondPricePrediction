import os
import sys
from dataclasses import dataclass

from sklearn.impute import SimpleImputer # handling missing values
from sklearn.preprocessing import StandardScaler, OrdinalEncoder # Scaling and Ordinal Encoding

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig():
    preprocessor_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self, config: DataTransformationConfig= DataTransformationConfig()):
        self.data_transformation_config = config
    
    def get_data_transformation_object(self):
        logging.info("Starting the Data Transformation Stage")

        try:
            numerical_column = ['carat', 'depth', 'table', 'x', 'y', 'z']
            categorical_column = ['cut', 'color', 'clarity']


            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]
            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_column),
            ('cat_pipeline',cat_pipeline,categorical_column)
            ])
            logging.info("Data Transformation obejct created")

            return preprocessor

        except Exception as e:
            logging.info("An error occured in the Data Transformation stage - making object")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df= pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Read train and test data from the path provided")

            preprocessor_obj = self.get_data_transformation_object()

            target_column = 'price'
            drop_column = ['id', target_column]
            # dividing the dataset into train and test features
            input_feature_train_df = train_df.drop(columns=drop_column, axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=drop_column, axis=1)
            target_feature_test_df = test_df[target_column]

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("applied preprocessing to training and test data")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(self.data_transformation_config.preprocessor_file_path, preprocessor_obj)

            logging.info("saved the preprocessor obeject to appropriate path")
            logging.info("data transformation stage is complete")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path
            )
    
        except Exception as e:
            logging.info("an error occured tat the Data Transformation stage - initation")
            raise CustomException(e, sys)

