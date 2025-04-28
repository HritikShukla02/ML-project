import sys
import os

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exceptions import CustomError
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig

    def get_data_preprocessor_obj(self):
        """
        This method is responsible for data transformation.
        """
        try:
            num_columns = ['reading_score', 'writing_score']
            cat_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ('impute', SimpleImputer(strategy='median')),
                    ('scalar', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('impute', SimpleImputer(strategy='most_frequent')),
                    ('one_hot', OneHotEncoder())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_transformer', num_pipeline, num_columns),
                    ('cat_transformer', cat_pipeline, cat_columns)
                ]
            )
            logging.info('Preprocessor created.')
            return preprocessor
        except Exception as e:
            raise CustomError(e, sys)

    def initiate_data_transformer(self, train_path, test_path):
        try:

            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
            logging.info('Datasets read.')

            preprocessor_obj = self.get_data_preprocessor_obj()

            target_column = "math_score"

            train_input = train.drop(columns=target_column, axis=1)
            train_target = train[target_column]
            test_input = test.drop(columns=target_column)
            test_target = test[target_column]
            logging.info('Got hold of Targets and inputs for train and test sets.')

            input_features_train = preprocessor_obj.fit_transform(train_input)
            input_features_test = preprocessor_obj.transform(test_input)

            logging.info("Applied Transformations to train and test input features.")

            train_arr = np.c_[input_features_train, train_target]
            test_arr = np.c_[input_features_test, test_target]

            logging.info('Saving the preprocessor object as pikkle.')
            save_obj(
                self.data_transformation_config.preprocessor_file_path,
                preprocessor_obj
                )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path
            )
        except Exception as e:
            raise CustomError(e, sys)




