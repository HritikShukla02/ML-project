import sys
import os
from src.exceptions import CustomError
from src.logger import logging
from src.utils import load_object
import numpy as np
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            logging.info(features)
            data = preprocessor.transform(features)
            preds = model.predict(data) 
            logging.info("Score predicted successfully.")
            
            return preds
        except Exception as e:
            raise CustomError(e, sys)

class CustomData:
    def __init__(self,
                 gender:str,
                 race_ethnicity:str,
                 parental_level_of_education:str,
                 lunch:str,
                 test_preparation_course:str,
                 reading_score:int,
                 writing_score:int
                 ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education  =parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score= writing_score          

    def get_data_as_dataframe(self):
        try:
            data = {
            "gender": [self.gender],
            "race_ethnicity": [self.race_ethnicity],
            "parental_level_of_education":[self.parental_level_of_education],
            "lunch": [self.lunch],
            "test_preparation_course": [self.test_preparation_course],
            "reading_score": [self.reading_score],
            "writing_score": [self.writing_score], 
            }

            return pd.DataFrame(data=data)  
        except Exception as e:
            raise CustomError(e, sys)