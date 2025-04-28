import sys
import os

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exceptions import CustomError
from src.logger import logging
from src.utils import *

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def model_training(self, train_data, test_data):
        try:
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            model_list = []
            r2_list =[]

            logging.info("Splitting datasets into inputs and targets.")

            X_train, y_train, X_test, y_test = (
                train_data[:,:-1],
                train_data[:,-1],
                test_data[:,:-1],
                test_data[:,-1]
            ) 

            for i in range(len(list(models))):
                model = list(models.values())[i]
                model.fit(X_train, y_train) # Train model
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Evaluate Train and Test dataset
                # model_train_r2 = evaluate_model(y_train, y_train_pred)

                model_test_r2 = evaluate_model(y_test, y_test_pred)
                r2_list.append(model_test_r2)

                model_list.append(list(models.keys())[i])

            logging.info('Models trained.')
            return model_list, r2_list
        except Exception as e:
            raise CustomError(e,sys)


    def initiate_model_training(self, train_data, test_data):
        try:
            logging.info("Initiating model training.")
            models, r2 = self.model_training(train_data, test_data)

            logging.info('Getting best model.')
            best_score = np.argmax(r2)
            model = models[best_score]

            logging.info('Saving best model')
            save_obj(
                self.model_trainer_config.trained_model_path,
                model
            )

            return model, r2[best_score]


        except Exception as e:
            raise CustomError(e, sys)
