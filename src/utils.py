import sys
import os
import dill

from src.exceptions import CustomError
from src.logger import logging
from sklearn.metrics import r2_score

def save_obj(file_path, obj):
    try:
        # Ensure the directory where the file will be saved exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Now open the actual file to write the object
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        
        logging.info("Object dumped into .pkl file successfully.")

    except Exception as e:
        raise CustomError(e, sys)
    
def evaluate_model(true, predicted):
    # mae = mean_absolute_error(true, predicted)
    # mse = mean_squared_error(true, predicted)
    # rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return r2_square