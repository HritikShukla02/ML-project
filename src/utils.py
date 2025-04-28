import sys
import os
import dill

from src.exceptions import CustomError
from src.logger import logging


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
