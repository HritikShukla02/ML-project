import os
import sys
from src.exceptions import CustomError
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered Data ingestion method or component.')
        try:
            data = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as DataFrame.')
            os.makedirs('artifacts', exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train, test = train_test_split(data, test_size=0.2, random_state=42)

            train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data Ingestion Completed.')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomError(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path =obj.initiate_data_ingestion()

    transformer = DataTransformation()
    train_arr, test_arr, _ = transformer.initiate_data_transformer(train_path, test_path)

    model = ModelTrainer()
    print(model.initiate_model_training(train_arr, test_arr))

