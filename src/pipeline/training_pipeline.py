import os
import sys

PROJECT_ROOT = r"D:\pythonProject\Second hand car price estimator"
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import Dataingestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import Model_Trainer


if __name__ == '__main__':
    # Create an instance of the DataIngestion class
    dataingestion = Dataingestion()

    # Call the method to initiate data ingestion
    train_data_path,test_data_path = dataingestion.Initaite_Data_ingestion()

    #Create an instance of the Data Transformation class
    data_transformation = DataTransformation()

    # Call the method to initiate data Transformation
    train_arr,test_arr,_= data_transformation.Initiate_data_transformation(train_data_path,test_data_path)

    #Create an instance of the Model training class
    model_trainer = Model_Trainer()

    # Call the method to initiate model trainer
    model_trainer.initiate_model_training(train_arr,test_arr)
