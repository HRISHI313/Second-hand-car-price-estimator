import os
import sys

PROJECT_ROOT = r"D:\pythonProject\Second hand car price estimator"
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))

from src.components.data_ingestion import Dataingestion
# from src.components.data_transformation import DataTransformation
# from src.components.Model_Trainer import ModelTrainer

if __name__ == '__main__':
    # Create an instance of the DataIngestion class
    dataingestion = Dataingestion()

    # Call the method to initiate data ingestion
    train_data_path,test_data_path = dataingestion.Initaite_Data_ingestion()
