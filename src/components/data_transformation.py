import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import pandas as pd
import numpy as np

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_traformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        logging.info('Initiating Data Transformation')
        try:
            categorical_cols = ['car_name', 'brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']
            numerical_cols = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']

            car_name_cat = ['Hyundai i20', 'Maruti Swift Dzire', 'Maruti Swift', 'Maruti Alto', 'Honda City',
                            'Maruti Wagon R', 'Hyundai Grand', 'Toyota Innova', 'Hyundai Verna', 'Hyundai i10',
                            'Ford Ecosport', 'Volkswagen Polo', 'Maruti Baleno', 'Honda Amaze', 'Maruti Ciaz',
                            'Maruti Ertiga', 'Hyundai Creta', 'Mahindra XUV500', 'Renault KWID', 'Maruti Vitara',
                            'Mahindra Scorpio', 'Ford Figo', 'Volkswagen Vento', 'Maruti Celerio',
                            'Renault Duster', 'Mahindra Bolero', 'Toyota Fortuner', 'Skoda Rapid',
                            'Honda Jazz', 'BMW 3', 'Tata Tiago', 'Hyundai Santro', 'Maruti Eeco',
                            'Mercedes-Benz E-Class', 'BMW 5', 'Mercedes-Benz C-Class', 'Honda WR-V', 'Audi A4',
                            'Tata Safari', 'Skoda Superb', 'Datsun GO', 'Tata Nexon', 'Datsun RediGO',
                            'Maruti Ignis', 'Audi A6', 'Mahindra KUV', 'Ford Aspire', 'Mahindra Thar',
                            'Honda Civic', 'Skoda Octavia', 'Hyundai Venue', 'BMW X1', 'Jaguar XF',
                            'Hyundai Elantra', 'Land Rover Rover', 'Ford Endeavour', 'Tata Hexa', 'Jeep Compass',
                            'Tata Tigor', 'Mercedes-Benz GL-Class', 'Mercedes-Benz S-Class', 'BMW 7',
                            'Toyota Camry', 'Ford Freestyle', 'Honda CR-V', 'Kia Seltos', 'Mahindra KUV100',
                            'BMW X5', 'Mahindra Marazzo', 'Audi Q7', 'BMW X3', 'Tata Harrier', 'MG Hector',
                            'Maruti Dzire VXI', 'BMW 6', 'Mini Cooper', 'Toyota Yaris', 'Porsche Cayenne',
                            'Mahindra XUV300', 'Maruti S-Presso', 'Mercedes-Benz GLS', 'Renault Triber',
                            'Hyundai Tucson', 'Datsun redi-GO', 'Mercedes-Benz CLS', 'Nissan Kicks',
                            'Toyota Glanza', 'Volvo XC', 'Maruti XL6', 'Audi A8', 'BMW X4', 'Isuzu D-Max',
                            'BMW Z4', 'Lexus ES', 'Volvo XC60', 'Jaguar XE', 'Volvo XC90', 'Maruti Dzire ZXI',
                            'Kia Carnival', 'Volvo S90', 'Honda CR', 'Bentley Continental', 'Jaguar F-PACE',
                            'Nissan X-Trail', 'Porsche Panamera', 'Mahindra Alturas', 'Porsche Macan',
                            'ISUZU MUX', 'Lexus RX', 'Jeep Wrangler', 'Lexus NX', 'Maruti Dzire LXI',
                            'Isuzu MUX', 'Maserati Quattroporte', 'Rolls-Royce Ghost', 'Maserati Ghibli',
                            'Mercedes-AMG C', 'Tata Altroz', 'Ferrari GTC4Lusso', 'Hyundai Aura', 'Force Gurkha']

            brand_cat = ['Maruti', 'Hyundai', 'Honda', 'Mahindra', 'Toyota', 'Ford', 'Volkswagen', 'Renault',
                            'BMW', 'Tata', 'Skoda', 'Mercedes-Benz', 'Audi', 'Datsun', 'Jaguar', 'Land Rover',
                            'Jeep', 'Kia', 'Porsche', 'Volvo', 'MG', 'Mini', 'Nissan', 'Lexus', 'Isuzu', 'Bentley',
                            'Maserati', 'ISUZU', 'Ferrari', 'Mercedes-AMG', 'Rolls-Royce', 'Force']
            model_cat = ['i20', 'Swift Dzire', 'Swift', 'Alto', 'City', 'Wagon R', 'Grand', 'Innova', 'Verna', 'i10',
                            'Ecosport', 'Polo', 'Baleno', 'Amaze', 'Ciaz', 'Ertiga', 'Creta', 'XUV500', 'KWID', 'Vitara',
                            'Scorpio', 'Figo', 'Vento', 'Celerio', 'Duster', 'Bolero', 'Fortuner', 'Rapid', 'Jazz', '3',
                            'Tiago', 'Santro', 'Eeco', 'E-Class', '5', 'C-Class', 'WR-V', 'A4', 'Safari', 'Superb', 'GO',
                            'Nexon', 'RediGO', 'Ignis', 'A6', 'KUV', 'Aspire', 'Thar', 'Civic', 'Octavia', 'Venue', 'X1',
                            'XF', 'Rover', 'Elantra', 'Endeavour', 'Hexa', 'Compass', 'Tigor', '7', 'GL-Class', 'S-Class',
                            'Camry', 'Freestyle', 'CR-V', 'Seltos', 'KUV100', 'X5', 'Marazzo', 'X3', 'Q7', 'Harrier',
                            'Hector', '6', 'Cooper', 'Dzire VXI', 'Yaris', 'Cayenne', 'XUV300', 'S-Presso', 'GLS', 'Triber',
                            'redi-GO', 'Tucson', 'CLS', 'Glanza', 'Kicks', 'XC', 'XL6', 'Z4', 'D-Max', 'X4', 'A8', 'XC60',
                            'ES', 'Carnival', 'S90', 'XE', 'Dzire ZXI', 'XC90', 'CR', 'Alturas', 'Panamera', 'X-Trail', 'MUX',
                            'Continental', 'F-PACE', 'Macan', 'Wrangler', 'Dzire LXI', 'NX', 'RX', 'GTC4Lusso', 'Aura',
                            'Altroz', 'Ghibli', 'C', 'Ghost', 'Quattroporte', 'Gurkha']
            seller_types_cat = ['Dealer', 'Individual', 'Trustmark Dealer']
            fuel_type_cat = ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']
            transmission_type_cat = ['Manual','Automatic']

            logging.info('Pipeline Initiated')

            num_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy = 'median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy = 'most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories = [car_name_cat,brand_cat,model_cat,
                                                                   seller_types_cat,fuel_type_cat,
                                                                   transmission_type_cat])),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            return preprocessor

            logging.info('Data pipeline completed')

        except Exception as e:
            logging.exception('Error occurred in Data ingestion Config: %s', e)


    def Initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train DataFrame Head : \n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head : \n{test_df.head().to_string()}')

            logging.info('Obtaining Preprocessing object')

            preprocessing_obj = self.get_data_transformation()

            target_column_name = 'selling_price'
            drop_column = [target_column_name]

            input_feature_train_df = train_df.drop(columns = drop_column,axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = drop_column,axis = 1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing object on training dataframe and testing dataframe')

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_traformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info('Preprocessor pickle is created and saved')

            return (
                train_arr,
                test_arr,
                self.data_traformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.exception('Error occurred in Data ingestion Config: %s', e)

