import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except CustomException as e:
            logging.exception('Exception occurred in the predict function', e)
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 car_name: str,
                 brand: str,
                 model: str,
                 vehicle_age: int,
                 km_driven: float,
                 seller_type: str,
                 fuel_type: str,
                 transmission_type: str,
                 mileage: float,
                 engine: int,
                 max_power: float,
                 seats: int):
        self.car_name = car_name
        self.brand = brand
        self.model = model
        self.vehicle_age = vehicle_age
        self.km_driven = km_driven
        self.seller_type = seller_type
        self.fuel_type = fuel_type
        self.transmission_type = transmission_type
        self.mileage = mileage
        self.engine = engine
        self.max_power = max_power
        self.seats = seats

    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                'car_name': [self.car_name],
                'brand': [self.brand],
                'model': [self.model],
                'vehicle_age': [self.vehicle_age],
                'km_driven': [self.km_driven],
                'seller_type': [self.seller_type],
                'fuel_type': [self.fuel_type],
                'transmission_type': [self.transmission_type],
                'mileage': [self.mileage],
                'engine': [self.engine],
                'max_power': [self.max_power],
                'seats': [self.seats]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            logging.exception('Exception occurred in get_data_as_frame', e)
            raise CustomException(e, sys)
