import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
import pickle 


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        logging.error("Error in save_object: %s", e)
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            r2_square = r2_score(y_test, y_test_pred)
            report[model_name] = r2_square
            logging.info(f"{model_name} R2 Score: {r2_square}")
        return report
    except Exception as e:
        logging.error("Error in evaluate_models: %s", e)
        raise CustomException(e, sys)