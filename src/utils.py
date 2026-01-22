import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
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

def evaluate_models(X_train, y_train, X_test, y_test, models ,params):
    try:
        report = {}
        
        for i in range(len(models)):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            # Here you can implement hyperparameter tuning using GridSearchCV or RandomizedSearchCV
            # For simplicity, we are skipping that part
            
            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            # Fit the model
            #model.fit(X_train, y_train)
            
            # Predicting the test set results
            y_test_pred = model.predict(X_test)
            
            # Calculating r2 score
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    except Exception as e:
        logging.error("Error in evaluate_models: %s", e)
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)