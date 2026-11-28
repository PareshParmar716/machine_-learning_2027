import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 


import os 
import sys
from dataclasses import dataclass

from sklearn.metrics import r2_score


from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    HuberRegressor,
    SGDRegressor,
    RANSACRegressor,
    TheilSenRegressor,
    ARDRegression,
    PoissonRegressor,
    TweedieRegressor,
    GammaRegressor
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
    VotingRegressor,
    StackingRegressor,
    BaggingRegressor,
    RandomForestRegressor,
    IsolationForest
)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.neural_network import MLPRegressor



from xgboost import XGBClassifier, XGBRegressor
# from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from src.expesion import CustomException
from src.componets.logger import logging
from src.utils import save_object, evalute_models




@dataclass
class ModelTrainerConnfig:
    trained_model_file_path  = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConnfig()
        
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
           X_train, y_train, X_test, y_test = (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
              test_array[:, -1]
)

           models = {
               "Random Forest" : RandomForestRegressor(),
               "Decision Tree": DecisionTreeRegressor(),
               "GradientBoosting": GradientBoostingRegressor(),
               "Linear Regresion": LinearRegression(),
               "KNeighborsRegressor": KNeighborsRegressor(),
               "XGBRegressor": XGBRegressor(),
               " CatBoostRegressor":  CatBoostRegressor(verbose=False),
               "AdaBoost Regressor" :AdaBoostRegressor()
           }

           model_report :dict=evalute_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)




           ### to get best model score from dica
           best_model_score = max(sorted(model_report.values()))

           best_model_name = list(model_report.keys())[
               list(model_report.values()).index(best_model_score)
           ]
           best_model = models[best_model_name]

           if best_model_score<0.6:
               raise CustomException("No Best model found", sys)

           logging.info(f"Best found model on both training and Testing  datasets")

           save_object(
               file_path=self.model_trainer_config.trained_model_file_path,
               obj = best_model


               

        )

           predicted = best_model.predict(X_test)
           model_r2_score = r2_score(y_test,predicted)
           return model_r2_score




        except Exception as e:
            raise CustomException(e,sys)
           


