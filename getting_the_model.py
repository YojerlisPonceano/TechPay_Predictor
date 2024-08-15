#data processing libraries
import pandas as pd
import numpy as np

#machine learning libraries
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

#to save the model
import joblib

#to get the data
from get_data import get_and_clean_data

cleaned_data, raw_data = get_and_clean_data()

def perform_GridSearchCV(X_train, y_train, model) -> GridSearchCV:

    #hyperparameter for tuning
    search_parameters = {
        "n_estimators" : [100, 200, 500],
        "max_depth" : [3, 6, 9],
        "gamma" : [0.001,0.01, 0.1],
        "learning_rate" : [0.001, 0.01, 0.1, 1],
        "min_child_weight" : [1,6,9],
        "reg_lambda" : [1,6]
     }
    #GridSearchCV object
    GS = GridSearchCV(estimator = model,
                      param_grid = search_parameters,
                      scoring = "r2",
                      refit = True,
                      cv = 5,
                      verbose = 4)

    return GS.fit(X_train, y_train)


def get_best_model(cleaned_data) -> (XGBRegressor, float):
    X = cleaned_data.iloc[:,:12].values
    y = cleaned_data["Salary"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=11)

    xgb_model = XGBRegressor(random_state=11)
    gridSearchCV_object = perform_GridSearchCV(X_train, y_train, xgb_model)

    xgb_best_model = gridSearchCV_object.best_estimator_
    r2_score = np.round(gridSearchCV_object.best_score_*100,2)
    
    return (xgb_best_model.fit(X_train, y_train), r2_score)


xgb_best_model, r2_score = get_best_model(cleaned_data)

joblib.dump(xgb_best_model,r"assets\\xgb_model.joblib")

with open("assets\\model_score.txt", "w") as file:
    file.write(str(r2_score)+ "%")