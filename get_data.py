#data processing libraries
import pandas as pd
import numpy as np

#machine learning libraries
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def get_and_clean_data() -> (pd.DataFrame, pd.DataFrame):
    data = pd.read_csv("https://raw.githubusercontent.com/YojerlisPonceano/TechPay_Predictor/main/assets/employee_data.csv")

    #encode the gender and position fields
    label_enc = LabelEncoder() 
    hot_enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop ='first').set_output(transform = 'pandas')
    data['Gender']= label_enc.fit_transform(data['Gender']) 
    transformed_labels = hot_enc.fit_transform(data[['Position']])

    cleaned_data = pd.concat([transformed_labels, data], axis=1).drop(columns=["Position","ID"])
    return cleaned_data, data
