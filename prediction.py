import pandas as pd
import os
import numpy as np

currentPath = os.path.abspath(os.path.dirname(__file__))
df_before_prediction = pd.read_csv(os.path.join(currentPath,'data/signals_before_prediction.csv'))
df_after_prediction = pd.read_csv(os.path.join(currentPath,'data/signals_after_prediction.csv'))

def before_predict_table():
    df = df_before_prediction.copy()
    return df

def after_predict_table():
    df = df_after_prediction.copy()
    return df