import pandas as pd
import os
import numpy as np
import datetime

currentPath = os.path.abspath(os.path.dirname(__file__))
df_before_prediction = pd.read_csv(os.path.join(currentPath,'data/signals_before_prediction.csv'))
df_after_prediction = pd.read_csv(os.path.join(currentPath,'data/signals_after_prediction.csv'))
df_after_prediction_all = pd.read_csv(os.path.join(currentPath,'data/whole_dataset_with_class.csv'))

def before_predict_table():
    df = df_before_prediction.copy()
    return df

def after_predict_table():
    df = df_after_prediction.copy()
    return df

def after_predict_table_all(day,start,end):

    df_time = df_after_prediction_all.copy()
    df_time['time'] = pd.to_datetime(df_time['start_time'], format='%Y-%m-%d %H:%M:%S')
    f = '%Y-%m-%d %H:%M:%S'
    if start < 10 and end >= 10:
        time1 = day.strftime('%Y-%m-%d') + ' 0' + str(start) + ':00:00'
        time2 = day.strftime('%Y-%m-%d') + ' ' + str(end) + ':00:00'
    elif start >= 10 and end >= 10:
        time1 = day.strftime('%Y-%m-%d') + ' ' + str(start) + ':00:00'
        time2 = day.strftime('%Y-%m-%d') + ' ' + str(end) + ':00:00'
    elif start < 10 and end < 10:
        time1 = day.strftime('%Y-%m-%d') + ' 0' + str(start) + ':00:00'
        time2 = day.strftime('%Y-%m-%d') + ' 0' + str(end) + ':00:00'
    else:
        time1 = day.strftime('%Y-%m-%d') + ' ' + str(start) + ':00:00'
        time2 = day.strftime('%Y-%m-%d') + ' 0' + str(end) + ':00:00'
    df_time = df_time[(datetime.datetime.strptime(time1,f)<= df_time.time) & (df_time.time<=datetime.datetime.strptime(time2,f))]
    df_time.drop(['time'],inplace=True,axis=1)


    return df_time