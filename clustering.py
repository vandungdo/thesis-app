import pandas as pd
import os
import datetime

currentPath = os.path.abspath(os.path.dirname(__file__))


df1 = pd.read_csv(os.path.join(currentPath,'data/merged_classes_with_frequency2.csv'))


def picture(label):
    df_signal = df[df.label == label]
    time = []
    for i in df_signal.index:
        t = df_signal.start_time[i][:-6].replace(' ','T')
        time.append(t + '-00-00_' +  df_signal.band[i]+'.csv.png')
    time = list(set(time))
    return time

def new_df():
    a = df1
    a.drop(['time','band'],inplace=True,axis=1)
    return a

def new_signal_class(label):
    df_signal = df1[df1.label == label]
    df_signal.drop(['time','band'],inplace=True,axis=1)
    return df_signal

def new_signal_class_time(day,start,end):
    df_time = df1.copy()
    df_time.drop(['time','band'],inplace=True,axis=1)
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

def picture1(label):
    df_signal = df1[df1.label == label]
    time = []
    for i in df_signal.index:
        t = df_signal.start_time[i][:-6].replace(' ','T')
        time.append(t + '-00-00_' +  df_signal.band[i]+'.csv.png')
    time = list(set(time))
    return time

def picture2(label,day,start,end):
    df_time = df1.copy()
    df_time.drop(['time'],inplace=True,axis=1)
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

    if label == 'all labels':
        df_time = df_time[(datetime.datetime.strptime(time1,f)<= df_time.time) & (df_time.time<=datetime.datetime.strptime(time2,f))]
    else:
        df_time = df_time[(datetime.datetime.strptime(time1,f)<= df_time.time) & (df_time.time<=datetime.datetime.strptime(time2,f))]
        df_time = df_time[df_time.label == label]
    time = []
    for i in df_time.index:
        t = df_time.start_time[i][:-6].replace(' ','T')
        time.append(t + '-00-00_' +  df_time.band[i]+'.csv.png')
    time = list(set(time))

    return time