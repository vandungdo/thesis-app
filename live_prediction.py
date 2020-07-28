import numpy as np
from PIL import Image
import cv2
import pandas as pd
import datetime
from scipy.signal import find_peaks
from sklearn.preprocessing import normalize
import math
import reverse_geocoder as revgc
from joblib import dump, load
import pickle
import os

from sklearn.cluster import KMeans
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
# from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler


df_sample = pd.read_csv('data/maconisstadb.csv')  # get lat, long from database
# csv data to change the country to regions
df_regions = pd.read_csv('data/all.csv')
model = pickle.load(open('data/model.obj', 'rb'))  # prediction model
enc = load('data/enc.joblib')  # one hot encode
with open('data/factor.pickle', 'rb') as f:  # factor to exchange class name with the number
    factor = pickle.load(f)

im_path = 'data/images/'

# getting threshold from boxplot method


def get_threshold_boxplot_method(val):
    val = sorted(val)
    q1, q3 = np.percentile(val, [25, 75])
    iqr = (q3-q1)*1.5
    lower_bound = q1-iqr
    upper_bound = q3+iqr

    return upper_bound


def predict_one_file(file, band):
    df = pd.read_csv(file, header=None)
    date_time = []
    form = '%Y-%m-%d %H:%M:%S'
    for i in range(df.shape[0]):
        date_time.append(datetime.datetime.strptime(
            df[0][i]+' '+df[1][i], form))

    values = df.iloc[:, 6:].values
    fr = (df[3][0]-df[2][0])/values.shape[1]
    bw = list(np.arange(df[2][0], df[3][0], fr))

    index_of_min = np.where(values == np.amin(values))
    min_line = values[index_of_min[0][0], :]
    new_values = values - min_line
    normalized_values = normalize(new_values, axis=0).ravel()
    normalized_values = normalized_values.reshape(new_values.shape)

    threshold = get_threshold_boxplot_method(normalized_values.flat)

    bool_array = normalized_values > threshold
    image = Image.fromarray(bool_array)
    image.save(im_path+'bool.png')
    # image.show()
    im = cv2.imread(im_path+'bool.png', cv2.IMREAD_UNCHANGED)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    closing = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    im2 = Image.fromarray(closing)
    im2.save(im_path+'closed.png')
    close_im = cv2.imread(im_path+'closed.png', cv2.IMREAD_UNCHANGED)

    opening = cv2.morphologyEx(close_im, cv2.MORPH_OPEN, kernel)
    im3 = Image.fromarray(opening)
    im3.save(im_path+'opened.png')
    open_im = cv2.imread(im_path+'opened.png', cv2.IMREAD_UNCHANGED)

    contour_list = cv2.findContours(open_im, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]

    rect_contour = cv2.boundingRect(contour_list[1])

    position_of_signal = []
    for i in range(len(contour_list)):
        position_of_signal.append(cv2.boundingRect(contour_list[i]))

    duration = []
    bandwidth = []
    start_time = []
    end_time = []
    start_bandwidth = []
    end_bandwidth = []
    for i in position_of_signal:
        try:
            duration.append((date_time[i[1]+i[3]] - date_time[i[1]]).seconds)
            bandwidth.append((fr*i[2])/1000000)
            start_time.append(date_time[i[1]-1])
            end_time.append(date_time[i[1]+i[3]-1])
            start_bandwidth.append(bw[i[0]]/1000000)
            end_bandwidth.append((bw[i[0]+i[2]-1]+fr)/1000000)
        except:
            duration.append(
                (date_time[i[1]+i[3]-1] - date_time[i[1]-1]).seconds)
            bandwidth.append((fr*i[2])/1000000)
            start_time.append(date_time[i[1]-1])
            end_time.append(date_time[i[1]+i[3]-1])
            start_bandwidth.append(bw[i[0]]/1000000)
            end_bandwidth.append((bw[i[0]+i[2]-1]+fr)/1000000)

    signal_power = []
    signal_var = []
    number_of_peaks = []
    max_gradient = []
    for i in position_of_signal:
        tmp = new_values[i[1]:i[1]+i[3]+1, i[0]:i[0]+i[2]+1]
        box = tmp
        first_line = tmp[0]
        tmp = tmp[tmp > threshold]
        try:
            peaks, _ = find_peaks(tmp, height=threshold)
            first_line_peaks, _ = find_peaks(first_line, height=threshold)
        except:
            peaks = tmp
            first_line_peaks = first_line
        signal_power.append(np.mean(tmp[peaks]))
        number_of_peaks.append(len(peaks))
        signal_var.append(np.var(first_line[first_line_peaks]))
        try:
            max_gradient.append(np.amax(np.gradient(box)))
        except:
            max_gradient.append(0)

    df_signals = pd.DataFrame({'start_time': start_time[:-1], 'end_time': end_time[:-1], 'duration[s]': duration[:-1], 'bandwidth[MHz]': bandwidth[:-1], 'signal_power[dB]': signal_power[:-1],
                               'start_bandwidth': start_bandwidth[:-1], 'end_bandwidth': end_bandwidth[:-1]})

    df_signals['origin_band'] = [band]*df_signals.shape[0]

    df_signals['peaks_number'] = number_of_peaks[:-1]
    df_signals['max_gradient'] = max_gradient[:-1]

    df_sample['timestamp'] = pd.to_datetime(
        df_sample['timestamp'], format='%Y-%m-%d %H:%M:%S')
    df1 = pd.merge(df_signals, df_sample, left_on='start_time',
                   right_on='timestamp', how='left')
    df1.drop('timestamp', inplace=True, axis=1)
    df1 = df1.rename(columns={'lat': 'start_lat', 'lon': 'start_lon'})
    df2 = pd.merge(df1, df_sample, left_on='end_time',
                   right_on='timestamp', how='left')
    df2.drop('timestamp', inplace=True, axis=1)
    df2 = df2.rename(columns={'lat': 'end_lat', 'lon': 'end_lon'})
    df2 = df2[df2.peaks_number >= 5]

    start_location = []
    end_location = []
    for i in df2.index:
        try:
            a = revgc.search((df2.start_lat[i], df2.start_lon[i]))
            b = revgc.search((df2.end_lat[i], df2.end_lon[i]))
            start_location.append(a[0]['cc'])
            end_location.append(b[0]['cc'])
        except:
            start_location.append(np.nan)
            end_location.append(np.nan)
    df2['start_location'] = start_location
    df2['end_location'] = end_location

    df3 = df2.copy()
    df3 = df3.reset_index(inplace=False, drop=True)
    df3.drop(['start_lat', 'start_lon', 'end_lat',
              'end_lon'], inplace=True, axis=1)

    df_reg = df_regions[['alpha-2', 'sub-region']]
    df_reg = df_reg.dropna(axis=0)
    df_end = pd.merge(df3, df_reg, left_on='start_location',
                      right_on='alpha-2', how='left')
    df_end = pd.merge(df_end, df_reg, left_on='end_location',
                      right_on='alpha-2', how='left')

    df_end.drop(['alpha-2_x', 'alpha-2_y'], axis=1, inplace=True)
    df_end = df_end.rename(
        columns={'sub-region_x': 'start_region', 'sub-region_y': 'end_region'})
    df_end = df_end.dropna(axis=0).reset_index(drop=True)
    df_predict = df_end.copy()
    df_predict.dropna(inplace=True, axis=0)
    df_predict = df_predict[['origin_band', 'start_region', 'end_region', 'duration[s]',
                             'bandwidth[MHz]', 'signal_power[dB]', 'peaks_number', 'max_gradient']]
    v = df_predict.values[:, 0:3]
    v = enc.transform(v).toarray()
    v1 = df_predict.values[:, 3:]
    v = np.concatenate((v, v1), axis=1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(v)
    y_pred = model.predict(data_scaled)
    definitions = factor[1]
    labels = factor[0]
    reversefactor = dict(zip(range(len(definitions)), definitions))
    y_pred_reversed = np.vectorize(reversefactor.get)(y_pred)
    df_end['label'] = y_pred_reversed
    df_end = df_end[['label', 'origin_band', 'start_region', 'end_region', 'duration[s]',
                     'bandwidth[MHz]', 'start_bandwidth', 'end_bandwidth', 'signal_power[dB]', 'peaks_number', 'max_gradient']]

    files = os.listdir(im_path)
    for f in files:
        os.remove(os.path.join(im_path, f))

    return df_end
