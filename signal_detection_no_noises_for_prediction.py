#!/usr/bin/env python
# coding: utf-8

# ## purpose of the script is generating the data for signal classification
# ###### detect the signal automatically from the spectrum
# ###### get as more signal features as possible: duration, bandwidth, power, location of ISS, number of peaks,
# ######    shape of signals, max gradient, mesurement band, distance between global minima/maxima

# In[1]:


import numpy as np
from PIL import Image
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy.signal import find_peaks
from sklearn.preprocessing import normalize
import math
import sqlite3
import reverse_geocoder as revgc



# folder path in that black white images will be saved when the script excecuted
im_path = 'data/images/'


# In[5]:


# database path, it is normally in tmp folder on ubuntu.
database_path = '/tmp/marconissta.db'


# In[6]:


# the folder path of generated data. these csv files will be used in next step - signal clustering
generated_data_folder = '/home/lab/Desktop/marconista/generated_data/'


# ###### get the name of band VHF,UHF,L,S

# In[7]:


p = file.find('_')
band_name = 'VHF'


# In[8]:


df = pd.read_csv(folder+file,header=None)


# In[9]:


df.shape


# In[10]:


df.head()


# ### get date time

# In[11]:


date_time = []
form = '%Y-%m-%d %H:%M:%S'
for i in range(df.shape[0]):
    date_time.append(datetime.datetime.strptime(df[0][i]+' '+df[1][i],form))


# In[12]:


len(date_time)


# In[13]:


values = df.iloc[:,6:].values


# In[14]:


values.shape


# ### get frequency

# In[15]:


fr = (df[3][0]-df[2][0])/values.shape[1]


# In[16]:


bw = list(np.arange(df[2][0],df[3][0],fr))
print(len(bw))
bw


# In[17]:


np.amax(values)


# In[18]:


index_of_min = np.where(values == np.amin(values))


# In[19]:


min_line = values[index_of_min[0][0],:]


# In[20]:


new_values = values - min_line


# In[21]:


print(np.amax(new_values))
print(np.amin(new_values))
print(new_values.shape)


# ### get normalization of values

# In[22]:


normalized_values = normalize(new_values,axis=0).ravel()
normalized_values = normalized_values.reshape(new_values.shape)


# ### get threshold function using boxplot method

# In[23]:


def get_threshold_boxplot_method(val):
    val = sorted(val)
    q1,q3 = np.percentile(val,[25,75])
    iqr = (q3-q1)*1.5
    lower_bound = q1-iqr
    upper_bound = q3+iqr
    
    return upper_bound


# In[24]:


threshold = get_threshold_boxplot_method(normalized_values.flat)


# In[25]:


threshold # threshold of normalized values in csv file to make boolean image


# ### get boolean array to make black-white picture

# In[26]:


bool_array = normalized_values > threshold
bool_array


# In[27]:


bool_array


# ### create black-white picture from boolean array

# In[28]:


image = Image.fromarray(bool_array)
image.save(im_path+file+'.png')
# image.show()
im = cv2.imread(im_path+file+'.png',cv2.IMREAD_UNCHANGED)


# In[29]:


kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))


# ### close the white areas in picture

# In[30]:


closing = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
im2 = Image.fromarray(closing)
im2.save(im_path+file+'_closed'+'.png')
close_im = cv2.imread(im_path+file+'_closed'+'.png',cv2.IMREAD_UNCHANGED)
# im2.show()


# ### open the white areas in picture

# In[31]:


opening = cv2.morphologyEx(close_im, cv2.MORPH_OPEN, kernel)
im3 = Image.fromarray(opening)
im3.save(im_path+file+'_opened'+'.png')
open_im = cv2.imread(im_path+file+'_opened'+'.png',cv2.IMREAD_UNCHANGED)
# im2.show()


# ### get the contour of the white area

# In[32]:


contour_list = cv2.findContours(open_im, cv2.RETR_TREE,  
                    cv2.CHAIN_APPROX_SIMPLE)[-2]


# ### get the rectangle of each contour

# In[33]:


rect_contour = cv2.boundingRect(contour_list[1])


# In[34]:


position_of_signal = []
for i in range(len(contour_list)):
    position_of_signal.append(cv2.boundingRect(contour_list[i]))


# ### calculate duration, bandwidth, time of signals

# In[35]:


duration = []
bandwidth = []
start_time = []
end_time= []
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
        duration.append((date_time[i[1]+i[3]-1] - date_time[i[1]-1]).seconds)
        bandwidth.append((fr*i[2])/1000000)
        start_time.append(date_time[i[1]-1])
        end_time.append(date_time[i[1]+i[3]-1])
        start_bandwidth.append(bw[i[0]]/1000000)
        end_bandwidth.append((bw[i[0]+i[2]-1]+fr)/1000000)


# In[36]:


bw[511]/1000000


# In[37]:


print(len(duration))
print(len(bandwidth))
print(len(start_time))
print(len(end_time))
print(len(start_bandwidth))
print(len(end_bandwidth))


# In[38]:


print(start_bandwidth[1])
print(end_bandwidth[1])
print(bandwidth[1])


# ### calculate the power of signal
# ##### y is 2nd number, x is the 1st number. the 3rd number is width, the 4th number is high

# In[39]:


signal_power = []
signal_var = []
number_of_peaks = []
max_gradient = []
for i in position_of_signal:
    tmp = new_values[i[1]:i[1]+i[3]+1,i[0]:i[0]+i[2]+1]
    box = tmp
    first_line = tmp[0]
    tmp = tmp[tmp > threshold]
    try:
        peaks, _ = find_peaks(tmp, height=threshold)
        first_line_peaks,_ = find_peaks(first_line, height=threshold)
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


# In[40]:


plt.figure(figsize=(8,5))
plt.scatter(bandwidth[:-1],duration[:-1],alpha=0.5,s=100,c='red')
plt.xlabel('bandwidth [MHz]',fontsize=16)
plt.ylabel('duration [s]',fontsize=16)
plt.grid()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('bandwidth-duration of signals',fontsize=16)


# In[41]:


df_signals = pd.DataFrame({'start_time':start_time[:-1],'end_time':end_time[:-1],'duration[s]':duration[:-1],'bandwidth[MHz]':bandwidth[:-1],'signal_power[dB]':signal_power[:-1],
                          'start_bandwidth':start_bandwidth[:-1],'end_bandwidth':end_bandwidth[:-1]})


# In[42]:


pd.set_option('display.max_rows', None)


# In[43]:


print('number of signals is:', df_signals.shape[0])


# ### get the measurement band

# In[44]:


measurement_band = [band_name]*df_signals.shape[0]


# In[45]:


df_signals['band'] = measurement_band
# df_signals['var'] = signal_var[:-1]


# ### get the number of peaks in signal

# In[46]:


df_signals['peaks_number'] = number_of_peaks[:-1]


# ### get maxgradient of signal

# In[47]:


df_signals['max_gradient'] = max_gradient[:-1]


# ### get the location of ISS from database (lat,long)
# ##### by merging the dataframe of signals and the sample table in the database
# ##### then get the location (country) of ISS using (lat,lon)

# In[48]:



connect = sqlite3.connect(database_path)
df_sample = pd.read_sql_query("SELECT timestamp as timestamp, lat as lat, lon as lon FROM sample",connect)
df_sample.head()


# In[49]:


df_sample['timestamp'] = pd.to_datetime(df_sample['timestamp'],format='%Y-%m-%d %H:%M:%S')


# In[50]:


df1 = pd.merge(df_signals,df_sample,left_on='start_time',right_on='timestamp',how='left')


# In[51]:


df1.drop('timestamp',inplace=True,axis=1)


# In[52]:


df1 = df1.rename(columns={'lat':'start_lat','lon':'start_lon'})


# In[53]:


df2 = pd.merge(df1,df_sample,left_on='end_time',right_on='timestamp',how='left')
df2.drop('timestamp',inplace=True,axis=1)
df2 = df2.rename(columns={'lat':'end_lat','lon':'end_lon'})


# In[54]:


df1.shape


# In[55]:


df2 = df2[df2.peaks_number >= 5]


# In[56]:


df2


# In[57]:


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


# In[58]:


df3 = df2.copy()


# In[59]:


df3 = df3.reset_index(inplace=False,drop=True)


# In[60]:


df3.drop(['start_lat','start_lon','end_lat','end_lon'],inplace=True,axis=1)


# In[61]:


df3.to_csv(generated_data_folder+file,index=False)


# In[62]:


print('number of signals: {}'.format(df3.shape[0]))


# In[63]:


df3


# In[64]:


#### change the countries to regions


# In[65]:


df_regions = pd.read_csv('/home/lab/Desktop/marconista/document/ISO-3166-Countries-with-Regional-Codes-master/all/all.csv')
df_reg = df_regions[['alpha-2','sub-region']]
df_reg = df_reg.dropna(axis=0)


# In[66]:


df_end = pd.merge(df3,df_reg,left_on='start_location',right_on='alpha-2',how='left')
df_end = pd.merge(df_end,df_reg,left_on='end_location',right_on='alpha-2',how='left')


# In[67]:


df_end


# In[68]:


df_end.drop(['alpha-2_x','alpha-2_y'],axis=1,inplace=True)
df_end = df_end.rename(columns={'sub-region_x':'start_region','sub-region_y':'end_region'})


# In[69]:


df_end = df_end.dropna(axis=0).reset_index(drop=True)


# In[70]:


df_end


# In[71]:


df_end.to_csv('/home/lab/Desktop/marconista/code/classification/files/data_to_predict1.csv',index=False)


# In[ ]:




