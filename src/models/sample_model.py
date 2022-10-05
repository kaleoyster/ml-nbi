import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import zscore
from math import radians, cos, sin, asin, sqrt

import pydot
import seaborn as sns
import keras
from keras import metrics
from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

# Label 
label_column = 'price'

# Import dataset
kc_raw_data = pd.read_csv('data/kc_house_data.csv')

# Data slicing 

kc_raw_data['sale_yr'] = pd.to_numeric(kc_raw_data.date.str.slice(0, 4))
kc_raw_data['sale_month'] = pd.to_numeric(kc_raw_data.date.str.slice(4, 6))
kc_raw_data['sale_day'] = pd.to_numeric(kc_raw_data.date.str.slice(6, 8))
kc_data = pd.DataFrame(kc_raw_data,
                       columns = [
                           'sale_yr',
                           'sale_month',
                           'sale_day',
                           'view',
                           'waterfront',
                           'lat',
                           'long',
                           'bedrooms',
                           'bathrooms',
                           'sqft_living',
                           'sqft_lot',
                           'floors',
                           'condition',
                           'grade',
                           'sqft_above',
                           'sqft_basement',
                           'yr_built',
                           'yr_renovated',
                           'zipcode',
                           'sqft_living15',
                           'sqft_lot15',
                           'price'
                       ]
)

kc_data = kc_data.sample(frac=1)
train = kc_data.sample(frac=0.8)
validate = kc_data.sample(frac=0.1)
test = kc_data.sample(frac=0.1)

# Model
t_model = Sequential()
t_model.add(Dense(100, activation='relu', input_shape=(xsize, )))
t_model.add(Dense(50, activation='relu'))
t_model.add(Dense(ysize))
t_model.compile(
    loss='mean_squared_error',
    optimizer=Adam(lr=0.001),
    metrics=[metrics.mae]
)

epochs = 500
batch = 128

cols = list(train.columns)
cols.remove(label_column)
history = model.fit(
    train[cols], train[label_column],
    batch_size=batch,
    epochs=epochs,
    shuffle=True,
    verbose=1,
    validation_data=(validate[cols],validate[label_column]),
    callbacks=keras_callbacks
)
score = model.evaluate(test[cols], test[label_column], verbose=0)

print('score:', score)
train_mean = train[cols].mean(axis=0)
train_std = train[cols].std(axis=0)
train[cols] = (train[cols] - train_mean) / train_std
validate[cols] = (validate[cols] - train_mean) / train_std
test[cols] = (test[cols] - train_mean) / train_std
