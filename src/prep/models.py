"""
Description:
    Function that houses all the available deep learning models
Author:
    Akshay Kale
Date:
    August 13th, 2021
"""
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
import plotly.express as px

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.models as models
import tensorflow.keras.layers as layer
import tensorflow.keras.datasets as dataset
import tensorflow.keras.optimizers as optimizers

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GaussianNoise

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Metrics and stats
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# test drivers
def leNet(in_shape):
    """
    Description:
    Args:
    Returns:
    """
    input_img = Input(shape=in_shape)
    x = Conv2D(filters=6, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(input_img)
    x = AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid')(x)
    x = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
    x = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)
    x = Conv2D(filters=120, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(units=84, activation='relu')(x)
    x = Dense(units=2, activation='softmax')(x)
    model =  Model(input_img, x)
    model.compile(loss='binary_crossentropy', optimizer='sgd',metrics=['accuracy'])
    plot_model(model, show_shapes=True, to_file='LeNetModel.png')
    return model

def iris_model(in_shape):
    """
    Description: A simple implementation of mobilenet
    Args: in_shape
    Returns: Returns model
    """
    return model

def mobile_net(in_shape):
    """
    Description: A simple implementation of mobilenet
    Args: in_shape
    Returns: Returns model
    """
    input_img = Input(shape=in_shape)
    x = ZeroPadding2D(padding=(2, 2))(input_img)
    x = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='tanh', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = sub_layers_mobinet(x, 64)
    x = sub_layers_mobinet(x, 128, stridesNo=(2, 2))
    x = sub_layers_mobinet(x, 128)
    x = sub_layers_mobinet(x, 256, stridesNo=(2, 2))
    x = sub_layers_mobinet(x, 256)
    x = sub_layers_mobinet(x, 512, stridesNo=(2, 2))

    for _ in range(5):
        x = sub_layers_mobinet(x, 512)

    x = sub_layers_mobinet(x, 1024, stridesNo=(2, 2))
    x = sub_layers_mobinet(x, 1024)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=2, activation='softmax')(x)
    x = Activation('softmax')(x)
    model = Model(input_img, x)
    opt = SGD(lr=0.1)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    plot_model(model, show_shapes=True, to_file='MobileNet.png')
    return model

def sub_layers_mobinet(x, filtersNo, stridesNo=(1, 1)):
    """
    Description:
        A simple implementation
    Args:
    Returns:
    """
    layer = DepthwiseConv2D(kernel_size=(3,3), strides=stridesNo, activation='tanh', padding='same')(x)
    layer = BatchNormalization()(x)
    layer = Activation('relu')(x)
    layer = Conv2D(filters=filtersNo, kernel_size=(3, 3), strides=(1, 1), activation='tanh', padding='same')(x)
    layer = BatchNormalization()(x)
    layer = Activation('relu')(x)
    return x

# define cnn model
def define_model():
    """
    Descriptions:
    Args:
    Returns:
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax'),
        ]
    )

    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


    #model = Sequential()
    #model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(8, 12, 1)))
    #model.add(MaxPooling2D((2, 2)))
    #model.add(Flatten())
    #model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    #model.add(Dense(2, activation='softmax'))
    # compile model
    #opt = SGD(lr=0.1, momentum=0.9)
    #model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def model2():
    """
    Description:
    Args:
    Returns:
    """
    model=Sequential()
    model.add(Conv2D(28,(5,5),padding='same',input_shape=(10, 10, 1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(28,(5,5)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    # model.add(Conv2D(32,(5,5),padding='same',input_shape=(10, 10, 1)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(32,(5,5)))
    # model.add(Activation('relu'))
    # model.add(MaxPool2D(pool_size=(2,2)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    """
    Description:
        Performs a training and testing
        using cross-validation of the desired
        model

    Args:
        dataX (Training set):
        dataY (testing set) :

    Returns:
        scores training and testing model scores
        histories training and testing
   """
    #oversample = RandomOverSampler(sampling_strategy='minority')
    #print(dataX[:5])
    #dataX, dataY = oversample.fit_resample(dataX, dataY)
    scores, histories = list(), list()

    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)

    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        # Define model
        #model = define_model()
        #model = model2()
        #model = leNet((8, 12, 1))
        model = mobile_net((8, 12, 1))

        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit model
        history = model.fit(trainX,
                            trainY,
                            epochs=10,
                            batch_size=32,
                            validation_data=(testX, testY), verbose=2)
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))

        # stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories

def main():
    pass

if __name__=='__main__':
   main()
