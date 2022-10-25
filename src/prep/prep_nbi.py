"""
Decription:
    Data preparation script

Author:
    Akshay Kale
Date:
    August 1, 2021
"""
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
import plotly.express as px

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

from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from collections import Counter
from prep.models import *
from prep.kmeans import *
from prep.decision_tree import *

# ConvertInt
def convertInt(df, columns):
    """
    from sklearn import preprocessing
    Descriptions:
    Args:
    Returns:
    """
    for feature in columns:
        df[feature] = df[feature].astype(int)
    return df

# Normalize
def normalize(df, columns):
    """
    Descriptions:
    Args:
    Returns:
    """
    for feature in columns:
        max_value = df[feature].max()
        min_value = df[feature].min()
        df[feature] = (df[feature] - min_value) \
                     / (max_value - min_value)
    return df

# Convert labels
def convert_labels(groundList):
    """
    Descriptions:
        Preprocessing function to convert
        into groundList

    Args:
        groundlist(list)
    Returns:
        Transformed groundlist
    """
    le = preprocessing.LabelEncoder()
    le.fit(groundList)
    groundList = le.transform(groundList)
    return groundList

# Scale pixels
def prep_pixels(train, test):
    """
    Descriptions:
        Function
    Args:
    Returns:
    """
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    # return normalized images
    return train_norm, test_norm

def display(dataset, targetSize, n=9):
    """
    Description:
        Displays the images in a grid
    Args:
        n (int): number of tiles in a grid
        dataset (vector): images in vector format
    """
    X, y =  dataset
    plt.figure(figsize=(30, 10))
    for i in range(n):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(X[i].reshape(targetSize))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

# plot diagnostic learning curves
def summarize_diagnostics(histories):
    """
    Description:
        Displays performance and
    Args:
        n (int): number of tiles in a grid
        dataset (vector): images in vector format
    """
    colorTrain = ['orange', 'red', 'purple', 'brown', 'black']
    colorTest = ['blue', 'green', 'yellow', 'aqua', 'pink']
    pyplot.figure(figsize=(15, 15))
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss', fontsize=25)
        pyplot.plot(histories[i].history['loss'], color=colorTrain[i], label='train ' + str(i))
        pyplot.plot(histories[i].history['val_loss'], color=colorTest[i], label='test '+ str(i))
        plt.ylabel("Loss", fontsize=20)
        plt.xlabel("Epoch", fontsize=20)
        pyplot.legend()

        # plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy', fontsize=25)
        pyplot.plot(histories[i].history['accuracy'], color=colorTrain[i], label='train ' + str(i))
        pyplot.plot(histories[i].history['val_accuracy'], color=colorTest[i], label='test ' + str(i))
        plt.ylabel("Accuracy", fontsize=20)
        plt.xlabel("Epoch", fontsize=20)
        pyplot.legend()
    pyplot.show()

# Summarize model performance
def summarize_performance(scores):
    """
    Description:
    Args:
    Returns:
    """
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores)*(100), np.std(scores)*(100), len(scores)))

    # box and whisker plots of results
    pyplot.figure(figsize=(10, 10))
    pyplot.title('Distribution of Performance', fontsize=25)
    pyplot.boxplot(scores)
    pyplot.show()

def prepare_xy(newDf, column, groundtruth):
    """
    Description:
       Prepare the factors and groundtruth

    Args:
        newDF:
        column:
        groundtruth:

    Returns:
        data: nbi dataset
        groundtruthlist: semantic labeling
    """
    groundtruthList = list()
    data = list()
    column.remove('structureNumber')
    column.remove('year')
    column.remove(groundtruth)
    counter = 0
    selectedCounter = 0
    for record in newDf.groupby('structureNumber'):
        structureNumber, rec = record
        temp = pd.DataFrame(rec)
        temp.drop_duplicates(subset=['year'], keep='first', inplace=True)
        rows, cols = temp.shape
        counter = counter + 1
        if rows > 7:
          selectedCounter = selectedCounter + 1
          deteriorationScore = list(temp[groundtruth])[0]
          b = temp.head(8)[column]
          data.append(b.values)
          groundtruthList.append(deteriorationScore)

    return data, groundtruthList

def categorize_groundtruth(scores):
    """
    Description:
        categorize scores
    """
    _min = np.min(scores)
    _max = np.max(scores)
    mean = np.mean(scores)
    std = np.std(scores)

    negStd = mean - std
    posStd = mean + std

    cat = list()
    for score in scores:
        if score >= _min and score < negStd:
          cat.append('0')
        elif score >= negStd and score < mean:
          cat.append('0')
        elif score >= mean and score < posStd:
          cat.append('1')
        elif score:
          cat.append('1')
    return cat

def data_preprocessing():
    """
    Description:
        Pipeline for create images of
    """
    df = pd.read_csv("../data/nebraska_deep.csv",
                     index_col=None,
                     low_memory=False)

    # Remove null values
    df = df.dropna(subset=['deck', 'substructure', 'superstructure'])

    # Remove values encoded as N
    df = df[~df['deck'].isin(['N'])]
    df = df[~df['substructure'].isin(['N'])]
    df = df[~df['superstructure'].isin(['N'])]
    df = df[~df['deckStructureType'].isin(['N'])]
    df = df[~df['scourCriticalBridges'].isin(['N', 'U', np.nan])]
    #df = df[~df['baseDifferenceScore'].isin(['N', 'U', np.nan])]
    df = df[~df['subNumberIntervention'].isin(['N', 'U', np.nan])]
    df = df[~df['deckNumberIntervention'].isin(['N', 'U', np.nan])]


    # Fill the null values with -1
    df.snowfall.fillna(value=-1, inplace=True)
    df.precipitation.fillna(value=-1, inplace=True)
    df.freezethaw.fillna(value=-1, inplace=True)

    # Create a new column 
    df['age'] = df['year'] - df['yearBuilt']

    #---------------------- Semantic Labeling -----------------------#
    # Perform semantic labeling
    #features = ["supNumberIntervention",
    #            "subNumberIntervention",
    #            "deckNumberIntervention"]

    #sLabels = semantic_labeling(df[features], name="")
    #df['cluster'] = sLabels
    #df = create_labels(df, 'No Substructure - High Deck - No Superstructure')
    #---------------------- Semantic Labeling -----------------------#

    # TODO:
    # Create label of substructure:
    print("\n printing list of substructure")
    df = create_condition_label(df, 'substructure')

    # Select columns for conversion and normalization
    columns = [
               'yearBuilt',
               'averageDailyTraffic',
               'avgDailyTruckTraffic',
               'designLoad',
               'numberOfSpansInMainUnit',
               'structureLength',
               'deck',
               'material',
               #'structureType',
               #'wearingSurface',
               'substructure',
               'superstructure',
               'operatingRating',
               #'bridgeImprovementCost',
               #'totalProjectCost',
               #'futureAvgDailyTraffic',
               'precipitation',
               'snowfall',
               'freezethaw',
               'age',
               'subNumberIntervention',
               'deckNumberIntervention',
               'label'
                ]

    # Columns in the dataset:
    columns = [#'year',
              #'structureNumber',
              'latitude',
              'longitude',
              'toll',
              'owner',
              'yearBuilt',
              'averageDailyTraffic',
              'designLoad',
              'skew',
              'numberOfSpansInMainUnit',
              'lengthOfMaximumSpan',
              'structureLength',
              'substructure',
              'deck',
              'superstructure',
              'operatingRating',
              'designatedInspectionFrequency',
              #'structureNumber',
              'year',
              'deckStructureType',
              'avgDailyTruckTraffic',
              'scourCriticalBridges',
              'lanesOnStructure',
              'typeOfDesign',
              'material',
              #'baseDifferenceScore',
              #'precipitation',
              #'snowfall',
              'freezethaw',
              'label']

    columnsFinal = ['year',
                    #'baseDifferenceScore',
                    'longitude',
                    'latitude',
                    #'owner',
                    'numberOfSpansInMainUnit',
                    #'structureNumber',
                    #'material',
                    #'toll',
                    'lanesOnStructure',
                    'scourCriticalBridges',
                    'averageDailyTraffic',
                    'avgDailyTruckTraffic',
                    'deckStructureType',
                    'substructure',
                    'superstructure',
                    'deck',
                    #'designLoad',
                    #'operatingRating',
                    #'designatedInspectionFrequency',
                    #'skew',
                    'yearBuilt',
                    #'deckNumberIntervention',
                    #'subNumberIntervention',
                    #'supNumberIntervention'
                   ]

    columns = columnsFinal
    df = convertInt(df, columns)
    df = normalize(df, columns)

    columns.append('structureNumber')
    columns.append('label')
    groundtruth = 'label'
    newDf = df[columns]

    #---- Selection of the structure number and groundtruth --------#
    structSet = [(structNo, groundtruth) for structNo, groundtruth in zip(newDf['structureNumber'], newDf['label'])]
    print("\nPrinting the dataframe")
    print(structSet)
    #---- Selection of the structure number and groundtruth --------#

    ## Remove null values from groundtruth
    newDf.dropna(subset=[groundtruth], inplace=True)

    ## Preparing the overall dataset
    data, groundtruthList = prepare_xy(newDf,
                                      columns,
                                      groundtruth)

    ### Display Boxplot and distribution plot
    groundTruth = np.array(groundtruth)
    #fig = px.box(pd.DataFrame(groundTruth,
                              #columns=['score']),
                              #y="score")
    #fig.show()

    ## Categorization of the ground truth
    #cat = categorize_groundtruth(groundtruthList)
    #cat = df[groundTruth]

    groundtruthList = convert_labels(groundtruthList)
    cat = groundtruthList
    randomData = list()
    for iter in range(len(data)):
        randomData.append(list(np.random.rand(10, 10)))

    #TODO: Create a function to generate random dataset
            # For comparison of performance
    #randomData = np.array(randomData)
    #randomData = list()
    #for iter in range(len(data)):
    #    randomData.append(list(np.random.rand(10, 10)))
    #randomData = np.array(randomData)

    # Train test split (training  set = 70%, testing set = 30%)
    trainX, testX, trainY, testY = train_test_split(data,
                                                    cat,
                                                    test_size=0.3,
                                                    random_state=42)
    # Converting into numpy array
    trainX = np.array(trainX)
    testX = np.array(testX)
    trainY = np.array(trainY)
    testY = np.array(testY)

    # Reshaping the values of the numpy array
    trainX = trainX.reshape((trainX.shape[0], 8, 12, 1))
    testX = testX.reshape((testX.shape[0], 8, 12, 1))

    # Treating ground truth as categorical variable
    #trainY = to_categorical(trainY)
    #testY = to_categorical(testY)

    # Preparing pixels
    trainX, testX = prep_pixels(trainX, testX)
    print("\nTraining label distribution:")
    print(Counter(trainY))
    print("\nTesting label distribution:")
    print(Counter(testY))

    # Display a sample of images
    #display([trainX, trainY], (8, 12))
    return trainX, trainY

def main():
    trainX, trainY  = data_preprocessing()

if __name__=='__main__':
    main()
