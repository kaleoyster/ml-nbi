"""
Description:
    This script provides commonly used tools for
    deep learning.

TODO:
    Divide this into:
    -- Models
    -- Visualization
    -- Sampling

Author: Akshay Kale
"""

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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.utils import resample

def sub_layers_mobinet(x, filtersNo, stridesNo=(1, 1)):
    """
    Description:
        A sublayer of mobinet
    Args:
        x (input layer)
        filterNo (int)
        strideNo=(1, 1)
    Returns:
        returns a set of layers
    """
    layer = DepthwiseConv2D(kernel_size=(3,3), strides=stridesNo, activation='tanh', padding='same')(x)
    layer = BatchNormalization()(x)
    layer = Activation('relu')(x)
    layer = Conv2D(filters=filtersNo, kernel_size=(3, 3), strides=(1, 1), activation='tanh', padding='same')(x)
    layer = BatchNormalization()(x)
    layer = Activation('relu')(x)
    return x

def mobile_net(in_shape):
    """
    Description:
        A simple implementation of mobilenet
    Args:
        in_shape
    Returns:
        Returns model
    """
    input_img = Input(shape=in_shape)
    x = ZeroPadding2D(padding=(2, 2))(input_img)
    x = Conv2D(filters=32,
               kernel_size=(3,3),
               strides=(1,1),
               activation='tanh',
               padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = sub_layers_mobinet(x, 64)
    x = sub_layers_mobinet(x, 128,
                           stridesNo=(2, 2))
    x = sub_layers_mobinet(x, 128)
    x = sub_layers_mobinet(x, 256,
                           stridesNo=(2, 2))
    x = sub_layers_mobinet(x, 256)
    x = sub_layers_mobinet(x, 512,
                           stridesNo=(2, 2))

    for _ in range(5):
        x = sub_layers_mobinet(x, 512)

    x = sub_layers_mobinet(x, 1024,
                           stridesNo=(2, 2))
    x = sub_layers_mobinet(x, 1024)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=10, activation='softmax')(x)
    x = Activation('softmax')(x)
    model = Model(input_img, x)
    opt = SGD(lr=0.001)
    # Alternate
        # loss='sparse_categorical_crossentropy', 
        # optimizer='adam',
        # metrics=['accuracy']
    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['accuracy'])
    plot_model(model,
               show_shapes=True,
               to_file='MobileNet.png')
    return model


def leNet(in_shape):
    """
    Description:
        A simple implementation of leNet
    Args:
        inshape (set)
    Returns:
        model (tensorflow)
    """
    input_img = Input(shape=in_shape)
    x = Conv2D(filters=6, kernel_size=(3,3), strides=(1,1), activation='tanh', padding='same')(input_img)
    x = AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid')(x)
    x = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='tanh', padding='same')(x)
    x = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)
    x = Conv2D(filters=120, kernel_size=(3,3), strides=(1,1), activation='tanh', padding='same')(x)
    x = Flatten()(x)
    x = Dense(units=84, activation='tanh')(x)
    x = Dense(units=10, activation='softmax')(x)
    model =  Model(input_img, x)
    model.compile(loss='mean_squared_error', optimizer='sgd',metrics=['accuracy'])
    plot_model(model, show_shapes=True, to_file='LeNetModel.png')
    return model

def mobilenet_functional(in_shape):
    """
    Description:
        A simple implementation of leNet
    Args:
        in_shape (set)
    Returns:
    """
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(ZeroPadding2D(padding=(2, 2), input_shape=in_shape))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='tanh', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model = sub_layers(model, 64)
    model = sub_layers(model, 128, stridesNo=(2, 2))
    model = sub_layers(model, 128)
    model = sub_layers(model, 256, stridesNo=(2, 2))
    model = sub_layers(model, 256)
    model = sub_layers(model, 512, stridesNo=(2, 2))
    for _ in range(5):
        model = sub_layers(model, 512)

    model = sub_layers(model, 512 )
    model = sub_layers(model, 1024, stridesNo=(2, 2))
    model = sub_layers(model, 1024)

    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10))
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accruacy'])
    plot_model(model, show_shapes=True, to_file='MobileNet.png')
    return model

def sub_layers(model, filtersNo, stridesNo=(1, 1)):
    model.add(DepthwiseConv2D(kernel_size=(3,3), strides=stridesNo, activation='tanh', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=filtersNo, kernel_size=(3, 3), strides=(1, 1), activation='tanh', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    return model

def summarize_acc_magnitude(layer1, layer2, layer3):
    fig, ax = plt.subplots()
    plt.plot(layer1)
    plt.plot(layer2)
    plt.plot(layer3)
    plt.title("Accuracy vs. Magnitude")
    plt.ylabel("Accuracy")
    plt.xlabel("Noise Magnitude")
    ax.set_xticklabels([0, 0.10, 0.20, 0.3, 0.4, 0.5, 0.6, 0.7 ,0.80, 0.90, 1, 1.1, 1.2])
    plt.show()

def summarize_loss_magnitude(layer1, layer2, layer3):
    fig, ax = plt.subplots()
    plt.plot(layer1)
    plt.plot(layer2)
    plt.plot(layer3)
    plt.title("Loss vs. Magnitude")
    plt.ylabel("Loss")
    plt.xlabel("Noise Magnitude")
    ax.set_xticklabels([0, 0.10, 0.20, 0.3, 0.4, 0.5, 0.6, 0.7 ,0.80, 0.90, 1, 1.1, 1.2])
    #plt.legend(["Layer1", "Layer2", "Layer3"], loc="upper left")
    plt.show()

def summarize_train_loss(lossPerNoiseLevel, label):
    for modelLoss in lossPerNoiseLevel:
        plt.plot(modelLoss)
    plt.title(label)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Layer1", "Layer2", "Layer3", "Baseline"], loc="upper left")
    plt.show()

def summarize_train_acc(accPerNoiseLevel, label):
    for modelAcc in accPerNoiseLevel:
        plt.plot(modelAcc)
    #plt.plot(acc)
    plt.title(label)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Layer1", "Layer2", "Layer3", "Baseline"], loc="upper left")
    plt.show()

def summarize_model_loss(lossTrain, lossValidation, label):
    plt.plot(lossTrain)
    plt.plot(lossValidation)
    plt.title(label)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()

def summarize_model_accuracy(accTrain, accValidation, label):
    plt.plot(lossTrain)
    plt.plot(lossValidation)
    plt.title(label)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()

def get_loss(model):
    trainLoss = model.history['loss']
    testLoss = model.history['val_loss']
    return trainLoss, testLoss

def get_acc(model):
    trainAcc = model.history['accuracy']
    testAcc = model.history['val_accuracy']
    return trainAcc, testAcc

def load_mnist():
    """
    Description:
        Loads a total of 50K training and 10K testing MNIST data
    Args:
        None
    Returns:
        xTrain (list): image vectors
        xTest (list): image vectors

    """
    (xTrain, yTrain),  (xTest, yTest) = mnist.load_data()
    xTrain = xTrain.astype("float32")
    xTest = xTest.astype("float32")

    xTrain = xTrain / 255.0
    xTest = xTest / 255.0

    return xTrain, yTrain, xTest, yTest

def load_cifar10():
    """
    Description:
        Loads a total of 50K training and 10K testing Cifar10 data
    Args:
        None
    Returns:
        xTrain (list): image vectors
        xTest (list): image vectors

    """
    (xTrain, yTrain),  (xTest, yTest) = cifar10.load_data()
    xTrain = xTrain.astype("float32")
    xTest = xTest.astype("float32")

    xTrain = xTrain / 255.0
    xTest = xTest / 255.0

    return xTrain, yTrain, xTest, yTest

def create_batch(xDataset, yDataset, size):
    """
    TODO:
            -- Create a batch of labels [Done]
            -- Remove the random assortment for every batch [Done]
            -- Only randomly assort for the last batch [Done]

    Description:
        Generates batch of image specifiec size variable
        Random assortment for all batches will create a bootstrap sampling.
        Thus, creating a yield function stratergy is better than just
        returning it as a list.

    Args:
        xDataset (list): training sample image vectors
        yDataet (list): labels of the training image vectors
        size (int): size of the label

    Return:
        yields a batch of the size
    """
    length = len(xDataset)
    indices = np.arange(length)

    for start in range(0, length, size):
        end = start + size
        if end < length:
            index = indices[start:end]
            yield xDataset[index], yDataset[index]
        else:
            np.random.shuffle(indices)
            start = 0
            end = 64
            index = indices[start:end]
            yield xDataset[index], yDataset[index]


def display(dataset, targetSize, n=16):
    """
    TODO:
            -- Accept dataset with image data and labels [Done]
            -- Separate images from labels and dataset [Done]
            -- Display images with labels -- Not important

    Description:
        Displays the images in a grid
    Args:
        n (int): number of tiles in a grid
        dataset (vector): images in vector format
    """
    X, y =  dataset
    plt.figure(figsize=(30, 10))
    for i in range(n):
        ax = plt.subplot(4, 4, i+1)
        plt.imshow(X[i].reshape(targetSize))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def divide_kFold(xDataset, yDataset):
    """
    Description:
        divide the dataset into training and validation set
        using kFold validation for training and validation set
    Args:
        xDataset (list): image vectors
        yDataset (list): labels
    """
    kfolds = KFold(n_splits=10, shuffle=False, random_state=None)
    trainingSets = list()
    listOfTrainLabels = list()

    testingSets = list()
    listOfTestLabels = list()

    print("Total Size of the Data:", len(xDataset))
    for train, test in kfolds.split(xDataset):
        trainingSet = xDataset[train]
        trainLabels = yDataset[train]

        testingSet = xDataset[test]
        testLabels = yDataset[test]

        trainingSets.append(trainingSet)
        listOfTrainLabels.append(trainLabels)

        testingSets.append(testingSet)
        listOfTestLabels.append(testLabels)

    print("(K fold) K: ", len(trainingSets))
    for eachIteration in range(len(trainingSets)):
        print("(K fold) Number of Training Sample ", eachIteration,":", len(trainingSets[eachIteration]))

    return trainingSets, listOfTrainLabels, testingSets, listOfTestLabels

def divide_bootstrap(xDataset, yDataset):
    """
    Description:
        - Divide the dataset into training and validation set
        using bootstrap validation for training and validation set

        Bootstrap:
           1. multiple train

    Args:
        xDataset (list): image vectors
        yDataset (list): labels
    """
    iterations = 10
    size = 45000

    freeSet = xDataset[:size]
    freeSetLabels = yDataset[:size]

    testSet = xDataset[size:]
    testSetLabel = yDataset[size:]
    bootstrapSize = 10000

    xTrainSet = list()
    yTrainSet = list()

    for eachIteration in range(iterations):
        train, label = resample(freeSet, freeSetLabels, n_samples=bootstrapSize)
        xTrainSet.append(train)
        yTrainSet.append(label)

    print("(Bootstrap) Total Number of Bootstrap samples : ", len(xTrainSet))
    for eachIteration in range(iterations):
        print("(Bootstrap) Number of Training Sample: ", len(xTrainSet[eachIteration]))

    return xTrainSet, yTrainSet, testSet, testSetLabel

# Sampling techniques
def divide_holdout(xDataset, yDataset):
    """
    Description:
        - Divide the dataset into training and validation set
        using holdout validation.
        - This function will generate a list of 10 training set.
    Args:
        xDataset (list): image vector
        yDataset (list): label

    Returns:
        trainingSet (list): list of Training set
        testSet (list): list of Testing set
    """
    size = 45000

    # Train set
    trainSet = xDataset[:size]
    labelTrainSet  = yDataset[:size]

    # Test set
    testSet = xDataset[size:]
    labelTestSet = yDataset[size:]

    print("(Holdout) Total length of the training set: ", len(trainSet))
    print("(Holdout) Total length of the testing set: ", len(testSet))
    print("\n")

    return trainSet, labelTrainSet, testSet, labelTestSet
