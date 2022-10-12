"""
Description:
    Xgboost model to predict the future maintenance of bridges

Date:
   October 3rd, 2022
"""

import sys
import csv
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pydotplus
from xgboost import XGBRegressor
from sklearn.model_selection import KFold

# Metrics and stats
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Preprocessing
from preprocessing import *

def convert_prediction_to_binary(predictions, threshold=0.5):
    """
    Description:

    Args:
        predictions:
        threshold:
    Returns:
        binary_conversion
    """
    binary_conversion = []
    for value in predictions:
        if value <= threshold:
            val_binary = 0
            binary_conversion.append(val_binary)
        else:
            val_binary = 1
            binary_conversion.append(val_binary)
    return binary_conversion

def xgb_utility(train_x, trainy,
                 test_x, testy, cols):
    """
    Description:
        Performs the modeling and returns performance metrics

    Args:
        trainX: Features of Training Set
        trainy: Ground truth of Training Set
        testX: Features of Testing Set
        testy: Ground truth of Testing Set

    Return:
        acc: Accuracy
        cm: Confusion Report
        cr: Classification Report
        kappa: Kappa Value
        model: Random Forest Model
    """
    model = XGBRegressor(objective='reg:squarederror')
    #cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    model.fit(train_x, trainy)
    prediction = model.predict(test_x)
    b_prediction = convert_prediction_to_binary(prediction)

    #_mse = mean_squared_error(prediction, testy)
    _acc = accuracy_score(testy, b_prediction)
    _cm = confusion_matrix(testy, b_prediction)
    _cr = classification_report(testy, b_prediction, zero_division=0)
    #_fi = dict(zip(cols, model.feature_importances_))
    _kappa = cohen_kappa_score(b_prediction, testy,
                              weights='quadratic')
    fpr, tpr, threshold = roc_curve(testy, prediction, pos_label=2)
    _auc = auc(fpr, tpr)
    return _acc, _cm, _cr, _kappa

def main():
    X, y, cols = preprocess()

    # Convert y into 0, 1
    # If positive = 1
    # If negative = 0
    new_y = []
    for value in y:
        if value == 'positive':
            new_val = 1
        else:
            new_val = 0
        new_y.append(new_val)
    y = np.array(new_y)

    kfold = KFold(5, shuffle=True, random_state=1)

    #X is the dataset
    for foldTrainX, foldTestX in kfold.split(X):
        trainX, trainy, testX, testy = X[foldTrainX], y[foldTrainX], \
                                           X[foldTestX], y[foldTestX]

        # Check the distribution
        # structure numbers
        acc, cm, cr, kappa = xgb_utility(trainX, trainy, testX, testy, cols)

main()
