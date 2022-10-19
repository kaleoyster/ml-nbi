"""
Description:
    Light boost model to predict the future maintenance of bridges

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
import lightgbm as lgb
from sklearn.model_selection import KFold

# Metrics and stats
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Preprocessing
from preprocessing import *

def light_boost_utility(train_x, trainy,
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
    model = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-5, random_state=42)
    #cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    model.fit(train_x, trainy, eval_set=[(test_x, testy), (train_x, trainy)], verbose=20, eval_metric='logloss')
    #testing_accuracy = model.score(test_x, testy)
    prediction = model.predict(test_x)
    _acc = accuracy_score(testy, prediction)
    _cm = confusion_matrix(testy, prediction)
    _cr = classification_report(testy, prediction, zero_division=0)
    _fi = dict(zip(cols, model.feature_importances_))
    kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')

    #fpr, tpr, threshold = metrics.roc_curve(testy, prediction, pos_label=2)
    #_auc = metrics.auc(fpr, tpr)
    return _acc, _cm, _cr, kappa

def main():
    X, y, cols = preprocess()

    # Convert y into 0, 1
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
    performance = defaultdict(list)
    for foldTrainX, foldTestX in kfold.split(X):
        trainX, trainy, testX, testy = X[foldTrainX], y[foldTrainX], \
                                           X[foldTestX], y[foldTestX]

        # structure numbers
        #gacc, gcm, gcr, gkappa, gmodel = xgb_utility(trainX, trainy,
        #                                          testX, testy, cols)
        acc, cm, cr, kappa = light_boost_utility(trainX, trainy, testX, testy, cols)

        performance['accuracy'].append(acc)
        performance['kappa'].append(kappa)
        performance['confusion_matrix'].append(cm)
        performance['classification_report'].append(cr)

    return performance

if __name__ =='__main__':
    main()
