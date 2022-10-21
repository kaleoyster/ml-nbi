"""
Description:
    Support vector machine to predict the future maintenance of bridges

Date:
   October 3rd, 2022
"""

import sys
import sys
import csv
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import export_graphviz
from collections import defaultdict
from tqdm import tqdm
import pydotplus

# Model 
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

def support_vector_utility(train_x, trainy,
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
        model: Support vector machine Model
    """
    model = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    model.fit(train_x, trainy)
    prediction = model.predict(test_x)
    prediction_prob = model.predict_proba(test_x)
    acc = accuracy_score(testy, prediction)
    _cm = confusion_matrix(testy, prediction)
    _cr = classification_report(testy, prediction, zero_division=0)
    #_fi = dict(zip(cols, model.feature_importances_))
    kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')
#    fpr, tpr, threshold = roc_curve(testy, prediction, pos_label=2)
#    _auc = auc(fpr, tpr)

    return acc, _cm, _cr, kappa, model

def main():
    X, y, cols = preprocess()
    kfold = KFold(5, shuffle=True, random_state=1)

    # X is the dataset
    performance = defaultdict(list)
    for foldTrainX, foldTestX in kfold.split(X):
        trainX, trainy, testX, testy = X[foldTrainX], y[foldTrainX], \
                                          X[foldTestX], y[foldTestX]

        # structure numbers
        gacc, gcm, gcr, gkappa, gmodel = support_vector_utility(trainX, trainy,
                                                 testX, testy, cols)
        performance['accuracy'].append(gacc)
        performance['kappa'].append(gkappa)
        performance['confusion_matrix'].append(gcr)
        performance['classification_report'].append(gcr)

    print('Performance metrics:')
    print(performance['accuracy'])
    print(np.mean(performance['accuracy']))
    print(performance['kappa'])
    print(np.mean(performance['kappa']))

    return performance

if __name__ == '__main__':
    main()
