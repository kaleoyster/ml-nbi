"""
Description:
    Regression model to predict the future maintenance of bridges

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
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

# SHAP
import shap
from shap import KernelExplainer
from shap import summary_plot

# Metrics and stats
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Preprocessing
from preprocessing import *

def logistic_regression_utility(train_x, trainy,
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
        model: Logistic Regression Model
    """
    model = LogisticRegression(random_state=0)
    model.fit(train_x, trainy)
    prediction_prob = model.predict_proba(test_x)

    # Shap
    # TODO: Don't know how does this work
    testing_data = shap.sample(train_x, 1)
    log_exp = KernelExplainer(model=model.predict_proba, data=testing_data)

    log_sv = np.array(log_exp.shap_values(train_x))
    log_ev = np.array(log_exp.expected_value)

    log_sv = log_exp.shap_values(train_x)
    log_ev = log_exp.expected_value

    # Cat boost:
    print("Shape of the RF values:", log_sv[0])
    #print("Shape of the Light boost Shap Values")
    #summary_plot(log_sv, train_x)

    prediction = model.predict(test_x)
    acc = accuracy_score(testy, prediction)
    _cm = confusion_matrix(testy, prediction)
    _cr = classification_report(testy, prediction, zero_division=0)
    #fpr, tpr, threshold = roc_curve(prediction, prediction_prob, pos_label=2)
    #_auc = auc(fpr, tpr)
    #_auc = metrics.roc_auc_score(testy, prediction_prob)
    #_fi = dict(zip(cols, model.feature_importances_))
    _kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')
    return acc, _cm, _cr, _kappa, model

def main():
    X, y, cols = preprocess()
    kfold = KFold(5, shuffle=True, random_state=1)

    # X is the dataset
    performance = defaultdict(list)
    for foldTrainX, foldTestX in kfold.split(X):
        trainX, trainy, testX, testy = X[foldTrainX], y[foldTrainX], \
                                          X[foldTestX], y[foldTestX]

        # structure numbers
        gacc, gcm, gcr, gkappa, gmodel = logistic_regression_utility(trainX, trainy,
                                                 testX, testy, cols)
        performance['accuracy'].append(gacc)
        performance['kappa'].append(gkappa)
        performance['confusion_matrix'].append(gcm)
        performance['classification_report'].append(gcr)

    print('Performance metrics:')
    print(performance['accuracy'])
    print(np.mean(performance['accuracy']))
    print(performance['kappa'])
    print(np.mean(performance['kappa']))

    return performance

if __name__ =='__main__':
    main()
