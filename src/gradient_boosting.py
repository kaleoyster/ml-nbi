"""
Description:
    Gradient boosting model to predict the future maintenance of bridges

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
from sklearn.ensemble import GradientBoostingClassifier

# SHAP
import shap
from shap import TreeExplainer
from shap import summary_plot

# LIME
import lime
from lime import lime_tabular

# Metrics and stats
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

# Preprocessing
from preprocessing import *

def gradient_boosting_utility(train_x, trainy,
                 test_x, testy, cols, max_depth=7):
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
    #TODO: add column names to new dataframe
    X_train = pd.DataFrame(train_x)

    model = GradientBoostingClassifier(n_estimators=100,
                                       learning_rate=1.0,
                                       max_depth=max_depth,
                                       random_state=0)
    model.fit(X_train, trainy)
    grad_exp_lime = lime_tabular.LimeTabularExplainer(
        training_data = np.array(X_train),
        feature_names = X_train.columns,
        class_names=['Repair', 'No Repair'],
        mode='classification'
    )

    # Explaining the LIME for specific instance
    instance_exp = grad_exp_lime.explain_instance(
        data_row = X_train.iloc[4],
        predict_fn = model.predict_proba
    )

    fig = instance_exp.as_pyplot_figure()
    fig.savefig('grad_lime_report.jpg')

    #model.fit(train_x, trainy)
    g_exp = TreeExplainer(model)
    g_sv = np.array(g_exp.shap_values(train_x))
    g_ev = np.array(g_exp.expected_value)

    g_sv = g_exp.shap_values(train_x)
    g_ev = g_exp.expected_value

    summary_plot(g_sv, train_x)

    prediction = model.predict(test_x)
    acc = accuracy_score(testy, prediction)
    _cm = confusion_matrix(testy, prediction)
    _cr = classification_report(testy, prediction, zero_division=0)
    #_fi = dict(zip(cols, model.feature_importances_))
    kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')
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
        gacc, gcm, gcr, gkappa, gmodel = gradient_boosting_utility(trainX, trainy,
                                                 testX, testy, cols, max_depth=7)

        performance['accuracy'].append(gacc)
        performance['kappa'].append(gkappa)
        performance['confusion_matrix'].append(gcm)
        performance['classification_report'].append(gcr)
#
    print('Performance metrics:')
    print(performance['accuracy'])
    print(np.mean(performance['accuracy']))
    print(performance['kappa'])
    print(np.mean(performance['kappa']))

    return performance

if __name__ =='__main__':
    main()
