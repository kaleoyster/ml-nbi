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

# Permutation importance
from sklearn.inspection import PartialDependenceDisplay

# SHAP
import shap
from shap import KernelExplainer
from shap import summary_plot

# LIME
import lime
from lime import lime_tabular

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
    X_train = pd.DataFrame(train_x)
    model = LogisticRegression(random_state=0)
    model.fit(train_x, trainy)

    # Partial dependency
    features = [0, 1]
    PartialDependenceDisplay.from_estimator(model, train_x, features)
    print("PartialDependenceDisplay Working OK")


    # SHAP
    explainer = shap.Explainer(model, test_x)
    shap_values = explainer(test_x)
    int_shap = np.array(shap_values.values, dtype=int)

    # LIME:
    log_exp_lime = lime_tabular.LimeTabularExplainer(
        training_data = train_x,
        feature_names = X_train.columns,
        class_names=['Repair', 'No Repair'],
        #mode='regression'
        discretize_continuous = True
    )

    ## Explaining the instances using LIME
    instance_exp = log_exp_lime.explain_instance(
        data_row = X_train.values[4],
        predict_fn = model.predict_proba
    )

    fig = instance_exp.as_pyplot_figure()
    fig.savefig('lg_lime_report.jpg')

    summary_plot(int_shap, feature_names=cols)

    prediction_prob = model.predict_proba(test_x)
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
    return acc, _cm, _cr, _kappa, model, instance_exp, int_shap

def main():
    X, y, cols = preprocess()
    kfold = KFold(2, shuffle=True, random_state=1)

    # X is the dataset
    performance = defaultdict(list)
    for foldTrainX, foldTestX in kfold.split(X):
        trainX, trainy, testX, testy = X[foldTrainX], y[foldTrainX], \
                                          X[foldTestX], y[foldTestX]

        # structure numbers
        gacc, gcm, gcr, gkappa, gmodel, lr_sv, lr_lime = logistic_regression_utility(trainX, trainy,
                                                 testX, testy, cols)
        performance['accuracy'].append(gacc)
        performance['kappa'].append(gkappa)
        performance['confusion_matrix'].append(gcm)
        performance['classification_report'].append(gcr)
        performance['shap_values'].append(lr_sv)
        performance['lime_val'].append(lr_lime)

    print('Performance metrics:')
    print(performance['accuracy'])
    print(np.mean(performance['accuracy']))
    print(performance['kappa'])
    print(np.mean(performance['kappa']))

    return performance

if __name__ =='__main__':
    main()
