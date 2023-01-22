"""
Description:
    Gradient boosting model to predict the future maintenance of bridges

Date:
   October 3rd, 2022
"""

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

# Permutation importance and pdp
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay

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
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

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
    train_x = np.array(train_x, dtype='f')
    X_train = pd.DataFrame(train_x)
    model = GradientBoostingClassifier(n_estimators=100,
                                       learning_rate=1.0,
                                       max_depth=max_depth,
                                       random_state=0)
    model.fit(X_train, trainy)

    # Permutation mean of the feature importance
    #p_imp = permutation_importance(model,
    #                               test_x,
    #                               testy,
    #                               n_repeats=10,
    #                            random_state=0)
    #p_imp_mean = p_imp.importances_mean
    #p_imp_std = p_imp.importances_std

    ## Partial dependency
    #features = [0, 1]
    #PartialDependenceDisplay.from_estimator(model, X_train, features)
    #print("PartialDependenceDisplay Working OK")

    ## LIME
    #grad_exp_lime = lime_tabular.LimeTabularExplainer(
    #    training_data = np.array(X_train),
    #    feature_names = X_train.columns,
    #    class_names=['Repair', 'No Repair'],
    #    mode='classification'
    #)

    ## Explaining the LIME for specific instance
    #instance_exp = grad_exp_lime.explain_instance(
    #    data_row = X_train.iloc[4],
    #    predict_fn = model.predict_proba
    #)

    #fig = instance_exp.as_pyplot_figure()
    #fig.savefig('grad_lime_report.jpg')

    ##model.fit(train_x, trainy)
    #g_exp = TreeExplainer(model)
    g_exp = shap.Explainer(model, train_x)
    g_sv = g_exp(train_x, check_additivity=False)
    g_sv = g_sv.values

    #g_sv = np.array(g_exp.shap_values(train_x))
    #g_ev = np.array(g_exp.expected_value)

    #print("Printing the sv values")
    #print(np.shape(g_sv))

    # Calculating mean shap values also known as SHAP feature importance
    mean_shap = np.mean(g_sv, axis=0)
    mean_shap_features = {column:shap_v for column, shap_v in zip(cols, mean_shap)}

    #g_sv = g_exp.shap_values(train_x)
    #g_ev = g_exp.expected_value
    #summary_plot(g_sv, train_x, feature_names=cols)

    prediction_prob = model.predict_proba(test_x)[::, 1]
    prediction = model.predict(test_x)
    acc = accuracy_score(testy, prediction)
    _cm = confusion_matrix(testy, prediction)
    _cr = classification_report(testy, prediction, zero_division=0)

    class_label = {'negative':0,
                   'positive':1}

    testy_num = [class_label[i] for i in testy]
    fpr, tpr, threshold = roc_curve(testy_num, prediction_prob)
    _auc = auc(fpr, tpr)
    #print("Printing area under curve")
    #print(_auc)

    _fi = dict(zip(cols, model.feature_importances_))
    _kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')
    instance_exp = []
    g_sv = []

    return acc, _cm, _cr, _kappa, _auc, fpr, tpr, model, _fi, instance_exp, g_sv, mean_shap_features

def main():

    # States
    states = [
             # 'wisconsin_deep.csv',
             # 'colorado_deep.csv',
             # 'illinois_deep.csv',
             # 'indiana_deep.csv',
             # 'iowa_deep.csv',
             # 'minnesota_deep.csv',
             # 'missouri_deep.csv',
             # 'ohio_deep.csv',
              'nebraska_deep.csv',
             # 'indiana_deep.csv',
             # 'kansas_deep.csv',
             ]

    temp_dfs = list()
    for state in states:
        state_file = '../data/' + state
        X, y, cols = preprocess(csv_file=state_file)
        kfold = KFold(5, shuffle=True, random_state=1)

        # X is the dataset
        performance = defaultdict(list)
        for foldTrainX, foldTestX in kfold.split(X):
            trainX, trainy, testX, testy = X[foldTrainX], y[foldTrainX], \
                                              X[foldTestX], y[foldTestX]

            # Training 
            gacc, gcm, gcr, gkappa, gauc, gfpr, gtpr, gmodel, fi, gb_lime, gb_sv, mean_shap_features = gradient_boosting_utility(trainX,
                                                                                           trainy,
                                                     testX, testy, cols, max_depth=7)
            state_name = state[:-9]
            performance['state'].append(state_name)
            performance['accuracy'].append(gacc)
            performance['kappa'].append(gkappa)
            performance['auc'].append(gauc)
            performance['fpr'].append(gfpr)
            performance['tpr'].append(gtpr)
            performance['confusion_matrix'].append(gcm)
            performance['classification_report'].append(gcr)
            performance['feature_importance'].append(fi)
            performance['shap_values'].append(mean_shap_features)
            performance['lime_val'].append(gb_lime)

            # Create a dataframe
            temp_df = pd.DataFrame(performance, columns=['state',
                                                     'accuracy',
                                                     'kappa',
                                                     'auc',
                                                     'fpr',
                                                     'tpr',
                                                     'confusion_matrix',
                                                     'shap_values',
                                                     'lime_val',
                                                    ])
        temp_dfs.append(temp_df)
    performance_df = pd.concat(temp_dfs)
    return performance

if __name__ =='__main__':
    main()
