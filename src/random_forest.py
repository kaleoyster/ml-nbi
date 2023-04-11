"""
Description:
    Random forest model to predict the future maintenance of bridges

Start Date:
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
from sklearn.ensemble import RandomForestClassifier

# Permutation Feature Importance
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay

# From Shap
import shap
from shap import TreeExplainer
from shap import summary_plot

# Import Lime:
import lime
from lime import lime_tabular

# Metrics and Stats
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Preprocessing
from preprocessing import *

def random_forest_utility(train_x, trainy,
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
        fpr: false positive rate
        tpr: true positive rate
        mean_shap_features: shapley values
        model: Random Forest Model
    """
    X_merged = np.concatenate((train_x, test_x))
    X_merged = np.array(X_merged, dtype='f')

    # Training and testing model 
    X_train = pd.DataFrame(train_x)
    train_x = np.array(train_x, dtype='f')

    # Initialize model
    model = RandomForestClassifier(max_depth=max_depth,
                                   random_state=0)
    # Fit model
    model.fit(X_train, trainy)

    # p_imp = permutation_importance(model,
    #                                test_x,
    #                                testy,
    #                                n_repeats=10,
    #                             random_state=0)

    # p_imp_mean = p_imp.importances_mean
    # p_imp_std = p_imp.importances_std

    # # Partial dependency
    # features = [0, 1]
    # PartialDependenceDisplay.from_estimator(model, train_x, features)
    # print("PartialDependenceDisplay Working OK")

    # ## Lime explainer
    # rf_exp_lime = lime_tabular.LimeTabularExplainer(
    #     training_data = np.array(X_train),
    #     feature_names = X_train.columns,
    #     class_names=['Repair', 'No Repair'],
    #     mode='classification'
    # )

    # ## Explaining the instances using LIME
    # instance_exp = rf_exp_lime.explain_instance(
    #     data_row = X_train.iloc[4],
    #     predict_fn = model.predict_proba
    # )

    # fig = instance_exp.as_pyplot_figure()
    # fig.savefig('lime_report.jpg')
    #rf_exp_lime.show_in_notebook(show_table=True)

    # SHAP explainer
    #rf_exp = shap.Explainer(model, X_merged)
    #rf_sv = rf_exp(X_merged, check_additivity=False)
    #mean_shap = np.mean(abs(rf_sv.values), axis=0).mean(1)

    # Calculating mean shap values also known as SHAP feature importance
    #mean_shap_features = {column:shap_v for column, shap_v in zip(cols, mean_shap)}
    mean_shap_features = {}

    # Predictions
    prediction = model.predict(test_x)
    prediction_prob = model.predict_proba(test_x)[::, 1]

    # Computing metrics
    acc = accuracy_score(testy, prediction)
    _cm = confusion_matrix(testy, prediction)
    _cr = classification_report(testy, prediction, zero_division=0)

    class_label = {'negative':0,
                   'positive':1}

    testy_num = [class_label[i] for i in testy]
    fpr, tpr, threshold = roc_curve(testy_num, prediction_prob)
    _auc = auc(fpr, tpr)

    _fi = dict(zip(cols, model.feature_importances_))
    kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')

    return acc, _cm, _cr, kappa, _auc, fpr, tpr, model, _fi, mean_shap_features

def main():

    # States
    states = ['nebraska_deep.csv']

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
            gacc, gcm, gcr, gkappa, gauc, gfpr, gtpr, gmodel, fi, mean_shap_features = random_forest_utility(trainX, trainy,
                     testX, testy, cols, max_depth=10)
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

            # Create a temp dataframe
            temp_df = pd.DataFrame(performance, columns=['state',
                                                     'accuracy',
                                                     'kappa',
                                                     'auc',
                                                     'fpr',
                                                     'tpr',
                                                     'confusion_matrix',
                                                     'shap_values',
                                                    ])
        temp_dfs.append(temp_df)
    performance_df = pd.concat(temp_dfs)
    return performance_df

if __name__ =='__main__':
    main()
