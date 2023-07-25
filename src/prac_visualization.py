"""
Description:
    Regression model to predict the future maintenance of bridges
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
        auc: Area Under Curve
        fpr: False Positive Rate
        tpr: True Positive Rate
        model: Logistic Regression Model
    """
    X_merged = np.concatenate((train_x, test_x))
    X_merged = np.array(X_merged, dtype='f')
    X_train = pd.DataFrame(train_x)

    # initialize Model
    model = LogisticRegression(random_state=0)

    # Fit Model
    model.fit(train_x, trainy)

    # SHAP
    explainer = shap.Explainer(model, X_merged)
    shap_values = explainer(X_merged)
    int_shap = np.array(shap_values.values,
                        dtype=int)

    # Calculating mean shap values also known as SHAP feature importance
    mean_shap = np.mean(abs(shap_values.values), axis=0)
    mean_shap_features = {column:shap_v for column, shap_v in zip(cols, mean_shap)}
    #mean_shap_features = {}

    # Partial dependency
    #features = [0, 1]
    #PartialDependenceDisplay.from_estimator(model, train_x, features)
    #print("PartialDependenceDisplay Working OK")


    # # LIME:
    # log_exp_lime = lime_tabular.LimeTabularExplainer(
    #     training_data = train_x,
    #     feature_names = X_train.columns,
    #     class_names=['Repair', 'No Repair'],
    #     #mode='regression'
    #     discretize_continuous = True
    # )

    # ## Explaining the instances using LIME
    # instance_exp = log_exp_lime.explain_instance(
    #     data_row = X_train.values[4],
    #     predict_fn = model.predict_proba
    # )

    # fig = instance_exp.as_pyplot_figure()
    # fig.savefig('lg_lime_report.jpg')
    # summary_plot(int_shap, train_x, feature_names=cols)

    prediction_prob = model.predict_proba(test_x)[::, 1]
    prediction = model.predict(test_x)
    acc = accuracy_score(testy, prediction)
    _cm = confusion_matrix(testy, prediction)
    _cr = classification_report(testy,
                                prediction,
                                zero_division=0)
    class_label = {
                   'negative':0,
                   'positive':1
                  }

    testy_num = [class_label[i] for i in testy]
    fpr, tpr, threshold = roc_curve(testy_num, prediction_prob)
    _auc = auc(fpr, tpr)
    _kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')
    return acc, _cm, _cr, _kappa, _auc, fpr, tpr, model, mean_shap_features

def main():

    # States
    states = ['nebraska_deep.csv']
    temp_dfs = list()
    for state in states:
        state_file = '../data/' + state
        X, y, cols = preprocess(csv_file=state_file)
        kfold = KFold(5, shuffle=True, random_state=42)
        X = X[:, :3]
        cols = cols[:3]


        # X is the dataset
        performance = defaultdict(list)
        for foldTrainX, foldTestX in kfold.split(X):
            trainX, trainy, testX, testy = X[foldTrainX], y[foldTrainX], \
                                              X[foldTestX], y[foldTestX]

            # structure numbers
            gacc, gcm, gcr, gkappa, gauc, gfpr, gtpr, gmodel, mean_shap_features = logistic_regression_utility(trainX, trainy,
                                                     testX, testy, cols)
            state_name = state[:-9]
            performance['state'].append(state_name)
            performance['accuracy'].append(gacc)
            performance['kappa'].append(gkappa)
            performance['auc'].append(gauc)
            performance['fpr'].append(gfpr)
            performance['tpr'].append(gtpr)
            performance['confusion_matrix'].append(gcm)
            performance['classification_report'].append(gcr)
            performance['shap_values'].append(mean_shap_features)

            # Create a dataframe
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

    # Performance dataframe
    performance_df = pd.concat(temp_dfs)
    df_perf = performance_df[['accuracy', 'kappa', 'auc']]

    # Create FPR dataframe
    fprs = [fpr for fpr in performance_df['fpr']]
    fprs_df = pd.DataFrame(fprs).transpose()
    fprs_df.columns=['k1', 'k2', 'k3', 'k4', 'k5']

    # Create TPR dataframe
    tprs = [tpr for tpr in performance_df['tpr']]
    tprs_df = pd.DataFrame(tprs).transpose()
    tprs_df.columns=['k1', 'k2', 'k3', 'k4', 'k5']

    # Combine the dictionaries for shap values
    dict1, dict2, dict3, dict4, dict5 = performance_df['shap_values']

    # Combined dictionary
    combined_dict = defaultdict()
    for key in dict1.keys():
        vals = []
        val1 = dict1[key]
        val2 = dict2[key]
        val3 = dict3[key]
        val4 = dict4[key]
        val5 = dict5[key]
        mean_val = np.mean([val1, val2, val3, val4, val5])
        combined_dict[key] = mean_val

    # Convert the dictionary into a pandas DataFrame
    df = pd.DataFrame.from_dict(combined_dict, orient='index', columns=['values'])

    # Reset index and rename column
    df = df.reset_index().rename(columns={'index': 'features'})
    print(df_perf)
    #df.to_csv('logistic_regression_shap_values_deck.csv')
    #df_perf.to_csv('logistic_regression_performance_values_deck.csv')
    #fprs_df.to_csv('logistic_regression_fprs_deck.csv')
    #tprs_df.to_csv('logistic_regression_tprs_deck.csv')

    return performance_df

if __name__ =='__main__':
    main()
