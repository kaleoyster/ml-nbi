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
    #features = [0, 1]
    #PartialDependenceDisplay.from_estimator(model, train_x, features)
    #print("PartialDependenceDisplay Working OK")

    ## SHAP
    explainer = shap.Explainer(model, train_x)
    shap_values = explainer(train_x)
    int_shap = np.array(shap_values.values,
                        dtype=int)

    # Calculating mean shap values also known as SHAP feature importance
    mean_shap = np.mean(abs(shap_values.values), axis=0)
    mean_shap_features = {column:shap_v for column, shap_v in zip(cols, mean_shap)}

    #print("printing the shape of mean_shap")
    #print(np.shape(mean_shap))

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
    #print("Shapes: ", np.shape(testy_num), np.shape(prediction_prob))
    fpr, tpr, threshold = roc_curve(testy_num, prediction_prob)

    #print(testy_num[:10])
    #print(prediction_prob[:10])

    #print("Checking dimensions")
    #print(np.shape(testy_num), np.shape(prediction_prob))
    #print(np.shape(fpr), np.shape(tpr))
    #print("printing fpr and tpr")
    #print(fpr, tpr)
    _auc = auc(fpr, tpr)
    #print("Printing area under curve")
    #print(_auc)
    #_auc = metrics.roc_auc_score(testy, prediction_prob)
    #_fi = dict(zip(cols, model.feature_importances_))

    instance_exp = []
    int_shap = []
    _kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')
    return acc, _cm, _cr, _kappa, _auc, fpr, tpr, model, instance_exp, int_shap, mean_shap_features

def main():

    # States
    states = [
            # 'wisconsin_deep.csv',
            #  'colorado_deep.csv',
            #  'illinois_deep.csv',
            #  'indiana_deep.csv',
            #  'iowa_deep.csv',
            #  'minnesota_deep.csv',
            #  'missouri_deep.csv',
            #  'ohio_deep.csv',
              'nebraska_deep.csv',
            #  'indiana_deep.csv',
            #  'kansas_deep.csv',
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

            # structure numbers
            gacc, gcm, gcr, gkappa, gauc, gfpr, gtpr, gmodel, lr_sv, lr_lime, mean_shap_features = logistic_regression_utility(trainX, trainy,
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
            performance['lime_val'].append(lr_lime)

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
    return performance_df

if __name__ =='__main__':
    main()
