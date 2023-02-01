"""
Description:
    Support vector machine to predict the future maintenance of bridges

Date:
   October 3rd, 2022
"""

# Essentials
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
from sklearn import svm

# PDP
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
    # Training and testing data
    train_x = np.array(train_x, dtype='f')
    X_train = pd.DataFrame(train_x, columns=cols)

    # Model initialization
    model = make_pipeline(StandardScaler(),
                          SVC(gamma='auto',
                              probability=True))
    # Fit model
    #model = svm.SVC()
    model.fit(train_x, trainy)
    features = [0, 1]

    #PartialDependenceDisplay.from_estimator(model, train_x, features)
    #print("PartialDependenceDisplay Working OK")

    #data_sample = shap.sample(test_x, 20)
    #svm_exp = shap.Explainer(model.predict_proba, data_sample)
    #print("printing shap values")
    #print(svm_exp)
    #svm_sv = svm_exp(train_x)


    #data_sample = shap.kmeans(test_x[:100], 100)
    #data_sample = shap.sample(test_x, 100)
    #svm_exp = shap.KernelExplainer(model=model.predict_proba,
    #                               data=data_sample,
    #                               link='logit')
    #print(svm_exp)
    #shap_values = svm_exp.shap_values(X=train_x, nsamples=100)
    #svm_sv = svm_exp(train_x)

    #print(svm_exp.shap_values(test_x))

    #svm_sv = np.array(svm_exp.shap_values(train_x))
    #print(svm_exp)

    #svm_ev = np.array(svm_exp.expected_value)
    #svm_sv = svm_exp.shap_values(train_x)

    # Calculating mean shap values also known as SHAP feature importance
    #mean_shap = np.mean(svm_sv, axis=0)
    #mean_shap_features = {column:shap_v for column, shap_v in zip(cols, mean_shap)}

    ##svm_ev = svm_exp.expected_value

    ## Partial dependency
    #features = [0, 1]
    #PartialDependenceDisplay.from_estimator(model, X_train, features)
    #print("PartialDependenceDisplay Working OK")

    ## LIME:
    #svm_exp_lime = lime_tabular.LimeTabularExplainer(
    #    training_data = np.array(X_train),
    #    feature_names = X_train.columns,
    #    class_names=['Repair', 'No Repair'],
    #    mode='classification'
    #)

    ### Explaining the instances using LIME
    #instance_exp = svm_exp_lime.explain_instance(
    #    data_row = X_train.values[4],
    #    predict_fn = model.predict_proba
    #)

    #fig = instance_exp.as_pyplot_figure()
    #fig.savefig('svm_lime_report.jpg')

    #summary_plot(svm_sv, train_x, feature_names=cols)

    prediction = model.predict(test_x)
    prediction_prob = model.predict_proba(test_x)[::, 1]
    acc = accuracy_score(testy, prediction)
    _cm = confusion_matrix(testy, prediction)
    _cr = classification_report(testy, prediction, zero_division=0)

    class_label = {'negative':0,
                   'positive':1 }

    testy_num = [class_label[i] for i in testy]
    fpr, tpr, threshold = roc_curve(testy_num, prediction_prob)
    _auc = auc(fpr, tpr)
    #print("Printing area under curve")
    #print(_auc)

    #_fi = dict(zip(cols, model.feature_importances_))
    _kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')
    #_auc = auc(fpr, tpr)

    instance_exp = []
    svm_sv = []
    mean_shap_features = []
    return acc, _cm, _cr, _kappa, _auc, fpr, tpr, model, instance_exp, svm_sv, mean_shap_features

def main():

    # States
    states = [
              #'wisconsin_deep.csv',
              #'colorado_deep.csv',
              #'illinois_deep.csv',
              #'indiana_deep.csv',
              #'iowa_deep.csv',
              #'minnesota_deep.csv',
              #'missouri_deep.csv',
              #'ohio_deep.csv',
              'nebraska_deep.csv',
              #'indiana_deep.csv',
              #'kansas_deep.csv',
             ]

    temp_dfs = list()
    for state in states:
        state_file = '../data/' + state
        X, y, cols = preprocess(csv_file=state_file)

        # K-fold cross validation
        kfold = KFold(5, shuffle=True, random_state=1)

        gmodels = []
        # X is the dataset
        performance = defaultdict(list)
        for foldTrainX, foldTestX in kfold.split(X):
            trainX, trainy, testX, testy = X[foldTrainX], y[foldTrainX], \
                                              X[foldTestX], y[foldTestX]
            # Training
            gacc, gcm, gcr, gkappa, gauc, gfpr, gtpr, gmodel, svm_lime, svm_sv, mean_shap_features = support_vector_utility(trainX, trainy, testX, testy, cols)
            gmodels.append(gmodel)

        state_name = state[:-9]
        performance['accuracy'].append(gacc)
        performance['kappa'].append(gkappa)
        performance['auc'].append(gauc)
        performance['fpr'].append(gfpr)
        performance['tpr'].append(gtpr)
        performance['confusion_matrix'].append(gcr)
        performance['classification_report'].append(gcr)
        performance['shap_values'].append(mean_shap_features)
        performance['lime_val'].append(svm_lime)

        # Create a temp dataframe
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
    sample = np.array(X, dtype=float)
    #data_sample = shap.sample(X, 10)
    svm_exp = shap.Explainer(gmodels[1].predict_proba, sample)
    svm_sv = svm_exp(sample)

    # Counter
    temp_mean_values = []
    # for each observartion:
    for observation in svm_sv:
        mean_shap_ob_val = []
        # For each feature there is the value:
        for ob, feat in zip(observation, cols):
            mean_shap_o_v = np.mean(np.abs(ob.values))
            mean_shap_ob_val.append(mean_shap_o_v)
        temp_mean_values.append(mean_shap_ob_val)
    #print(np.shape(mean_shap_ob_val))
    mean_values = np.mean(temp_mean_values, axis=0)
    dictionary_svm_shap = dict(zip(cols, mean_values))
    performance_df = pd.concat(temp_dfs)
    shap_series = pd.Series(dictionary_svm_shap)
    shap_series.to_csv("svm_shap_1.csv")
    return performance_df

if __name__ == '__main__':
    main()
