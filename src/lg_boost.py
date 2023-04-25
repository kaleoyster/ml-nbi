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

# Permutation feature importance
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay

# SHAP
import shap
from shap import TreeExplainer
from shap import summary_plot
from sklearn.model_selection import KFold

# LIME
import lime
from lime import lime_tabular

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
        fpr: False positive rate
        tpr: True positive rate
        mean_shap_features: Shap values
        model: Light boost Model
    """
    # Merging the train_x and test_x
    X_merged = np.concatenate((train_x, test_x))
    X_merged = np.array(X_merged, dtype='f')

    train_x = np.array(train_x, dtype='f')
    X_train = pd.DataFrame(train_x, columns=cols)
    model = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-5, random_state=42)

    # Model Fit 
    model.fit(train_x, trainy,
              eval_set=[(test_x, testy), (train_x, trainy)],
              verbose=20, eval_metric='logloss')

    # Permutation mean of the feature importance
    #p_imp = permutation_importance(model,
    #                               test_x,
    #                               testy,
    #                               n_repeats=10,
    #                            random_state=0)

    #p_imp_mean = p_imp.importances_mean
    #p_imp_std = p_imp.importances_std

    #Partial dependency
    #features = [0, 1]
    #PartialDependenceDisplay.from_estimator(model, train_x, features)
    #print("PartialDependenceDisplay Working OK")

    # SHAP
    lg_exp = shap.Explainer(model, X_merged)
    lg_sv = lg_exp(X_merged, check_additivity=False)
    mean_shap = np.mean(abs(lg_sv.values), 0)

    # Calculating mean shap values also known as SHAP feature importance
    # Have for two classes
    mean_shap_features = {column:shap_v for column, shap_v in zip(cols, mean_shap)}
    mean_shap_features = {}

    # LIME:
    lg_exp_lime = lime_tabular.LimeTabularExplainer(
        training_data = np.array(X_train),
        feature_names = X_train.columns,
        class_names=['Repair', 'No Repair'],
        mode='regression'
    )

    # Explaining the instances using LIME:
    instance_exp = lg_exp_lime.explain_instance(
        data_row = X_train.values[4],
        predict_fn = model.predict
    )

    #fig = instance_exp.as_pyplot_figure()
    #fig.savefig('lg_lime_report.jpg')

    #summary_plot(lg_sv, train_x, feature_names=cols)
    #shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])

    # Computing metrics
    prediction_prob = model.predict_proba(test_x)[::, 1]
    prediction = model.predict(test_x)
    _acc = accuracy_score(testy, prediction)
    _cm = confusion_matrix(testy, prediction)
    _cr = classification_report(testy, prediction, zero_division=0)
    fpr, tpr, threshold = roc_curve(testy, prediction_prob)
    _auc = auc(fpr, tpr)
    _fi = dict(zip(cols, model.feature_importances_))
    kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')

    return _acc, _cm, _cr, kappa, _auc, fpr, tpr, mean_shap_features

def main():

    # States
    states = ['nebraska_deep.csv']

    temp_dfs = list()
    for state in states:
        state_file = '../data/' + state
        X, y, cols = preprocess(csv_file=state_file)

        # Convert y into 0, 1
        new_y = []
        for value in y:
            if value == 'positive':
                new_val = 1
            else:
                new_val = 0
            new_y.append(new_val)
        y = np.array(new_y)

        # K fold cross validation
        kfold = KFold(5, shuffle=True, random_state=1)

        # X is the dataset
        # Loop through 5 times (K = 5)
        performance = defaultdict(list)
        for foldTrainX, foldTestX in kfold.split(X):
            trainX, trainy, testX, testy = X[foldTrainX], y[foldTrainX], \
                                              X[foldTestX], y[foldTestX]
            # Training 
            acc, cm, cr, kappa, auc, fpr, tpr, mean_shap_features = light_boost_utility(trainX, trainy, testX, testy, cols)

            state_name = state[:-9]
            performance['state'].append(state_name)
            performance['accuracy'].append(acc)
            performance['kappa'].append(kappa)
            performance['auc'].append(auc)
            performance['fpr'].append(fpr)
            performance['tpr'].append(tpr)
            performance['confusion_matrix'].append(cm)
            performance['classification_report'].append(cr)
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
    performance_df = pd.concat(temp_dfs)
    print(performance_df['shap_values'])
    return performance_df

if __name__ =='__main__':
    main()
