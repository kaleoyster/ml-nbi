"""
Description:
    Xgboost model to predict the future maintenance of bridges

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
from matplotlib import pyplot

# XGBoost
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from xgboost import plot_importance
from sklearn.model_selection import KFold

# Permutation importance
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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Preprocessing
from preprocessing import *

def convert_prediction_to_binary(predictions, threshold=0.5):
    """
    Description:

    Args:
        predictions:
        threshold:
    Returns:
        binary_conversion
    """
    binary_conversion = []
    for value in predictions:
        if value <= threshold:
            val_binary = 0
            binary_conversion.append(val_binary)
        else:
            val_binary = 1
            binary_conversion.append(val_binary)
    return binary_conversion

def xgb_utility(train_x, trainy,
                 test_x, testy, cols):
    """
    Description:
        Performs the modeling and returns performance metrics

    Args:
        trainX: Features of Training Set
        tr_fainy: Ground truth of Training Set
        testX: Features of Testing Set
        testy: Ground truth of Testing Set

    Return:
        acc: Accuracy
        cm: Confusion Report
        cr: Classification Report
        kappa: Kappa Value
        model: XGB boost Model
    """
    # Merging the train_x and test_x
    X_merged = np.concatenate((train_x, test_x))
    X_merged = np.array(X_merged, dtype='f')

    # Training and testing dataset
    train_x = np.array(train_x, dtype='f')
    X_train = pd.DataFrame(train_x, columns=cols)
    y_train = pd.DataFrame(trainy)

    #model = XGBRegressor(objective='reg:squarederror')
    model =  xgb.XGBClassifier()
    model.fit(train_x, trainy,
              eval_set=[(test_x, testy), (train_x, trainy)],
              verbose=20, eval_metric='logloss')

    #p_imp = permutation_importance(model, test_x,
    #                               testy, n_repeats=10,
    #                               random_state=0)

    #p_imp_mean = p_imp.importances_mean
    #p_imp_std = p_imp.importances_std

    ## Partial dependency
    #features = [0, 1]
    #PartialDependenceDisplay.from_estimator(model, train_x, features)
    #print("PartialDependenceDisplay Working OK")

    #lg_exp = TreeExplainer(model)

    #importance_type = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    #_fi = model.get_booster().get_score(importance_type='gain')
    #_fn = model.get_booster().feature_names

    ##print(model.feature_importances_)
    #plot_importance(model)

    # SHAP
    xgb_exp = shap.Explainer(model, X_merged)
    xgb_sv = xgb_exp(X_merged, check_additivity=False)

    # Calculating mean shap values also known as SHAP feature importance 
    # Shape = (11360, 49)
    mean_shap = np.mean(abs(xgb_sv.values), axis=0)
    mean_shap_features = {column:shap_v for column, shap_v in zip(cols, mean_shap)}

    # LIME:
    #xgb_exp_lime = lime_tabular.LimeTabularExplainer(
    #    training_data = np.array(X_train),
    #    feature_names = X_train.columns,
    #    class_names=['Repair', 'No Repair'],
    #    mode='regression'
    #)

    ## Explaining the instances using LIME
    #instance_exp = xgb_exp_lime.explain_instance(
    #    data_row = X_train.values[4],
    #    predict_fn = model.predict
    #)

    #fig = instance_exp.as_pyplot_figure()
    #fig.savefig('xgb_lime_report.jpg')

    ##xgb_boost_lime.show_in_notebook(show_table=True)
    ## RF
    ##print("Shape of the RF values:", xgb_sv[0])
    ##print("Shape of the XGB Shap values:", xgb_sv.shape)
    #summary_plot(xgb_sv, train_x, feature_names=cols)

    # Predictions
    prediction = model.predict(test_x)
    prediction_prob = model.predict_proba(test_x)[::, 1]

    # Computing metrics
    _acc = accuracy_score(testy, prediction)
    _cm = confusion_matrix(testy, prediction)
    _cr = classification_report(testy,
                                prediction,
                                zero_division=0)

    fpr, tpr, threshold = roc_curve(testy, prediction_prob)
    _auc = auc(fpr, tpr)
    _kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')
    return _acc, _cm, _cr, _kappa, _auc, fpr, tpr, mean_shap_features

def main():
    # States
    states = ['nebraska_deep.csv']

    temp_dfs = list()
    for state in states:
        state_file = '../data/' + state
        X, y, cols = preprocess(csv_file=state_file)
         # Convert y into 0, 1
         # If positive = 1
         # If negative = 0
        new_y = []
        for value in y:
            if value == 'positive':
                new_val = 1
            else:
                new_val = 0
            new_y.append(new_val)

        y = np.array(new_y)

        # K-fold cross validation
        kfold = KFold(5, shuffle=True, random_state=1)

        # X is the dataset
        performance = defaultdict(list)
        for foldTrainX, foldTestX in kfold.split(X):
            trainX, trainy, testX, testy = X[foldTrainX], y[foldTrainX], \
                                              X[foldTestX], y[foldTestX]
            # Training
            acc, cm, cr, kappa, auc, fpr, tpr, mean_shap_features = xgb_utility(trainX, trainy, testX, testy, cols)

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
    print("printing performance_df", performance_df)
    return performance_df

if __name__ =='__main__':
    main()
