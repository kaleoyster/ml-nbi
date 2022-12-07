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
        model: Random Forest Model
    """
    # Train model
    X_train = pd.DataFrame(train_x, columns=cols)
    y_train = pd.DataFrame(trainy)

    #model = XGBRegressor(objective='reg:squarederror')
    model =  xgb.XGBClassifier()

    #cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # dtrain = model.DMatrix(train_x, label=train_y)
    # watchlist = [(dtrain, 'train')]
    # param = {'max_depth': 6, 'learning_rate': 0.03}
    # num_round = 200
    # bst = model.train(param, dtrain, num_round, watchlist)

   # model.fit(train_x, trainy)
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

    ##model.fit(X_train, y_train)
    ##print(Counter(trainy))
    ##importance_type = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    #print("printing feature importance")
    #_fi = model.get_booster().get_score(importance_type='gain')
    #_fn = model.get_booster().feature_names
    #print(len(_fi))
    #print(_fn)

    ##print(model.feature_importances_)
    #plot_importance(model)
    ##pyplot.show()
    ##model.fit(train_x, trainy)

    ## SHAP
    #xgb_exp = TreeExplainer(model)
    #xgb_sv = xgb_exp.shap_values(train_x)
    #xgb_ev = xgb_exp.expected_value

    ## LIME:
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

    #Predictions
    prediction = model.predict(test_x)
    prediction_prob = model.predict_proba(test_x)[::, 1]
    #b_prediction = convert_prediction_to_binary(prediction)

    #_mse = mean_squared_error(prediction, testy)
    _acc = accuracy_score(testy, prediction)
    _cm = confusion_matrix(testy, prediction)
    _cr = classification_report(testy,
                                prediction,
                                zero_division=0)

    fpr, tpr, threshold = roc_curve(testy, prediction_prob)
    _auc = auc(fpr, tpr)
    print("Printing area under curve")
    print(_auc)

    #_fi = dict(zip(cols, model.feature_importances_))
    _kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')
#    fpr, tpr, threshold = roc_curve(testy, prediction, pos_label=2)
#    _auc = auc(fpr, tpr)
    print("printing the auc", _auc)

    instance_exp = []
    xgb_sv = []
    return _acc, _cm, _cr, _kappa, _auc, fpr, tpr, instance_exp, xgb_sv

def main():
    X , y, cols = preprocess()
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
    kfold = KFold(5, shuffle=True, random_state=1)

    #X is the dataset
    performance = defaultdict(list)
    for foldTrainX, foldTestX in kfold.split(X):
        trainX, trainy, testX, testy = X[foldTrainX], y[foldTrainX], \
                                           X[foldTestX], y[foldTestX]
        # structure numbers
        acc, cm, cr, kappa, auc, fpr, tpr, xgb_lime, xgb_sv = xgb_utility(trainX, trainy, testX, testy, cols)
        performance['accuracy'].append(acc)
        performance['kappa'].append(kappa)
        performance['auc'].append(auc)
        performance['fpr'].append(fpr)
        performance['tpr'].append(tpr)
        performance['confusion_matrix'].append(cm)
        performance['classification_report'].append(cr)
        performance['shap_values'].append(xgb_sv)
        performance['lime_val'].append(xgb_lime)

    # Performance metrics
    print('Performance metrics:')
    print(performance['accuracy'])
    print(np.mean(performance['accuracy']))
    print(performance['kappa'])
    print(np.mean(performance['kappa']))

    return performance

if __name__ =='__main__':
    main()
