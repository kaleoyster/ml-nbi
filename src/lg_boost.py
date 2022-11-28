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

# Permutation importance
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
        model: Random Forest Model
    """

    X_train = pd.DataFrame(train_x, columns=cols)
    model = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-5, random_state=42)
    #cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    model.fit(train_x, trainy,
              eval_set=[(test_x, testy), (train_x, trainy)],
              verbose=20, eval_metric='logloss')

    print("printing feature importance")
    _fi = model.feature_importances_
    #_fn = model.feature_names
    print(_fi)
    print(len(cols))
    #print(_fn)
    # Permutation mean of the feature importance
    p_imp = permutation_importance(model,
                                   test_x,
                                   testy,
                                   n_repeats=10,
                                random_state=0)

    p_imp_mean = p_imp.importances_mean
    p_imp_std = p_imp.importances_std

    # Partial dependency
    features = [0, 1]
    PartialDependenceDisplay.from_estimator(model, train_x, features)
    print("PartialDependenceDisplay Working OK")

    lg_exp = TreeExplainer(model)
    #lg_exp = shap.Explainer(model)
    #lg_sv = explainer(train_x)

    lg_sv = lg_exp.shap_values(train_x)
    lg_ev = lg_exp.expected_value

    # LIME:
    lg_exp_lime = lime_tabular.LimeTabularExplainer(
        training_data = np.array(X_train),
        feature_names = X_train.columns,
        class_names=['Repair', 'No Repair'],
        mode='regression'
    )

    ## Explaining the instances using LIME
    instance_exp = lg_exp_lime.explain_instance(
        data_row = X_train.values[4],
        predict_fn = model.predict
    )

    fig = instance_exp.as_pyplot_figure()
    fig.savefig('lg_lime_report.jpg')

    #print("Shape of the RF values:", lg_sv[0])
    #print("Shape of the Light boost Shap Values")

    summary_plot(lg_sv, train_x, feature_names=cols)
    #shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])

    #shap.plots.waterfall(lg_sv[0])
    #testing_accuracy = modelX_test.score(test_x, testy)
    prediction = model.predict(test_x)
    _acc = accuracy_score(testy, prediction)
    _cm = confusion_matrix(testy, prediction)
    _cr = classification_report(testy, prediction, zero_division=0)
    _fi = dict(zip(cols, model.feature_importances_))
    kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')

    #fpr, tpr, threshold = metrics.roc_curve(testy, prediction, pos_label=2)
    #_auc = metrics.auc(fpr, tpr)
    return _acc, _cm, _cr, kappa, instance_exp, lg_sv

def main():
    X, y, cols = preprocess()

    # Convert y into 0, 1
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
        #gacc, gcm, gcr, gkappa, gmodel = xgb_utility(trainX, trainy,
        #                                          testX, testy, cols)
        acc, cm, cr, kappa, lg_lime, lg_sv = light_boost_utility(trainX, trainy, testX, testy, cols)

        performance['accuracy'].append(acc)
        performance['kappa'].append(kappa)
        performance['confusion_matrix'].append(cm)
        performance['classification_report'].append(cr)
        performance['shap_values'].append(lg_sv)
        performance['lime_val'].append(lg_lime)

    print('Performance metrics:')
    print(performance['accuracy'])
    print(np.mean(performance['accuracy']))
    print(performance['kappa'])
    print(np.mean(performance['kappa']))

    return performance

if __name__ =='__main__':
    main()
