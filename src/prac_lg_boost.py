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
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import plot_partial_dependence
from mpl_toolkits.mplot3d import Axes3D


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

def plot_samples(X_train, Y_train, feature_names, classifier):
    """
    Plot samples across all three dimensions
    """
    # Dataframes
    df = pd.DataFrame(X_train, columns=feature_names)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Map classes
    map_class = {
        1: True,
        0: False,
    }

    y_train = []
    # Target class
    for target in Y_train:
        y_train.append(map_class[target])

    # Create scatter plots for each pair of features
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    feature_1 = feature_names[0]
    feature_2 = feature_names[1]
    feature_3 = feature_names[2]

    print("Printing the feature: \n")
    print(feature_1, feature_2, feature_3)

    # Feature1 vs Feature2
    title = feature_1 + ' Vs. '  + feature_2
    axes[0].scatter(df[feature_1], df[feature_2], c=y_train, cmap='viridis', alpha=0.5)
    axes[0].set_xlabel(feature_1)
    axes[0].set_ylabel(feature_2)
    axes[0].set_title(title)

    # Feature2 vs Feature3
    title = feature_2 + ' Vs. ' + feature_3
    axes[1].scatter(df[feature_2], df[feature_3], c=y_train, cmap='viridis', alpha=0.5)
    axes[1].set_xlabel(feature_2)
    axes[1].set_ylabel(feature_3)
    axes[1].set_title(title)

    # Feature1 vs Feature3
    title = feature_1 + ' Vs. ' + feature_3
    axes[2].scatter(df[feature_1], df[feature_3], c=y_train, cmap='viridis', alpha=0.5)
    axes[2].set_xlabel(feature_1)
    axes[2].set_ylabel(feature_3)
    axes[2].set_title(title)

    plt.tight_layout()

    #plt.show()
    plt.savefig("sample_plot_lg_boost.png", dpi=300)


    print("Printing the dataframes:\n")
    print(df.head())

    # Specify the features for which you want to create the 3D PDP plot
    features_to_plot = [(0, 1), (1, 2), (0, 2)]  # Pairs of feature indices for 3D PDP plot

    # Create the 3D PDP plot
    fig = plt.figure(figsize=(12, 8))
    plot_partial_dependence(classifier, df, features_to_plot, grid_resolution=100)

    plt.subplots_adjust(top=0.9)  # Adjust the position of the title
    plt.suptitle('3D Partial Dependency Plot (Light Boost)', fontsize=16)

    #plt.show()
    plt.savefig("PDP_plot_lg_boost.png", dpi=300)


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
    #lg_exp = shap.Explainer(model, X_merged)
    #lg_sv = lg_exp(X_merged, check_additivity=False)
    #mean_shap = np.mean(abs(lg_sv.values), 0)

    # Calculating mean shap values also known as SHAP feature importance
    # Have for two classes
    #mean_shap_features = {column:shap_v for column, shap_v in zip(cols, mean_shap)}
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

    return _acc, _cm, _cr, kappa, _auc, fpr, tpr, model, mean_shap_features

def main():

    # States
    states = ['nebraska_deep.csv']

    temp_dfs = list()
    for state in states:
        state_file = '../data/' + state
        X, y, cols = preprocess(csv_file=state_file)
        X = X[:, :3]
        cols = cols[:3]

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
            acc, cm, cr, kappa, auc, fpr, tpr, model, mean_shap_features = light_boost_utility(trainX, trainy, testX, testy, cols)
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
            break
        temp_dfs.append(temp_df)
    plot_samples(trainX, trainy, cols, model)

    # Performance dataframe
    performance_df = pd.concat(temp_dfs)
    df_perf = performance_df[['accuracy', 'kappa', 'auc']]

    ## Create FPR dataframe
    #fprs = [fpr for fpr in performance_df['fpr']]
    #fprs_df = pd.DataFrame(fprs).transpose()
    #fprs_df.columns=['k1', 'k2', 'k3', 'k4', 'k5']

    ## Create TPR dataframe
    #tprs = [tpr for tpr in performance_df['tpr']]
    #tprs_df = pd.DataFrame(tprs).transpose()
    #tprs_df.columns=['k1', 'k2', 'k3', 'k4', 'k5']

    ## Combine the dictionaries for shap values
    #dict1, dict2, dict3, dict4, dict5 = performance_df['shap_values']

    ## Combined dictionary
    #combined_dict = defaultdict()
    #for key in dict1.keys():
    #    vals = []
    #    val1 = dict1[key]
    #    val2 = dict2[key]
    #    val3 = dict3[key]
    #    val4 = dict4[key]
    #    val5 = dict5[key]
    #    mean_val = np.mean([val1, val2, val3, val4, val5])
    #    combined_dict[key] = mean_val

    ## Convert the dictionary into a pandas DataFrame
    #df = pd.DataFrame.from_dict(combined_dict, orient='index', columns=['values'])

    ## Reset index and rename column
    #df = df.reset_index().rename(columns={'index': 'features'})
    print(df_perf)
    #df.to_csv('lg_boost_shap_values_superstructure.csv')
    #df_perf.to_csv('lg_boost_performance_values_superstructure.csv')
    #fprs_df.to_csv('lg_boost_tree_fprs_superstructure.csv')
    #tprs_df.to_csv('lg_boost_tree_tprs_superstructure.csv')


    performance_df = pd.concat(temp_dfs)
    return performance_df

if __name__ =='__main__':
    main()
