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
def plot_samples(X_train, Y_train, feature_names, classifier):
    """
    Plot samples across all three dimensions
    """
    # Dataframes
    df = pd.DataFrame(X_train, columns=feature_names)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Map classes
    map_class = {
        'positive': True,
        'negative': False,
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
    plt.savefig("sample_plot_gradient_boosting.png", dpi=300)


    # Specify the features for which you want to create the 3D PDP plot
    features_to_plot = [(0, 1), (1, 2), (0, 2)]  # Pairs of feature indices for 3D PDP plot

    # Create the 3D PDP plot
    fig = plt.figure(figsize=(12, 8))
    plot_partial_dependence(classifier, df, features_to_plot, grid_resolution=100)

    plt.subplots_adjust(top=0.9)  # Adjust the position of the title
    plt.suptitle('3D Partial Dependency Plot (Gradient boosting)', fontsize=16)

    #plt.show()
    plt.savefig("PDP_plot_gradient_boosting.png", dpi=300)


def gradient_boosting_utility(train_x, trainy,
                 test_x, testy, cols, max_depth=30):
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
        mean_shap_features: Shapley values for each features
        model: Random Forest Model
    """
    X_merged = np.concatenate((train_x, test_x))
    X_merged = np.array(X_merged, dtype='f')

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

    # Compute SHAP
    #g_exp = shap.Explainer(model, X_merged)
    #g_sv = g_exp(X_merged, check_additivity=False)
    #g_sv = g_sv.values

    ### Calculating mean shap values also known as SHAP feature importance
    #mean_shap = np.mean(g_sv, axis=0)
    #mean_shap_features = {column:shap_v for column, shap_v in zip(cols, mean_shap)}
    mean_shap_features = {}

    prediction_prob = model.predict_proba(test_x)[::, 1]
    prediction = model.predict(test_x)

    # Compute metrics
    acc = accuracy_score(testy, prediction)
    _cm = confusion_matrix(testy, prediction)
    _cr = classification_report(testy, prediction, zero_division=0)

    class_label = {'negative':0,
                   'positive':1}

    testy_num = [class_label[i] for i in testy]
    fpr, tpr, threshold = roc_curve(testy_num, prediction_prob)
    _auc = auc(fpr, tpr)
    _fi = dict(zip(cols, model.feature_importances_))
    _kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')
    return acc, _cm, _cr, _kappa, _auc, fpr, tpr, model, _fi, mean_shap_features

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
            gacc, gcm, gcr, gkappa, gauc, gfpr, gtpr, gmodel, fi, mean_shap_features = gradient_boosting_utility(trainX,
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
    plot_samples(trainX, trainy, cols, gmodel)

    # Performance dataframe
    performance_df = pd.concat(temp_dfs)
    #df_perf = performance_df[['accuracy', 'kappa', 'auc']]

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

    # Combine dictionary
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

    # Convert the dictionary into a pandas DataFrame
    #df = pd.DataFrame.from_dict(combined_dict, orient='index', columns=['values'])

    # Reset index and rename column
    #df = df.reset_index().rename(columns={'index': 'features'})
    #df.to_csv('gradient_boosting_shap_values_substructure.csv')
    #df_perf.to_csv('gradient_boosting_performance_values_substructure.csv')
    #fprs_df.to_csv('gradient_boosting_fprs_substructure.csv')
    #tprs_df.to_csv('gradient_boosting_tprs_substructure.csv')
    print(performance_df)

    return performance

if __name__ =='__main__':
    main()
