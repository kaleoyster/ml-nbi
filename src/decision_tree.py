"""
Description:
    Decision tree model to predict future maintenance of bridges

Date:
    30th September, 2022
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
from sklearn.tree import DecisionTreeClassifier

# Permutation importance
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay

# Shap
import shap
from shap import TreeExplainer
from shap import summary_plot

# Import LIME
import lime
from lime import lime_tabular

# LOFO
from lofo import LOFOImportance, dataset, plot_importance

# Metrics and stats
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Preprocessing
from preprocessing import *

def tree_utility(train_x, trainy,
                 test_x, testy, cols,
                 criteria='entropy', max_depth=30):
    """
    Description:
        Performs decision tree modeling and returns performance metrics

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
        model: Decision Tree Model
    """

    # Merging the train_x and test_x
    X_merged_integer = np.concatenate((train_x, test_x))
    X_merged = np.array(X_merged_integer, dtype='f')

    # Float Conversion for the model (numpy array)
    train_x = np.array(train_x, dtype='f')

    # X_train (Dataframe)
    X_train = pd.DataFrame(train_x)

    # Initialize model
    model = DecisionTreeClassifier(criterion=criteria, max_depth=max_depth)
    cv = KFold(n_splits=5, shuffle=False)

    # Fit model
    model.fit(X_train, trainy)

    #lofo_importance = LOFOImportance(X_train,
    #                                 cv=cv,
    #                                 scoring='roc_auc',
    #                                 model=model)
    #importance_df = lofo_importance.get_importance()

    # plot the means and standard deviations of the importances
    #plot_importance(importance_df, figsize=(12, 20))

    ## Permutation mean of the feature importance
    p_imp = permutation_importance(model,
                                   test_x,
                                   testy,
                                   n_repeats=10,
                                random_state=0)

    p_imp_mean = p_imp.importances_mean
    p_imp_std = p_imp.importances_std

    ## Partial dependency
    #features = [0, 1]
    #PartialDependenceDisplay.from_estimator(model, X_train, features)
    #print("PartialDependenceDisplay Working OK")

    #dt_exp_lime = lime_tabular.LimeTabularExplainer(
    #    training_data = np.array(X_train),
    #    feature_names = X_train.columns,
    #    class_names=['Repair', 'No Repair'],
    #    mode='classification'
    #)

    ### Explaining the instances using LIME
    #instance_exp = dt_exp_lime.explain_instance(
    #    data_row = X_train.iloc[4],
    #    predict_fn = model.predict_proba
    #)

    #fig = instance_exp.as_pyplot_figure()
    #instance_exp.save_to_file('dt_lime_report.html')

    # Computing SHAP values
    dt_exp = shap.Explainer(model, X_merged)
    dt_sv = dt_exp(X_merged, check_additivity=False)
    mean_shap = np.mean(abs(dt_sv.values), 0).mean(1)

    # Calculating mean shap values also known as SHAP feature importance
    mean_shap_features = {column:shap_v for column, shap_v in zip(cols, mean_shap)}
    prediction_prob = model.predict_proba(test_x)[::, 1]
    prediction = model.predict(test_x)
    acc = accuracy_score(testy, prediction)
    _cm = confusion_matrix(testy, prediction)
    _cr = classification_report(testy, prediction, zero_division=0)

    # Define classes
    class_label = {
                    'negative':0,
                    'positive':1
                    }

    testy_num = [class_label[i] for i in testy]
    fpr, tpr, threshold = roc_curve(testy_num, prediction_prob)

    # Compute AUC
    _auc = auc(fpr, tpr)

    # Compute feature importance
    _fi = dict(zip(cols, model.feature_importances_))

    # Compute Kappa
    _kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')

    return acc, _cm, _cr, _kappa, _auc, fpr, tpr, model, _fi, mean_shap_features

# Decision Tree
def decision_tree(X, y, features, label, all_data, nFold=5):
    """
    Description:
        Performs training-testing split
        Train model for various depth level
        Train model for both Entropy and GiniIndex

    Args:
        df (Dataframe)
    """
    # Kfold Cross Validation
    kfold = KFold(nFold, shuffle=True, random_state=1)

    # For storing Confusion Matrix
    confusionMatrixsEntropy = []
    confusionMatrixsGini = []

    # For storing Classification Report
    classReportsEntropy = []
    classReportsGini = []

    # Scores
    scoresGini = []
    scoresEntropy = []

    # ROC AUC 
    eRocs = []
    gRocs = []

    # Kappa values
    gKappaValues = []
    eKappaValues = []

    # Converting columns into to numpy array
    cols = X.columns
    X = np.array(X)
    y = np.array(y)

    # Store models
    eModels = []
    gModels = []

    # Feature importance
    eFeatures = []
    gFeatures = []

    for depth in tqdm(range(1, 31), desc='\n Modeling DT'):
        tempG = []
        tempE = []
        for foldTrainX, foldTestX in KFold.split(X):
            trainX, trainy, testX, testy = X[foldTrainX], y[foldTrainX], \
                                          X[foldTestX], y[foldTestX]

            # structure numbers
            # Gini
            gacc, gcm, gcr, gkappa, gmodel, gfi = tree_utility(trainX, trainy,
                                                 testX, testy, cols,
                                                 criteria='gini',
                                                 max_depth=depth
                                                 )

            # Entropy
            eacc, ecm, ecr, ekappa, emodel, efi = tree_utility(trainX, trainy,
                                                  testX, testy, cols,
                                                  criteria='entropy',
                                                  max_depth=depth )
            tempG.append(gacc)
            tempE.append(eacc)

        # Accuracies
        scoresGini.append(np.mean(tempG))
        scoresEntropy.append(np.mean(tempE))

        # Confusion Matrix
        confusionMatrixsEntropy.append(ecm)
        confusionMatrixsGini.append(gcm)

        # Classification Report
        classReportsEntropy.append(ecr)
        classReportsGini.append(gcr)

        # Kappa Values 
        eKappaValues.append(ekappa)
        gKappaValues.append(gkappa)

        # ROC AUC values
        # Models
        eModels.append(emodel)
        gModels.append(gmodel)

        # Feature importance
        eFeatures.append(efi)
        gFeatures.append(gfi)

    # Performance Summarizer
    depths = list(range(1, 31))

    # Create Dictionaries
    # Kappa
    eKappaDict = dict(zip(eKappaValues, depths))
    gKappaDict = dict(zip(gKappaValues, depths))

    # Confusion Matrix 
    eConfDict = dict(zip(depths, confusionMatrixsEntropy))
    gConfDict = dict(zip(depths, confusionMatrixsGini))

    # Classification Report
    eClassDict = dict(zip(depths, classReportsEntropy))
    gClassDict = dict(zip(depths, classReportsGini))

    # Scores (Accuracy)
    eScoreDict = dict(zip(scoresEntropy, depths))
    gScoreDict = dict(zip(scoresGini, depths))

    # Models
    eModelsDict = dict(zip(depths, eModels))
    gModelsDict = dict(zip(depths, gModels))

    # Feature Importance
    eFeatureDict = dict(zip(depths, eFeatures))
    gFeatureDict = dict(zip(depths, gFeatures))

    # Swap X_test with all_data
    kappaVals, accVals, featImps, models = performance_summarizer(eKappaDict, gKappaDict,
                                           eConfDict, gConfDict,
                                           eClassDict, gClassDict,
                                           eScoreDict, gScoreDict,
                                           eModelsDict, gModelsDict,
                                           eFeatureDict, gFeatureDict,
                                           testX, cols, label, all_data)

    # Return the average kappa value for state
    eBestModel, gBestModel = models

    return kappaVals, accVals, featImps, models

def main():

    # States
    states = ['nebraska_deep.csv']

    # Loop through all states
    temp_dfs = list()
    for state in states:
        state_file = '../data/' + state
        X, y, cols = preprocess(csv_file=state_file)
        kfold = KFold(5, shuffle=True, random_state=1)

        # X is the dataset
        # Loop through 5 times (K=5)
        performance = defaultdict(list)
        for foldTrainX, foldTestX in kfold.split(X):
            trainX, trainy, testX, testy = X[foldTrainX], y[foldTrainX], \
                                              X[foldTestX], y[foldTestX]

            # Entropy model
            acc, cm, cr, kappa, auc, fpr, tpr, model, fi, mean_shap_features = tree_utility(trainX, trainy,
                                                     testX, testy, cols,
                                                     criteria='entropy',
                                                     max_depth=30)
            state_name = state[:-9]
            performance['state'].append(state_name)
            performance['accuracy'].append(acc)
            performance['kappa'].append(kappa)
            performance['auc'].append(auc)
            performance['fpr'].append(fpr)
            performance['tpr'].append(tpr)
            performance['confusion_matrix'].append(cm)
            performance['classification_report'].append(cr)
            performance['feature_importance'].append(fi)
            performance['shap_values'].append(mean_shap_features)

            # Create a dataframe for all the perfomance metric
            temp_df = pd.DataFrame(performance, columns=['state',
                                                         'accuracy',
                                                         'kappa',
                                                         'auc',
                                                         'fpr',
                                                         'tpr',
                                                         'confusion_matrix',
                                                         'shap_values',
                                                        ])
        # Concatenate all dataframes together
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
    df.to_csv('decision_tree_shap_values_superstructure.csv')
    df_perf.to_csv('decision_tree_performance_values_superstructure.csv')
    fprs_df.to_csv('decision_tree_fprs_superstructure.csv')
    tprs_df.to_csv('decision_tree_tprs_superstructure.csv')


    return performance_df

if __name__ == '__main__':
    main()
