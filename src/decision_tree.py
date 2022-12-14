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
                 criteria='entropy', max_depth=15):
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
        model: Decision Tree Model
    """
    # new dataframes
    #X_train = pd.DataFrame(train_x, columns=cols)
    X_train = pd.DataFrame(train_x)
    model = DecisionTreeClassifier(criterion=criteria, max_depth=max_depth)
    cv = KFold(n_splits=5, shuffle=False)

    #lofo_importance = LOFOImportance(X_train,
    #                                 cv=cv,
    #                                 scoring='roc_auc',
    #                                 model=model)
    ##print(lofo_importance.get_importance)
    ## Get the mean and standard deviation of the importances in pandas format
    ## Fix this error
    ##importance_df = lofo_importance.get_importance()

    ## plot the means and standard deviations of the importances
    ##plot_importance(importance_df, figsize=(12, 20))

    model.fit(X_train, trainy)
    ##model.fit(train_x, trainy)

    ## Permutation mean of the feature importance
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

    #dt_exp = TreeExplainer(model)
    #dt_sv = np.array(dt_exp.shap_values(train_x))
    #dt_ev = np.array(dt_exp.expected_value)

    #dt_sv =  dt_exp.shap_values(train_x)
    #dt_ev =  dt_exp.expected_value

    ##print("Shape of the RF values:", dt_sv[0])
    ##print("Shape of the Light boost Shap Values")
    #summary_plot(dt_sv, train_x, feature_names=cols)

    prediction_prob = model.predict_proba(test_x)[::, 1]
    prediction = model.predict(test_x)
    acc = accuracy_score(testy, prediction)
    _cm = confusion_matrix(testy, prediction)
    _cr = classification_report(testy, prediction, zero_division=0)
    class_label = {
                    'negative':0,
                    'positive':1
                    }
    testy_num = [class_label[i] for i in testy]
    fpr, tpr, threshold = roc_curve(testy_num, prediction_prob)
    #print(testy_num[:100])
    #print(prediction_prob[:100])
    print("Checking dimensions")
    print(np.shape(testy_num), np.shape(prediction_prob))
    print(np.shape(fpr), np.shape(tpr))
    print("printing fpr and tpr")
    print(fpr, tpr)
    _auc = auc(fpr, tpr)
    print("printing auc", _auc)
    _fi = dict(zip(cols, model.feature_importances_))
    _kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')
    #fpr, tpr, threshold = roc_curve(testy, prediction, pos_label=2)
    #_auc = metrics.auc(fpr, tpr)
    instance_exp = []
    dt_sv = []

    return acc, _cm, _cr, _kappa, _auc, fpr, tpr, model, _fi, instance_exp, dt_sv

# Decision Tree
def decision_tree(X, y, features, label, all_data, nFold=5):
    """
    #TODO: We can do things together.
    def decision_tree(X, y, features, label, nFold=5):
    Description:
        Performs training-testing split
        Train model for various depth level
        Train model for both Entropy and GiniIndex

    Args:
        df (Dataframe)
    """
    # Kfold cross validation
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

    # Converting them to array
    cols = X.columns
    X = np.array(X)
    y = np.array(y)

    # Converting all data into array
    #all_sub_data =  all_data[cols]
    #structure_numbers = all_data['structureNumber']
    #all_sub_data = np.array(X)
    #structure_numbers = np.array(structure_numbers)

    # Store models:
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

        # Kappa Values (TODO: select average of Kappa Value)
        eKappaValues.append(ekappa)
        gKappaValues.append(gkappa)

        # ROC AUC values(TODO: select average of Kappa Value)
        #eRocs.append(eroc)
        #gRocs.append(groc)

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

    #TODO:  Scores (ROCs) doesn't work
    #eRocsDict = dict(zip(eRocs, depths))
    #gRocsDict = dict(zip(gRocs, depths))

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
                                           #eRocsDict, gRocsDict,
                                           eModelsDict, gModelsDict,
                                           eFeatureDict, gFeatureDict,
                                           testX, cols, label, all_data)

    # Return the average kappa value for state
    eBestModel, gBestModel = models
    #leaves = find_leaves(eBestModel)
    #splitNodes = print_split_nodes(leaves, eBestModel, features)

    return kappaVals, accVals, featImps, models

def main():
    # States
    states = [
              'wisconsin_deep.csv',
              'colorado_deep.csv',
              'illinois_deep.csv',
              'indiana_deep.csv',
              'iowa_deep.csv',
              'minnesota_deep.csv',
              'missouri_deep.csv',
              'ohio_deep.csv',
              'nebraska_deep.csv',
              'indiana_deep.csv',
              'kansas_deep.csv',
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


            # Entropy
            acc, cm, cr, kappa, auc, fpr, tpr, model, fi, dt_lime, dt_sv= tree_utility(trainX, trainy,
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
            performance['shap_values'].append(dt_sv)
            performance['lime_val'].append(dt_lime)

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
    print(performance_df)
    return performance

if __name__ == '__main__':
    main()
