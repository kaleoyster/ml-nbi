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
                 criteria='gini', max_depth=7):
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
    model = DecisionTreeClassifier(criterion=criteria, max_depth=max_depth)
    model.fit(train_x, trainy)
    prediction_prob = model.predict_proba(test_x)
    prediction = model.predict(test_x)
    acc = accuracy_score(testy, prediction)
    _cm = confusion_matrix(testy, prediction)
    _cr = classification_report(testy, prediction, zero_division=0)
    _fi = dict(zip(cols, model.feature_importances_))
    _kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')
    #fpr, tpr, threshold = roc_curve(testy, prediction, pos_label=2)
    #_auc = metrics.auc(fpr, tpr)

    return acc, _cm, _cr, _kappa, model, _fi

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
        for foldTrainX, foldTestX in kfold.split(X):
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

    X, y, cols = preprocess()
    kfold = KFold(5, shuffle=True, random_state=1)

    # X is the dataset
    performance = defaultdict(list)
    for foldTrainX, foldTestX in kfold.split(X):
        trainX, trainy, testX, testy = X[foldTrainX], y[foldTrainX], \
                                          X[foldTestX], y[foldTestX]

        # Entropy
        acc, cm, cr, kappa, model, fi = tree_utility(trainX, trainy,
                                                 testX, testy, cols,
                                                 criteria='entropy',
                                                 max_depth=30)
        performance['accuracy'].append(acc)
        performance['kappa'].append(kappa)
        performance['confusion_matrix'].append(cm)
        performance['classification_report'].append(cr)
        performance['feature_importance'].append(fi)


    print('Performance metrics:')
    print(performance['accuracy'])
    print(np.mean(performance['accuracy']))
    print(performance['kappa'])
    print(np.mean(performance['kappa']))

    return performance

if __name__ == '__main__':
    main()
