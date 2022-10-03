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

def read_csv(csv_file):
    """
    Read csv files using pandas
    """
    _df = pd.read_csv(csv_file)
    return _df

def create_label(intervention_columns):
    """
    Return positive or negative function
    """
    labels = []
    for value in intervention_columns:
        if value == 0:
            label_val = 'negative'
            labels.append(label_val)
        else:
            label_val = 'positive'
            labels.append(label_val)
    return labels

def normalize(_df, columns):
    """
    Function for normalizing the data

    Args:
        _df (dataframe)
        columns (features)
    """
    for feature in columns:
        _df[feature] = _df[feature].astype(int)
        max_value = _df[feature].max()
        min_value = _df[feature].min()
        _df[feature] = (_df[feature] - min_value) / (max_value - min_value)
    return _df

def remove_null_values(_df):
    """
    Description: return a new df with null values removed
    Args:
        df (dataframe): the original dataframe to work with
    Returns:
        df (dataframe): dataframe
    """
    for feature in _df:
        if feature != 'structureNumber':
            try:
                _df = _df[~_df[feature].isin([np.nan])]
            except:
                print("Error: ", feature)
    return _df

def remove_duplicates(_df, column_name='structureNumbers'):
    """
    Description: return a new df with duplicates removed
    Args:
        df (dataframe): the original dataframe to work with
        column (string): columname to drop duplicates by
    Returns:
        newdf (dataframe)
    """
    temp = []
    for group in _df.groupby(['structureNumber']):
        structure_number, grouped_df = group
        grouped_df = grouped_df.drop_duplicates(subset=['structureNumber'],
                               keep='last'
                               )
        temp.append(grouped_df)
    new_df = pd.concat(temp)
    return new_df

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
    prediction = model.predict(test_x)
    acc = accuracy_score(testy, prediction)
    _cm = confusion_matrix(testy, prediction)
    _cr = classification_report(testy, prediction, zero_division=0)
    _fi = dict(zip(cols, model.feature_importances_))
    #rocAuc = roc_auc_score(testy, prediction, multi_class='ovr')
    kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')
    return acc, _cm, _cr, kappa, model, _fi # rocAuc, model


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
    csv_file = '../../data/nebraska_deep.csv'
    df = read_csv(csv_file)

    # Remove null values:
    df = df.dropna(subset=['deck',
                           'substructure',
                           'deckNumberIntervention',
                           'subNumberIntervention',
                           'supNumberIntervention'
                          ])

    df = remove_duplicates(df)

    # Remove values encoded as N:
    df = df[~df['deck'].isin(['N'])]
    df = df[~df['substructure'].isin(['N'])]
    df = df[~df['superstructure'].isin(['N'])]
    df = df[~df['material'].isin(['N'])]
    df = df[~df['scourCriticalBridges'].isin(['N', 'U', np.nan])]
    df = df[~df['deckStructureType'].isin(['N', 'U'])]

    # Fill the null values with -1:
    df.snowfall.fillna(value=-1, inplace=True)
    df.precipitation.fillna(value=-1, inplace=True)
    df.freezethaw.fillna(value=-1, inplace=True)

    df.toll.fillna(value=-1, inplace=True)
    df.designatedInspectionFrequency.fillna(value=-1, inplace=True)
    df.deckStructureType.fillna(value=-1, inplace=True)
    df.typeOfDesign.fillna(value=-1, inplace=True)

    # Normalize features:
    columns_normalize = [
                        "deck",
                        "yearBuilt",
                        "superstructure",
                        "substructure",
                        "averageDailyTraffic",
                        "avgDailyTruckTraffic",
                        "supNumberIntervention",
                        "subNumberIntervention",
                        "deckNumberIntervention",
                        "latitude",
                        "longitude",
                        "skew",
                        "numberOfSpansInMainUnit",
                        "lengthOfMaximumSpan",
                        "structureLength",
                        "bridgeRoadwayWithCurbToCurb",
                        "operatingRating",
                        "scourCriticalBridges",
                        "lanesOnStructure",
                        ]

    # Select final columns:
    columns_final = [
                    "deck",
                    "yearBuilt",
                    "superstructure",
                    "substructure",
                    "averageDailyTraffic",
                    "avgDailyTruckTraffic",
                    "material",
                    "designLoad",
                    "snowfall",
                    "freezethaw",
                    "supNumberIntervention",
                    "subNumberIntervention",
                    "deckNumberIntervention",

                    "latitude",
                    "longitude",
                    "skew",
                    "numberOfSpansInMainUnit",
                    "lengthOfMaximumSpan",
                    "structureLength",
                    "bridgeRoadwayWithCurbToCurb",
                    "operatingRating",
                    "scourCriticalBridges",
                    "lanesOnStructure",

                    "toll",
                    "designatedInspectionFrequency",
                    "deckStructureType",
                    "typeOfDesign",

                    "deckDeteriorationScore",
                    "subDeteriorationScore",
                    "supDeteriorationScore"
                ]

    cols = columns_normalize
    data_scaled = normalize(df, columns_normalize)
    X = data_scaled[columns_final]
    X = remove_null_values(X)

    deckLabels = X['deckDeteriorationScore']
    y = create_label(deckLabels)

    # Convert them into arrays
    X = np.array(X)
    y = np.array(y)
    kfold = KFold(5, shuffle=True, random_state=1)

    # X is the dataset
    for foldTrainX, foldTestX in kfold.split(X):
        trainX, trainy, testX, testy = X[foldTrainX], y[foldTrainX], \
                                          X[foldTestX], y[foldTestX]

        # structure numbers
        # Gini
        gacc, gcm, gcr, gkappa, gmodel, gfi = tree_utility(trainX, trainy,
                                                 testX, testy, cols,
                                                 criteria='entropy',
                                                 max_depth=5)
        print(gcr)


    # TODO: Need to create the positive and negative

    #X, y = data_scaled[columns_final], data_scaled[label]
    #decision_tree(X, y, features, label, all_data, nFold=5)

    #dataScaled, lowestCount, centroids, counts = kmeans_clustering(data_scaled,
    #                                                               list_of_parameters,
    #                                                               state=state)

main()
