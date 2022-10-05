"""--------------------------------------------
Descriptions:
    Data preprocessing file for decision tree

Author: Akshay Kale
Date: May 11th, 2021

TODO:
    1. Create Folders for the ouput [Done]
    2. Create Random forest model [Done]
    3. Complexity Parameters
    4. Select the important variables
    5. Characterization of the clusters
    6. Computing deterioration scores,
        and intervention
-----------------------------------------------"""

# Data structures
import pandas as pd
import numpy as np
from collections import Counter
from collections import defaultdict
from tqdm import tqdm

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Model
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# Metrics and stats
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score

# Visualization
import plotly
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
#import graphviz

# Function for normalizing
def normalize(df, columns):
    """
    Function for normalizing the data
    """
    for feature in columns:
        df[feature] = df[feature].astype(int)
        maxValue = df[feature].max()
        minValue = df[feature].min()
        df[feature] = (df[feature] - minValue) / (maxValue - minValue)
    return df

# Summarize features
def summarize_features(df, columns):
    """
    return df
    """
    for feature in columns:
        print("Feature :", feature)
        values = df[feature].astype(int)
        print_box_plot(values, filename=feature, col=feature)
        if feature == 'deteriorationScore':
            print("\n ",  Counter(df[feature]))

# Function for removing duplicates
def remove_duplicates(df, columnName='structureNumbers'):
    """
    Description: return a new df with duplicates removed
    Args:
        df (dataframe): the original dataframe to work with
        column (string): columname to drop duplicates by
    Returns:
        newdf (dataframe)
    """
    temp = list()
    for group in df.groupby(['structureNumber']):
        structureNumber, groupedDf = group
        groupedDf = groupedDf.drop_duplicates(subset=['structureNumber'],
                               keep='last'
                               )
        temp.append(groupedDf)
    newdf = pd.concat(temp)
    return newdf

def remove_null_values(df):
    """
    Description: return a new df with null values removed
    Args:
        df (dataframe): the original dataframe to work with
    Returns:
        df (dataframe): dataframe
    """
    for feature in df:
        df = df[~df[feature].isin([np.nan])]
    return df

def create_labels(df, label):
    """
    Description:
        Create binary categories from
        multiple categories.
    Args:
        df (dataframe)
        label (string): primary label
    Returns:
        df (dataframe): a dataframe with additional
        attributes
    """
    positiveClass = df[df['cluster'].isin([label])]
    negativeClass = df[~df['cluster'].isin([label, 'No intervention'])]
    positiveClass['label'] = ['positive']*len(positiveClass)
    negativeClass['label'] = ['negative']*len(negativeClass)
    df = pd.concat([positiveClass, negativeClass])
    return df

def create_condition_label(df, label):
    """
    Description:
        Create condition labels

    Args:
        df (dataframe)
        label (string): usually an attribute that
        will be used as a groundtruth

    Returns:
        label (list of string): Binary classification
        of returned label
    """
    labelList = list(df[label].astype(int))
    meanVal = np.mean(labelList)
    labelClass = list()

    for value in labelList:
        if value > meanVal:
            labelClass.append('highSub')
        else:
            labelClass.append('lowSub')
    df['label'] = labelClass
    return df

def categorize_attribute(df, fieldname, category=2):
    """
    Description:
        Categerize numerical values by normal distribution

    Args:
        df (Dataframe)
        category of attributes: Divide the attributes types
        either into a total number of 2 categories
        or 4 categories

    Returns:
       categories (list)
    """
    categories = list()
    mean = np.mean(df[fieldname])
    std = np.std(df[fieldname])
    if category == 4:
        for value in df[fieldname]:
            if value > mean and value < (mean + std):
                categories.append('Good')
            elif value > mean + std:
                categories.append('Good')
            elif value < (mean - std):
                categories.append('Bad')
            else:
                categories.append('Bad')

    elif category == 2:
        for value in df[fieldname]:
            if value > mean:
                categories.append("Good")
            else:
                categories.append("Bad")
    else:
        categories = list()

    return categories

# Confusion Matrix
def conf_matrix(cm, filename=''):
    """
    Description:
        Confusion matrix on validation set
    """
    indexList = list()
    columnList = list()
    filename = 'results/' + filename +  'ConfusionMatrix.png'

    for row in range(0, np.shape(cm)[0]):
        indexString = 'Actual' + str(row+1)
        columnString = 'Predicted' + str(row+1)
        indexList.append(indexString)
        columnList.append(columnString)

    dfCm = pd.DataFrame(cm,
                        index=indexList,
                        columns=columnList
                        )

    plt.subplots(figsize=(8, 8))
    sns.heatmap(dfCm, annot=True, fmt='g', cmap='BuPu')
    plt.savefig(filename)

# Box plot
def box_plot(scores, filename='', col='accuracy'):
    """
    Boxplot of training accuracy
    """
    filename = 'results/' + filename + 'AccuracyBoxPlot.png'
    dfScore = pd.Series(scores)

    font = {'weight': 'bold',
            'size': 25
            }

    plt.figure(figsize=(10, 10))
    plt.title("Performance: Accuracy", **font)
    sns.boxplot(y=dfScore, orient='v')
    plt.savefig(filename)

# Line plot
def line_plot(scores, filename='', col='accuracy'):
    """
    lineplot of training accuracy
    """
    filename = "results/" + filename + 'AccuracyLinePlot.png'
    dfScore = pd.Series(scores)

    font = {'weight': 'bold',
            'size': 25
            }

    plt.figure(figsize=(10, 10))
    plt.title("Performance: Accuracy", **font)
    sns.lineplot(data=dfScore)
    plt.savefig(filename)

# Plot decision trees
def plot_decision_tree(model, filename=''):
    """
    Decision Tree
    """
    filename = "results/" + filename + "DecisionTree.png"
    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(model,
                   filled=True)
    fig.savefig(filename)

# To summarize performance
def performance_summarizer(eKappaDict, gKappaDict,
                          eConfDict, gConfDict,
                          eClassDict, gClassDict,
                          eAccDict, gAccDict,
                          #eRocsDict, gRocsDict,
                          eModelsDict, gModelsDict,
                          eFeatureDict, gFeatureDict):

    """
    Description:
        Summarize the prformance of the decision tree

    Args:
        Kappa Values (list):
        Confusion Matrix (list):
        Accuracy Values (list):

    Returns:
        Prints a summary of Model performance with respect to
        Entropy and Kappa Value
    """
    # Entropy
    eBestKappa = max(eKappaDict.keys())
    eBestAcc = max(eAccDict.keys())
    eBestDepth = eKappaDict.get(eBestKappa)
    ecm = eConfDict.get(eBestDepth)
    efi = eFeatureDict.get(eBestDepth)
    efi = dict(sorted(efi.items(), key=lambda item: item[1]))

    print("""\n
            -------------- Performance of Entropy ---------------
            \n""")
    print("\n Best Kappa Values: ", eBestKappa)
    print("\n Best Accuracy: ", eBestAcc)
    print("\n Best Depth: ", eBestDepth)
    print("\n Classification Report: \n", eClassDict.get(eBestDepth))
    print("\n Confusion Matrix: \n", ecm)
    print("\n Feature Importance: \n", efi)
    #print("\n AUC: ", eRocsDict[eBestDept])

    # GiniIndex 
    gBestKappa = max(gKappaDict.keys())
    gBestAcc = max(gAccDict.keys())
    gBestDepth = gKappaDict.get(gBestKappa)
    gcm = gConfDict.get(gBestDepth)
    gfi = gFeatureDict.get(gBestDepth)
    gfi = dict(sorted(gfi.items(), key=lambda item: item[1]))

    print("""\n
             ----------- Performance with GiniIndex ------------
             \n""")

    print("\n Best Kappa Values: ", gBestKappa)
    print("\n Best Accuracy: ", gBestAcc)
    print("\n Best Depth: ", gBestDepth)
    print("\n Classification Report: \n", gClassDict.get(gBestDepth))
    print("\n Confusion Matrix: \n", gcm)
    print("\n Feature Importance: \n", efi)
    #print("\n AUC: ", gRocsDict[gBestDept])

    # Plot Confusion Matrix
    conf_matrix(gcm, 'Gini')
    conf_matrix(ecm, 'Entropy')

    # Box plot of training accuracies
    scoresGini = list(gAccDict.keys())
    scoresEntropy = list(eAccDict.keys())

    box_plot(scoresGini, 'Gini')
    box_plot(scoresEntropy, 'Entropy')

    ## Line plot
    line_plot(scoresGini, 'Gini')
    line_plot(scoresEntropy, 'Entropy')

    # Get best models (entropy and gini models)
    eBestModel = eModelsDict.get(eBestDepth)
    gBestModel = gModelsDict.get(gBestDepth)

    # rint decision tree of the Best Model
    # Entropy
    print("\n Saving decision trees \n")
    eTextRepresentation = tree.export_text(eBestModel)
    with open("models/entropy_decision_tree.log", "w") as fout:
        fout.write(eTextRepresentation)

    # Gini
    gTextRepresentation = tree.export_text(gBestModel)
    with open("models/gini_decision_tree.log", "w") as fout:
        fout.write(gTextRepresentation)

    print("\n Plotting decision trees \n")
    plot_decision_tree(eBestModel, filename='Entropy')
    plot_decision_tree(gBestModel, filename='Gini')
    return (eBestKappa, gBestKappa),  (eBestAcc, gBestAcc), (efi, gfi)

def tree_utility(trainX, trainy, testX, testy, cols, criteria='gini', maxDepth=7):
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
    # TODO: Implement Recursive feature elimination
    model = DecisionTreeClassifier(criterion=criteria, max_depth=maxDepth)
    model.fit(trainX, trainy)
    prediction = model.predict(testX)
    acc = accuracy_score(testy, prediction)
    cm = confusion_matrix(testy, prediction)
    cr = classification_report(testy, prediction, zero_division=0)
    fi = dict(zip(cols, model.feature_importances_))
    #rocAuc = roc_auc_score(testy, prediction, multi_class='ovr')
    kappa = cohen_kappa_score(prediction, testy, weights='quadratic')
    return acc, cm, cr, kappa, model, fi# rocAuc, model

# Decision Tree
def decision_tree(X, y, nFold=5):
    """
    Description:
        Performs training testing split
        Train model for various depth level
        Train model for both Entropy and GiniIndex

    Args:
        df (Dataframe)
    """
    # Kfold cross validation
    kfold = KFold(nFold, shuffle=True, random_state=1)

    # For storing Confusion Matrix
    confusionMatrixsEntropy = list()
    confusionMatrixsGini = list()

    # For storing Classification Report
    classReportsEntropy = list()
    classReportsGini = list()

    # Scores
    scoresGini = list()
    scoresEntropy = list()

    # ROC AUC 
    eRocs = list()
    gRocs = list()

    # Kappa values
    gKappaValues = list()
    eKappaValues = list()

    # Converting them to array
    cols = X.columns
    X = np.array(X)
    y = np.array(y)

    # Store models:
    eModels = list()
    gModels = list()

    # Feature importance
    eFeatures = list()
    gFeatures = list()

    for depth in tqdm(range(1, 31), desc='\n Modeling DT'):
        tempG = list()
        tempE = list()
        for foldTrainX, foldTestX in kfold.split(X):
            trainX, trainy, testX, testy = X[foldTrainX], y[foldTrainX], \
                                          X[foldTestX], y[foldTestX]

            # Gini
            gacc, gcm, gcr, gkappa, gmodel, gfi = tree_utility(trainX, trainy,
                                                 testX, testy, cols,
                                                 criteria='gini',
                                                 maxDepth=depth
                                                 )

            # Entropy
            eacc, ecm, ecr, ekappa, emodel, efi = tree_utility(trainX, trainy,
                                                  testX, testy, cols,
                                                  criteria='entropy',
                                                  maxDepth=depth )
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

    # Scores (ROCs)
    #eRocsDict = dict(zip(eRocs, depths))
    #gRocsDict = dict(zip(gRocs, depths))

    # Models
    eModelsDict = dict(zip(depths, eModels))
    gModelsDict = dict(zip(depths, gModels))

    # Feature Importance
    eFeatureDict = dict(zip(depths, eFeatures))
    gFeatureDict = dict(zip(depths, gFeatures))

    kappaVals, accVals, featImps = performance_summarizer(eKappaDict, gKappaDict,
                                           eConfDict, gConfDict,
                                           eClassDict, gClassDict,
                                           eScoreDict, gScoreDict,
                                           #eRocsDict, gRocsDict,
                                           eModelsDict, gModelsDict,
                                           eFeatureDict, gFeatureDict)

    # Return the average kappa value for state
    return kappaVals, accVals, featImps

def plot_centroids(states, centroidDf, metricName):
    """
    Description:

    Args:
        states (states)
        listOfMetricsValues (list of list)
        metricName (list)

    Returns:
        saves a 3d scatter plot
    """
    filename = metricName + ".html"
    title = "3D representation of centroids for the midwestern states"

    fig = px.scatter_3d(centroidDf,
                      x='subNumInt',
                      y='supNumInt',
                      z='deckNumInt',
                      color='name',
                      symbol='state',
                      title=title)

    plotly.offline.plot(fig, filename=filename)


def plot_overall_performance(states, listOfMetricValues, metricName, state):
    """
    Description:
        plots a barchart of all states and their metrics values
    Args:
        states (list)
        listOfMetricValues (list of list)
        metricName (list of names)
    Returns:
        saves a barchart
    """
    filename = metricName + '.png'

    # Values
    eMetricValues = list()
    gMetricValues = list()
    for metricVal in listOfMetricValues:
        eMetric, gMetric = metricVal
        eMetricValues.append(eMetric)
        gMetricValues.append(gMetric)

    height = np.array(range(0, len(states)))

    # Make the plot
    plt.figure(figsize=(10, 8))
    plt.title("Overall Performance")
    plt.bar(height, eMetricValues, color='#7f6d5f', width=0.25, label='gini')
    plt.bar(height + 0.25, gMetricValues,color='#557f2d', width=0.25, label='entropy')
    plt.xticks(height, states, rotation=45)
    plt.legend()
    plt.savefig(filename)

    print("\n" + metricName + "Table: ")
    dataFrame = pd.DataFrame()
    dataFrame['state'] = states
    dataFrame['gini'] = gMetricValues
    dataFrame['entropy'] = eMetricValues
    print(dataFrame)

def to_csv(listOfDataFrames):
    """
    Description:
        Convert the dataframe into csv files
    Args:
        listOfDataFrames: (list of dataframe)
    """
    concatDf = pd.concat(listOfDataFrames)
    concatDf.to_csv('allFiles.csv', sep=',', index=False)
