"""
Description:
    Support vector machine to predict the future maintenance of bridges

Date:
   October 3rd, 2022
"""

# Essentials
import sys
import csv
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pydotplus

# Model 
import sklearn
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# PDP
from sklearn.inspection import PartialDependenceDisplay

# SHAP
import shap
from shap import KernelExplainer
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

#def visualize_support_vectors(support_vectors):
#    ""

def support_vector_utility(train_x, trainy,
                 test_x, testy, cols):
    """
    Description:
        Performs SVM the modeling and returns performance metrics

    Args:
        trainX: Features of Training Set
        trainy: Ground truth of Training Set
        testX: Features of Testing Set
        testy: Ground truth of Testing Set

    Return:
        acc: Accuracy
        cm: Confusion Report
        cr: Classification Report
        kappa: Inter-rater measurement
        tpr: True positive rate
        fpr: False positive rate
        model: Support vector machine model
    """
    # Training and testing data
    train_x = np.array(train_x, dtype='f')
    X_train = pd.DataFrame(train_x, columns=cols)

    # Model initialization
    model = make_pipeline(StandardScaler(),
                          SVC(
                              #kernel='linear',
                              #gamma='auto',
                          probability=True))

    svc_model = model.named_steps['svc']

    # Fit model
    model.fit(train_x, trainy)
    #print("printing model support vectors:\n")
    #print(svc_model.coef_)

    ## Partial dependency
    #features = [0, 1]
    #PartialDependenceDisplay.from_estimator(model, X_train, features)
    #print("PartialDependenceDisplay Working OK")

    ## LIME:
    #svm_exp_lime = lime_tabular.LimeTabularExplainer(
    #    training_data = np.array(X_train),
    #    feature_names = X_train.columns,
    #    class_names=['Repair', 'No Repair'],
    #    mode='classification'
    #)

    ## Explaining the instances using LIME
    #instance_exp = svm_exp_lime.explain_instance(
    #    data_row = X_train.values[4],
    #    predict_fn = model.predict_proba
    #)

    #fig = instance_exp.as_pyplot_figure()
    #fig.savefig('svm_lime_report.jpg')
    #summary_plot(svm_sv, train_x, feature_names=cols)

    prediction = model.predict(test_x)
    prediction_prob = model.predict_proba(test_x)[::, 1]
    acc = accuracy_score(testy, prediction)
    _cm = confusion_matrix(testy, prediction)
    _cr = classification_report(testy, prediction, zero_division=0)

    class_label = {'negative':0,
                   'positive':1 }

    testy_num = [class_label[i] for i in testy]
    fpr, tpr, threshold = roc_curve(testy_num, prediction_prob)
    _auc = auc(fpr, tpr)
    _kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')
    #_support_vectors = model.support_vectors_
    #print("Printing support vectors")
    #print(_support_vectors)
    #print("\n")

    return acc, _cm, _cr, _kappa, _auc, fpr, tpr, model

def main():

    # States
    states = ['nebraska_deep.csv']

    temp_dfs = list()
    for state in states:
        state_file = '../data/' + state
        X, y, cols = preprocess(csv_file=state_file)
        X = X[:, :3]
        cols = cols[:3]

        # K-fold cross validation
        kfold = KFold(5, shuffle=True, random_state=1)

        # Contains all models
        gmodels = []

        # X is the dataset
        performance = defaultdict(list)
        for foldTrainX, foldTestX in kfold.split(X):
            trainX, trainy, testX, testy = X[foldTrainX], y[foldTrainX], \
                                              X[foldTestX], y[foldTestX]
            # Training
            gacc, gcm, gcr, gkappa, gauc, gfpr, gtpr, gmodel = support_vector_utility(trainX, trainy, testX, testy, cols)

            # Appending all models
            gmodels.append(gmodel)

            state_name = state[:-9]
            performance['accuracy'].append(gacc)
            performance['kappa'].append(gkappa)
            performance['auc'].append(gauc)
            performance['fpr'].append(gfpr)
            performance['tpr'].append(gtpr)
            performance['confusion_matrix'].append(gcr)
            performance['classification_report'].append(gcr)

            # Create a temp dataframe
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

    ## Select all data from X
    for model_no in range(5):
        X = np.array(X, dtype=float)
        #svm_exp = shap.Explainer(gmodels[model_no].predict_proba, X)
        #svm_sv = svm_exp(X)

        ## Counter
        #temp_mean_values = []

        ## for each observartion:
        #for observation in svm_sv:
        #    mean_shap_ob_val = []
        #    # For each feature there is the value:
        #    for ob, feat in zip(observation, cols):
        #        mean_shap_o_v = np.mean(np.abs(ob.values))
        #        mean_shap_ob_val.append(mean_shap_o_v)
        #    temp_mean_values.append(mean_shap_ob_val)

        ## Averaging shap values across all the observation
        #mean_values = np.mean(temp_mean_values, axis=0)

        # Shap dictionary
        #dictionary_svm_shap = dict(zip(cols, mean_values))

        # Concatenate all models together 
        #performance_df = pd.concat(temp_dfs)

        # Export SHAP Feature
        #filename = 'svm_shap_deck' + '_'+ str(model_no) + '.csv'
        #shap_series = pd.Series(dictionary_svm_shap)
        #shap_series.to_csv(filename)

    # Concatenate all models together 
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
    #dict1, dict2, dict3, dict4, dict5 = performance_df['shap_values']

    # Combined dictionary
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

    #df.to_csv('svm_shap_values_substructure.csv')
    #df_perf.to_csv('svm_performance_values_deck.csv')
    #fprs_df.to_csv('svm_fprs_deck.csv')
    #tprs_df.to_csv('svm_tprs_deck.csv')
    print("Performance:")
    print(df_perf)

    return performance_df

if __name__ == '__main__':
    main()
