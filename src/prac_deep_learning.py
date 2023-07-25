import tensorflow as tf
import pandas as pd
import numpy as np
from collections import Counter

from tensorflow.keras import layers
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Metric
import tensorflow_addons as tfa

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import plot_partial_dependence
from mpl_toolkits.mplot3d import Axes3D

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from prep.prep_nbi import *
from preprocessing import *

import shap

"""
   Follow: https://towardsdatascience.com/deep-learning-model-interpretation-using-shap-a21786e91d16
"""

def plot_samples(X_train, Y_train, feature_names, classifier):
    """
    Plot samples across all three dimensions
    """
    # Dataframes
    df = pd.DataFrame(X_train, columns=feature_names)

    # Map classes
    map_class = {
        1: True,
        0: False,
    }

    y_train = []
    # Target class
    print("printing the Y_trains")
    print(Counter(Y_train))
    #for target in Y_train:
    #    print(target)
        #y_train.append(map_class[target])

    # Create scatter plots for each pair of features
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    feature_1 = feature_names[0]
    feature_2 = feature_names[1]
    feature_3 = feature_names[2]

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
    plt.savefig("sample_plot_deep_learning.png", dpi=300)


    # Specify the features for which you want to create the 3D PDP plot
    features_to_plot = [(0, 1), (1, 2), (0, 2)]  # Pairs of feature indices for 3D PDP plot

    # Create the 3D PDP plot
    fig = plt.figure(figsize=(12, 8))
    plot_partial_dependence(classifier, df, features_to_plot, grid_resolution=50)

    plt.subplots_adjust(top=0.9)  # Adjust the position of the title
    plt.suptitle('3D Partial Dependency Plot', fontsize=16)

    #plt.show()
    plt.savefig("PDP_plot_deep_learning.png", dpi=300)


def bridge_model(X_train):
    """
    Description:
        Performs the modeling (Deep learning) and returns performance metrics

    Args:
        trainX: Features of Training Set

    Return:
        model: Deep learning machine model
    """

    # Definition
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))

    # Compile
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    """
    Description:
        Performs the modeling (Deep learning) and returns performance metrics

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
        auc: area under curve
        fpr: false positive rate
        tpr: true positive rate
        model: Deep learning machine learning model
    """
    # States
    states = ['nebraska_deep.csv']

    temp_dfs = list()
    for state in states:
        state_file = '../data/' + state

        # Preprocess dataset
        bridge_X, bridge_y, cols =  preprocess(csv_file=state_file)

        # Label encoder
        encoder = LabelEncoder()
        y1 = encoder.fit_transform(bridge_y)
        Y = pd.get_dummies(y1).values

        # Conversion
        bridge_X = np.asarray(bridge_X).astype('float32')

        # K-fold cross validation
        kFold = KFold(5, shuffle=True, random_state=1)
        performance = defaultdict(list)
        model_accuracy_list = []

        # Run model
        run_number = 0
        bridge_X = bridge_X[:, :3]
        cols = cols[:3]

        for foldTrainX, foldTestX in kFold.split(bridge_X):
            X_train, y_train, X_test, y_test = bridge_X[foldTrainX], Y[foldTrainX], \
                                            bridge_X[foldTestX], Y[foldTestX]

            train_x = np.array(X_train, dtype='f')

            # Initialize model
            model = bridge_model(X_train)

            # Model fit
            model.fit(X_train, y_train, batch_size=64, epochs=100)

            ## Compute SHAP Values
            #explainer = shap.Explainer(model, bridge_X)
            #shap_values = explainer(bridge_X)
            #mean_shap = np.mean(abs(shap_values.values), axis=0).mean(1)
            #mean_shap_features = {column:shap_v for column, shap_v in zip(cols, mean_shap)}
            #mean_shap_features = {}

            # Evaluate model
            loss, acc =  model.evaluate(X_test, y_test, verbose=0)

            # Predict test model
            y_pred = model.predict(X_test)
            y_pred_r = y_pred[::, 1]
            y_test_r = y_test[::, 1]

            # Compute metrics
            kappa_metric = tfa.metrics.CohenKappa(num_classes=2,
                                                  sparse_labels=False)

            kappa_metric.update_state(y_test_r, y_pred_r)
            kappa_result = kappa_metric.result()
            _kappa = kappa_result.numpy()

            # Computing AUC and ROC value
            fpr, tpr, threshold_keras = roc_curve(y_test_r, y_pred_r)
            _auc = auc(fpr, tpr)

            actual = np.argmax(y_test, axis=1)
            predict = np.argmax(y_pred, axis=1)

            state_name = state[:-9]
            performance['state'].append(state_name)
            performance['accuracy'].append(acc)
            performance['kappa'].append(_kappa)
            performance['auc'].append(_auc)
            performance['fpr'].append(fpr)
            performance['tpr'].append(tpr)
            #performance['shap_values'].append(mean_shap_features)

            model_accuracy_set = (acc, model)
            model_accuracy_list.append(model_accuracy_set)

            # Create a dataframe
            temp_df = pd.DataFrame(performance, columns=['state',
                                                         'accuracy',
                                                         'kappa',
                                                         'auc',
                                                         'fpr',
                                                         'tpr',
            #                                             'shap_values'
                                                        ])
            run_number = run_number + 1
            #print("running model:", run_number)
            break

        best_model = max(model_accuracy_list, key=lambda item: item[0])
        temp_dfs.append(temp_df)
    plot_samples(X_train, y_train, cols, model)
   # performance_df = pd.concat(temp_dfs)

   # # Performance dataframe
   # df_perf = performance_df[['accuracy', 'kappa', 'auc']]

   # # Create FPR dataframe
   # fprs = [fpr for fpr in performance_df['fpr']]
   # fprs_df = pd.DataFrame(fprs).transpose()
   # fprs_df.columns=['k1', 'k2', 'k3', 'k4', 'k5']

   # # Create TPR dataframe
   # tprs = [tpr for tpr in performance_df['tpr']]
   # tprs_df = pd.DataFrame(tprs).transpose()
   # tprs_df.columns=['k1', 'k2', 'k3', 'k4', 'k5']

   # # Combine the dictionaries for shap values
   # dict1, dict2, dict3, dict4, dict5 = performance_df['shap_values']

   # # Combined dictionary
   # combined_dict = defaultdict()
   # for key in dict1.keys():
   #     vals = []
   #     val1 = dict1[key]
   #     val2 = dict2[key]
   #     val3 = dict3[key]
   #     val4 = dict4[key]
   #     val5 = dict5[key]
   #     mean_val = np.mean([val1, val2, val3, val4, val5])
   #     combined_dict[key] = mean_val

   # # Convert the dictionary into a pandas DataFrame
   # df = pd.DataFrame.from_dict(combined_dict, orient='index', columns=['values'])

   # # Reset index and rename column
   # #df = df.reset_index().rename(columns={'index': 'features'})
   # #df.to_csv('deep_learning_shap_values_deck.csv')
   # #df_perf.to_csv('deep_learning_performance_values_deck.csv')
   # #fprs_df.to_csv('deep_learning_fprs_deck.csv')
   # #tprs_df.to_csv('deep_learning_tprs_deck.csv')
   # print(df_perf)

    return performance_df

if __name__=='__main__':
    main()
