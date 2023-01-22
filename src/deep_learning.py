import tensorflow as tf
import pandas as pd
import numpy as np

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

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from prep.prep_nbi import *
from preprocessing import *

import shap

"""
   Follow: https://towardsdatascience.com/deep-learning-model-interpretation-using-shap-a21786e91d16
"""

def main():

    # States
    states = [
              #'wisconsin_deep.csv',
              #'colorado_deep.csv',
              #'illinois_deep.csv',
              #'indiana_deep.csv',
              #'iowa_deep.csv',
              #'minnesota_deep.csv',
              #'missouri_deep.csv',
              #'ohio_deep.csv',
              'nebraska_deep.csv',
              #'indiana_deep.csv',
              #'kansas_deep.csv',
             ]

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

        for foldTrainX, foldTestX in kFold.split(bridge_X):
            X_train, y_train, X_test, y_test = bridge_X[foldTrainX], Y[foldTrainX], \
                                            bridge_X[foldTestX], Y[foldTestX]

            train_x = np.array(X_train, dtype='f')

            # Simple sequential model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, activation='relu'),
                tf.keras.layers.Dense(10, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax'), # 2 / 3
            ])

            #X_train = pd.DataFrame(X_train, columns=cols)

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

            #model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            #model = tf.keras.Model(inputs=input, outputs=output, name='Deck Maintenance Model')

            # Compile
            model.compile(optimizer='rmsprop',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            # Model fit
            model.fit(X_train, y_train, batch_size=64, epochs=100)

            # Implement this as a separate functions
            # Compute SHAP Values
            #explainer = shap.DeepExplainer(model, X_train)
            explainer = shap.Explainer(model, train_x)

            #explainer = shap.KernelExplainer(model, X_train[:5])
            shap_values = explainer(train_x)
            #shap_values = explainer.shap_values(X_test)

            #shap.summary_plot(shap_values[0], plot_type='bar', feature_names=cols)

            # Calculating mean shap values also known as SHAP feature importance

            #mean_shap = []
            #for target_class in shap_values:
            #    mean_shap.append(np.mean(target_class, axis=0)) # Averaging shap values across all values row

            #mean_shap_2 = np.mean(mean_shap, axis=0)
            mean_shap = np.mean(abs(shap_values.values), axis=0).mean(1)
            mean_shap_features = {column:shap_v for column, shap_v in zip(cols, mean_shap)}

            # Evaluate model
            loss, acc =  model.evaluate(X_test, y_test, verbose=0)
            #print("Test loss: ", loss)
            #print("Test accuracy:", acc)

            # Predict test model
            y_pred = model.predict(X_test)
            y_pred_r = y_pred[::, 1]
            y_test_r = y_test[::, 1]

            kappa_metric = tfa.metrics.CohenKappa(num_classes=2,
                                                  sparse_labels=False)

            kappa_metric.update_state(y_test_r, y_pred_r)
            kappa_result = kappa_metric.result()
            _kappa = kappa_result.numpy()

            # Kappa
            #print("Test Kappa:", _kappa)

            # Computing AUC and ROC value
            fpr, tpr, threshold_keras = roc_curve(y_test_r, y_pred_r)
            _auc = auc(fpr, tpr)
            #print("AUC: ", _auc)

            actual = np.argmax(y_test, axis=1)
            predict = np.argmax(y_pred, axis=1)

            #print(f'Actual:{actual}')
            #print(f'Predicted:{predict}')

            state_name = state[:-9]
            performance['state'].append(state_name)
            performance['accuracy'].append(acc)
            performance['kappa'].append(_kappa)
            performance['auc'].append(_auc)
            performance['fpr'].append(fpr)
            performance['tpr'].append(tpr)
            performance['shap_values'].append(mean_shap_features)

            # Create a dataframe
            temp_df = pd.DataFrame(performance, columns=['state',
                                                         'accuracy',
                                                         'kappa',
                                                         'auc',
                                                         'fpr',
                                                         'tpr',
                                                         'shap_values'
                                                        ])
        temp_dfs.append(temp_df)
        performance_df = pd.concat(temp_dfs)
        return performance

if __name__=='__main__':
    main()
