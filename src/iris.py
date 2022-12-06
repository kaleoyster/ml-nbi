import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

from keras.wrappers.scikit_learn import KerasClassifier
from prep.prep_nbi import *
from preprocessing import *

# Preprocess dataset
bridge_X, bridge_y, cols =  preprocess()

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

    # Training testing split
    #X_train, X_test, y_train, y_test =  train_test_split(bridge_X, Y, test_size=0.2, random_state=0)

    # Simple sequential model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax'), # 2 / 3
    ])

    # Compile
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Model fit
    model.fit(X_train, y_train, batch_size=50, epochs=3)

    # Evaluate model
    loss, acc =  model.evaluate(X_test, y_test, verbose=0)
    print("Test loss: ", loss)
    print("Test accuracy:", acc)

    # Predict test model
    y_pred = model.predict(X_test)
    y_pred_r = y_pred[::, 1]
    y_test_r = y_test[::, 1]

    # Kappa
    print("Printing samples")
    print(y_pred[:5])
    print(y_test[:5])

    y_test_classes = model.predict_classes(X_test)
    kappa = cohen_kappa_score(y_pred, y_test)
    print("Test Kappa:", kappa)

    # Computing AUC and ROC value
    fpr, tpr, threshold_keras = roc_curve(y_test_r, y_pred_r)
    auc = auc(fpr, tpr)
    print("AUC: ", auc)

    actual = np.argmax(y_test, axis=1)
    predict = np.argmax(y_pred, axis=1)

    print(f'Actual:{actual}')
    print(f'Predicted:{predict}')

    preformance['accuracy'].append(acc)
    preformance['kappa'].append(kappa)
    preformance['auc'].append(auc)

    #preformance['confusion_matrix'].append(cm)
    #preformance['classificaltion_report'].append(cr)
    #preformance['shap_values'].append(sv)
    #preformance['lime_val'].append(sv)

