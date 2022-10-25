import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from prep.prep_nbi import *
from preprocessing import *

#data_folder = 'input/'
#data_file = datafolder + "iris.csv"
#data_file = 'iris.csv'
#print(data_file)

#df = pd.read_csv(data_file)
3df.head()

#X = df.iloc[:, 0:4].values
#y = df.iloc[:, 4].values

#print(X[0:5])
#print(y[0:5])

#print(X.shape)
#print(y.shape)

#X, y = data_preprocessing()
bridge_X, bridge_y, cols =  preprocess()
print("Comparing features")
print(X[:1])
print(bridge_X[:1])
#
print("Comparing labels")
print(y[:1])
print(bridge_y[:1])

encoder = LabelEncoder()
y1 = encoder.fit_transform(bridge_y)

Y = pd.get_dummies(y1).values
#print(Y[0:5])

# Simple sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax'), # 2 / 3
])

bridge_X = np.asarray(bridge_X).astype('float32')

X_train, X_test, y_train, y_test =  train_test_split(bridge_X, Y, test_size=0.2, random_state=0)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=50, epochs=100)

## Evaluate model
loss, acc =  model.evaluate(X_test, y_test, verbose=0)
print("Test loss: ", loss)
print("Test accuracy:", acc)

## Predict test model
y_pred = model.predict(X_test)
print(y_pred)

actual = np.argmax(y_test, axis=1)
predict = np.argmax(y_pred, axis=1)

print(f'Actual:{actual}')
print(f'Predicted:{predict}')
