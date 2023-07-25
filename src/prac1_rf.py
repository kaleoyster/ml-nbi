import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import plot_partial_dependence

# Generate sample data
np.random.seed(42)
n_samples = 1000
X = np.random.rand(n_samples, 3)
y = (X[:, 0] + X[:, 1] + X[:, 2] + np.random.normal(0, 0.2, n_samples)) > 1.5

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Create a DataFrame with feature names
feature_names = ['Feature1', 'Feature2', 'Feature3']
df = pd.DataFrame(X_train, columns=feature_names)

# Specify the features for which you want to create the 3D PDP plot
features_to_plot = [(0, 1), (1, 2), (0, 2)]  # Pairs of feature indices for 3D PDP plot

# Create the 3D PDP plot
#fig = plt.figure(figsize=(12, 8))
plot_partial_dependence(rf_classifier, df, features_to_plot, grid_resolution=50)

plt.subplots_adjust(top=0.9)  # Adjust the position of the title
plt.suptitle('3D Partial Dependency Plot', fontsize=16)
plt.show()
