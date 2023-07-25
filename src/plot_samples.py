import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Generate sample data (same as the previous example)
np.random.seed(42)
n_samples = 1000
X = np.random.rand(n_samples, 3)
y = (X[:, 0] + X[:, 1] + X[:, 2] + np.random.normal(0, 0.2, n_samples)) > 1.5

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_train)

# Create a DataFrame with feature names
feature_names = ['Feature1', 'Feature2', 'Feature3']
df = pd.DataFrame(X_train, columns=feature_names)

# Create scatter plots for each pair of features
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Feature1 vs Feature2
axes[0].scatter(df['Feature1'], df['Feature2'], c=y_train, cmap='viridis', alpha=0.5)
axes[0].set_xlabel('Feature1')
axes[0].set_ylabel('Feature2')
axes[0].set_title('Feature1 vs Feature2')

# Feature2 vs Feature3
axes[1].scatter(df['Feature2'], df['Feature3'], c=y_train, cmap='viridis', alpha=0.5)
axes[1].set_xlabel('Feature2')
axes[1].set_ylabel('Feature3')
axes[1].set_title('Feature2 vs Feature3')

# Feature1 vs Feature3
axes[2].scatter(df['Feature1'], df['Feature3'], c=y_train, cmap='viridis', alpha=0.5)
axes[2].set_xlabel('Feature1')
axes[2].set_ylabel('Feature3')
axes[2].set_title('Feature1 vs Feature3')

plt.tight_layout()
plt.show()

