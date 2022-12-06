"""
Skorch: Give Scikit learn like api to your pytorch networks
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetRegressor

X, Y = datasets.load_boston(return_X_y=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=123)



from torch import tensor

X_train = tensor(X_train, dtype=torch.float32)
X_test = tensor(X_test, dtype=torch.float32)
Y_train = tensor(Y_train.reshape(-1,1), dtype=torch.float32)
Y_test = tensor(Y_test.reshape(-1,1), dtype=torch.float32)

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()

        self.first_layer = nn.Linear(13, 26)
        self.second_layer = nn.Linear(26,52)
        self.final_layer = nn.Linear(52,1)

    def forward(self, x_batch):
        X = self.first_layer(x_batch)
        X = F.relu(X)

        X = self.second_layer(X)
        X = F.relu(X)

        return self.final_layer(X)

skorch_regressor = NeuralNetRegressor(module=Regressor,
                                         optimizer= optim.Adam,
                                         max_epochs=500,
                                         verbose=0)
print(X_train.shape, Y_train.shape)
skorch_regressor.fit(X_train, Y_train)
y_preds = skorch_regressor.predict(X_test)
print(y_preds[:5])

from sklearn.metrics import mean_squared_error

print("Train MSE : {}".format(mean_squared_error(Y_train, skorch_regressor.predict(X_train).reshape(-1))))
print("Test  MSE : {}".format(mean_squared_error(Y_test, skorch_regressor.predict(X_test).reshape(-1))))

print("\nTrain R^2 : {}".format(skorch_regressor.score(X_train, Y_train)))
print("Test  R^2 : {}".format(skorch_regressor.score(X_test, Y_test)))

skorch_regressor.history[:, ("train_loss", "valid_loss")][-5:]

