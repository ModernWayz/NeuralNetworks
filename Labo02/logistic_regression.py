import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate, epsilon):
        self.lr = learning_rate
        self.epsilon = epsilon
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        while True:
            # approximate y with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            # apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            new_w = self.weights - self.lr * dw
            new_b = self.bias - self.lr * db

            magnitude_change = abs(new_b - self.bias)

            for i in range(len(new_w)):
                magnitude_change += abs(new_w[i] - self.weights[i])

            print(magnitude_change)
            print(self.epsilon)
            if magnitude_change < self.epsilon:
                break
            else:
                # update parameters
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    # Imports
    import pandas as pd
    #from sklearn.model_selection import train_test_split
    #from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy


    testDat = pd.read_csv("Labo02/data/test.csv", header=None, sep=';')
    trainDat = pd.read_csv("Labo02/data/train.csv", header=None, sep=';')

    y_train = trainDat.iloc[:, 0]
    X_train = trainDat.iloc[:, 1:7]
    X_test = testDat.iloc[:, 1:7]
    y_test = testDat.iloc[:, 0]



    print(X_train)
    regressor = LogisticRegression(learning_rate=0.0042, epsilon=0.0001)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    print("LR classification accuracy:", accuracy(y_test, predictions))