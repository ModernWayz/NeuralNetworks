import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

cols = ['x', 'y']
dat = pd.read_csv("Labo01/src/train_fictief.csv", names=cols, header=None, sep=';')

print(dat)

dat.plot(kind='scatter', x='x', y='y')

#w = 0.43271063933252424
#b = 4.355605491877763
#Y_pred = w*dat.x.values + b
#plot.plot([min(dat.x.values), max(dat.x.values)], [min(Y_pred), max(Y_pred)], color='red')
#plot.show()

class LinearModel:
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def predict(self, x):
        return self.slope * x + self.intercept

    def calcCost(self, y_hat, y):
        return np.sum(np.square(y_hat - y)) / len(y)

model = LinearModel(1, 1.8)

y_hat = model.predict(dat.x.values)
print(model.calcCost(y_hat=y_hat, y=dat.y.values))

class LinearRegression:
    def __init__(self, model):
        self.model = model

    def gradientDescent(self, alpha, w, b, epsilon):
        self.model
        x = dat.x.values
        y = dat.y.values
        n = float(len(x))
        print("Loading plot...")

        stop = False
        while stop == False:
            y_hat = w * x + b
            new_w = w - alpha * ((-2 / n) * sum(x * (y - y_hat)))
            new_b = b - alpha * ((-2 / n) * sum(y - y_hat))

            if (abs(w-new_w) < epsilon) and (abs(b - new_b) < epsilon):
                # Show plot
                Y_pred = w * dat.x.values + b
                #plot.plot([min(dat.x.values), max(dat.x.values)], [min(Y_pred), max(Y_pred)], color='red')
                plot.plot(dat.x.values , Y_pred, color='red')
                plot.show()

                stop = True
                return w, b
            else:
                w = new_w
                b = new_b

gd = LinearRegression(model= model)
print(gd.gradientDescent(0.0001, 0, 0, 0.0000000001))