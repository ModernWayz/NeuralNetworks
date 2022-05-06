# Bronvermelding: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

# Slide 5
import pandas
import matplotlib.pyplot as plt
dataframe = pandas.read_csv('Labo_Week_09_10/data/airline-passengers.csv', usecols=[1], engine='python') 
#plt.plot(dataset)
#plt.show()

# Slide 7 & 8
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error

# Slide 9
numpy.random.seed(7)
dataset = dataframe.values
dataset = dataset.astype('float32')

# Slide 10
scaler = MinMaxScaler(feature_range=(0, 1)) 
datasetScaler = scaler.fit_transform(dataset)

# Slide 11
# Split in 2 delen, 70% train 30% test
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:] 
print(len(train), len(test))

# Slide 13
def create_matrix(dataset, look_back=1): 
    x, y = [], []
    for i in range(len(dataset)-look_back-1): 
        a = dataset[i:(i+look_back), 0] 
        x.append(a)
        y.append(dataset[i + look_back, 0]) 
    return numpy.array(x), numpy.array(y)

x, y = create_matrix(dataset)
print(x,y)