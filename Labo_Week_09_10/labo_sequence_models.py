# Imports
import pandas
import matplotlib.pyplot as plt
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Settings
FILE_PATH = 'Labo_Week_09_10/data/airline-passengers.csv' # CSV file locatie
LOOK_BACK = 1
TRAIN_SPLIT = 0.7 # Percentage van de training dataset tov de hele dataset (schaal van 0 tot 1)
SEED = 7
EPOCHS = 100
BATCH_SIZE = 1
VERBOSE = 2
OPTIMIZER = 'adam'
LOSS = 'mean_squared_error'

# Dataset inladen
dataframe = pandas.read_csv(FILE_PATH, usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

numpy.random.seed(SEED)

# Dataset normaliseren in een range van 0 tot 1
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split dataset in train en test set
train_size = int(len(dataset) * TRAIN_SPLIT)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# Array naar matrix numpy array omzetten
def create_matrix(dataset, look_back = 1):
	x, y = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		x.append(a)
		y.append(dataset[i + look_back, 0])
	return numpy.array(x), numpy.array(y)

# Matrix aanmaken
train_x, train_y = create_matrix(train, LOOK_BACK)
test_x, test_y = create_matrix(test, LOOK_BACK)

# Reshape data
train_x = numpy.reshape(train_x, (len(train_x), 1, LOOK_BACK))
test_x = numpy.reshape(test_x, (len(test_x), 1, LOOK_BACK))

# LSTM network aanmaken
lstmModel = Sequential()
lstmModel.add(LSTM(4, input_shape=(1, LOOK_BACK)))
lstmModel.add(Dense(1))
lstmModel.compile(loss = LOSS, optimizer = OPTIMIZER)

# LSTM network fitten
history = lstmModel.fit(train_x, train_y, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = VERBOSE)

# Model predicten
predict_train = lstmModel.predict(train_x)
predict_test = lstmModel.predict(test_x)

# Predictions en data terug inverten naar originele scale
predict_train = scaler.inverse_transform(predict_train)
predict_test = scaler.inverse_transform(predict_test)

train_y = scaler.inverse_transform([train_y])
test_y = scaler.inverse_transform([test_y])

# RMSE (Root Mean Squared Error) berekenen
trainScore = mean_squared_error(train_y[0], predict_train, squared = False)
print('RMSE train:', trainScore)

testScore = mean_squared_error(test_y[0], predict_test, squared = False)
print('RMSE test:', testScore)

plt.plot(test_y[0])
plt.plot(predict_test)
plt.show()