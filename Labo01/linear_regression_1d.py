import matplotlib.pyplot as plt
import csv
import numpy as np

class TrainingSample:
  def __init__(self, feature, label):
    self.feature = feature
    self.label = label

class TrainingSet:
  def __init__(self):
    self.training_samples = []
  
  def append(self, training_sample):
    self.training_samples.append(training_sample)

from random import uniform

class LinearModel:
  def __init__(self, slope, intercept):
    self.slope = slope
    self.intercept = intercept
  
  def forward(self, input):
    return self.slope * input + self.intercept
  
  def slope_derivative(self, training_set):
    m = len(training_set.training_samples)
    s = 0
    for training_sample in training_set.training_samples:
      prediction = self.forward(training_sample.feature)
      s += training_sample.feature * (training_sample.label - prediction)

    return (-2 / m) * s

  def intercept_derivative(self, training_set):
    m = len(training_set.training_samples)
    s = 0
    for training_sample in training_set.training_samples:
      prediction = self.forward(training_sample.feature)
      s += (training_sample.label - prediction)

    return (-2 / m) * s

  # Perfect fit --> cost 0. Higher cost is poorer fit
  def cost_of_fit(self, training_set):
    m = len(training_set.training_samples)

    cost = 0
    for training_sample in training_set.training_samples:
      predicted_y = self.forward(training_sample.feature)

      # L2 norm
      cost += pow(training_sample.label - predicted_y, 2)

    return cost / m

class LinearRegression:
  def __init__(self, learning_rate, convergence_limit):
    self.learning_rate = learning_rate
    self.convergence_limit = convergence_limit

  def perform(self, training_set):
    # Initialize model with random values, prediction will be purely guesswork. But we will get there
    model = LinearModel(uniform(0, 10), uniform(0, 10))

    iteration = 0
    while(True):
      iteration += 1

      # Follow the steepest slope (given by derivative) in steps defined by magnitude of learning rate
      new_slope = model.slope - self.learning_rate * model.slope_derivative(training_set)
      new_intercept = model.intercept - self.learning_rate * model.intercept_derivative(training_set)

      # Check for convergence
      magnitude_change = abs(new_slope - model.slope) + abs(new_intercept - model.intercept)
      if (magnitude_change < self.convergence_limit):
        # We are at a maximum/minimum
        break
      else:
        # Update values and continue
        model = LinearModel(new_slope, new_intercept)

      print(model.cost_of_fit(training_set))
      print(magnitude_change)
    
    return model

def read_labeled_csv(path):
  training_set = TrainingSet()
  with open(path, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
      float_row = list(map(lambda str: float(str), row))
      training_set.append(TrainingSample(float_row[0], float_row[1]))
  return training_set

def plot(model, training_set):
  x_arr = list(map(lambda s: s.feature, training_set.training_samples))
  y_arr = list(map(lambda s: s.label, training_set.training_samples))

  model_x = np.linspace(0, 10, 100)

  plt.scatter(x_arr, y_arr, label='Samples')
  plt.plot(model_x, list(map(lambda x: model.forward(x), model_x)), color='g')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title('Linear regression')
  plt.legend()
  plt.show()

training_set = read_labeled_csv('Labo01/data/train_fictief.csv')
optimization_fn = LinearRegression(1e-2, 1e-5)
best_fit = optimization_fn.perform(training_set)
print(f'{best_fit.slope}*x + {best_fit.intercept}')
plot(best_fit, training_set)
