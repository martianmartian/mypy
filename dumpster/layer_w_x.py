# import numpy as np
from numpy import dot, random

"""
:param layers: A list containing the number of units in each layer.
Should be at least two values
:param activation: The activation function to be used. Can be
"logistic" or "tanh"
"""
class NeuralNetwork:
  def __init__(self, layers, activation='tanh'):
    if activation == 'logistic':
      self.activation = logistic
      self.activation_deriv = logistic_derivative
    elif activation == 'tanh':
      self.activation = tanh
      self.activation_deriv = tanh_deriv

    self.weights = []
    for i in range(1, len(layers) - 1):
      self.weights.append((2*random.random((layers[i - 1] + 1, layers[i]+ 1))-1)*0.25)
    self.weights.append((2*random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)

