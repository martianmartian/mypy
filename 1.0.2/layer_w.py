import numpy as np


def tanh(x):
    return np.tanh(x)
def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2
def logistic(x):
    return 1/(1 + np.exp(-x))
def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))


""" extra weight on each layer should be specified at definition """
class Layer_w_tanh:
  def __init__(self, weights):
    self.eta = 0.2
    self.response = tanh
    self.response_deriv = tanh_deriv
    self.w = np.random.rand(weights)*0.25
  # step one:
  def stimulate(self,x):
    print self.w, x, np.dot(self.w, x) 
    return np.dot(self.w, x) 

  def output(self,x):
    charge = self.stimulate(x)
    return self.response(charge)

  def updateWeights(self,x,error):
    # self.w += self.eta * error * x
    print "not right"


# class Layer_w_logistic:
#   def __init__(self, weights):
#     self.eta = 0.2
#     self.response = logistic
#     self.response_deriv = logistic_derivative
#     self.w = np.random.rand(weights)*0.25
#   # step one:
#   def stimulate(self,x):
#     return dot(self.w, x) 

#   def output(self,x):
#     charge = self.stimulate(x)
#     return self.response(charge)

#   def updateWeights(self,x,error):
#     # self.w += self.eta * error * x


class Layer_w_1:
  """ w currently is the property of current layer """
  def __init__(self):
    self.w = random.rand(3)
    self.eta = 0.2
    self.response = lambda x: 0 if x < 0 else 1

  # step one:
  def stimulate(self,x):
    return dot(self.w, x) 

  def output(self,x):
    charge = self.stimulate(x)
    return self.response(charge)

  def updateWeights(self,x,error):
    self.w += self.eta * error * x



