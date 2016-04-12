import matplotlib.pyplot as plt
import numpy as np

class Plot_errors:
  """ can import library be inside of class? so that saves capcaity"""
  def __init__(self,errors):
    self.errors = errors
    plt.plot(self.errors, marker='o')
    plt.title('Perceptron')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# errors = np.arange(-5,5,0.1)
# Plot_errors(errors)



class Plot_function:
  def __init__(self,fn):
    self.fn = fn
    # self.nnrange = np.arange(-5,5,0.1)
    self.nnrange = np.linspace(-5,5,100,endpoint=True)
    y = self.fn(self.nnrange)
    plt.plot(self.nnrange,y)
    plt.title('function plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()



# def tester(x):
#     return 1/np.exp(-x)
# def tester(x):
#     return 1/(1 + np.exp(-x))
# def logistic(x):
#     return 1/(1 + np.exp(-x))
# def tanh(x):
#     return np.tanh(x)

# def logistic_derivative(x):
#     return logistic(x)*(1-logistic(x))
# Plot_function(logistic_derivative)
# Plot_function()


