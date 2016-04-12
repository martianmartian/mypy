''' linear regression, single/multiple variable'''
# batch_gradient_descent_algo.py
import numpy as np
from sklearn.datasets.samples_generator import make_regression 
import pylab


def gradient_descent_2(alpha, x, y, numIterations):
    m = x.shape[0]
    theta = np.ones(2)
    for iter in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        diff = hypothesis - y
        # J = np.sum(diff ** 2) / (2 * m)
        gradient = np.dot(x.T, diff) / m
        theta = theta - alpha * gradient  # update
    return theta

if __name__ == '__main__':

    x, y = make_regression(n_samples=100, n_features=1, n_informative=1, 
                        random_state=0, noise=35)
    m, n = x.shape
    x = np.c_[ np.ones(m), x] # insert column
    alpha = 0.01 # learning rate
    theta = gradient_descent_2(alpha, x, y, 1000)

    for i in range(x.shape[1]):
        y_predict = theta[0] + theta[1]*x
    pylab.plot(x[:,1],y,'o')
    pylab.plot(x,y_predict,'k-')
    pylab.show()
    print "Done!"


================


import numpy as np
from sklearn.datasets.samples_generator import make_regression
import matplotlib.pyplot as plt
# x,y = make_regression(n_samples=500, n_features=1, n_informative=1,random_state=1, noise=35)
x = np.linspace(-5,5,50)
y = np.linspace(0,5,50) + np.array(np.random.random(50))

alpha = 0.01
m = x.shape[0]
ones = np.ones(x.shape[0])
J = []
X52 = np.c_[ones,x]
theta12 = np.random.random(X52.shape[1])

for i in range(1000):
  hypothesis = np.dot(X52,theta12.T)
  diff = hypothesis - y
  J.append(np.sum(diff**2)/(2*m))
  gradients = np.dot(diff, X52)/m
  theta12 = theta12 - alpha*gradients

plt.plot(x,y,'o')
plt.plot(x,np.dot(X52,theta12.T),'-')
# plt.plot(J,'o')
plt.show()


==================

import numpy as np
from numpy import linspace,array,random
import matplotlib.pyplot as plt
m = 50
n = 3
x = linspace(0,5,m)
randomX1 = x + array(random.random(m))
randomX2 = x + array(random.random(m))*5+10
randomX3 = x + array(random.random(m))*0.1-10
# plt.plot(x,randomX1,'o')
# plt.plot(x,randomX2,'x')
# plt.plot(x,randomX3,'-')
# plt.show()
y = np.linspace(0,5,50) + np.array(np.random.random(50))


ones = np.ones(m)
X = np.c_[ones, randomX1,randomX2,randomX3]
alpha = 0.0002
J = []
theta = np.random.random(n+1)  # print theta.shape #(4,)

for i in range(100):
  hypothesis = np.dot(X,theta)
  diff = hypothesis - y
  J.append(np.sum(diff**2)/(2*m))
  gradients = np.dot(diff,X)/m
  theta = theta - alpha * gradients


plt.plot(J,'o')
plt.show()

==================================
screate abitary X,y matrix 
create random linearly increasing X, y
scale / normalze X with std() 
==========


import numpy as np
from numpy import linspace,array,random
import matplotlib.pyplot as plt


def create_X_y(m,n):
  ones = np.ones(m)
  x = np.random.random((m,n))
  count = 0
  while (count < n):
    x[:,count] += np.arange(m)*np.random.random()*np.random.randint(10)
    count = count + 1
  X = np.c_[ones,x]
  y = np.linspace(0,5,m) + np.random.random(m)
  return X,y

def scale_std(matrix):
  m, n = X.shape
  count = 1
  while (count<n):
    array = matrix[:,count]
    matrix[:,count] = (array - array.mean())/array.std()
    count = count +1
  return matrix

m = 40
n = 3
X,y = create_X_y(m,n)
X = scale_std(X)
alpha = 0.02
J = []
theta = np.random.random(n+1)  # print theta.shape #(4,)

for i in range(100):
  hypothesis = np.dot(X,theta)
  diff = hypothesis - y
  J.append(np.sum(diff**2)/(2*m))
  gradients = np.dot(diff,X)/m
  theta = theta - alpha * gradients
# print theta
plt.plot(J,'o')
plt.show()

============== test which alpha converges ============

import numpy as np
from numpy import linspace,array,random
import matplotlib.pyplot as plt


def create_X_y(m,n):
  ones = np.ones(m)
  x = np.random.random((m,n))
  count = 0
  while (count < n):
    x[:,count] += np.arange(m)*np.random.random()*np.random.randint(10)
    count = count + 1
  X = np.c_[ones,x]
  y = np.linspace(0,5,m) + np.random.random(m)
  return X,y

def scale_std(matrix):
  m, n = X.shape
  count = 1
  while (count<n):
    array = matrix[:,count]
    matrix[:,count] = (array - array.mean())/array.std()
    count = count +1
  return matrix

m = 4
n = 2
X,y = create_X_y(m,n)
X = scale_std(X)
# alpha = 0.01
# J=[5]
theta = np.random.random(n+1)  # print theta.shape #(4,)

def testALPHA(alpha,J,X,y,m,n,theta):
  n = 0
  while J[-1]>1:
    n = n+1
    if n == 200:
      break
    hypothesis = np.dot(X,theta)
    diff = hypothesis - y
    J.append(np.sum(diff**2)/(2*m))
    gradients = np.dot(diff,X)/m
    theta = theta - alpha * gradients
  print J[-1]

alphas = np.arange(0.0005,0.02,0.0005)
for alpha in alphas:
  testALPHA(alpha,[5],X,y,m,n,theta)

# plt.plot(J,'o')
# plt.show()
# print np.array(J).shape[0]  # number of iterations done

=============== sovle for theta numerically ==================


import numpy as np
from numpy import linspace,array,random
import matplotlib.pyplot as plt
from numpy.linalg import pinv

def create_X_y(m,n):
  ones = np.ones(m)
  x = np.random.random((m,n))
  count = 0
  while (count < n):
    x[:,count] += np.arange(m)*np.random.random()*np.random.randint(10)
    count = count + 1
  X = np.c_[ones,x]
  y = np.linspace(0,5,m) + np.random.random(m)
  return X,y

def scale_std(matrix):
  m, n = X.shape
  count = 1
  while (count<n):
    array = matrix[:,count]
    matrix[:,count] = (array - array.mean())/array.std()
    count = count +1
  return matrix

m = 3
n = 2
X,y = create_X_y(m,n)
X = scale_std(X)
alpha = 0.02
J = []
theta = np.random.random(n+1)  # print theta.shape #(4,)

for i in range(10000):
  hypothesis = np.dot(X,theta)
  diff = hypothesis - y
  J.append(np.sum(diff**2)/(2*m))
  gradients = np.dot(diff,X)/m
  theta = theta - alpha * gradients
print theta
plt.plot(J,'o')
plt.show()

XT = X.T
thetaXT = pinv(XT.dot(X)).dot(XT).dot(y)
print thetaXT


================== new tricks  ===========================
# Python imports
import numpy  # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
# Allow matplotlib to plot inside this notebook
# %matplotlib inline
# Set the seed of the numpy random number generator so that the tutorial is reproducable
numpy.random.seed(seed=1)

# Define the vector of input samples as x, with 20 values sampled from a uniform distribution
# between 0 and 1
x = numpy.random.uniform(0, 1, 20)
def f(x): return x * 2
# Create the targets t with some gaussian noise
noise_variance = 0.2  # Variance of the gaussian noise
# Gaussian noise error for each sample in x
noise = numpy.random.randn(x.shape[0]) * noise_variance
# Create targets t
t = f(x) + noise
'''# # Plot the target t versus the input x'''
# plt.plot(x, t, 'o', label='t')
# # Plot the initial line
# plt.plot([0, 1], [f(0), f(1)], 'b-', label='f(x)')
# plt.xlabel('$x$', fontsize=15)
# plt.ylabel('$t$', fontsize=15)
# plt.ylim([0,2])
# plt.title('inputs (x) vs targets (t)')
# plt.grid()
# plt.legend(loc=2)
# plt.show()

'''# Define the neural network function y = x * w'''
def nn(x, w): return x * w

# Define the cost function
def cost(y, t): return ((t - y)**2).sum()

''' # Plot the cost vs the given weight w'''
# # Define a vector of weights for which we want to plot the cost
ws = numpy.linspace(0, 4, num=100)  # weight values
cost_ws = numpy.vectorize(lambda w: cost(nn(x, w) , t))(ws)  # cost for each weight in ws

# # Plot
# plt.plot(ws, cost_ws, 'r-')
# plt.xlabel('$w$', fontsize=15)
# plt.ylabel('$\\xi$', fontsize=15)
# plt.title('cost vs. weight')
# plt.grid()
# plt.show()

'''# define the gradient function. Remember that y = nn(x, w) = x * w'''
def gradient(w, x, t): 
    return 2 * x * (nn(x, w) - t)

# define the update function delta w
def delta_w(w_k, x, t, learning_rate):
    return learning_rate * gradient(w_k, x, t).sum()

# Set the initial weight parameter
w = 0
# Set the learning rate
learning_rate = 0.1

# Start performing the gradient descent updates, and print the weights and cost:
nb_of_iterations = 10  # number of gradient descent updates
w_cost = [(w, cost(nn(x, w), t))] # List to store the weight,costs values
for i in range(nb_of_iterations):
    dw = delta_w(w, x, t, learning_rate)  # Get the delta w update
    w = w - dw  # Update the current weight parameter
    w_cost.append((w, cost(nn(x, w), t)))  # Add weight,cost to list

# Print the final w, and cost
for i in range(0, len(w_cost)):
    print('w({}): {:.4f} \t cost: {:.4f}'.format(i, w_cost[i][0], w_cost[i][1]))


'''# Plot the first 2 gradient descent updates'''
# plt.plot(ws, cost_ws, 'r-')  # Plot the error curve
# # Plot the updates
# for i in range(1, len(w_cost)-2):
#     w1, c1 = w_cost[i-1]
#     w2, c2 = w_cost[i]
#     plt.plot(w1, c1, 'bo')
#     plt.plot([w1, w2],[c1, c2], 'b-')
#     plt.text(w1, c1+0.5, '$w({})$'.format(i))

# # Plot the last weight, axis, and show figure
# w1, c1 = w_cost[len(w_cost)-3]
# plt.plot(w1, c1, 'bo')
# plt.text(w1, c1+0.5, '$w({})$'.format(nb_of_iterations))  
# plt.xlabel('$w$', fontsize=15)
# plt.ylabel('$\\xi$', fontsize=15)
# plt.title('Gradient descent updates plotted on cost function')
# plt.grid()
# plt.show()

'''# Plot the fitted line agains the target line'''
# Plot the target t versus the input x
plt.plot(x, t, 'o', label='t')
# Plot the initial line
plt.plot([0, 1], [f(0), f(1)], 'b-', label='f(x)')
# plot the fitted line
plt.plot([0, 1], [0*w, 1*w], 'r-', label='fitted line')
plt.xlabel('input x')
plt.ylabel('target t')
plt.ylim([0,2])
plt.title('input vs. target')
plt.grid()
plt.legend(loc=2)
plt.show()
