

======================''' simple drawing of the cost function result'''======================
import matplotlib.pyplot as plt
plt.plot(J,'o')
plt.show()

======================''' 3d drawing'''======================
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
# X0, X1, Z = axes3d.get_test_data(0.05)

scale = 20
order = 4
X0 = np.arange(-scale*0.5, scale*0.5, 0.25)
X1 = np.arange(-scale*0.5, scale*0.5, 0.25)
X0, X1 = np.meshgrid(X0, X1)
Z = X0 + X1
ax.plot_surface(X0, X1, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(X0, X1, Z, zdir='z', offset=-scale, cmap=cm.coolwarm)
cset = ax.contour(X0, X1, Z, zdir='x', offset=-scale, cmap=cm.coolwarm)
cset = ax.contour(X0, X1, Z, zdir='y', offset=scale, cmap=cm.coolwarm)

ax.set_xlabel('X0')
ax.set_xlim(-scale, scale)
ax.set_ylabel('X1')
ax.set_ylim(-scale, scale)
ax.set_zlabel('Z')
ax.set_zlim(-scale, scale)

plt.show()


===================''' scatter plot data'''=====================
'''2-d'''
a_I = np.array([[0,0],[0,1],[1,0],[1,1]])
t_K = np.array([[0],[1],[1],[0]])

x1 = a_I[:,0]
x2 = a_I[:,1]
y = t_K[:,0]

import matplotlib.pyplot as plt
plt.scatter(x1, x2, c=y, s=100, cmap=plt.cm.cool, edgecolors='None', alpha=0.75)
plt.colorbar()
plt.show()

'''3-d'''
a_I = np.array([[0,0],[0,1],[1,0],[1,1]])
t_K = np.array([[0],[1],[1],[0]])

x1 = a_I[:,0]
x2 = a_I[:,1]
y = t_K[:,0]

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.cm.get_cmap("hot")
l = ax.scatter(x1,x2,y, c=y, cmap=cmhot)
fig.colorbar(l)
plt.show()

===================''' complex: weights and dataset'''=================
'''v-0.0.1'''
'''decision boundary ploting for XOR gate'''

lines drawn from layer_J themselves are the decision boundaries...

import sys
import PIL.Image
import scipy.misc, scipy.io, scipy.optimize, scipy.special
import numpy as np

import pylab
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlaba

import timeit
import cProfile, pstats, StringIO


def logis(x,theta):
  syn = np.dot(x,theta)
  # syn = syn*syn
  return scipy.special.expit(syn)
def logisDeri(x):
  return x*(1-x)
def addBias(source):
  ones = np.ones((source.shape[0],1))
  return np.c_[source,ones]
def getWeights(width1,width2):
  return np.random.random((width1,width2))*2 - 1
def projectToStandardField(source):
  '''project to standard field '''
  m,n = source.shape
  ones = np.ones((m, 25*25-n))*0.001
  source = np.c_[source,ones]
  return source
def plot_E(E):
  # print E
  import matplotlib.pyplot as plt
  E = np.array(E)/m
  plt.plot(E,'o')
  plt.show()
def get_accuracy(a_I, theta_J, theta_K):
  a_J = logis(a_I,theta_J)
  a_K = logis(a_J,theta_K)
  print a_K
def plot_lines_for_theta_(theta_, a_, t_K):
  # print a_
  '''plot_lines_for_theta_ '''
  # theta_K || L = np.array([[ 8.30783206, 6.88747479,-2.57799283],
  #                        [-5.24990104, 6.7286029, 8.0803377 ],
  #                         [ 1.49909855,-1.39486048,-1.08465038]])
  import matplotlib.pyplot as plt
  x2 = a_[0][2] # 1    # after 2nd, rest is bias-like for next layer
  countx0x1 = 0
  for i in range(theta_.shape[1]):
    w = theta_[:,i]
    x1 = np.linspace(-2,2,20)
    x0 = -1  * (w[1] * x1 + w[2] * x2) / w[0]
    plt.plot(x0,x1,'b-')
  
  if(countx0x1<2):  # fist 2 lines special
    plt.plot(x0,x1,'r-')
  else:             # after 2nd, rest is bias-like for next layer
    plt.plot(x0,x1,'b-')
  countx0x1 += 1

  # '''plot data sets'''
  x0 = a_[:,0]
  x1 = a_[:,1]
  y = t_K[:,0]

  plt.scatter(x0, x1, c=y, s=100, edgecolors='None')
  plt.colorbar()
  plt.grid()
  plt.show()
def plot_final_shape(theta_J, theta_K):
  '''plot_final_shape'''
  import matplotlib.pyplot as plt
  x0 = np.arange(-20, 20, 0.1)
  x1 = np.arange(-20, 20, 0.1)
  a_i = np.zeros((len(x0)*len(x1),2))
  for i in range(len(x0)):
    for j in range(len(x1)):
      a_i[i*len(x1)+j] = (x0[i],x1[j])
  a_i = addBias(a_i)
  a_j = logis(a_i, theta_J)
  a_k = logis(a_j, theta_K)
  for i in range(len(a_k)):
    if(a_k[i][0]) >=0.5:
      a_k[i][0] = 1
    else:
      a_k[i][0] = 0
  x0 = a_i[:,0]
  x1 = a_i[:,1]
  plt.scatter(x0, x1, c=a_k, s=100, cmap=plt.cm.cool, edgecolors='None', alpha=0.75)
  plt.colorbar()
  plt.show()

a_I = np.array([[0,0],[0,1],[1,0],[1,1]])
a_I = addBias(a_I)
# a_I = projectToStandardField(a_I)
m = a_I.shape[0]
t_K = np.array([[0],[1],[1],[0]])
alpha = 0.1
E = []

I_width = a_I.shape[1]
J_width = 3
K_width = t_K.shape[1]

theta_J = getWeights(I_width, J_width)
theta_K = getWeights(J_width,K_width)
counter = 0
for i in range(100):
  a_J = logis(a_I,theta_J)
  a_K = logis(a_J,theta_K)

  delta_K = t_K - a_K
  delta_J = delta_K.dot(theta_K.T)*a_J*(1-a_J)

  cost = -t_K * np.log(a_K+0.00000001) - (1-t_K)*np.log(1-a_K+0.00000001)
  cost = np.sum(cost)
  E.append(cost)

  theta_K += a_J.T.dot(delta_K)
  theta_J += a_I.T.dot(delta_J)
  
  counter += 0
  if(counter == 5):
    counter = 0
    plot_lines_for_theta_(theta_J, a_I, t_K)
  counter += 1


# plot_E(E)
plot_final_shape(theta_J, theta_K)
# plot_lines_for_theta_(theta_J, a_I, t_K)
# get_accuracy(a_I, theta_J, theta_K)
# plot_lines_for_theta_(theta_K, logis(a_I,theta_J),t_K)

