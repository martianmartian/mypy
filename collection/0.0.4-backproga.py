=================== Classic 11-lines  ===========================

# http://iamtrask.github.io/2015/07/12/basic-python-network/
'''how many neurons are there in each layers??'''
'''batch learning'''
'Each column corresponds to one of our input nodes. '
'Thus, we have 3 input nodes to the network and 4 training examples.'

import numpy as np

E = []

X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
syn0 = 2*np.random.random((3,4)) - 1 
syn1 = 2*np.random.random((4,1)) - 1
for j in xrange(4000):
    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    E.append(np.sum((y - l2)**2)/2)
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)


# test = np.array([[1,1,1]])
# l1 = 1/(1+np.exp(-(np.dot(test,syn0))))
# l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
# print l2

import matplotlib.pyplot as plt
plt.plot(E,'o')
plt.show()

=================== begin here  ===========================


          '''How should one tweak the input slightly to increase the output?'''
              

                  '''Strategy #1: Random Local Search'''

import numpy as np
def forwardMultiplyGate(x,y):
  return x*y
x = -2
y = 3

tweak_amount = 0.01
best_out = -100
best_x = x
best_y = y
for k in range(0,100):
  print k
  x_try = x + tweak_amount * (np.random.random() * 2 - 1)
  y_try = y + tweak_amount * (np.random.random() * 2 - 1)
  out = forwardMultiplyGate(x_try,y_try)
  if(out > best_out):
    best_out = out
    best_x = x_try
    best_y = y_try
print best_out,best_x,best_y




======================= Online learning; layers of theta defined by case.  ===============================
                            '''forever: Ax = y, use vertical 1-D [[ARRAY]] as standard'''
                            '''a = activation'''
                            '''I = input'''
                            '''J,K are random layer labels'''
'''J_width = 10   # there's an optimal range for J numbers ...'''
''' essentialy there is not bias unit, but changing the J_width will help to converge... '''

import numpy as np
import random
def logistic(x):
  return 1/(1+np.exp(-x))
def logisticDe(x):
  return x*(1-x)

training_data = [ 
  (np.array([[0,0]]), np.array([[0]])), 
  (np.array([[0,1]]), np.array([[1]])), 
  (np.array([[1,0]]), np.array([[1]])), 
  (np.array([[1,1]]), np.array([[0]]))
]
E = []
alpha = 0.1
a_I,t_K = random.choice(training_data) # print a_I,t_K

I_width = a_I.shape[1]  # print 'I_width',I_width
J_width = 10   # there's an optimal range...
K_width = t_K.shape[0]  # print K_width

theta_J = np.random.random((J_width,I_width))
theta_K = np.random.random((K_width,J_width))

for i in range(30000):
  a_I,t_K = random.choice(training_data)

  a_J = logistic(np.dot(theta_J,a_I.T))
  a_K = logistic(np.dot(theta_K,a_J))

  diff = t_K - a_K
  E.append(np.sum(diff**2)/J_width)
  dumpy_K = diff*logisticDe(a_K)
  gradient_K = np.dot(dumpy_K,a_J.T)
  dumpy_J = np.dot(dumpy_K,theta_K)*logisticDe(a_J).T
  gradient_J = np.dot(dumpy_J.T,a_I)
  theta_J = theta_J + gradient_J*alpha
  theta_K = theta_K + gradient_K*alpha

import matplotlib.pyplot as plt
plt.plot(E,'o')
plt.show()




================== XOR and 'bias?..'... not sure about the bias  ==================
wrong 'bias unit' here

import numpy as np
import random
def logistic(x):
  return 1/(1+np.exp(-x))
def logisticDe(x):
  return x*(1-x)

training_data = [ 
  (np.array([[0,0,1]]), np.array([[0]])), 
  (np.array([[0,1,1]]), np.array([[1]])), 
  (np.array([[1,0,1]]), np.array([[1]])), 
  (np.array([[1,1,1]]), np.array([[0]]))
]
E = []
alpha = 0.1

a_I,t_K = random.choice(training_data)

I_width = a_I.shape[1]  # print 'I_width',I_width
J_width = 2+1
K_width = t_K.shape[0]  # print K_width

theta_J = np.random.random((J_width-1,I_width))*2 - 1
ones = np.ones((1,I_width))
theta_J = np.append(theta_J,ones,axis = 0)
theta_K = np.random.random((K_width,J_width))*2 - 1

for i in range(30000):
  a_I,t_K = random.choice(training_data)
  a_J = logistic(np.dot(theta_J,a_I.T))
  a_K = logistic(np.dot(theta_K,a_J))
  diff = t_K - a_K
  E.append(np.sum(diff**2)/J_width)
  dumpy_K = diff*logisticDe(a_K)
  gradient_K = np.dot(dumpy_K,a_J.T)
  dumpy_J = np.dot(dumpy_K,theta_K)*logisticDe(a_J).T
  gradient_J = np.dot(dumpy_J.T,a_I)
  print theta_J
  print gradient_J
  theta_J = theta_J + gradient_J*alpha
  theta_K = theta_K + gradient_K*alpha

import matplotlib.pyplot as plt
plt.plot(E,'o')
plt.show()

========================== XNOR||XOR gate ==========================
'''working version for XNOR/XOR'''

import numpy as np
import random
def logistic(x):
  return 1/(1+np.exp(-x))
def logisticDe(x):
  return x*(1-x)

training_data = [ 
  (np.array([[0,0,1]]), np.array([[1]])), 
  (np.array([[0,1,1]]), np.array([[0]])), 
  (np.array([[1,0,1]]), np.array([[0]])), 
  (np.array([[1,1,1]]), np.array([[1]]))
]
E = []
alpha = 0.1
a_I,t_K = random.choice(training_data) # print a_I,t_K

I_width = a_I.shape[1]  # print 'I_width',I_width
J_width = 3   # there's an optimal range... it's also called, 'fake bias'
K_width = t_K.shape[0]  # print K_width

theta_J = np.random.random((J_width,I_width))
theta_K = np.random.random((K_width,J_width))

for i in range(30000):
  a_I,t_K = random.choice(training_data)
  a_J = logistic(np.dot(theta_J,a_I.T))
  a_K = logistic(np.dot(theta_K,a_J))
  diff = t_K - a_K
  E.append(np.sum(diff**2)/J_width)
  dumpy_K = diff*logisticDe(a_K)
  gradient_K = np.dot(dumpy_K,a_J.T)
  dumpy_J = np.dot(dumpy_K,theta_K)*logisticDe(a_J).T
  gradient_J = np.dot(dumpy_J.T,a_I)
  theta_J = theta_J + gradient_J*alpha
  theta_K = theta_K + gradient_K*alpha

import matplotlib.pyplot as plt
plt.plot(E,'o')
plt.show()

==========================  plotting decision scatter... ==========================

''' plotting decision scatter... '''
import matplotlib.pyplot as plt
from pylab import scatter, show, legend, xlabel, ylabel

a = np.array([[0,0]])
b = np.array([[0]])

for i in range(10000):
  a_I,t_K = random.choice(training_data)
  a_J = logistic(np.dot(theta_J,a_I.T))
  a_K = logistic(np.dot(theta_K,a_J))
  a_I = np.delete(a_I,2,axis = 1)
  a = np.append(a, a_I, axis = 0)
  b = np.append(b,a_K, axis = 1)

# print a
# print b

pos = np.where(b > 0.5)
neg = np.where(b < 0.5)
plt.scatter(a[pos, 0], a[pos, 1], marker='o', c='b')
plt.scatter(a[neg, 0], a[neg, 1], marker='x', c='r')
plt.show()


================ for digit recognition ================
''' discovery: the above approach couldnt converge for the image related'''
'''question. in brief words, it is because derivative of logistic function at far '''
'''from origin is 0, giving the term a_K*(1-a_K)=> 0, invalidates the  "sum-squared error" approach'''
'''therefore a different cost function should be used... '''
'''trick is about finding the right theta_J,theta_K to start with'''
import sys
import PIL.Image
import scipy.misc, scipy.optimize, scipy.io, scipy.special
# from numpy import *
import numpy as np
import random

import pylab
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab

import timeit
import cProfile, pstats, StringIO


def displayData( X, theta1 = None, theta2 = None ):
  m, n = np.shape( X )
  width = np.sqrt( n )
  rows, cols = 5, 5

  out = np.zeros(( width * rows, width*cols ))

  rand_indices = np.random.permutation( m )[0:rows * cols]

  counter = 0
  for y in range(0, rows):
    for x in range(0, cols):
      start_x = x * width
      start_y = y * width
      out[start_x:start_x+width, start_y:start_y+width] = X[rand_indices[counter]].reshape(width, width).T
      counter += 1

  img   = scipy.misc.toimage( out )
  figure  = pyplot.figure()
  axes    = figure.add_subplot(111)
  axes.imshow( img )


  if theta1 is not None and theta2 is not None:
    result_matrix   = []
    
    for idx in rand_indices:
      result = predict( X[idx], theta1, theta2 )
      result_matrix.append( result )

    result_matrix = array( result_matrix ).reshape( rows, cols ).transpose()
    print result_matrix

  pyplot.show( )

def recodeLabel( y, k ):
  m = np.shape(y)[0]
  out = np.zeros( ( k, m ) )
  
  for i in range(0, m):
    out[y[i]-1, i] = 1

  return out

def logistic(x):
  return 1/(1+np.exp(-x))
def logisticDe(x):
  return x*(1-x)

'''body is here'''
E = []
alpha = 0.1
mat = scipy.io.loadmat( "data/ex3data1.mat" )
All_a_I, All_t_K = mat['X'],mat['y'] #(5000, 400) (5000, 1)
m = 50
index = np.arange(m)
indexD = np.r_[index,index+500,index+1000,index+1500,index+2000]
All_a_I = np.take(All_a_I,indexD,axis=0)
# displayData(All_a_I)

rand_index = np.random.randint(m, size=1)
a_I = np.c_[All_a_I[rand_index],1] #(1, 401) 
t_K = recodeLabel(All_t_K[rand_index],10) #(10, 1)

I_width = a_I.shape[1]  # 400+1
J_width = 25+1          # 25+1
K_width = t_K.shape[0]  # 10

theta_J = np.random.random((J_width,I_width)) #(26, 401)
theta_K = np.random.random((K_width,J_width)) #(10, 26)

for i in range(100):
  rand_index = np.random.randint(m, size=1)
  a_I = np.c_[All_a_I[rand_index],1] #(1, 401) 
  # print a_I
  # print np.dot(theta_J,a_I.T)  all around 25....
  t_K = recodeLabel(All_t_K[rand_index],10) #(10, 1)
  a_J = logistic(np.dot(theta_J,a_I.T)) #(25+1,1)
  # print a_J  # -> 1
  a_K = logistic(np.dot(theta_K,a_J))  #(10,1)
  diff = t_K - a_K
  fixed = diff*a_K*(1-a_K)
  E.append(np.sum(diff**2)/J_width)
  # gradient_K = fixed*a_J.T
  gradient_K = np.dot(fixed,a_J.T)
  # print gradient_K  super small
  # print fixed.shape,fixed #(10,1)
  # print a_J.shape,a_J #(26,1)
  # print gradient_K.shape
  gradient_J = np.transpose(fixed).dot(theta_K).T*a_J*(1-a_J)*a_I
  theta_J = theta_J - gradient_J*alpha #(26, 401)
  theta_K = theta_K - gradient_K*alpha

# print E
# import matplotlib.pyplot as plt
# plt.plot(E,'o')
# plt.show()

================ normalize/scale doesnt help ================
'''new problem: the results are treated as the same since they'''
'''all have 1 only'''
'''there must be an erro in my code that eliminates the specificity'''
'''*2 - 1 should be given to all the weights '''
'''solved: the problem was normalizing it'''
import sys
import PIL.Image
import scipy.misc, scipy.optimize, scipy.io, scipy.special
# from numpy import *
import numpy as np
import random

import pylab
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab

import timeit
import cProfile, pstats, StringIO


def displayData( X, theta1 = None, theta2 = None ):
  m, n = np.shape( X )
  width = np.sqrt( n )
  rows, cols = 5, 5

  out = np.zeros(( width * rows, width*cols ))

  rand_indices = np.random.permutation( m )[0:rows * cols]

  counter = 0
  for y in range(0, rows):
    for x in range(0, cols):
      start_x = x * width
      start_y = y * width
      out[start_x:start_x+width, start_y:start_y+width] = X[rand_indices[counter]].reshape(width, width).T
      counter += 1

  img   = scipy.misc.toimage( out )
  figure  = pyplot.figure()
  axes    = figure.add_subplot(111)
  axes.imshow( img )


  if theta1 is not None and theta2 is not None:
    result_matrix   = []
    
    for idx in rand_indices:
      result = predict( X[idx], theta1, theta2 )
      result_matrix.append( result )

    result_matrix = array( result_matrix ).reshape( rows, cols ).transpose()
    print result_matrix

  pyplot.show( )

def recodeLabel( y, k ):
  m = np.shape(y)[0]
  out = np.zeros( ( k, m ) )
  
  for i in range(0, m):
    out[y[i]-1, i] = 1

  return out

def logistic(x):
  return 1/(1+np.exp(-x+0.000001))
def logisticDe(x):
  return x*(1-x)

def normalize(x):
  return (x - x.mean())/(x.std()+0.0001)
'''body is here'''
E = []
alpha = 0.1
mat = scipy.io.loadmat( "data/ex3data1.mat" )
All_a_I, All_t_K = mat['X'],mat['y'] #(5000, 400) (5000, 1)
m = 50
index = np.arange(m)
# indexD = np.r_[index]
All_a_I = np.take(All_a_I,index,axis=0)
# displayData(All_a_I)

rand_index = np.random.randint(m, size=1)
a_I = np.c_[All_a_I[rand_index],1] #(1, 401) 
t_K = recodeLabel(All_t_K[rand_index],10) #(10, 1)

I_width = a_I.shape[1]  # 400+1
J_width = 25+1          # 25+1
K_width = t_K.shape[0]  # 10

theta_J = np.random.random((J_width,I_width)) #(26, 401)
theta_K = np.random.random((K_width,J_width)) #(10, 26)

for i in range(10):
  rand_index = np.random.randint(m, size=1)
  a_I = np.c_[All_a_I[rand_index],1] #(1, 401) 
  a_I = normalize(a_I)
  # print np.dot(theta_J,a_I.T)  all around 25....
  t_K = recodeLabel(All_t_K[rand_index],10) #(10, 1)
  a_J = logistic(np.dot(theta_J,a_I.T)) #(25+1,1)
  a_J = normalize(a_J)
  # print a_J  # -> 1
  a_K = logistic(np.dot(theta_K,a_J))  #(10,1)
  # a_K = normalize(a_K)
  diff = t_K - a_K
  fixed = diff*a_K*(1-a_K)
  E.append(np.sum(diff**2)/J_width)
  # gradient_K = fixed*a_J.T
  gradient_K = np.dot(fixed,a_J.T)
  # print gradient_K  super small
  # print fixed.shape,fixed #(10,1)
  # print a_J.shape,a_J #(26,1)
  # print gradient_K.shape
  gradient_J = np.transpose(fixed).dot(theta_K).T*a_J*(1-a_J)*a_I
  theta_J = theta_J - gradient_J*alpha #(26, 401)
  theta_K = theta_K - gradient_K*alpha


# # print E
# import matplotlib.pyplot as plt
# plt.plot(E,'o')
# plt.show()


'''new body is here'''
E = []
alpha = 0.1
mat = scipy.io.loadmat( "data/ex3data1.mat" )
All_a_I, All_t_K = mat['X'],mat['y'] #(5000, 400) (5000, 1)
m,n = np.shape(All_a_I)

# rand_index = np.random.randint(indexD.shape[0], size=1)
rand_index = np.random.randint(m, size=1)
a_I = np.c_[All_a_I[rand_index],1] #(1, 401) 
t_K = recodeLabel(All_t_K[rand_index],10) #(10, 1)

I_width = a_I.shape[1]  # 400+1
J_width = 30+1          # 25+1
K_width = t_K.shape[0]  # 10

'''make the weights between -1 and 1 to prevent 0 derivative!!'''
theta_J = np.random.random((J_width,I_width))*2 - 1 #(26, 401)
theta_K = np.random.random((K_width,J_width))*2 - 1 #(10, 26)

for i in range(10000):
  rand_index = np.random.randint(m, size=1)
  a_I = np.c_[All_a_I[rand_index],1] #(1, 401) 
  t_K = recodeLabel(All_t_K[rand_index],10) #(10, 1)
  a_J = logistic(np.dot(theta_J,a_I.T)) #(25+1,1)
  a_K = logistic(np.dot(theta_K,a_J))  #(10,1)
  diff = t_K - a_K
  fixed = diff*a_K*(1-a_K)
  E.append(np.sum(diff**2)/J_width)
  gradient_K = np.dot(fixed,a_J.T)
  gradient_J = np.transpose(fixed).dot(theta_K).T*a_J*(1-a_J)*a_I
  theta_J = theta_J + gradient_J*alpha #(26, 401)
  theta_K = theta_K + gradient_K*alpha


# rand_index = np.random.randint(m, size=1)
# print All_t_K[rand_index]
# a_I = np.c_[All_a_I[rand_index],1] #(1, 401) 
# t_K = recodeLabel(All_t_K[rand_index],10) #(10, 1)
# a_J = logistic(np.dot(theta_J,a_I.T)) #(25+1,1)
# a_K = logistic(np.dot(theta_K,a_J))  #(10,1)
# print a_K


# print E
import matplotlib.pyplot as plt
plt.plot(E,'o')
plt.show()

========================== XNOR||XOR gate ==========================
'''working version for XNOR/XOR'''
''' batch learning is pretty cooooll'''
'''this is the most recent version of backpropagation algo'''
'''using batch learning'''
''' no alpha is better, idk why but it is...'''

import numpy as np
import random
def logistic(x):
  return 1/(1+np.exp(-x))
def logisticDe(x):
  return x*(1-x)

a_I = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
# t_K = np.array([[0,1,1,0]]).T
t_K = np.array([[0,0],[1,0],[1,0],[0,0]])
m = a_I.shape[0]

E = []
# alpha = 0.1

I_width = a_I.shape[1]  #3
J_width = 3   # there's an optimal range... it's also called, 'fake bias'
K_width = t_K.shape[1]  # print K_width

theta_J = np.random.random((I_width,J_width))*2 - 1
theta_K = np.random.random((J_width,K_width))*2 - 1

'''bias should be added for all?'''
for i in range(1000):
  a_J = logistic(np.dot(a_I,theta_J))
  a_K = logistic(np.dot(a_J,theta_K))
  diff = t_K - a_K
  E.append(np.sum(diff**2)/m)
  fixed_K = diff*(a_K*(1-a_K))
  fixed_J = fixed_K.dot(theta_K.T)*(a_J*(1-a_J))
  gradient_K = a_J.T.dot(fixed_K)
  gradient_J = a_I.T.dot(fixed_J)
  theta_K += gradient_K
  theta_J += gradient_J


import matplotlib.pyplot as plt
plt.plot(E,'o')
plt.show()
========================== digit recog with 11-lines ==========================
'''failed'''


import sys
import PIL.Image
import scipy.misc, scipy.optimize, scipy.io, scipy.special
# from numpy import *
import numpy as np
import random

import pylab
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab

import timeit
import cProfile, pstats, StringIO


def displayData( X, theta1 = None, theta2 = None ):
  m, n = np.shape( X )
  width = np.sqrt( n )
  rows, cols = 5, 5

  out = np.zeros(( width * rows, width*cols ))

  rand_indices = np.random.permutation( m )[0:rows * cols]

  counter = 0
  for y in range(0, rows):
    for x in range(0, cols):
      start_x = x * width
      start_y = y * width
      out[start_x:start_x+width, start_y:start_y+width] = X[rand_indices[counter]].reshape(width, width).T
      counter += 1

  img   = scipy.misc.toimage( out )
  figure  = pyplot.figure()
  axes    = figure.add_subplot(111)
  axes.imshow( img )


  if theta1 is not None and theta2 is not None:
    result_matrix   = []
    
    for idx in rand_indices:
      result = predict( X[idx], theta1, theta2 )
      result_matrix.append( result )

    result_matrix = array( result_matrix ).reshape( rows, cols ).transpose()
    print result_matrix

  pyplot.show( )

'''this can totally be done in a remove API'''
'''can this be vectorized?'''
def recodeLabel( y, k ):
  m = np.shape(y)[0]
  out = np.zeros( ( m, k ) )
  for i in range(0, m):
    out[i, y[i]-1] = 1
  return out
  # m = np.shape(y)[0]
  # out = np.zeros( ( k, m ) )
  # for i in range(0, m):
  #   out[y[i]-1, i] = 1
  # return out

def logistic(x):
  return scipy.special.expit(x)
def logisticDe(x):
  return x*(1-x)

'''body is here'''
E = []
alpha = 0.005
mat = scipy.io.loadmat( "data/ex3data1.mat" )
All_a_I, All_t_K = mat['X'],mat['y'] #(5000, 400) (5000, 1)
All_t_K = recodeLabel(All_t_K,10)
m,n = np.shape(All_a_I)

I_width = All_a_I.shape[1] 
J_width = 26 
K_width = All_t_K.shape[1]  # print K_width

theta_J = np.random.random((I_width,J_width))*2 - 1 #(400, 26)
theta_K = np.random.random((J_width,K_width))*2 - 1 #(26, 10)

for i in range(100):
  # print "====="
  # print theta_J[300:305,:]
  All_a_J = logistic(np.dot(All_a_I,theta_J)) #(5000,26)
  # print All_a_J[0:5,:] =>1
  All_a_K = logistic(np.dot(All_a_J,theta_K)) #(5000,10)
  # print All_a_K[0:5,:]
  # print theta_K[0:5,:] =>-10
  diff = All_t_K - All_a_K #(5000,10)
  # print diff[0:5,0:5]
  E.append(np.sum(diff**2)/m)
  fixed_K = diff*logisticDe(All_a_K) #(5000,10)
  # print fixed_K[0:5,0:5] => -0.125
  fixed_J = fixed_K.dot(theta_K.T)*logisticDe(All_a_J)
  # print fixed_J[0:5,0:5]
  gradient_K = All_a_J.T.dot(fixed_K)
  # print gradient_K.shape
  # print gradient_K[0:5,0:5]
  gradient_J = All_a_I.T.dot(fixed_J)
  # print gradient_J[0:5,0:5]
  theta_K += gradient_K
  theta_J += gradient_J


# print E
import matplotlib.pyplot as plt
plt.plot(E,'o')
plt.show()


========================== Cross-entropy ======================
'''output>1
cross-entropy
11-line
XOR||XNOR
batch'''

import sys
import PIL.Image
import scipy.misc, scipy.io, scipy.optimize, scipy.special
from numpy import *
import numpy as np

import pylab
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlaba

# from util import Util
import timeit
import cProfile, pstats, StringIO


def logis(x,theta):
  syn = np.dot(x,theta)
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
  ones = np.ones((m, 30*30-n))*0.001
  source = np.c_[source,ones]
  return source


a_I = np.array([[0,0],[0,1],[1,0],[1,1]])
a_I = addBias(a_I)
# a_I = projectToStandardField(a_I)
m = a_I.shape[0]
t_K = np.array([[0,0],[1,0],[1,0],[0,0]])
alpha = 0.1
E = []

I_width = a_I.shape[1]
J_width = 4
K_width = t_K.shape[1]

theta_J = getWeights(I_width, J_width)
theta_K = getWeights(J_width,K_width)

for i in range(1000):
  a_J = logis(a_I,theta_J)
  a_K = logis(a_J,theta_K)

  delta_K = t_K - a_K
  delta_J = delta_K.dot(theta_K.T)*a_J*(1-a_J)

  cost = -t_K * np.log(a_K) - (1-t_K)*np.log(1-a_K)
  cost = np.sum(cost)
  E.append(cost)

  theta_K += a_J.T.dot(delta_K)
  theta_J += a_I.T.dot(delta_J)

# print E
import matplotlib.pyplot as plt
E = np.array(E)/m
plt.plot(E,'o')
plt.show()

'''online version'''
# def logis(x,theta):
#   syn = np.dot(x,theta)
#   return scipy.special.expit(syn)
# def logisDeri(x):
#   return x*(1-x)
# def addBias(source):
#   ones = np.ones((source.shape[0],3))
#   return np.c_[source,ones]
# def getWeights(width1,width2):
#   return np.random.random((width1,width2))*2 - 1

# training_data = [ 
#   (np.array([[0,0]]), np.array([[0]])), 
#   (np.array([[0,1]]), np.array([[1]])), 
#   (np.array([[1,0]]), np.array([[1]])), 
#   (np.array([[1,1]]), np.array([[0]]))
# ]

# a_I,t_K = random.choice(training_data)
# a_I = addBias(a_I)
# m = a_I.shape[0]
# alpha = 0.1
# E = []

# I_width = a_I.shape[1]
# J_width = 8
# K_width = t_K.shape[1]

# theta_J = getWeights(I_width, J_width)
# theta_K = getWeights(J_width,K_width)

# for i in range(5000):
#   a_I,t_K = random.choice(training_data)
#   a_I = addBias(a_I)

#   a_J = logis(a_I,theta_J)
#   a_K = logis(a_J,theta_K)

#   delta_K = t_K - a_K
#   delta_J = delta_K.dot(theta_K.T)*a_J*(1-a_J)

#   cost = -t_K * np.log(a_K) - (1-t_K)*np.log(1-a_K)
#   cost = np.sum(cost)/m
#   E.append(cost)

#   theta_K += a_J.T.dot(delta_K)
#   theta_J += a_I.T.dot(delta_J)

# # print E
# import matplotlib.pyplot as plt
# plt.plot(E,'o')
# plt.show()

=================== Cross-entropy for digit recog ==========
'''ones>1 columns, no bias added. extra neuron added
11-line
digit recog
average cost by m*n 
alpha'''

import sys
import PIL.Image
import scipy.misc, scipy.io, scipy.optimize, scipy.special
from numpy import *
import numpy as np

import pylab
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlaba

import timeit
import cProfile, pstats, StringIO


def logis(x,theta):
  syn = np.dot(x,theta)
  return scipy.special.expit(syn)
def logisDeri(x):
  return x*(1-x)
def addBias(source):
  ones = np.ones((source.shape[0],3))
  return np.c_[source,ones]
'''be careful the bias maybe not just one column of ones'''
def getWeights(width1,width2):
  return np.random.random((width1,width2))*2 - 1
def recodeLabel( y, k ):
  m = np.shape(y)[0]
  out = np.zeros( ( m, k ) )
  for i in range(0, m):
    out[i, y[i]-1] = 1
  return out
def projectToStandardField(source):
  '''project to standard field '''
  m,n = source.shape
  ones = np.ones((m, 25*25-n))*0.001
  source = np.c_[source,ones]
  return source

mat = scipy.io.loadmat( "data/ex3data1.mat" )
a_I, t_K = mat['X'],mat['y']
# displayData(a_I)
# a_I = addBias(a_I)
a_I = projectToStandardField(a_I)
t_K = recodeLabel(t_K,10)
m,n = a_I.shape
E = []

I_width = a_I.shape[1]
J_width = 30
K_width = t_K.shape[1]

theta_J = getWeights(I_width, J_width)
theta_K = getWeights(J_width,K_width)

for i in range(200):
  a_J = logis(a_I,theta_J)
  a_K = logis(a_J,theta_K)

  delta_K = t_K - a_K
  delta_J = delta_K.dot(theta_K.T)*a_J*(1-a_J)

  cost = -t_K * np.log(a_K+0.00000001) - (1-t_K)*np.log(1-a_K+0.00000001)
  cost = np.sum(cost)
  E.append(cost)

  theta_K += a_J.T.dot(delta_K)*0.001
  theta_J += a_I.T.dot(delta_J)*0.001

# print E
E = np.array(E)/m
import matplotlib.pyplot as plt
plt.plot(E,'o')
plt.show()

'''online version'''
'''beautiful 5 lines'''

# def logis(x,theta):
#   syn = np.dot(x,theta)
#   return scipy.special.expit(syn)
# def logisDeri(x):
#   return x*(1-x)
# def addBias(source):
#   ones = np.ones((source.shape[0],3))
#   return np.c_[source,ones]
# '''be careful the bias maybe not just one column of ones'''
# def getWeights(width1,width2):
#   return np.random.random((width1,width2))*2 - 1
# def recodeLabel( y, k ):
#   m = np.shape(y)[0]
#   out = np.zeros( ( m, k ) )
#   for i in range(0, m):
#     out[i, y[i]-1] = 1
#   return out

# mat = scipy.io.loadmat( "data/ex3data1.mat" )
# All_a_I, All_t_K = mat['X'],mat['y']
# # displayData(a_I)
# All_a_I = addBias(All_a_I)
# All_t_K = recodeLabel(All_t_K,10)
# # m,n = All_a_I.shape
# m = 5
# E = []

# rand_index = np.random.randint(m, size=1)
# a_I = All_a_I[rand_index]
# t_K = All_t_K[rand_index]

# I_width = a_I.shape[1]
# J_width = 26
# K_width = t_K.shape[1]

# theta_J = getWeights(I_width, J_width)
# theta_K = getWeights(J_width,K_width)

# for i in range(1000):
#   rand_index = np.random.randint(m, size=1)
#   a_I = All_a_I[rand_index]
#   t_K = All_t_K[rand_index]

#   a_J = logis(a_I,theta_J)
#   a_K = logis(a_J,theta_K)

#   delta_K = t_K - a_K
#   delta_J = delta_K.dot(theta_K.T)*a_J*(1-a_J)

#   cost = -t_K * np.log(a_K+0.00001) - (1-t_K)*np.log(1-a_K+0.00001)
#   cost = np.sum(cost)
#   E.append(cost)

#   theta_K += a_J.T.dot(delta_K)*0.001
#   theta_J += a_I.T.dot(delta_J)*0.001

# E = np.array(E)/m
# import matplotlib.pyplot as plt
# plt.plot(E,'o')
# plt.show()



=================== Backprop from cs231n ==========

