
import pylab
import scipy.misc, scipy.optimize, scipy.io, scipy.special
from numpy import *
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab

def displayData( X, theta = None ):
  width = 20
  rows, cols = 10, 10
  out = zeros(( width * rows, width*cols ))

  rand_indices = random.permutation( 5000 )[0:rows * cols]

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

  img   = scipy.misc.toimage( out )
  figure  = pyplot.figure()
  axes    = figure.add_subplot(111)
  axes.imshow( img )

  if theta is not None:
    result_matrix   = []
    X_biased    = c_[ ones( shape(X)[0] ), X ]
    
    for idx in rand_indices:
      result = (argmax( theta.T.dot(X_biased[idx]) ) + 1) % 10
      result_matrix.append( result )

    result_matrix = array( result_matrix ).reshape( rows, cols ).transpose()
    print result_matrix
  pyplot.show()



mat = scipy.io.loadmat( "./data/ex3data1.mat")
X, y      = mat['X'], mat['y']
displayData( X )


================ Nearest Neighbor ================

import numpy as np;
import matplotlib.pyplot as plt;

from cs231nlib.classifier import NearestNeighbor;
from cs231nlib.utils import load_CIFAR10;
from cs231nlib.utils import visualize_CIFAR;

## load dataset
Xtr, Ytr, Xte, Yte=load_CIFAR10("data_image/cifar-10-batches-py");
# print Xtr.shape # (50000, 32, 32, 3)
"""Converting Image data set to Raw Date Format"""
Xtr_rows=Xtr.reshape(Xtr.shape[0],Xtr.shape[1]*Xtr.shape[2]*Xtr.shape[3])
Xte_rows=Xte.reshape(Xte.shape[0],Xte.shape[1]*Xte.shape[2]*Xte.shape[3])


# ## plot configuration
# plt.rcParams['figure.figsize']=(10.0, 8.0);
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

# visualize_CIFAR(X_train=Xtr, y_train=Ytr, samples_per_class=10);

# # Testing for Nearest Neighbor Function
# nn=NearestNeighbor();
# nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
# Yte_predict = nn.predict(Xte_rows) 
# print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )


# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50,000 x 3072 matrix
Xval_rows = Xtr_rows[:1000, :] # take first 1000 for validation
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :] # keep last 49,000 for train
Ytr = Ytr[1000:]

# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:

  # use a particular value of k and evaluation on validation data
  nn = NearestNeighbor()
  nn.train(Xtr_rows, Ytr)
  # here we assume a modified NearestNeighbor class that can take a k as input
  Yval_predict = nn.predict(Xval_rows, k = k)
  acc = np.mean(Yval_predict == Yval)
  print 'accuracy: %f' % (acc,)

  # keep track of what works on the validation set
  validation_accuracies.append((k, acc))

