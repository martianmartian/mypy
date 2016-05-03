

================ softmax begin ================


import random
import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10
from cs231n.data_utils import load_CIFAR10_mini
import time


# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'



def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
  """
  Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
  it for the linear classifier. These are the same steps as we used for the
  SVM, but condensed to a single function.  
  """
  # Load the raw CIFAR-10 data
  cifar10_dir = 'data_image/cifar-10-batches-py'
  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
  
  # subsample the data
  mask = range(num_training, num_training + num_validation)
  X_val = X_train[mask]
  y_val = y_train[mask]
  mask = range(num_training)
  X_train = X_train[mask]
  y_train = y_train[mask]
  mask = range(num_test)
  X_test = X_test[mask]
  y_test = y_test[mask]
  
  # Preprocessing: reshape the image data into rows
  X_train = np.reshape(X_train, (X_train.shape[0], -1))
  X_val = np.reshape(X_val, (X_val.shape[0], -1))
  X_test = np.reshape(X_test, (X_test.shape[0], -1))
  
  # Normalize the data: subtract the mean image
  mean_image = np.mean(X_train, axis = 0)
  X_train -= mean_image
  X_val -= mean_image
  X_test -= mean_image
  
  # add bias dimension and transform into columns
  X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T
  X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))]).T
  X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]).T
  
  return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
# print 'Train data shape: ', X_train.shape
# print 'Train labels shape: ', y_train.shape
# print 'Validation data shape: ', X_val.shape
# print 'Validation labels shape: ', y_val.shape
# print 'Test data shape: ', X_test.shape
# print 'Test labels shape: ', y_test.shape


# First implement the naive softmax loss function with nested loops.
# Open the file cs231n/classifiers/softmax.py and implement the
# softmax_loss_vectorized function.

from cs231n.classifiers.softmax import softmax_loss_vectorized
import time

# Generate a random softmax weight matrix and use it to compute the loss.
W = np.random.randn(10, 3073) * 0.0001

loss, grad = softmax_loss_vectorized(W, X_train, y_train, 0.01)

'''# As a rough sanity check, our loss should be something close to -log(0.1).'''
# print 'loss:'
# print loss
# print 'sanity check: %f' % (-np.log(0.1))


'''gradient check'''
# # Complete the implementation of softmax_loss_vectorized and implement a (naive)
# # version of the gradient that uses nested loops.
# loss, grad = softmax_loss_vectorized(W, X_train, y_train, 0.0)

# # As we did for the SVM, use numeric gradient checking as a debugging tool.
# # The numeric gradient should be close to the analytic gradient.
# from cs231n.gradient_check import grad_check_sparse
# f = lambda w: softmax_loss_vectorized(w, X_train, y_train, 0.0)[0]
# grad_numerical = grad_check_sparse(f, W, grad, 10)

