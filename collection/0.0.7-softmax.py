
def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # Get shapes
  num_classes = W.shape[0]
  num_train = X.shape[1]

  # Compute scores
  f = np.dot(W, X)
  f -= np.max(f,axis=0,keepdims=True)
  expf = np.exp(f)
  sum_expf = expf.sum(axis=0,keepdims=True)

  q = expf/sum_expf
  qyi = q[y, range(num_train)]
  # print qyi
  qyi+=0.00001
  loss = -np.mean(np.log(qyi))

  p = np.zeros(q.shape)
  # print 'p =================='
  # print p

  p[y, range(num_train)] = 1
  dW = np.dot((q-p), X.T)  #this is original
  dW /= num_train

  # Regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  return loss, dW


================ softmax begin ================

import random
import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10
from cs231n.data_utils import load_CIFAR10_mini




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

# from cs231n.classifiers.softmax import softmax_loss_vectorized


# Generate a random softmax weight matrix and use it to compute the loss.
W = np.random.randn(10, 3073) * 0.0001

# loss, grad = softmax_loss_vectorized(W, X_train, y_train, 0.01)

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

'''plot error function for a single set of parameter'''
# lr = 1e-2
# rs = 1e-2
# iters = 10

# from cs231n.classifiers.linear_classifier import Softmax
# softmax = Softmax(W)
# loss_history = softmax.train(X_train, y_train, learning_rate=lr, reg=rs, num_iters=iters)
# # print loss_history
# # y_val_pred = softmax.predict(X_test)
# # acc_val = np.mean(y_test == y_val_pred)
# # print acc_val

# import matplotlib.pyplot as plt
# plt.plot(loss_history,'o')
# plt.show()






'''cross validation'''
# # Use the validation set to tune hyperparameters (regularization strength and
# # learning rate). You should experiment with different ranges for the learning
# # rates and regularization strengths; if you are careful you should be able to
# # get a classification accuracy of over 0.35 on the validation set.
# from cs231n.classifiers.linear_classifier import Softmax
# results = {}
# best_val = -1
# best_softmax = None
# learning_rates = np.logspace(-10, 10, 10) # np.logspace(-10, 10, 8) #-10, -9, -8, -7, -6, -5, -4
# regularization_strengths = np.logspace(-3, 6, 10) # causes numeric issues: np.logspace(-5, 5, 8) #[-4, -3, -2, -1, 1, 2, 3, 4, 5, 6]

# ################################################################################
# # Use the validation set to set the learning rate and regularization strength. #
# # This should be identical to the validation that you did for the SVM; save    #
# # the best trained softmax classifer in best_softmax.                          #
# ################################################################################
# iters = 2000 #100
# for lr in learning_rates:
#     for rs in regularization_strengths:
#         softmax = Softmax(W)
#         softmax.train(X_train, y_train, learning_rate=lr, reg=rs, num_iters=iters)
        
#         y_train_pred = softmax.predict(X_train)
#         acc_train = np.mean(y_train == y_train_pred)
#         y_val_pred = softmax.predict(X_val)
#         acc_val = np.mean(y_val == y_val_pred)
        
#         results[(lr, rs)] = (acc_train, acc_val)
        
#         if best_val < acc_val:
#             best_val = acc_val
#             best_softmax = softmax
    
# # Print out results.
# for lr, reg in sorted(results):
#     train_accuracy, val_accuracy = results[(lr, reg)]
#     print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
#                 lr, reg, train_accuracy, val_accuracy)
    
# print 'best validation accuracy achieved during cross-validation: %f' % best_val



================ softmax end ================
================ updating begin ================

import numpy as np
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *

class LinearClassifier:

  def __init__(self,W):
    # self.W = None
    self.W = W

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=True):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: D x N array of training data. Each training point is a D-dimensional
         column.
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    dim, num_train = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
      # lazily initialize W
      self.W = np.random.randn(num_classes, dim) * 0.001

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
      idx = np.random.choice(num_train, batch_size, replace=True)
      X_batch = X[:,idx]
      y_batch = y[idx]

      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)

      # perform parameter update
      #########################################################################
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################
      self.W -= learning_rate * grad

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

    return loss_history

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: D x N array of training data. Each column is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[1])
    ###########################################################################
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    y_pred = np.argmax(np.dot(self.W,X), axis=0)
    return y_pred
  
  def loss(self, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - X_batch: D x N array of data; each column is a data point.
    - y_batch: 1-dimensional array of length N with labels 0...K-1, for K classes.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
    """
    pass


class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, X_batch, y_batch, reg):
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

