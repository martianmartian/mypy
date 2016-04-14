


================ SVM begin ================

import numpy as np
from cs231nlib.classifier import NearestNeighbor;
from cs231nlib.utils import load_CIFAR10;
from cs231nlib.utils import visualize_CIFAR;


def L_i(x, y, W):
  """
  unvectorized version. Compute the multiclass svm loss for a single example (x,y)
  - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
    with an appended bias dimension in the 3073-rd position (i.e. bias trick)
  - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
  - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
  """
  delta = 1.0 # see notes about delta later in this section
  scores = W.dot(x) # scores becomes of size 10 x 1, the scores for each class
  correct_class_score = scores[y]
  D = W.shape[0] # number of classes, e.g. 10
  loss_i = 0.0
  for j in xrange(D): # iterate over all wrong classes
    if j == y:
      # skip for the true class to only loop over incorrect classes
      continue
    # accumulate loss for the i-th example
    loss_i += max(0, scores[j] - correct_class_score + delta)
  return loss_i


def L_i_vectorized(x, y, W):
  """
  A faster half-vectorized implementation. half-vectorized
  refers to the fact that for a single example the implementation contains
  no for loops, but there is still one loop over the examples (outside this function)
  """
  delta = 1.0
  scores = W.dot(x)
  # compute the margins for all classes in one vector operation
  margins = np.maximum(0, scores - scores[y] + delta)
  # on y-th position scores - scores[y] canceled and gave delta. We want
  # only consider margin on max wrong class
  # margins=[max(0,13-13+1)]
  margins[y] = 0
  loss_i = np.sum(margins)
  return loss_i


def L(X, y, W):
  """
  fully-vectorized implementation :
  - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
  - y is array of integers specifying correct class (e.g. 50,000-D array)
  - W are weights (e.g. 10 x 3073)
  """
  # evaluate loss over all examples in X without using any for loops
  # left as exercise to reader in the assignment
  delta = 1.0
  # W = np.zeros([category, X.shape[1]])

  scores = X.dot(W.T)    # (50000,10)
  # print 'scores',scores.shape,scores

  # print scores
  index = range(scores.shape[0])   # [0,1, ..., 50000-1] type: list, print all..
  # print 'index',index
  y_list = y.tolist()   # np.array (50000,) => list [6 9 9 ..., 9 1 1]
  # print 'y',y.shape,y   # (50000,) [6 9 9 ..., 9 1 1], numpy array
  # print 'y_list',y_list    #  type: list, print all..
  yi = scores[index, y_list]   # fancy index... fetching correct scores
  yi = yi[:, np.newaxis]
  margin = np.maximum(0, scores - yi + delta)
  loss_i = np.sum(margin, axis=1)
  total_loss = np.sum(loss_i)
  return total_loss



Xtr, Ytr, Xte, Yte=load_CIFAR10("data_image/cifar-10-batches-py");
# print Xtr.shape # (50000, 32, 32, 3)
# print Ytr.shape  (50000,)  # print Ytr  [6 9 9 ..., 9 1 1]
Xtr_rows=Xtr.reshape(Xtr.shape[0],Xtr.shape[1]*Xtr.shape[2]*Xtr.shape[3])
Xte_rows=Xte.reshape(Xte.shape[0],Xte.shape[1]*Xte.shape[2]*Xte.shape[3])

Xtr_means = np.mean(Xtr_rows, axis=0)
Xtr_rows -= Xtr_means
Xtr_rows /= 127.0
Xtr_totals = np.ones((Xtr_rows.shape[0], Xtr_rows.shape[1] + 1))
Xtr_totals[:, :Xtr_totals.shape[1]-1] = Xtr_rows # added bias
# print Xtr_totals.shape

category = 10
W = np.random.random([category, Xtr_rows.shape[1]+1])
def CIFAR10_loss_fun(W):
    return L(Xtr_totals, Ytr, W)
CIFAR10_loss_fun(W)