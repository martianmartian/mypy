


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
  syi = scores[index, y_list]   # fancy index... fetching correct scores
  syi = syi[:, np.newaxis]
  margin = np.maximum(0, scores - syi + delta)
  loss_i = np.sum(margin, axis=1)  #-1?
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


================ SVM full ================
import random
import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10
from cs231n.data_utils import load_CIFAR10_mini
import time


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


cifar10_dir = 'data_image/cifar-10-batches-py'
# X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
X_train, y_train, X_test, y_test = load_CIFAR10_mini(cifar10_dir)

# # As a sanity check, we print out the size of the training and test data.
# print 'Training data shape: ', X_train.shape
# print 'Training labels shape: ', y_train.shape
# print 'Test data shape: ', X_test.shape
# print 'Test labels shape: ', y_test.shape

# Subsample the data for more efficient code execution in this exercise.
num_training = 2000
num_validation = 10
num_test = 10

# Our validation set will be num_validation points from the original
# training set.
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# Our training set will be the first num_train points from the original
# training set.
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# We use the first num_test points of the original test set as our
# test set.
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# print 'Train data shape: ', X_train.shape # (49000, 3072)
# print 'Train labels shape: ', y_train.shape # (49000,)
# print 'Validation data shape: ', X_val.shape
# print 'Validation labels shape: ', y_val.shape
# print 'Test data shape: ', X_test.shape
# print 'Test labels shape: ', y_test.shape

# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# # As a sanity check, print out the shapes of the data
# print 'Training data shape: ', X_train.shape
# print 'Validation data shape: ', X_val.shape
# print 'Test data shape: ', X_test.shape

# Preprocessing: subtract the mean image
# first: compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0)
# print mean_image[:10] # print a few of the elements
# plt.figure(figsize=(4,4))
# plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # visualize the mean image
# plt.show()

# second: subtract the mean image from train and test data
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image

# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
# Also, lets transform both data matrices so that each image is a column.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))]).T
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]).T
print X_train.shape, X_val.shape, X_test.shape
# (3073, 49000) (3073, 1000) (3073, 1000)

# Evaluate the naive implementation of the loss we provided for you:
# from cs231n.classifiers.linear_svm import svm_loss_naive, svm_loss_vectorized
from cs231n.classifiers.linear_svm import svm_loss_vectorized

# generate a random SVM weight matrix of small numbers
W = np.random.randn(10, 3073) * 0.0001 
# loss, grad = svm_loss_naive(W, X_train, y_train, 0.00001)
loss, grad = svm_loss_vectorized(W, X_train, y_train, 0.00001)

# print 'loss: %f' % (loss, )



# Now implement SGD in LinearSVM.train() function and run it with the code below
from cs231n.classifiers import LinearSVM
svm = LinearSVM(W)
tic = time.time()
loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=5e4,
                      num_iters=1500, verbose=True)
toc = time.time()
print 'That took %fs' % (toc - tic)


# Visualize the learned weights for each class.
# Depending on your choice of learning rate and regularization strength, these may
# or may not be nice to look at.
w = W[:,:-1] # strip out the bias
w = w.reshape(10, 32, 32, 3)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in xrange(10):
  plt.subplot(2, 5, i + 1)
    
  # Rescale the weights to be between 0 and 255
  wimg = 255.0 * (w[i].squeeze() - w_min) / (w_max - w_min)
  plt.imshow(wimg.astype('uint8'))
  plt.axis('off')
  plt.title(classes[i])

plt.show()

