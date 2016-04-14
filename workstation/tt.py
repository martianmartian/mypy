

#    ) )
#    ( (
#  ........
# [|      |
#  \      /
#   `----'


# SVM finished??... get it from Andrew NG..


import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

# from cs231nlib.classifier import NearestNeighbor;
# from cs231nlib.utils import load_CIFAR10;
# from cs231nlib.utils import visualize_CIFAR;

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


cifar10_dir = 'data_image/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)


# # As a sanity check, we print out the size of the training and test data.
# print 'Training data shape: ', X_train.shape
# print 'Training labels shape: ', y_train.shape
# print 'Test data shape: ', X_test.shape
# print 'Test labels shape: ', y_test.shape


# # Visualize some examples from the dataset.
# # We show a few examples of training images from each class.
# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# num_classes = len(classes)
# samples_per_class = 7
# for y, cls in enumerate(classes):
#     idxs = np.flatnonzero(y_train == y)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(X_train[idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls)
# plt.show()


# Subsample the data for more efficient code execution in this exercise
num_training = 500
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 5
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows

# CV: each row is a data vector of 3072 items.
# 5k for training and 500 for testing

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print X_train.shape, X_test.shape



from cs231n.classifiers import KNearestNeighbor

# CV: Just saves the data

# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)


# Open cs231n/classifiers/k_nearest_neighbor.py and implement
# compute_distances_no_loops.

# Test your implementation:
dists = classifier.compute_distances_no_loops(X_test)
print dists.shape
print dists

# We can visualize the distance matrix: each row is a single test example and
# its distances to training examples
plt.imshow(dists, interpolation='nearest')
plt.show()


# Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
print 'Training data shape: ', X_train.shape
print 'Training labels shape: ', y_train.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape
y_test_pred = classifier.predict_labels(dists, k=5)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)