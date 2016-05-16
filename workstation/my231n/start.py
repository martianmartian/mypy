

#    ) )
#    ( (
#  ........
# [|      |
#  \      /
#   `----'

import numpy as np

import matplotlib.pyplot as plt
import library.pickledb as pickle

from nets231.neural_net import TwoLayerNet
from facilities.faci_start import *

from facilities.data_utils import get_CIFAR10_data_mini
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data_mini()

''' ========= special settings =========  '''
np.random.seed(0)
''' 
done: 
  no normalize=True

working on:
  X_train shape changed to : D x N
  p-q, not q-p
'''
''' ========= end =========  '''
# X_shape="DxN"


input_size = 32 * 32 * 3
hidden_size = 180
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

# print X_train.shape
# scores = net.loss(X_train)
# print scores.shape

# loss, _ = net.loss(X_train, y_train, reg=0.1)
# print loss


# Train the network
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=100, batch_size=400,
            learning_rate=1e-3, learning_rate_decay=0.98,
            reg=.001, verbose=True, dropout_fraction = .5)

# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
test_acc = (net.predict(X_test) == y_test).mean()


print 'Validation accuracy: ', val_acc
print 'Test accuracy: ', test_acc

#input_size = 32 * 32 * 3
#hidden_size = 170 #165
#num_classes = 10
#net = TwoLayerNet(input_size, hidden_size, num_classes)

# Train the network
#stats = net.train(X_train, y_train, X_val, y_val,
#            num_iters=1800, batch_size=2000,
#            learning_rate=1e-3, learning_rate_decay=0.996,
#            reg=.7, verbose=True)

# Predict on the validation set
#val_acc = (net.predict(X_val) == y_val).mean()
#print 'Validation accuracy: ', val_acc



