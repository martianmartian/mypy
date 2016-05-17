

#    ) )
#    ( (
#  ........
# [|      |
#  \      /
#   `----'

import numpy as np
import matplotlib.pyplot as plt
from facilities.faci_start import *


from nets231.TwoLayerNet import TwoLayerNet

from facilities.data_utils import get_CIFAR10_data_mini
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data_mini()
# X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data_mini(normalize=True)

''' ========= special settings =========  '''
np.random.seed(0)
''' 
done: 
  no normalize=True
  pass X_NxD, to acoid X.T in loss function
  p-q, not q-p
  X_train shape changed to : D x N

working on:
  dropout on weights. postpone it to next level
'''
''' ========= end =========  '''
# X_shape="DxN"

# this value should be decided by the data... eventually
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
            num_iters=4000, batch_size=400,
            learning_rate=1e-3, learning_rate_decay=0.98,
            reg=.001, verbose=True, dropout_fraction = .5)

# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show()


# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
test_acc = (net.predict(X_test) == y_test).mean()

print 'Validation accuracy: ', val_acc
print 'Test accuracy: ', test_acc

show_weights(net)