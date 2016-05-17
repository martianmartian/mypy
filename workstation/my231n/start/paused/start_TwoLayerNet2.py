

#    ) )
#    ( (
#  ........
# [|      |
#  \      /
#   `----'




# A bit of setup

import numpy as np
import matplotlib.pyplot as plt
from facilities.faci_start import *


# Create some toy data to check your implementations
input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
  model = {}
  model['W1'] = np.linspace(-0.2, 0.6, num=input_size*hidden_size).reshape(input_size, hidden_size)
  model['b1'] = np.linspace(-0.3, 0.7, num=hidden_size)
  model['W2'] = np.linspace(-0.4, 0.1, num=hidden_size*num_classes).reshape(hidden_size, num_classes)
  model['b2'] = np.linspace(-0.5, 0.9, num=num_classes)
  return model

def init_toy_data():
  X = np.linspace(-0.2, 0.5, num=num_inputs*input_size).reshape(num_inputs, input_size)
  y = np.array([0, 1, 2, 2, 1])
  return X, y

model = init_toy_model()
X, y = init_toy_data()




from nets231.TwoLayerNet2 import TwoLayerNet2

scores = TwoLayerNet2(X, model)
print scores
correct_scores = [[-0.5328368, 0.20031504, 0.93346689],
 [-0.59412164, 0.15498488, 0.9040914 ],
 [-0.67658362, 0.08978957, 0.85616275],
 [-0.77092643, 0.01339997, 0.79772637],
 [-0.89110401, -0.08754544, 0.71601312]]

# the difference should be very small. We get 3e-8
print 'Difference between your scores and correct scores:'
print np.sum(np.abs(scores - correct_scores))


reg = 0.1
loss, _ = TwoLayerNet2(X, model, y, reg)
correct_loss = 1.38191

# should be very small, we get 5e-12
print 'Difference between your loss and correct loss:'
print np.sum(np.abs(loss - correct_loss))


net_numeric_gradient(TwoLayerNet2, X, model, y, reg)


from cs231n.classifier_trainer import ClassifierTrainer
model = init_toy_model()
trainer = ClassifierTrainer()
# call the trainer to optimize the loss
# Notice that we're using sample_batches=False, so we're performing Gradient Descent (no sampled batches of data)
best_model, loss_history, _, _ = trainer.train(X, y, X, y,
                                             model, two_layer_net,
                                             reg=0.001,
                                             learning_rate=1e-1, momentum=0.0, learning_rate_decay=1,
                                             update='sgd', sample_batches=False,
                                             num_epochs=100,
                                             verbose=False)
print 'Final loss with vanilla SGD: %f' % (loss_history[-1], )





# import numpy as np
# import matplotlib.pyplot as plt


# from nets231.TwoLayerNet import TwoLayerNet

# from facilities.data_utils import get_CIFAR10_data_mini
# X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data_mini()
# # X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data_mini(normalize=True)

# ''' ========= special settings =========  '''
# np.random.seed(0)
# ''' 
# done: 
#   no normalize=True
#   pass X_NxD, to acoid X.T in loss function
#   p-q, not q-p
#   X_train shape changed to : D x N

# working on:
#   dropout on weights. postpone it to next level
# '''
# ''' ========= end =========  '''
# # X_shape="DxN"

# # this value should be decided by the data... eventually
# input_size = 32 * 32 * 3
# hidden_size = 180  
# num_classes = 10
# net = TwoLayerNet(input_size, hidden_size, num_classes)

# # print X_train.shape
# # scores = net.loss(X_train)
# # print scores.shape

# # loss, _ = net.loss(X_train, y_train, reg=0.1)
# # print loss



# # Train the network
# stats = net.train(X_train, y_train, X_val, y_val,
#             num_iters=4000, batch_size=400,
#             learning_rate=1e-3, learning_rate_decay=0.98,
#             reg=.001, verbose=True, dropout_fraction = .5)

# # Plot the loss function and train / validation accuracies
# plt.subplot(2, 1, 1)
# plt.plot(stats['loss_history'])
# plt.title('Loss history')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')

# plt.subplot(2, 1, 2)
# plt.plot(stats['train_acc_history'], label='train')
# plt.plot(stats['val_acc_history'], label='val')
# plt.title('Classification accuracy history')
# plt.xlabel('Epoch')
# plt.ylabel('Clasification accuracy')
# plt.show()


# # Predict on the validation set
# val_acc = (net.predict(X_val) == y_val).mean()
# test_acc = (net.predict(X_test) == y_test).mean()

# print 'Validation accuracy: ', val_acc
# print 'Test accuracy: ', test_acc

# show_weights(net)