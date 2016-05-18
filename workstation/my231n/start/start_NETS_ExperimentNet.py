

#    ) )
#    ( (
#  ........
# [|      |
#  \      /
#   `----'


# ''' ========= special settings =========  '''
#   ''' 
#   done: 
#     no normalize=True
#     pass X_NxD, to acoid X.T in loss function
#     p-q, not q-p
#     X_train shape changed to : D x N

#   working on:
#     dropout on weights. postpone it to next level
#   '''
# ''' ========= end =========  '''



import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from facilities.faci_start import *



'''get data ready'''
from facilities.data_utils import get_CIFAR10_data_mini
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data_mini()
# X_shape="DxN"



'''get layer ready for small experiments'''
# this value should be decided by the data... eventually
input_size = 32 * 32 * 3
hidden_size = 180  
num_classes = 10
from LAYERS.NETS.ExperimentNet import TwoLayerNet
net = TwoLayerNet(input_size, hidden_size, num_classes)

# print X_train.shape
# scores = net.loss(X_train)
# print scores.shape

# loss, _ = net.loss(X_train, y_train, reg=0.1)
# print loss


# Train the network
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=400,
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







# '''tuning hyper prarameters'''
# from LAYERS.NETS.ExperimentNet import TwoLayerNet
# input_size = 32 * 32 * 3
# hidden_size = 165
# num_classes = 10
# learning_rates = [(5e-3,2000),(1e-3,1500),( 5e-2,1000)]
# regularization_strengths = [0, 1, 10]
# results ={}
# best_val = 0
# for learning_rate in learning_rates:
#     for reg in regularization_strengths:
#         print("LR",learning_rate,"reg",reg)
#         net = TwoLayerNet(input_size, hidden_size, num_classes)

#         # Train the network
#         lost_hist = net.train(X_train, y_train, X_val, y_val,
#             num_iters=700, batch_size=400,
#             learning_rate=1e-3, learning_rate_decay=0.996,
#             reg=.7, verbose=True)
#         val_acc = (net.predict(X_val) == y_val).mean()
#         print val_acc
#         results[(learning_rate[0],reg)] = val_acc
        
#         if best_val < val_acc:
#             best_val = val_acc
#             best_parameters = { 'LR':learning_rate[0], 'reg': reg}

# # store the best model into this 
# best_net = TwoLayerNet(input_size, hidden_size, num_classes)
# best_net.train(X_train, y_train, X_val, y_val,
#             num_iters=1400, batch_size=400,
#             learning_rate=best_parameters['LR'], learning_rate_decay=0.996,
#             reg=best_parameters['reg'], verbose=True)

# test_acc = (best_net.predict(X_test) == y_test).mean()
# print 'Test accuracy: ', test_acc
# show_weights(best_net)