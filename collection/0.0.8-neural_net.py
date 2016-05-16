bulk part is in other files....



import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data, get_CIFAR10_data_mini
from cs231n.solver import Solver
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array



def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

''' _mini is to load one batch, smaller for testing'''
# # data = get_CIFAR10_data()
# # data = get_CIFAR10_data(num_training=49, num_validation=1, num_test=1)
# # data = get_CIFAR10_data_mini(num_training=49, num_validation=1, num_test=1)
# data = get_CIFAR10_data_mini()
# # for k, v in data.iteritems():
# #   print '%s: ' % k, v.shape

# # # '''# # Test the Solver and being able to overfit 50 training examples'''
# # num_train = 50
# # small_data = {
# #   'X_train': data['X_train'][:num_train],
# #   'y_train': data['y_train'][:num_train],
# #   'X_val': data['X_val'],
# #   'y_val': data['y_val'],
# # }

# weight_scale = 1e-2
# learning_rate = 1e-2
# model = FullyConnectedNet([100, 100],
#               weight_scale=weight_scale, dtype=np.float64)
# solver = Solver(model, small_data,
#                 print_every=10, num_epochs=20, batch_size=25,
#                 update_rule='sgd',
#                 optim_config={
#                   'learning_rate': learning_rate,
#                 }
#          )
# solver.train()

# plt.plot(solver.loss_history, 'o')
# plt.title('Training loss history')
# plt.xlabel('Iteration')
# plt.ylabel('Training loss')
# plt.show()


# # # '''# # Test five-layer Net overfit 50 training examples'''
# num_train = 50
# small_data = {
#   'X_train': data['X_train'][:num_train],
#   'y_train': data['y_train'][:num_train],
#   'X_val': data['X_val'],
#   'y_val': data['y_val'],
# }

# learning_rate = 2e-2
# weight_scale = 4e-2
# model = FullyConnectedNet([100, 100, 100, 100],
#                 weight_scale=weight_scale, dtype=np.float64)
# solver = Solver(model, small_data,
#                 print_every=10, num_epochs=20, batch_size=25,
#                 update_rule='sgd',
#                 optim_config={
#                   'learning_rate': learning_rate,
#                 }
#          )
# solver.train()

# plt.plot(solver.loss_history, 'o')
# plt.title('Training loss history')
# plt.xlabel('Iteration')
# plt.ylabel('Training loss')
# plt.show()


# '''## finding the best model'''
# best_model = None
# best_acc = 0
# best_params = []
# for lr in [1e-3]:
#     for lr_decay in [0.9]:
#         model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2,use_batchnorm=False)
#         solver = Solver(model,data,update_rule='adam',optim_config={'learning_rate':lr},
#                         lr_decay=lr_decay,num_epochs=10,batch_size=200,print_every=100
#                        )
#         solver.train()
#         val_acc = solver.check_accuracy(data['X_val'], data['y_val'])
#         if val_acc>best_acc:
#             best_acc = val_acc
#             best_model = model
#             best_params = [lr, lr_decay]

# print "best_acc = ", best_acc
# print "best_params = ", best_params
# y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
# y_val_pred = np.argmax(best_model.loss(data['X_val']), axis=1)
# print 'Validation set accuracy: ', (y_val_pred == data['y_val']).mean()
# print 'Test set accuracy: ', (y_test_pred == data['y_test']).mean()


'''testing modularized functions'''
# '''# # Test the affine_forward function'''
# num_inputs = 2
# input_shape = (4, 5, 6)
# output_dim = 3
# input_size = num_inputs * np.prod(input_shape)
# weight_size = output_dim * np.prod(input_shape)
# x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
# w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
# b = np.linspace(-0.3, 0.1, num=output_dim)
# out, _ = affine_forward(x, w, b)
# # correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
# #                         [ 3.25553199,  3.5141327,   3.77273342]])

# '''# # Test the affine_backward function'''
# x = np.random.randn(10, 2, 3)
# w = np.random.randn(6, 5)
# b = np.random.randn(5)
# dout = np.random.randn(10, 5)
# dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
# dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
# db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)
# _, cache = affine_forward(x, w, b)
# dx, dw, db = affine_backward(dout, cache)
# # The error should be around 1e-10
# print 'Testing affine_backward function:'
# print 'dx error: ', rel_error(dx_num, dx)
# print 'dw error: ', rel_error(dw_num, dw)
# print 'db error: ', rel_error(db_num, db)

# '''# # Test the relu_backward function'''
# x = np.random.randn(10, 10)
# dout = np.random.randn(*x.shape)
# dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)
# _, cache = relu_forward(x)
# dx = relu_backward(dout, cache)
# # The error should be around 1e-12
# print 'Testing relu_backward function:'
# print 'dx error: ', rel_error(dx_num, dx)


# '''# # Test the affine_relu_forward, affine_relu_backward function'''
# from cs231n.layer_utils import affine_relu_forward, affine_relu_backward
# x = np.random.randn(2, 3, 4)
# w = np.random.randn(12, 10)
# b = np.random.randn(10)
# dout = np.random.randn(2, 10)
# out, cache = affine_relu_forward(x, w, b)
# dx, dw, db = affine_relu_backward(dout, cache)

# # dx_num = eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0], x, dout)
# # dw_num = eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0], w, dout)
# # db_num = eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0], b, dout)
# # print 'Testing affine_relu_forward:'
# # print 'dx error: ', rel_error(dx_num, dx)
# # print 'dw error: ', rel_error(dw_num, dw)
# # print 'db error: ', rel_error(db_num, db)


# # '''# # Test the Softmax and SVM function'''
# num_classes, num_inputs = 10, 50
# x = 0.001 * np.random.randn(num_inputs, num_classes)
# y = np.random.randint(num_classes, size=num_inputs)

# dx_num = eval_numerical_gradient(lambda x: svm_loss(x, y)[0], x, verbose=False)
# loss, dx = svm_loss(x, y)
# # Test svm_loss function. Loss should be around 9 and dx error should be 1e-9
# print 'Testing svm_loss:'
# print 'loss: ', loss
# print 'dx error: ', rel_error(dx_num, dx)
# dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
# loss, dx = softmax_loss(x, y)
# # Test softmax_loss function. Loss should be 2.3 and dx error should be 1e-8
# print '\nTesting softmax_loss:'
# print 'loss: ', loss
# print 'dx error: ', rel_error(dx_num, dx)


# '''training loss'''
# N, D, H, C = 3, 5, 50, 7
# std=0.01
# model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)

# model.params['W1'] = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
# model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
# model.params['W2'] = np.linspace(-0.3, 0.4, num=H*C).reshape(H, C)
# model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
# X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
# y = np.asarray([0, 5, 1])
# model.reg = 1.0
# loss, grads = model.loss(X, y)





