'''all magics are called in this module'''


import numpy as np

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def rel_diff(x,y):
  '''e.g.: Difference between your scores and correct scores:'''
  return np.sum(np.abs(x - y))


""""""""""""""""""""""""
''' visualize grid'''
""""""""""""""""""""""""
def show_weights(net):
  '''# Visualize the weights of the network'''
  from magics.vis_utils import visualize_grid
  import matplotlib.pyplot as plt
  W1 = net.params['W1']
  # shape (N, H, W, C)
  W1 = W1.reshape(-1, 32, 32, 3)
  print W1.shape
  plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
  plt.gca().axis('off')
  plt.show()


""""""""""""""""""""""""
''' gradients ''' 
""""""""""""""""""""""""
def net_numeric_gradient(net, X, model, y, reg):
  ''' ? need change to get rid of X, model and shit'''
  from magics.gradient_check import eval_numerical_gradient
  loss, grads = net(X, model, y, reg)
  for param_name in grads:
      param_grad_num = eval_numerical_gradient(lambda W: net(X, model, y, reg)[0], model[param_name], verbose=False)
      print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))
  
def printParams(layer):
  print layer.params
    # make this a utility function
    # print in a nice way.