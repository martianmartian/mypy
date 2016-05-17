import numpy as np

eps = 0.0001
class Affine(object):
  """
  Basic definition of the type Affine
  X_shape = N x D
  
  """
  def __init__(self, K=1, D=1, std=1e-3):
    """
      ------------
      Initialize:
      Dictionary:
        Weights: small random values
        Biases: random ?zero 

      self.params:  (default for X_shape==(N x D))
        K: number of neurons (sometimes is C )
        D: number of weights of each neuron (i.e: Dimensions)

      rawGradients: if you use the neuron, they will always have the tendency to change.
    """
    self.params = {}
    self.params['W'] = std * np.random.randn(K , D)
    self.params['b'] = np.zeros((K, 1)) # + eps
    # self.params['b1'] = np.random.random((K, 1))

    # self.rawGradients = None
    # self.rawGradients = X.mean(axis=0, keepdims=True)
    self.cache = None

  def forward_1(self, X):
    # '''all X are already straightened into N x D.'''
    WT = self.params['W'].T
    b = self.params['b'].T
    out = np.dot(X, WT) + b

    self.cache = (X, out)
    return out

  def backward_1(self, dout=None, reg=0.001):
    '''buggy'''
    if(dout == None):
      W = self.params['W']
      X, out = self.cache
      dW = X.mean(axis=0, keepdims=True) - reg*W
      self.cache = None
      return dW

    else:
      W = self.params['W']
      X, out = self.cache
      N = X.shape[0]
      dW = np.dot(dout, X) / N


  # def forward_2_to_1(self, X):    
  #   # '''straightened X from N x (C x H x W) into N x D => D x N first'''
  #   '''buggy, also redudent procedure..?'''
  #   W = self.params['W']
  #   X = X.reshape(X.shape[0],-1).T
  #   out = np.dot(W, X)
  #   return out

  # def forward(self, x, w, b):
  #   out = None
  #   N = x.shape[0]
  #   out = x.reshape(N, np.prod(x.shape[1:])).dot(w)+b

  #   cache = (x, w, b)
  #   return out, cache
