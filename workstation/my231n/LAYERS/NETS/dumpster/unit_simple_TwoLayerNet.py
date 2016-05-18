import numpy as np
import matplotlib.pyplot as plt
import math, time



'''# relu = lambda x: x * (x > 0)'''
eps = 0.0001
class TwoLayerNet(object):
  """ 
    ------------
    Name: Two-layer fully-connected neural network
    Functions: Performs classification over C classes
    Architecture:
              input - fully connected layer - ReLU - fully connected layer - softmax
    ------------
    Dimensions: (default for X_shape=="DxN")
        # Input: N            e.g: ? (D x N)
        Hidden layer: H       e.g: (H x D)
        Output layer: C       e.g: (C x H)
    Regularization: 
        L2 on weight matrices
    Activation function: 
        first fully connected layer: ReLU
    Loss function: Softmax
  """

  def __init__(self, input_size, hidden_size, num_classes, std=1e-4):
    """
      ------------
      Initialize:
      Dictionary:
        Weights: small random values
        Biases: random ?zero 

      self.params:  (default for X_shape=="DxN")
        W1: First layer weights; has shape (H, D)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (C, H)
        b2: Second layer biases; has shape (C,)
    """
    # X_shape="DxN"
    self.params = {}
    self.params['W1'] = std * np.random.randn(hidden_size , input_size)
    self.params['b1'] = np.zeros((hidden_size,1))
    # self.params['b1'] = np.random.random(hidden_size)
    self.params['W2'] = std * np.random.randn(num_classes , hidden_size)
    self.params['b2'] = np.zeros((num_classes,1))
    # self.params['b2'] = np.random.random(num_classes)


  def loss(self, X, y=None, reg=0.0):
    """
      ------------
      Inputs:
      - X: (N, D). Each X[i] is a training sample.
      - y: (C, ) Vector of training labels. y[i] is the label for X[i], and each y[i] is
        an integer in the range 0 <= y[i] < C. 
      - reg: Regularization strength.
      ------------
      Returns:
      If y is None:
          return a matrix scores of shape (N, C)
      If y is not None:
          - loss: ...
          - grads: ...
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    D, N = X.shape
    C = b2.shape[0]

    '''# Compute the forward pass'''
    scores = None

    X1 = np.maximum(0, np.dot(W1,X)+b1)
    X2 = np.dot(W2,X1) + b2

    scores = X2
    
    if y is None:
      return scores
   
    loss = 0.0


    # p is the correct distribution
    p = np.zeros(scores.shape) 
    p[y, range(N)] = 1

    '''# Compute the loss'''
    
    scores -= np.max(scores, axis=0)
    exp_scores = pow(math.e, scores)
    q   = (exp_scores/np.sum(exp_scores, axis = 0))
    qyi = q[y, range(N)] + eps
    loss = -np.mean(np.log(qyi))
    loss += 0.5*reg*(np.sum(W1 * W1) + np.sum(W2 * W2))
    # Why mutiply the regularization loss by 0.5 ??  

    '''# Backward pass: compute gradients'''
    grads = {}

    # print 'X1[0:2,0:20]'
    # print X1[0:2,0:20]
    # print "=========="

    '''equation dW1 = (p-q) x W2 x indi x X.T'''
    distri_diff = p - q

    dW2 = np.dot(distri_diff, X1.T)/N - reg*W2
    db2 = distri_diff.mean(axis=1, keepdims=True)
    dX1 = np.dot(W2.T, distri_diff)
    dF1 = dX1*(X1>0)
    dW1 = np.dot(dF1, X.T)/N - reg*W1
    db1 = dX1.mean(axis=1, keepdims=True)
    # print db1[0:10]   0 ????
    # print db1.max()
    # '''attention: db1 is calculated after dX1 and relu_indi mutiplication
    #   the original version was before. see what's different''
      
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W1'] = dW1
    grads['b1'] = db1

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False ,dropout_fraction = 0):
    """
      SGD.
      Inputs:
      - learning_rate: Scalar giving learning rate for optimization.
      - learning_rate_decay: Scalar to decay the learning rate after each epoch.
      - reg: Scalar: regularization strength.
      - num_iters: iteration
      - batch_size: minibatch size
      - verbose: true: print progress
    """
    
    # #Dropout on X... 
    # '''change this to dropout on weights ...... '''
    # Binom_variables = np.random.choice([0, 1], size=X.shape, p=[dropout_fraction,1-dropout_fraction])
    # X = (X*Binom_variables)/(1-dropout_fraction)
    
    num_train = X.shape[1]
    iterations_per_epoch = max(num_train / batch_size, 1)

    val_acc = 0
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      '''minibatch'''
      batch_indicies = np.random.choice(num_train, batch_size, replace = False)
      X_batch = X[:,batch_indicies]
      y_batch = y[batch_indicies]

      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      for variable in self.params:
          self.params[variable] += learning_rate*grads[variable]  # this for p - q

      if verbose and it % 100 == 0:
        val_acc = (self.predict(X_val) == y_val).mean()
        print 'iteration %d / %d: loss %f, val_acc %f' % (it, num_iters, loss, val_acc)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    y_pred = None
    X1 = np.maximum(0, np.dot(self.params['W1'],X)+self.params['b1'])
    X2 = np.dot(self.params['W2'],X1) + self.params['b2']
    scores = X2
    scores -= np.max(scores, axis=0)
    exp_scores = pow(math.e, scores)
    q   = (exp_scores/np.sum(exp_scores, axis = 0))
    y_pred = np.argmax(q, axis = 0)
    return y_pred


