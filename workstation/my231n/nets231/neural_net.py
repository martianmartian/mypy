import numpy as np
import matplotlib.pyplot as plt
import math, time



'''# relu = lambda x: x * (x > 0)'''
eps = 0.00001
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


  def loss(self, X, X_NxD, y=None, reg=0.0):
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

    # Compute the forward pass
    scores = None
    #############################################################################
    # Perform the forward pass, computing the class scores for the input.       #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    X1 = np.maximum(0, np.dot(W1,X)+b1)
    X2 = np.dot(W2,X1) + b2

    scores = X2
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores
   
    loss = 0.0


    # p is the correct distribution
    p = np.zeros(scores.shape) 
    p[y, range(N)] = 1

    # Compute the loss
    
    #############################################################################
    # Why mutiply the regularization loss by 0.5 ??                                             #
    #############################################################################
    scores -= np.max(scores, axis=0)
    exp_scores = pow(math.e, scores)
    q   = (exp_scores/np.sum(exp_scores, axis = 0))
    qyi = q[y, range(N)] + eps
    loss = -np.mean(np.log(qyi))
    loss += 0.5*reg*(np.sum(W1 * W1) + np.sum(W2 * W2))

    # Backward pass: compute gradients
    grads = {}

    '''equation'''
    distri_diff = p - q
    dW2 = np.dot(distri_diff, X1.T)/N - reg*W2
    db2 = distri_diff.mean(axis=1, keepdims=True)
    dX1 = np.dot(W2.T, distri_diff) * (X1>0) # relu_indi
    dW1 = np.dot(dX1, X_NxD)/N - reg*W1
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
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    
    #Dropout
    Binom_variables = np.random.choice([0, 1], size=X.shape, p=[dropout_fraction,1-dropout_fraction])
    X = (X*Binom_variables)/(1-dropout_fraction)
    
    
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)
    val_acc = 0

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):


      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      batch_indicies = np.random.choice(num_train, batch_size, replace = False)
      X_batch = X[batch_indicies]
      y_batch = y[batch_indicies]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      for variable in self.params:
          self.params[variable] -= learning_rate*grads[variable]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        val_acc = (self.predict(X_val) == y_val).mean()
        print 'iteration %d / %d: loss %f, val_acc %f' % (it, num_iters, loss, val_acc)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
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
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    X1 = np.maximum( X.dot(self.params['W1']) + self.params['b1'], 0 )
    X2 = X1.dot(self.params['W2']) + self.params['b2']
    exp_X2 = pow(math.e, X2)
    scores   = (exp_X2.T/np.sum(exp_X2, axis = 1)).T
    
    y_pred = np.argmax(scores, axis = 1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################
    return y_pred


