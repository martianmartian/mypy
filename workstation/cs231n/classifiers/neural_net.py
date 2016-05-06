import numpy as np
import matplotlib.pyplot as plt
import math, time

'''copied from assignment 2 learning'''

# Helpers
relu = lambda x: x * (x > 0)

def init_two_layer_model(input_size, hidden_size, output_size):
  """
  Initialize the weights and biases for a two-layer fully connected neural
  network. The net has an input dimension of D, a hidden layer dimension of H,
  and performs classification over C classes. Weights are initialized to small
  random values and biases are initialized to zero.

  Inputs:
  - input_size: The dimension D of the input data
  - hidden_size: The number of neurons H in the hidden layer
  - ouput_size: The number of classes C

  Returns:
  A dictionary mapping parameter names to arrays of parameter values. It has
  the following keys:
  - W1: First layer weights; has shape (D, H)
  - b1: First layer biases; has shape (H,)
  - W2: Second layer weights; has shape (H, C)
  - b2: Second layer biases; has shape (C,)
  """
  # initialize a model
  model = {}
  model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model

def two_layer_net(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradients for a two layer fully connected neural network.
  The net has an input dimension of D, a hidden layer dimension of H, and
  performs classification over C classes. We use a softmax loss function and L2
  regularization the the weight matrices. The two layer net should use a ReLU
  nonlinearity after the first affine layer.

  The two layer net has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each
  class.

  Inputs:
  - X: Input data of shape (N, D). Each X[i] is a training sample.
  - model: Dictionary mapping parameter names to arrays of parameter values.
    It should contain the following:
    - W1: First layer weights; has shape (D, H)
    - b1: First layer biases; has shape (H,)
    - W2: Second layer weights; has shape (H, C)
    - b2: Second layer biases; has shape (C,)
  - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
  - reg: Regularization strength.

  Returns:
  If y not is passed, return a matrix scores of shape (N, C) where scores[i, c]
  is the score for class c on input X[i].

  If y is not passed, instead return a tuple of:
  - loss: Loss (data loss and regularization loss) for this batch of training
    samples.
  - grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function. This should have the same keys as model.
  """

  # unpack variables from the model dictionary
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, D = X.shape

  # compute the forward pass
  scores = None
  #############################################################################
  # Perform the forward pass, computing the class scores for the input.       #
  # Store the result in the scores variable, which should be an array of      #
  # shape (N, C).                                                             #
  #############################################################################
  layer1 = X.dot(W1)+b1
  layer2 = relu(layer1)
  layer3 = layer2.dot(W2) + b2

  scores = layer3 # no need to make two variables, but more readable this way. #relu(X.dot(W1)+b1).dot(W2) + b2
    
  # If the targets are not given then jump out, we're done
  if y is None:
    return scores

  #############################################################################
  # Compute Softmax classifier loss with L2 regularization                    #
  # log and e cancel out, sum splits                                          #
  # softmax loss is -sum(correct scores) + log(sum(e^[all scores (for each    #
  #  respective example)]))                                                   #
  #############################################################################
  rows = np.sum(np.exp(layer3), axis=1)
  layer4 = np.sum(-layer3[range(N), y]+np.log(rows) ) / N
  # loss = np.sum(-scores[range(N), y]+np.log(np.sum(np.exp(scores), axis=1)) ) / N
  # loss += 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2))
  loss = layer4 + 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2))

  # compute the gradients
  grads = {}

  #############################################################################
  # Compute the backward pass, computing the derivatives of the weights       #
  # and biases. Store the results in the grads dictionary. For example,       #
  # grads['W1'] should store the gradient on W1, and be a matrix of same size #
  #############################################################################

  dlayer4 = 1.0

  # Gradient of np.log(np.sum(np.exp(layer3), axis=1))
  dlayer3 = (np.exp(layer3).T / rows).T
  # Gradient of -layer3[range(N), y]:
  ys = np.zeros(dlayer3.shape)
  ys[range(N), y] = 1
  dlayer3 -= ys
  # / N term
  dlayer3 /= N
  # Chain rule
  dlayer3 *= dlayer4

  # Chain rule, element-wise multiplication works out
  dlayer2 = dlayer3.dot(W2.T)

  # Relu gradient
  dlayer1 = dlayer2 * (layer1>=0)

  dW1 = X.T.dot(dlayer1)
  dW2 = layer2.T.dot(dlayer3)

  # Same as matrix multiplication with 1-vector, chain rule works out
  db1 = np.sum(dlayer1, axis=0)
  db2 = np.sum(dlayer3, axis=0)

  # Regularization
  dW1 += reg * W1
  dW2 += reg * W2

  # Store
  grads['W1'] = dW1
  grads['W2'] = dW2
  grads['b1'] = db1
  grads['b2'] = db2

  return loss, grads



'''from first time learning'''
class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.random.random(hidden_size)
    # self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    # self.params['b2'] = np.zeros(output_size)
    self.params['b2'] = np.random.random(output_size)
    

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape  # NxD
    C = b2.shape[0]

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    # Values of the hidden layer

    X1 = np.maximum(0, X.dot(W1)+b1)
    X2 = X1.dot(W2) + b2

    scores = X2
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores
   
    loss = 0.0

    # Creating a output matrix from output vector X2
    # p is the correct distribution
    p = np.zeros(scores.shape) 
    p[range(N),y] = 1

    # Compute the loss
    
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5 ??                                             #
    #############################################################################
    
    #Exponentiating the X2 , then normalizing the exponentiated value
    '''# subtract max here?'''
    exp_scores = pow(math.e, scores) # NxC
    q   = (exp_scores.T/np.sum(exp_scores, axis = 1)).T
    qyi = q[range(N),y]
    qyi += 0.00001
    loss = -np.mean(np.log(qyi))
    loss += 0.5*reg*np.sum(W1 * W1) + 0.5*reg*np.sum(W2 * W2)
    # print loss


    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################

    #Back propagation

    dX2 = q - p
    dW2 = ((dX2.T).dot(X1)).T/N + reg*W2
    db2 = (dX2.T).dot(np.ones(X1.shape[0]))/N
    dX1 = dX2.dot(W2.T)
    dW1 = (((dX1*(X1>0)).T).dot(X)).T/N + reg*W1
    db1 = (((dX1*(X1>0)).T).dot(np.ones(X.shape[0]))).T/N
      
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W1'] = dW1
    grads['b1'] = db1
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

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


