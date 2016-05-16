import numpy as np
import pytest

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
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

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
    N, D = X.shape
    H, C = W2.shape

    weights = {
      'W1': W1,
      'W2': W2,
      'b1': b1,
      'b2': b2,
    }

    # Make properly shaped zeroed out template gradients
    grads = {}
    for k, W in weights.iteritems():
      grads[k] = np.zeros(W.shape)

    # Compute the forward pass
    scores = None
    #############################################################################
    # Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    ############################################################################

    first_matmul_out = np.matmul(X, W1) + b1 # shape (N, H)
    relus_out = np.maximum(first_matmul_out, np.zeros(first_matmul_out.shape)) # shape (N, H)
    scores = np.matmul(relus_out, W2) + b2 # shape (N, C)


    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = 0.0
    #############################################################################
    # Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################



    f = scores # shape = (N, C)

    # shift the values of f so that the highest number is 0 in any row:
    f -= np.max(f, axis=1)[:, np.newaxis] # shape (N, C)

    exp_scores = np.exp(f) # shape = (N, C)

    # Exponentiate and normalize to get confidence in each class
    p = exp_scores / np.sum(exp_scores, axis=1)[:, np.newaxis] # shape (N, C)
    probs = p # for pdb

    # We used the shift trick to get everything to small values
    # assert np.any(np.isnan(p)) == False

    correct_classes = [np.arange(y.shape[0]), y] # shape = (2, N)
    correct_class_scores = p[correct_classes] # shape (N)

    correct_class_scores = correct_class_scores[:, np.newaxis] # shape (N, 1)



    # This is np.log(0.0) when we're very confident in the wrong answer.
    # For numeric stability, I cap it at an arbitrary value
    correct_class_scores_maxed = np.maximum(correct_class_scores, 1e-20)

    # loss = - np.sum(np.log(correct_class_scores))


    loss = - np.sum(np.log(correct_class_scores_maxed))



    # Right now the loss and dW are sums over all training examples, but we
    # want them to be an average instead so we divide by N.
    loss /= N

    # Add regularization to the loss and the gradient.
    # TODO: check if broken

    reg_loss = 0.5 * reg * sum([
      np.sum(np.square(W))
      for W in weights.values()
    ])

    loss += reg_loss

    if np.any(correct_class_scores == 0):
      pytest.set_trace()


    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    #############################################################################
    # Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################

    # Update the gradients
    dscores = p
    dscores[correct_classes] -= 1 # Subtract 1 from all the correct class ps
    # dscores is now correct (N, C)


    # Local circuit:
    #
    #     relus_out (N, H) => dW2 (H, C) => dscores (N, C)
    #
    dW2s = np.matmul(relus_out[:, :, np.newaxis], dscores[:, np.newaxis, :]) # shape (N, H, C)
    grads['W2'] += np.sum(dW2s, axis=0)

    grads['b2'] += np.sum(dscores, axis=0)

    # Relus function
    # R(x) = x if (x > 0), else 0
    # dR(x)/dx = 1 if (x > 0), else 0
    #
    # Local circuit:
    #
    #     drelus_out (N, H) => dscores (N, C) => W2 (H, C)
    #


    drelus_out = np.matmul(dscores, W2.T) # shape (N, H)

    # Zero-out relu values that are below zero
    drelus_in = drelus_out * (relus_out > 0) # shape (N, H)

    # Local circuit:
    #
    #     X (N, D) => dW1 (D, H) => drelus_in (N, H)
    #

    dW1s = np.matmul(X[:, :, np.newaxis], drelus_in[:, np.newaxis, :]) # shape (N, D, H)
    grads['W1'] += np.sum(dW1s, axis=0)

    grads['b1'] += np.sum(drelus_in, axis=0)

    # For each weight Wij, the partial derivative dWij of the loss function above is:
    #     0.5 * reg * 2.0 * Wij
    #
    for key in grads.keys():
      grads[key] /= N
      grads[key] += reg * weights[key] # w.g dW2 += reg*W2

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
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
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.params
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################

      batch_indices = np.random.choice(range(num_train), batch_size, replace=True)
      X_batch = X[batch_indices]
      y_batch = y[batch_indices]

      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################

      for key in self.params.keys():
        self.params[key] -= grads[key] * learning_rate

      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

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
    # Implement this function; it should be VERY simple!                #
    ###########################################################################

    scores = self.loss(X) # shape (N, C)
    return np.argmax(scores, axis=1) # shape (N,)

    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred
