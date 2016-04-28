import numpy as np


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
        num_train = X.shape[0]

        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # Compute the forward pass
        # 1. first layer scores
        scores1 = X.dot(W1) + b1

        # 2. ReLU for the above scores
        scores1_relu = np.maximum(0, scores1)

        # 3. output layer scores
        scores2 = scores1_relu.dot(W2) + b2

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores2

        # Compute the loss
        # apply numeric stability trick
        scores2_stable = scores2 - np.max(scores2, axis=1, keepdims=True)
        scores2_stable_exp = np.exp(scores2_stable)
        scores2_stable_exp_sum = np.sum(scores2_stable_exp, axis=1, keepdims=True)

        # softmax scores (sum==1)
        scores2_softmax = scores2_stable_exp / scores2_stable_exp_sum

        loss = - np.sum(np.log(scores2_softmax[range(num_train), y]))
        loss /= num_train
        loss += 0.5 * reg * np.sum(W2 * W2)
        loss += 0.5 * reg * np.sum(W1 * W1)

        # Backward pass: compute gradients
        correct_softmax_scores = np.zeros_like(scores2_softmax)
        correct_softmax_scores[range(num_train), y] = 1
        dloss = scores2_softmax - correct_softmax_scores  # dL/dz = dL/dy * dy/dz (y = scores2_softmax; z = scores2)

        dW2 = scores1_relu.T.dot(dloss)
        dW2 /= num_train
        dW2 += W2 * reg

        db2 = np.sum(dloss, axis=0)
        db2 /= num_train

        dscores1_relu = dloss.dot(W2.T)

        drelu_local = np.zeros_like(scores1)
        drelu_local[scores1 > 0] = 1

        dscore1 = dscores1_relu * drelu_local

        dW1 = X.T.dot(dscore1)
        dW1 /= num_train
        dW1 += W1 * reg

        db1 = np.sum(dscore1, axis=0)
        db1 /= num_train

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5,
              sgd_momentum_decay=0.98,
              num_iters=100, batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data. (not used for actual training)
        - y_val: A numpy array of shape (N_val,) giving validation labels. (not used for actual training)
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

        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        # SGD momentum
        sgd_momentum_mu = 0.5   # friction
        dW1v = np.zeros_like(W1)
        db1v = np.zeros_like(b1)
        dW2v = np.zeros_like(W2)
        db2v = np.zeros_like(b2)

        for it in range(num_iters):
            batch_choice = np.random.choice(num_train, batch_size, replace=False)
            X_batch = X[batch_choice]
            y_batch = y[batch_choice]

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            dW1, db1 = grads['W1'], grads['b1']
            dW2, db2 = grads['W2'], grads['b2']

            # update weights using momentum
            dW1v = dW1v * sgd_momentum_mu - learning_rate * dW1
            db1v = db1v * sgd_momentum_mu - learning_rate * db1
            dW2v = dW2v * sgd_momentum_mu - learning_rate * dW2
            db2v = db2v * sgd_momentum_mu - learning_rate * db2

            W1 += dW1v
            b1 += db1v
            W2 += dW2v
            b2 += db2v

            # vanilla SGD
            # W1 += - learning_rate * dW1
            # b1 += - learning_rate * db1
            # W2 += - learning_rate * dW2
            # b2 += - learning_rate * db2

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay
                sgd_momentum_mu = min(0.99, sgd_momentum_mu / sgd_momentum_decay)

                # if verbose:
                #     print('epoch %d: train_acc %f: val_acc %f' % (it / iterations_per_epoch, train_acc, val_acc))

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
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # 1. first layer scores
        scores1 = X.dot(W1) + b1

        # 2. ReLU for the above scores
        scores1_relu = np.maximum(0, scores1)

        # 3. output layer scores
        scores2 = scores1_relu.dot(W2) + b2

        y_pred = np.argmax(scores2, axis=1)

        return y_pred
