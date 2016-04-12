import numpy as np


""" Vanilla Dropout: Not recommended implementation (see notes below) """

p = 0.5 # probability of keeping a unit active. higher = less dropout

def train_step(X):
  """ X contains the data """

  # forward pass for example 3-layer neural network
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  U1 = np.random.rand(*H1.shape) < p # first dropout mask
  H1 *= U1 # drop!
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = np.random.rand(*H2.shape) < p # second dropout mask
  H2 *= U2 # drop!
  out = np.dot(W3, H2) + b3

  # backward pass: compute gradients... (not shown)
  # perform parameter update... (not shown)

def predict(X):
  # ensembled forward pass
  H1 = np.maximum(0, np.dot(W1, X) + b1) * p # NOTE: scale the activations
  H2 = np.maximum(0, np.dot(W2, H1) + b2) * p # NOTE: scale the activations
  out = np.dot(W3, H2) + b3

if __name__=='__main__':
    data = np.random.rand(5, 3)
    cov = np.dot(data.T, data)/float(data.shape[0])
    U, S, D = np.linalg.svd(cov)
    # pca
    Xrot = np.dot(data, U)
    # whitening
    # The geometric interpretation of this transformation is that if the input data is a multivariable gaussian,
    # then the whitened data will be a gaussian with zero mean and identity covariance matrix.
    Xwhite = Xrot / np.sqrt(S + 1e-5)