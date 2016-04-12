import numpy as np
import datautils


class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X, k):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in xrange(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
      t_index = distances.argsort()
      vals = np.zeros(10)
      for v in t_index:
          min_index = self.ytr[v]
          vals[min_index] += 1
          if vals.max() >= k:
              Ypred[i] = np.argmax(vals)
              break
      # min_index = np.argmin(distances) # get the index with smallest distance
      # Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred


if __name__ == '__main__':
      Xtr, Ytr, Xte, Yte = datautils.load_CIFAR10('/home/aurora/workspace/cifar/cifar-10-batches-py/') # a magic function we provide
      # flatten out all images to be one-dimensional
      Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
      Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072
      # assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
      # recall Xtr_rows is 50,000 x 3072 matrix
      total_data = np.zeros([Xtr.shape[0], Xtr_rows.shape[1]+1])
      total_data[:, :3072] = Xtr_rows
      total_data[:, 3072] = Ytr

      validation_accuracies = np.zeros([5, 7])
      for index, i in enumerate(range(5)):
          np.random.shuffle(total_data)
          Xval_rows = total_data[:1000, :3072]   # validate data set contains 1000 datas
          Yval = total_data[:1000, 3072]         # validate label set contains 1000 labels
          Xtr_rows = total_data[1000:, :3072]    # training data set contains 49000 datas
          Ytr = total_data[1000:, 3072]          # training label set contains 49000 labels

          # find hyperparameters that work best on the validation set
          # validation_accuracies = []
          for ind, k in enumerate([1, 3, 5, 10, 20, 50, 100]):
              # use a particular value of k and evaluation on validation datann = NearestNeighbor()
              nn = NearestNeighbor()
              nn.train(Xtr_rows, Ytr)
              # here we assume a modified NearestNeighbor class that can take a k as input
              Yval_predict = nn.predict(Xval_rows, k)
              acc = np.mean(Yval_predict == Yval)
              print 'floders: %f,values of k: %f, accuracy: %f' % (index, ind, acc)
              # keep track of what works on the validation set
              # validation_accuracies.append((k, acc))
              validation_accuracies[index, ind] = acc
      np.save('accuracy', validation_accuracies)