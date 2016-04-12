import numpy as np
import tutorial_2.svm as t_svm
import datautils

def optimaize_1(X_train, Y_train):
    # assume X_train is the data where each column is an example (e.g. 3073 x 50,000)
    # assume Y_train are the labels (e.g. 1D array of 50,000)
    # assume the function L evaluates the loss function
    bestW = np.zeros((10, 3073))
    bestloss = float("inf")
    for num in xrange(1000):
        W = np.random.randn(10, 3073)*0.0001
        loss = t_svm.L(X_train, Y_train, W)
        if loss < bestloss:
            bestloss = loss
            bestW = W
        print 'in attempt %d the loss was %f, best %f' % (num, loss, bestloss)
    return bestW


def optimaize_2(Xtr_cols, Ytr):
    W = np.random.randn(10, 3073) * 0.001 # generate random starting W
    bestloss = float("inf")
    for i in xrange(1000):
        step_size = 0.0001
        Wtry = W + np.random.randn(10, 3073) * step_size
        loss = t_svm.L(Xtr_cols, Ytr, Wtry)
        if loss < bestloss:
            W = Wtry
            bestloss = loss
        print 'iter %d loss is %f' % (i, bestloss)
    return W


def eval_numerical_gradient(f, x):
  """
  a naive implementation of numerical gradient of f at x
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

  h = 1e-5
  # fx = f(x+h) # evaluate function value at original point
  grad = np.zeros(x.shape)


  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    # evaluate function at x+h
    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value + h # increment by h
    fxh = f(x) # evalute f(x + h)
    x[ix] = old_value - h
    fxl = f(x) # evalute f(x - h)
    x[ix] = old_value # restore to previous value (very important!)
    # compute the partial derivative
    grad[ix] = (fxh - fxl) / 2*h # the slope
    it.iternext() # step to next dimension
  return grad


def optimization_3():
    W = np.random.rand(10, 3073) * 0.001 # random weight vector
    df = eval_numerical_gradient(t_svm.CIFAR10_loss_fun, W) # get the gradient

    loss_original = t_svm.CIFAR10_loss_fun(W) # the original loss
    print 'original loss: %f' % (loss_original, )

    # lets see the effect of multiple step sizes
    for step_size_log in [-10, -9, -8, -7, -6, -5,-4,-3,-2,-1]:
        step_size = 10 ** step_size_log
        W_new = W - step_size * df # new position in the weight space
        loss_new = t_svm.CIFAR10_loss_fun(W_new)
        print 'for step size %f new loss: %f' % (step_size, loss_new)



def predictive(Xtest, Yte, W):
    loss_fun = np.dot(Xtest, W.T)
    # print loss_fun.shape
    predict = np.argmax(loss_fun, axis=1)
    # print predict.shape
    flag = predict == Yte
    print np.mean(flag)

if __name__ == '__main__':
    # # data pre_process
    # Xtr, Ytr, Xte, Yte = datautils.load_CIFAR10('/home/aurora/workspace/cifar/cifar-10-batches-py/') # a magic function we provide
    # # flatten out all images to be one-dimensional
    # Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
    # Xtr_means = np.mean(Xtr_rows, axis=0)
    #
    # Xtr_rows -= Xtr_means
    # Xtr_rows /= 127.0
    # # print Xtr_rows
    # Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072
    # Xtr_totals = np.ones((Xtr_rows.shape[0], Xtr_rows.shape[1] + 1))
    # Xtr_totals[:, :Xtr_totals.shape[1]-1] = Xtr_rows

    # print Xtr_totals
    optimization_3()

    # Xte_total = np.ones((Xte_rows.shape[0], Xte_rows.shape[1]+1))
    # Xte_total[:, :Xte_total.shape[1]-1] = Xte_rows
    # predictive(Xte_total, Yte, bestW)
