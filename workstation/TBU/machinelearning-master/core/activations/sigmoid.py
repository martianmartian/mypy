import numpy as np

class Sigmoid(object):
    def __init__(self):
        self.eps = 1e-6

    def forward(self, x):
        return 1. / (1. + np.exp(-x))

    def backward(self, y, target):
        return y - target

    def loss(self, y, target):
        return - np.sum( np.log(y+self.eps) * target + np.log(1.-y+self.eps) * (1.0 - target) )

    def error(self, y, taget):
        batch_size = np.prod( y.shape )
        y_dot = y.reshape( (batch_size, ) )
        t_dot = target.reshape( (batch_size, ) )
        return batch_size - np.sum( [(1.0 if t_dot[i] >= 0.5 and y_dot[i] >= 0.5 else 0.0) for i in range(batch_size)] )

    def __test(self):
        '''
        >>> x = np.log( np.array([[1, 2, 12], [1, 6, 4]]) )
        >>> t = np.array([[1, 0, 1], [0, 1, 0]])
        >>> f = Sigmoid()
        >>> y = f.forward( x )
        >>> print [['%.2f'%_ for _ in v] for v in y]
        [['0.50', '0.67', '0.92'], ['0.50', '0.86', '0.80']]
        >>> d = f.backward(y, t)
        >>> print [['%.2f'%_ for _ in v] for v in d]
        [['-0.50', '0.67', '-0.08'], ['0.50', '-0.14', '0.80']]
        >>> l = f.loss(y, t)
        >>> print '%.2f'%l
        4.33
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
