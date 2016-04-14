import numpy as np

class Identity(object):
    def __init__(self):
        pass

    def forward(self, x):
        return x

    def backward(self, y, target):
        return y - target

    def loss(self, y, target):
        return np.sum( np.square(y - target) ) / 2.0

    def error(self, y, target):
        return self.loss(y, target)

    def __test(self):
        '''
        >>> x = np.log( np.array([[1, 2, 12], [1, 6, 4]]) )
        >>> t = np.array([[1, 0, 1], [0, 1, 0]])
        >>> f = Identity()
        >>> y = f.forward( x )
        >>> print [['%.2f'%_ for _ in v] for v in y]
        [['0.00', '0.69', '2.48'], ['0.00', '1.79', '1.39']]
        >>> d = f.backward(y, t)
        >>> print [['%.2f'%_ for _ in v] for v in d]
        [['-1.00', '0.69', '1.48'], ['0.00', '0.79', '1.39']]
        >>> l = f.loss(y, t)
        >>> print '%.2f'%l
        3.12
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
